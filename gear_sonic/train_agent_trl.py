#!/usr/bin/env python3
# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Fix sys.path: when running as `python gear_sonic/train_agent_trl.py`, Python adds
# gear_sonic/ to sys.path[0], causing `from trl import ...` to resolve to our local
# gear_sonic/trl/ instead of the HuggingFace trl package. Replace with repo root.
import os
import sys

_script_dir = os.path.dirname(os.path.abspath(__file__))
_repo_root = os.path.dirname(_script_dir)
if _script_dir in sys.path:
    sys.path.remove(_script_dir)
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

try:
    import isaaclab  # noqa: F401
except ImportError:
    print(
        "\n"
        "ERROR: Isaac Lab is required for training but not installed.\n"
        "\n"
        "Isaac Lab is not a pip dependency — it must be installed separately.\n"
        "Follow the official guide:\n"
        "  https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html\n"
        "\n"
        "After installing, activate the Isaac Lab conda/venv environment\n"
        "before running this script.\n"
    )
    sys.exit(1)

import glob
import logging
import math
import os
from pathlib import Path
import re
import sys
import time

from filelock import FileLock, Timeout
import hydra
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from loguru import logger
from omegaconf import DictConfig, OmegaConf, open_dict
import wandb
import yaml

from gear_sonic.trl.utils import logger_utils
from gear_sonic.trl.utils.common import (
    custom_instantiate,
    distributed_stage,
    get_filtered_state_dict,
    materialize_lazy_params,
    verify_distributed_model_signatures,
    wandb_run_exists,
)
from gear_sonic.utils.common import seeding
from gear_sonic.utils.config_utils import register_rl_resolvers
from gear_sonic.utils.obs_utils import get_group_term_obs_shape

register_rl_resolvers()

DEFAULT_APP_LAUNCHER_LOCK_TIMEOUT_SECONDS = 1800.0
DEFAULT_DISTRIBUTED_TIMEOUT_SECONDS = 600.0


def _get_app_launcher_lock_timeout_seconds():
    raw_timeout = os.environ.get(
        "ISAACLAB_APP_LAUNCHER_LOCK_TIMEOUT_SECONDS",
        str(DEFAULT_APP_LAUNCHER_LOCK_TIMEOUT_SECONDS),
    )
    try:
        timeout_seconds = float(raw_timeout)
    except ValueError as exc:
        raise ValueError(
            "ISAACLAB_APP_LAUNCHER_LOCK_TIMEOUT_SECONDS must be a positive number, "
            f"got {raw_timeout!r}"
        ) from exc
    if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
        raise ValueError(
            "ISAACLAB_APP_LAUNCHER_LOCK_TIMEOUT_SECONDS must be a positive finite number, "
            f"got {raw_timeout!r}"
        )
    return timeout_seconds


def _get_distributed_timeout_seconds():
    raw_timeout = os.environ.get(
        "SONIC_DISTRIBUTED_TIMEOUT_SECONDS",
        str(DEFAULT_DISTRIBUTED_TIMEOUT_SECONDS),
    )
    try:
        timeout_seconds = float(raw_timeout)
    except ValueError as exc:
        raise ValueError(
            "SONIC_DISTRIBUTED_TIMEOUT_SECONDS must be a positive number, "
            f"got {raw_timeout!r}"
        ) from exc
    if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
        raise ValueError(
            "SONIC_DISTRIBUTED_TIMEOUT_SECONDS must be a positive finite number, "
            f"got {raw_timeout!r}"
        )
    return timeout_seconds


def launch_isaac_app_with_lock(
    app_launcher_cls,
    args_cli,
    *,
    global_rank,
    local_rank,
    lock_path="/tmp/isaaclab_app_launcher.lock",
    timeout_seconds=None,
    log_fn=None,
):
    """Serialize AppLauncher startup per node and fail fast on a stuck peer."""

    if timeout_seconds is None:
        timeout_seconds = _get_app_launcher_lock_timeout_seconds()
    if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
        raise ValueError(f"AppLauncher lock timeout must be positive, got {timeout_seconds!r}")

    def emit(message):
        if log_fn is None:
            print(message, flush=True)
        else:
            log_fn(message)

    rank_context = (
        f"global_rank={global_rank} local_rank={local_rank} pid={os.getpid()} "
        f"lock={lock_path} timeout_seconds={timeout_seconds:g}"
    )
    wait_started = time.monotonic()
    emit(f"[AppLauncherLock] waiting {rank_context}")
    try:
        with FileLock(lock_path, timeout=timeout_seconds):
            acquired_at = time.monotonic()
            emit(
                f"[AppLauncherLock] acquired {rank_context} "
                f"wait_seconds={acquired_at - wait_started:.3f}"
            )
            try:
                app_launcher = app_launcher_cls(args_cli)
            except Exception as exc:
                emit(
                    f"[AppLauncherLock] launcher_failed {rank_context} "
                    f"exception={type(exc).__name__}"
                )
                raise
            initialized_at = time.monotonic()
            emit(
                f"[AppLauncherLock] initialized {rank_context} "
                f"wait_seconds={acquired_at - wait_started:.3f} "
                f"init_seconds={initialized_at - acquired_at:.3f}"
            )
            return app_launcher
    except Timeout as exc:
        waited_seconds = time.monotonic() - wait_started
        message = (
            "Timed out waiting for a peer process to finish Isaac AppLauncher startup: "
            f"{rank_context} waited_seconds={waited_seconds:.3f}"
        )
        emit(f"[AppLauncherLock] timeout {message}")
        raise RuntimeError(message) from exc


def resume_training(config):
    explicit_checkpoint = config.get("checkpoint", None)
    resume_in_place = bool(config.get("resume_in_place", True))
    requested_experiment_dir = config.get("experiment_dir", None)
    if not resume_in_place and explicit_checkpoint is None:
        raise ValueError("resume_in_place=false requires an explicit checkpoint path")

    if explicit_checkpoint is not None:
        last_existing_checkpoint = config.checkpoint
    elif config.get("experiment_dir", None) is not None:
        last_existing_checkpoint = os.path.join(config.experiment_dir, "last.pt")
    else:
        # Use experiment_dir to find the checkpoint, rather than reconstructing
        # from config.project_name which can differ from the actual filesystem path.
        experiment_dir_base = re.sub(r"-\d{8}_\d{6}$", "", config.experiment_dir)
        checkpoints = sorted(glob.glob(os.path.join(f"{experiment_dir_base}-*", "last.pt")))
        if not checkpoints:
            print(f"No checkpoint found matching {experiment_dir_base}-*/last.pt, starting fresh")
            return
        last_existing_checkpoint = checkpoints[-1]
    checkpoint_experiment_dir = os.path.dirname(last_existing_checkpoint)
    if resume_in_place:
        config.experiment_dir = checkpoint_experiment_dir
    elif requested_experiment_dir is None:
        raise ValueError("resume_in_place=false requires an explicit experiment_dir")
    elif os.path.abspath(requested_experiment_dir) == os.path.abspath(
        checkpoint_experiment_dir
    ):
        raise ValueError(
            "resume_in_place=false requires experiment_dir to differ from the checkpoint directory"
        )
    config.checkpoint = last_existing_checkpoint
    print(  # noqa: T201
        f"Resuming training from {last_existing_checkpoint}; "
        f"outputs={config.experiment_dir}; in_place={resume_in_place}"
    )


def resume_checkpoint(config):
    config.checkpoint = config.checkpoint


def create_manager_env(config, device, args_cli):

    # import wandb

    from isaaclab.envs import (
        ManagerBasedRLEnv,
    )

    from gear_sonic.envs.wrapper.manager_env_wrapper import ManagerEnvWrapper

    env_instance_cfg = custom_instantiate(config.manager_env)

    # Iteratively check the difference in attribute of env_instance_cfg1 and env_instance_cfg, print out the difference
    def compare_attrs(obj1, obj2, prefix=""):
        # Only compare attributes that do not start with '__' and are not methods
        attrs1 = set(dir(obj1))
        attrs2 = set(dir(obj2))
        common_attrs = attrs1 & attrs2
        for attr in sorted(common_attrs):
            if (
                attr.startswith("__")
                or callable(getattr(obj1, attr))
                or callable(getattr(obj2, attr))
            ):
                continue
            try:
                val1 = getattr(obj1, attr)
                val2 = getattr(obj2, attr)
            except Exception:
                continue
            # Recursively compare if both are objects with __dict__ or are dicts
            if isinstance(val1, dict | DictConfig) and isinstance(val2, dict | DictConfig):
                compare_attrs(val1, val2, prefix + attr + ".")
            elif hasattr(val1, "__dict__") and hasattr(val2, "__dict__"):
                compare_attrs(val1, val2, prefix + attr + ".")
            else:
                if isinstance(val1, list):
                    val1 = tuple(val1)
                if isinstance(val2, list):
                    val2 = tuple(val2)
                if val1 != val2:
                    print(
                        f"\nDifference found at '{prefix}{attr}':\n"
                        f"  - env_instance_cfg1: {val1!r}\n"
                        f"  - env_instance_cfg : {val2!r}\n"
                    )

    env_instance_cfg.seed = config.seed
    env_instance_cfg.sim.device = device
    env_instance_cfg.config["headless"] = args_cli.headless
    env = ManagerBasedRLEnv(
        cfg=env_instance_cfg, render_mode="rgb_array" if not args_cli.headless else None
    )

    env = ManagerEnvWrapper(env, env_instance_cfg.config)
    return env


@hydra.main(config_path="config", config_name="base", version_base="1.1")
def main(config: OmegaConf):
    simulator_type = "IsaacSim"
    env_config = config.manager_env
    from transformers import HfArgumentParser
    from trl import ModelConfig, PPOConfig, ScriptArguments

    # Setup model components
    parser = HfArgumentParser((ScriptArguments, PPOConfig, ModelConfig))

    if config.get("resume", False):
        resume_training(config)
    elif config.get("checkpoint", None) is not None:
        resume_checkpoint(config)

    requested_report_to = config.algo.trl.get("report_to", None)
    use_tensorboard = logger_utils.report_to_contains(requested_report_to, "tensorboard")
    with open_dict(config.algo.trl):
        config.algo.trl.output_dir = str(Path(config.experiment_dir))
        if config.algo.trl.get("logging_dir", None) is None:
            config.algo.trl.logging_dir = str(Path(config.experiment_dir) / "tensorboard")

    trl_parser_config = OmegaConf.to_container(config.algo.trl, resolve=True)
    trl_parser_config["report_to"] = logger_utils.trainer_report_to(requested_report_to)
    script_args, training_args, model_args = parser.parse_dict(trl_parser_config)

    # Add exp_name from main config to training_args
    training_args.exp_name = config.experiment_name

    from datetime import timedelta

    from accelerate import Accelerator, DistributedDataParallelKwargs, InitProcessGroupKwargs
    from accelerate.utils import DDPCommunicationHookType
    import torch  # noqa: E402

    # Optional experiment: halve FP32 DDP gradient payloads while keeping model
    # and optimizer state in FP32. This changes reduction numerics, so 032 keeps
    # it disabled until a long-horizon quality/eval comparison is complete.
    use_fp16_ddp_comm_hook = bool(
        config.algo.config.get("use_fp16_ddp_comm_hook", False)
    )
    ddp_kwargs = DistributedDataParallelKwargs(
        find_unused_parameters=False,
        comm_hook=(
            DDPCommunicationHookType.FP16
            if use_fp16_ddp_comm_hook
            else DDPCommunicationHookType.NO
        ),
    )
    distributed_timeout_seconds = _get_distributed_timeout_seconds()
    kwargs = InitProcessGroupKwargs(
        timeout=timedelta(seconds=distributed_timeout_seconds)
    )
    accelerator = Accelerator(
        gradient_accumulation_steps=training_args.gradient_accumulation_steps,
        kwargs_handlers=[ddp_kwargs, kwargs],
    )
    print(
        "[DistributedStage] accelerator_ready "
        f"global_rank={accelerator.process_index} "
        f"local_rank={accelerator.local_process_index} "
        f"world_size={accelerator.num_processes} pid={os.getpid()} "
        f"timeout_seconds={distributed_timeout_seconds:g} "
        f"fp16_ddp_comm_hook={use_fp16_ddp_comm_hook}",
        flush=True,
    )

    device = str(accelerator.device)
    if device == "cuda":
        device = "cuda:0"
    config.multi_gpu = accelerator.num_processes > 1
    if config.multi_gpu:
        config.global_rank = accelerator.process_index
        config.seed += accelerator.process_index
        config.algo.config.global_rank = accelerator.process_index
        config.algo.config.world_size = accelerator.num_processes
    seeding(config.seed)

    meta_path = Path(config.experiment_dir) / "meta.yaml"
    if meta_path.exists():
        meta = yaml.safe_load(open(meta_path))
        wandb_run_id = meta.get("wandb_run") if meta else None
        if wandb_run_id is not None:
            config.wandb.wandb_id = wandb_run_id
            print(f"resume wandb from run: {config.wandb.wandb_id}")

    unresolved_conf = OmegaConf.to_container(config, resolve=False)
    if config.use_wandb and accelerator.is_main_process:
        project_name = f"{config.project_name}"
        run_name = config.experiment_dir.replace(f"{config.base_dir}/{project_name}/", "")
        wandb_dir = Path(config.wandb.wandb_dir)
        wandb_dir.mkdir(exist_ok=True, parents=True)
        wandb_group = None if config.wandb.wandb_id is not None else config.wandb.wandb_group
        logger.info(f"Saving wandb logs to {wandb_dir}")
        wandb.init(
            project=project_name,
            entity=config.wandb.wandb_entity,
            name=run_name,
            sync_tensorboard=False,
            config=unresolved_conf,
            dir=wandb_dir,
            id=config.wandb.wandb_id,
            group=wandb_group,
            resume="allow",
        )

    # Setup simulator similar to train_agent.py

    if simulator_type == "IsaacSim":
        try:
            with open("./rl/simulator/isaacsim/.isaacsim_version", encoding="utf-8") as f:
                DEFAULT_ISAACSIM_VERSION = f.read().strip()
        except FileNotFoundError:
            DEFAULT_ISAACSIM_VERSION = "4.5"

        if DEFAULT_ISAACSIM_VERSION == "4.5":
            from isaaclab.app import AppLauncher
        elif DEFAULT_ISAACSIM_VERSION == "4.2":
            logger.warning("Using IsaacSim 4.2, replacing isaaclab with omni.isaac.lab")
            from omni.isaac.lab.app import AppLauncher  # 4.2

            # from isaaclab.app import AppLauncher # not working
            # from omni.isaac.lab.app import AppLauncher

        import argparse

        parser = argparse.ArgumentParser(description="Train an RL agent with TRL.")
        AppLauncher.add_app_launcher_args(parser)

        ######################################################### ZL: fix isaacsim 4.5 rendering #########################################################
        args_cli, hydra_args = parser.parse_known_args()
        sys.argv = [sys.argv[0]] + hydra_args
        args_cli.num_envs = config.num_envs
        args_cli.seed = config.seed
        args_cli.env_spacing = env_config.config.env_spacing  # config.env_spacing
        args_cli.output_dir = config.output_dir
        # Enable cameras if enable_cameras, render_results, render_ego, or overview_camera is True
        args_cli.enable_cameras = (
            env_config.config.get("enable_cameras", False)
            or env_config.config.get("render_results", False)
            or env_config.config.get("render_ego", False)
            or env_config.config.get("overview_camera", False)
        )
        args_cli.headless = config.headless
        args_cli.multi_gpu = config.multi_gpu
        args_cli.distributed = config.multi_gpu
        args_cli.device = device

        # Base kit args (quiet logs)
        args_cli.kit_args = (
            "--/log/level=error --/log/fileLogLevel=error --/log/outputStreamLevel=error"
        )

        # AppLauncher can't handle multiple processes creating it at the same time. Keep the
        # per-node serialization, but do not let a stuck cold-cache startup block every rank forever.
        _local_rank = int(os.environ.get("LOCAL_RANK", 0))
        _global_rank = accelerator.process_index
        _rank_cache_path = Path.cwd() / ".isaacsim" / "registry-cache" / f"rank_{_global_rank}"
        print(
            f"[AppLauncherLock] cache_probe global_rank={_global_rank} "
            f"local_rank={_local_rank} path={_rank_cache_path} "
            f"exists={_rank_cache_path.is_dir()}",
            flush=True,
        )
        app_launcher = launch_isaac_app_with_lock(
            AppLauncher,
            args_cli,
            global_rank=_global_rank,
            local_rank=_local_rank,
        )

        simulation_app = app_launcher.app

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = False

    from gear_sonic.utils.logging import HydraLoggerBridge

    # resolve=False is important otherwise overrides
    # at inference time won't work properly
    # also, I believe this must be done before instantiation

    # logging to hydra log file
    hydra_log_path = os.path.join(HydraConfig.get().runtime.output_dir, "train.log")
    logger.remove()
    logger.add(hydra_log_path, level="DEBUG")
    console_log_level = os.environ.get("LOGURU_LEVEL", "INFO").upper()
    logger.add(sys.stdout, level=console_log_level, colorize=True)
    logging.basicConfig(level=logging.DEBUG)
    logging.getLogger().addHandler(HydraLoggerBridge())

    # Setup wandb if enabled
    os.chdir(hydra.utils.get_original_cwd())

    # Save config and meta BEFORE env creation so eval jobs can postprocess
    # checkpoint configs even if training crashes during env init.
    experiment_save_dir = Path(config.experiment_dir)
    if accelerator.is_main_process:
        experiment_save_dir.mkdir(exist_ok=True, parents=True)
        logger.info(f"Saving config file to {experiment_save_dir}")
        with open(experiment_save_dir / "config.yaml", "w") as file:
            OmegaConf.save(unresolved_conf, file)
        meta = {"wandb_run": wandb.run.id if wandb_run_exists() else None}
        meta["max_train_steps"] = config.algo.config.num_learning_iterations
        yaml.safe_dump(meta, open(meta_path, "w"))
        print("saved meta:", meta)

    # Initialize environment
    env_config.config.save_rendering_dir = str(Path(config.experiment_dir) / "renderings_training")
    env_config.config.experiment_dir = str(Path(config.experiment_dir))

    env = create_manager_env(config, device, args_cli)
    if config.get("replay", False):
        _save_video_path = config.get("replay_save_video", None)
        env.run_replay(
            start_time_step=-1,
            loop=config.get("replay_loop_num", True),
            save_video_path=_save_video_path,
            grid_spacing=config.get("replay_grid_spacing", 2.0),
        )
        os._exit(0)
    if config.get("vplanner_replay", False):
        vplanner_checkpoint = config.get("vplanner_checkpoint", None)
        if vplanner_checkpoint is None:
            raise ValueError("vplanner_checkpoint must be specified for vplanner_replay")
        env.run_vplanner_replay(
            checkpoint_path=vplanner_checkpoint,
            max_frames=config.get("vplanner_max_frames", 500),
            replan_interval=config.get("vplanner_replan_interval", 0),
            speed=config.get("vplanner_speed", 1.0),
            loop=config.get("vplanner_loop", True),
            save_images=config.get("vplanner_save_images", False),
            output_dir=config.get("vplanner_output_dir", None),
            dof_noise=config.get("vplanner_dof_noise", 0.0),
            dof_vel_noise=config.get("vplanner_dof_vel_noise", 0.0),
            quat_noise=config.get("vplanner_quat_noise", 0.0),
        )
        os._exit(0)

    ref_model = None
    value_model = None
    disc_model = None
    # import ipdb; ipdb.set_trace()

    if config.algo.config.get("use_new_actor_critic", False):
        module_dim_dict = getattr(config.algo.config, "module_dim", {})
        policy_backbone_kwargs = {}
        critic_backbone_kwargs = {}
        env.config["obs"]["obs_dims"]["actor_obs"] = env.env.observation_space["policy"].shape[-1]
        env.config["obs"]["obs_dims"]["critic_obs"] = env.env.observation_space["critic"].shape[-1]
        env.config["robot"]["algo_obs_dim_dict"]["actor_obs"] = env.env.observation_space[
            "policy"
        ].shape[-1]
        env.config["robot"]["algo_obs_dim_dict"]["critic_obs"] = env.env.observation_space[
            "critic"
        ].shape[-1]
        example_obs = env.reset(flatten_dict_obs=False)
        for key in env.env.observation_space:
            if key not in ["policy", "critic"]:
                group_obs_dims, group_obs_names, group_obs_total_dim = get_group_term_obs_shape(
                    example_obs, key
                )
                env.config["obs"]["group_obs_dims"][key] = group_obs_dims
                env.config["obs"]["group_obs_names"][key] = group_obs_names
                env.config["obs"]["obs_dims"][key] = group_obs_total_dim
                env.config["robot"]["algo_obs_dim_dict"][key] = group_obs_total_dim
        if config.manager_env.config.get("meta_action_dim", None) is not None:
            env.config["robot"]["actions_dim"] = config.manager_env.config.meta_action_dim
        else:
            env.config["robot"]["actions_dim"] = env.env.action_space.shape[-1]

        policy = custom_instantiate(
            config.algo.config.actor,
            env_config=env.config,
            algo_config=config.algo.config,
            module_dim_dict=module_dim_dict,
            backbone_kwargs=policy_backbone_kwargs,
            _resolve=False,
        ).to(device)

        if getattr(config.algo.config, "use_dagger", False):
            # Get teacher input key from config or default to "teacher"
            teacher_input_key = config.algo.config.get("teacher_input_key", "teacher")
            ref_model = custom_instantiate(
                config.algo.config.teacher_actor,
                env_config=env.config,
                algo_config=config.algo.config,
                module_dim_dict=module_dim_dict,
                _resolve=False,
                input_key=teacher_input_key,
            ).to(device)
        if not getattr(config.algo.config, "distill_only", False):
            value_model = custom_instantiate(
                config.algo.config.critic,
                env_config=env.config,
                algo_config=config.algo.config,
                module_dim_dict=module_dim_dict,
                backbone_kwargs=critic_backbone_kwargs,
                _resolve=False,
            ).to(device)
        if config.algo.config.get("use_amp", False):
            disc_model = custom_instantiate(
                config.algo.config.disc,
                env_config=env.config,
                algo_config=config.algo.config,
                module_dim_dict=module_dim_dict,
                _resolve=False,
            ).to(device)
    else:
        raise ValueError("No longer supported")

    materialize_lazy_params(policy, env)

    if config.get("checkpoint", None) is None:
        backbone = getattr(policy, "actor_module", None)
        if backbone is None:
            backbone = getattr(policy, "backbone", None)
        if hasattr(backbone, "apply_encoder_init_from"):
            backbone.apply_encoder_init_from(None)

    if config.algo.config.get("pretrained_model", None) is not None:
        pretrained_cfg = config.algo.config.pretrained_model
        sd_key = pretrained_cfg.get("state_dict_key", "state_dict")
        strict = pretrained_cfg.get("strict", True)
        state_dict = torch.load(pretrained_cfg.path, map_location=device, weights_only=False)[
            sd_key
        ]
        for (
            module_name,
            state_dict_key,
        ) in pretrained_cfg.module_mapping.items():
            module = eval(module_name)
            filtered_state_dict = get_filtered_state_dict(state_dict, state_dict_key)
            missing, unexpected = module.load_state_dict(filtered_state_dict, strict=strict)
            if missing:
                logger.info(f"Pretrained loading '{module_name}': missing keys: {missing}")
            if unexpected:
                logger.info(f"Pretrained loading '{module_name}': unexpected keys: {unexpected}")

    model_modules = {
        "policy": policy,
        "value_model": value_model,
        "disc_model": disc_model,
        "ref_model": ref_model,
    }
    distributed_stage(
        "model_signature_check",
        lambda: verify_distributed_model_signatures(model_modules, accelerator),
        accelerator=accelerator,
    )
    distributed_stage(
        "pre_trainer_barrier",
        accelerator.wait_for_everyone,
        accelerator=accelerator,
    )

    callbacks = []
    for callback in config.callbacks.values():
        callbacks.append(instantiate(callback))
    if use_tensorboard:
        from gear_sonic.trl.callbacks.tensorboard_callback import SonicTensorBoardCallback

        callbacks.append(
            SonicTensorBoardCallback(
                log_dir=training_args.logging_dir,
                flush_every_n_steps=config.algo.config.get(
                    "tensorboard_flush_every_n_steps", 1
                ),
            )
        )

    ################
    # Training
    ################
    trainer = distributed_stage(
        "trainer_init",
        lambda: custom_instantiate(
            config.trainer,
            args=training_args,
            config=config.algo.config,
            env=env,
            model=policy,
            disc_model=disc_model,
            value_model=value_model,
            ref_model=ref_model,
            use_ref_model=getattr(config.algo.config, "use_dagger", False),
            train_dataset=None,
            eval_dataset=None,
            callbacks=callbacks,
            checkpoint=config.checkpoint,
            resume=config.get("resume", False),
            local_seed=config.seed,
            log_dir=experiment_save_dir,
            accelerator=accelerator,
            _resolve=False,
        ),
        accelerator=accelerator,
    )

    # Training loop
    distributed_stage(
        "training_loop",
        trainer.train,
        accelerator=accelerator,
    )

    if simulator_type == "IsaacSim":
        os._exit(0)


if __name__ == "__main__":

    main()
