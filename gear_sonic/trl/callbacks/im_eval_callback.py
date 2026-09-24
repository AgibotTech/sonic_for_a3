import csv
from datetime import datetime
import gc
import json
import os
import time

import numpy as np
import torch
from tqdm import tqdm
from transformers import TrainerCallback
import wandb

from gear_sonic.evaluation.aggregator import aggregate_batches
from gear_sonic.evaluation.batch_store import BatchStore
from gear_sonic.evaluation.metrics import (
    build_batch_payload,
    compute_joint_mse_metrics,
    compute_metric_sums_and_counts,
    compute_metrics_lite_with_empty,
    compute_subset_metrics_lite,
)
from gear_sonic.evaluation.schemas import DatasetManifest, RunManifest
from gear_sonic.evaluation.valid_frames import valid_eval_frame_mask


def create_html_table(metrics_dict):
    """
    Create a sortable HTML table for metrics logging using DataTables.

    Args:
        metrics_dict: Dictionary containing metrics data with keys like 'mpjpe_g', 'mpjpe_l', 'mpjpe_pa',
                     'terminated', 'motion_keys', etc.

    Returns:
        str: HTML string containing the sortable table
    """
    if not metrics_dict or len(metrics_dict) == 0:
        return wandb.Html("<p>No metrics data available</p>")

    # Get the motion keys and number of motions
    motion_keys = metrics_dict.get("motion_keys", [])
    if len(motion_keys) == 0:
        return wandb.Html("<p>No motion data available</p>")

    num_motions = len(motion_keys)

    # Get metric names (excluding special keys)
    special_keys = {"terminated", "motion_keys"}
    metric_names = [key for key in metrics_dict.keys() if key not in special_keys]

    # Create table header
    html = """
<!-- HTML Table -->
<table id="my-table" class="display">
  <thead>
    <tr>
      <th>Motion Key</th>
      <th>Terminated</th>
"""

    # Add metric column headers
    for metric_name in metric_names:
        html += f"      <th>{metric_name}</th>\n"

    html += """    </tr>
  </thead>
  <tbody>
"""

    # Create table rows
    for i in range(num_motions):
        motion_key = motion_keys[i] if i < len(motion_keys) else f"Motion_{i}"
        terminated = "Yes" if metrics_dict.get("terminated", [True] * num_motions)[i] else "No"

        html += f"    <tr><td>{motion_key}</td><td>{terminated}</td>"

        # Add metric values
        for metric_name in metric_names:
            metric_values = metrics_dict[metric_name]
            if i < len(metric_values):
                value = metric_values[i]
                # Format the value appropriately
                if isinstance(value, int | float):
                    if abs(value) < 0.001:
                        formatted_value = f"{value:.6f}"
                    elif abs(value) < 1:
                        formatted_value = f"{value:.4f}"
                    else:
                        formatted_value = f"{value:.3f}"
                else:
                    formatted_value = str(value)
                html += f"<td>{formatted_value}</td>"
            else:
                html += "<td>N/A</td>"

        html += "</tr>\n"

    html += """  </tbody>
</table>

<!-- DataTables CSS and JS dependencies -->
<link rel="stylesheet" href="https://cdn.datatables.net/1.11.5/css/jquery.dataTables.min.css">
<script src="https://code.jquery.com/jquery-3.3.1.min.js"></script>
<script src="https://cdn.datatables.net/1.11.5/js/jquery.dataTables.min.js"></script>

<!-- DataTables Initialization -->
<script>
  $(document).ready(function () {
    $('#my-table').DataTable({
      pageLength: 100,
      order: [[0, 'asc']],
    });
  });
</script>
"""
    return wandb.Html(html)


class ImEvalCallback(TrainerCallback):
    """Callback to evaluate motion imtiation during training. Supports multigpu ."""

    def __init__(
        self,
        eval_frequency,
        empty_cache_freq=20,
        eval_only=False,
        output_dir=None,
        log_keys=None,
        max_render_steps=0,
        reward_curve_enabled=False,
        reward_curve_output_dir=None,
        foot_force_dump_dir=None,
        batch_results_dir=None,
        run_manifest_path=None,
        dataset_manifest_path=None,
        snapshot_dir=None,
        snapshot_every_n_batches=5,
        allow_partial=False,
    ):
        super().__init__()
        self.eval_frequency = eval_frequency
        self.empty_cache_freq = empty_cache_freq
        self.output_dir = output_dir
        self.eval_only = eval_only
        self.in_eval_mode = False
        self.render_only = False
        self.log_keys = log_keys
        self.max_render_steps = max_render_steps
        self.reward_curve_enabled = reward_curve_enabled
        self.reward_curve_output_dir = reward_curve_output_dir
        self.foot_force_dump_dir = foot_force_dump_dir
        self.batch_results_dir = batch_results_dir
        self.run_manifest_path = run_manifest_path
        self.dataset_manifest_path = dataset_manifest_path
        self.snapshot_dir = snapshot_dir
        self.snapshot_every_n_batches = int(snapshot_every_n_batches)
        self.allow_partial = bool(allow_partial)
        self.scalable_eval = batch_results_dir is not None
        if self.snapshot_every_n_batches < 1:
            raise ValueError("snapshot_every_n_batches must be at least 1")
        self._has_object = False

    def on_step_end(self, args, state, control, **kwargs):

        self.env = kwargs.get("env")
        self.model = kwargs.get("model")
        self.accelerator = kwargs.get("accelerator")
        self.device = self.accelerator.device
        self.args = args
        self.model.eval()

        if (state.global_step + 1) % self.eval_frequency == 0:
            metrics_eval = self.evaluate_policy()

    def save_metrics_eval(self, metrics_eval):
        if self.scalable_eval and not self.accelerator.is_main_process:
            return
        metrics_json = {}
        for k, v in metrics_eval.items():
            if k in ["eval/all_metrics_dict", "eval/failed_metrics_dict"]:
                metrics_json[k] = {}
                for kk, vv in v.items():
                    if isinstance(vv, np.ndarray):
                        metrics_json[k][kk] = vv.tolist()
                    else:
                        metrics_json[k][kk] = vv
            elif isinstance(v, np.ndarray):
                metrics_json[k] = v.tolist()
            else:
                metrics_json[k] = v

        os.makedirs(self.output_dir, exist_ok=True)
        with open(os.path.join(self.output_dir, "metrics_eval.json"), "w") as f:
            print(f"Saving metrics_eval to {os.path.join(self.output_dir, 'metrics_eval.json')}")
            if self.log_keys is not None:
                metrics_json["log_keys"] = self.log_keys
            json.dump(metrics_json, f, indent=4)

    @torch.no_grad()
    def evaluate_policy(self):

        self.accelerator.wait_for_everyone()
        with torch.no_grad():
            self._eval_mode()
            print(
                "============================================Evaluating policy============================================"
            )

            self._pre_evaluate_policy()
            actor_state = {"done_indices": [], "stop": False}
            step = 0
            self.eval_policy = self._get_inference_policy()
            obs_dict = self.env.reset_all(global_rank=self.args.global_rank)
            self.model.policy.init_rollout()

            init_actions = torch.zeros(
                self.env.num_envs, self.env.config.robot.actions_dim, device=self.env.device
            )
            actor_state.update({"obs": obs_dict, "actions": init_actions})
            actor_state = self._pre_eval_env_step(actor_state)

            if self.scalable_eval and self.resume_aggregate_only:
                actor_state = self._finish_scalable_run(actor_state)
            else:
                while not actor_state.get("end_eval", False):
                    self.env.render_results()
                    actor_state["step"] = step
                    actor_state = self._pre_eval_env_step(actor_state)
                    actor_state = self.env_step(actor_state)
                    actor_state = self._post_eval_env_step(actor_state)
                    step += 1

                    if self.max_render_steps > 0 and step >= self.max_render_steps:
                        print(f"Reached max_render_steps={self.max_render_steps}. Stopping render.")
                        self.render_only = True
                        self.env.end_render_results()
                        actor_state["end_eval"] = True
                        break

                    if step % self.empty_cache_freq == 0:
                        gc.collect()
                        torch.cuda.empty_cache()

            if self.render_only:
                self.env.end_render_results()
                return {}

            metrics_eval = self._post_evaluate_policy(actor_state)

            if self.eval_only:
                if self.output_dir is not None:
                    self.save_metrics_eval(metrics_eval)
            else:
                metrics_eval["eval/all_metrics_dict"] = create_html_table(
                    metrics_eval["eval/all_metrics_dict"]
                )
                metrics_eval["eval/failed_metrics_dict"] = create_html_table(
                    metrics_eval["eval/failed_metrics_dict"]
                )

        self._train_mode()
        self.model.policy.clear_rollout()
        if not self.eval_only:
            gc.collect()
            torch.cuda.empty_cache()
        if self.eval_frequency == 1:  # Exit if eval frequency is 1.
            os._exit(0)
        return metrics_eval

    def _post_evaluate_policy(self, eval_res):
        metrics_success = eval_res["metrics_success"]
        metrics_all = eval_res["metrics_all"]
        metrics_eval = {}
        for k, v in metrics_success.items():
            metrics_eval[f"eval/success/{k}"] = v
        for k, v in metrics_all.items():
            metrics_eval[f"eval/all/{k}"] = v

        # Add failed_keys to metrics_eval for wandb logging
        metrics_eval["eval/all_metrics_dict"] = eval_res["all_metrics_dict"]
        metrics_eval["eval/failed_metrics_dict"] = eval_res["failed_metrics_dict"]
        if self.reward_curve_enabled and "reward_curves" in eval_res:
            metrics_eval["reward_curves"] = eval_res["reward_curves"]
        if self.eval_only:
            metrics_eval["failed_keys"] = eval_res["failed_keys"]
            metrics_eval["failed_idxes"] = eval_res["failed_idxes"]
        if "eval_meta" in eval_res:
            metrics_eval["eval/meta"] = eval_res["eval_meta"]

        return metrics_eval

    def _get_inference_policy(self, device=None):
        self.model.policy.eval()  # switch to evaluation mode (dropout for example)
        if device is not None:
            self.model.policy.to(device)
        return self.model.policy.act_inference

    def _eval_mode(self):
        if self.eval_only and self.in_eval_mode:
            return
        self.in_eval_mode = True
        self.model.eval()
        if hasattr(self.model.policy, "eval_mode"):
            self.model.policy.eval_mode()  # For VAE, eval mode means that we are no longer sampling from the VAE but using the mean latent value.
        self.env.set_is_evaluating(True, global_rank=self.args.global_rank)

    def _train_mode(self):
        if self.eval_only and self.in_eval_mode:
            return
        self.in_eval_mode = False
        self.model.train()
        if hasattr(self.model.policy, "train_mode"):
            self.model.policy.train_mode()
        self.env.set_is_evaluating(False)
        self.env.set_is_training()

    def _pre_evaluate_policy(self, reset_env=True):
        if reset_env:
            _ = self.env.reset_all()

        self.num_total_env_eval_loops = int(
            np.ceil(
                self.env._motion_lib._num_unique_motions
                / (self.env.num_envs * self.args.world_size)
            )
        )
        if "max_render_envs" in self.env.config:
            self.num_total_env_eval_loops = 1
            self.render_only = True

        self.env_eval_loop_idx = 0
        self.pbar = tqdm(range(self.num_total_env_eval_loops), desc="Total evaluation progress")
        self.steps_pbar = None
        self.success_rate = 0
        self.curr_steps = 0
        # self.env.start_compute_metrics(global_rank=self.args.global_rank)
        self.terminate_state = torch.zeros(self.env.num_envs, device=self.env.device)
        self.progress_state = torch.zeros(self.env.num_envs, device=self.env.device)
        self.valid_frame_count = torch.zeros(
            self.env.num_envs, device=self.env.device, dtype=torch.long
        )
        self.terminate_memory = []
        self.progress_memory = []
        self.mpjpe, self.mpjpe_all = [], []
        self.gt_pos, self.gt_pos_all = [], []
        self.gt_rot, self.gt_rot_all = [], []
        self.pred_pos, self.pred_pos_all = [], []
        self.pred_rot, self.pred_rot_all = [], []
        self.sampled_motion_idx = []
        self.time_eval_start = time.time()

        # Object tracking metrics
        self._has_object = (
            hasattr(self.env, "env")
            and hasattr(self.env.env, "scene")
            and "object" in self.env.env.scene.rigid_objects
            and hasattr(self.env, "motion_command")
            and self.env.motion_command is not None
        )
        self.obj_pos_error, self.obj_pos_error_all = [], []
        self.obj_ori_error, self.obj_ori_error_all = [], []
        self.reward_curve_files_all = []
        self._reset_reward_curve_batch_buffers()
        self.foot_force_files_all = []
        self._reset_foot_force_batch_buffers()
        if self.scalable_eval:
            self._initialize_scalable_eval()

    def _initialize_scalable_eval(self):
        if self.run_manifest_path is None or self.dataset_manifest_path is None:
            raise ValueError(
                "scalable eval requires run_manifest_path and dataset_manifest_path"
            )
        self.run_manifest = RunManifest.read(self.run_manifest_path)
        self.dataset_manifest = DatasetManifest.read(self.dataset_manifest_path)
        motion_aliases = [str(key) for key in self.env._motion_lib._motion_data_keys]
        manifest_aliases = [entry.alias for entry in self.dataset_manifest.entries]
        if motion_aliases != manifest_aliases:
            raise ValueError(
                "motion library order does not match dataset_manifest aliases; "
                f"motion_lib[:5]={motion_aliases[:5]}, manifest[:5]={manifest_aliases[:5]}"
            )
        canonical_keys = tuple(entry.canonical_key for entry in self.dataset_manifest.entries)
        if canonical_keys != self.run_manifest.expected_motion_keys:
            raise ValueError("run manifest expected_motion_keys do not match dataset manifest")
        if self.run_manifest.world_size != self.args.world_size:
            raise ValueError(
                f"run manifest world_size={self.run_manifest.world_size} does not match "
                f"runtime world_size={self.args.world_size}"
            )

        self.batch_store = BatchStore(self.batch_results_dir, run_id=self.run_manifest.run_id)
        completed = self.batch_store.completed_indices(self.args.global_rank)
        completed_count = 0
        while completed_count in completed:
            completed_count += 1
        if completed != set(range(completed_count)):
            raise ValueError(
                f"rank {self.args.global_rank} has non-contiguous completed batches: "
                f"{sorted(completed)}"
            )
        if completed_count > self.num_total_env_eval_loops:
            raise ValueError(
                f"completed batch count {completed_count} exceeds total "
                f"{self.num_total_env_eval_loops}"
            )

        self.scalable_evaluated_count = 0
        self.scalable_terminated_count = 0
        self.scalable_progress_sum = 0.0
        for payload in self.batch_store.load_all():
            if int(payload["rank"]) != self.args.global_rank:
                continue
            terminated = np.asarray(payload["terminated"], dtype=bool)
            progress = np.asarray(payload["progress"], dtype=np.float64).copy()
            progress[~terminated] = 1.0
            self.scalable_evaluated_count += int(terminated.size)
            self.scalable_terminated_count += int(terminated.sum())
            self.scalable_progress_sum += float(progress.sum())

        self.env_eval_loop_idx = completed_count
        self.resume_aggregate_only = completed_count >= self.num_total_env_eval_loops
        if not self.resume_aggregate_only:
            for _ in range(completed_count):
                self.env.forward_motion_samples(self.args.global_rank, self.args.world_size)
        if completed_count:
            self.pbar.update(completed_count)
            self.pbar.refresh()
            print(
                f"[scalable_eval] rank={self.args.global_rank} resumed after "
                f"{completed_count}/{self.num_total_env_eval_loops} batches",
                flush=True,
            )

    def _write_scalable_snapshot(self, force=False):
        if self.snapshot_dir is None:
            return
        if not force and self.env_eval_loop_idx % self.snapshot_every_n_batches != 0:
            return
        snapshot_root = os.path.join(
            self.snapshot_dir, f"rank_{self.args.global_rank:03d}"
        )
        os.makedirs(snapshot_root, exist_ok=True)
        elapsed = time.time() - self.time_eval_start
        evaluated = self.scalable_evaluated_count
        payload = {
            "schema_version": 1,
            "run_id": self.run_manifest.run_id,
            "rank": int(self.args.global_rank),
            "completed_batches": int(self.env_eval_loop_idx),
            "total_batches": int(self.num_total_env_eval_loops),
            "evaluated_motion_count": int(evaluated),
            "success_rate": (
                1.0 - self.scalable_terminated_count / evaluated if evaluated else 0.0
            ),
            "progress_rate": (
                self.scalable_progress_sum / evaluated if evaluated else 0.0
            ),
            "elapsed_seconds": float(elapsed),
        }
        for filename in (
            "latest.json",
            f"batch_{self.env_eval_loop_idx:06d}.json",
        ):
            path = os.path.join(snapshot_root, filename)
            temporary = f"{path}.{os.getpid()}.tmp"
            with open(temporary, "w", encoding="utf-8") as stream:
                json.dump(payload, stream, ensure_ascii=False, indent=2)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)

    def _reset_scalable_batch_buffers(self):
        self.mpjpe = []
        self.gt_pos = []
        self.pred_pos = []
        self.obj_pos_error = []
        self.obj_ori_error = []
        self._reset_reward_curve_batch_buffers()
        self._reset_foot_force_batch_buffers()

    def _reset_reward_curve_batch_buffers(self):
        if not self.reward_curve_enabled:
            return
        if not hasattr(self, "reward_curve_files_all"):
            self.reward_curve_files_all = []
        self.reward_curve_term_names = None
        self.reward_curve_step_dt = None
        self.reward_curve_records = [[] for _ in range(self.env.num_envs)]
        self.reward_curve_closed = np.zeros(self.env.num_envs, dtype=bool)

    def _foot_force_enabled(self):
        return self.foot_force_dump_dir is not None

    def _reset_foot_force_batch_buffers(self):
        if not self._foot_force_enabled():
            return
        self.foot_force_body_names = None
        self.foot_force_body_ids = None
        self.foot_force_step_dt = None
        self.foot_force_records = [[] for _ in range(self.env.num_envs)]
        self.foot_force_closed = np.zeros(self.env.num_envs, dtype=bool)
        self.foot_force_missing_warning_printed = False

    def _get_foot_force_sensor(self):
        if not hasattr(self.env, "env") or not hasattr(self.env.env, "scene"):
            return None, None
        scene = self.env.env.scene
        if not hasattr(scene, "sensors") or "foot_end_contact" not in scene.sensors:
            return None, None
        sensor = scene["foot_end_contact"]
        body_names = list(getattr(sensor, "body_names", []))
        return sensor, body_names

    def _resolve_foot_force_body_ids(self, body_names):
        desired = [
            "left_foot_front_Link",
            "left_foot_mid_Link",
            "left_foot_rear_Link",
            "right_foot_front_Link",
            "right_foot_mid_Link",
            "right_foot_rear_Link",
        ]
        body_name_to_id = {name: idx for idx, name in enumerate(body_names)}
        missing = [name for name in desired if name not in body_name_to_id]
        if missing:
            raise RuntimeError(
                "foot_end_contact sensor is missing expected bodies: "
                f"{missing}; available={body_names}"
            )
        return desired, [body_name_to_id[name] for name in desired]

    def _collect_foot_force_step(self, actor_state):
        if not self._foot_force_enabled():
            return

        sensor, body_names = self._get_foot_force_sensor()
        if sensor is None:
            if not self.foot_force_missing_warning_printed:
                print("foot_force_dump_dir was set but foot_end_contact sensor is unavailable")
                self.foot_force_missing_warning_printed = True
            return

        if self.foot_force_body_ids is None:
            resolved_names, body_ids = self._resolve_foot_force_body_ids(body_names)
            self.foot_force_body_names = resolved_names
            self.foot_force_body_ids = body_ids
            base_env = getattr(self.env, "env", self.env)
            self.foot_force_step_dt = float(getattr(base_env, "step_dt", getattr(base_env, "dt", 1.0)))

        net_forces = sensor.data.net_forces_w.detach()
        if net_forces.ndim != 3 or net_forces.shape[-1] != 3:
            return

        fz = net_forces[:, self.foot_force_body_ids, 2].clamp_min(0.0).cpu().numpy()
        dones = actor_state["dones"].bool().detach().cpu().numpy()
        time_outs = actor_state["extras"]["time_outs"].bool().detach().cpu().numpy()
        terminated = np.logical_and(dones, np.logical_not(time_outs))

        for env_idx in range(min(self.env.num_envs, fz.shape[0])):
            if self.foot_force_closed[env_idx]:
                continue
            self.foot_force_records[env_idx].append(fz[env_idx].astype(np.float64))
            if terminated[env_idx]:
                self.foot_force_closed[env_idx] = True

    def _foot_force_output_base_dir(self):
        return self.foot_force_dump_dir

    def _write_foot_force_csv(self, csv_path, forces, step_dt):
        body_columns = [
            "Fz_left_front",
            "Fz_left_mid",
            "Fz_left_rear",
            "Fz_right_front",
            "Fz_right_mid",
            "Fz_right_rear",
        ]
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["step", "time_s", *body_columns])
            for step_idx, row in enumerate(forces):
                writer.writerow([step_idx, (step_idx + 1) * step_dt, *row.tolist()])

    def _finalize_foot_force_batch(self, env_motion_ids, valid_lengths):
        if not self._foot_force_enabled() or self.foot_force_body_names is None:
            return

        output_dir = self._foot_force_output_base_dir()
        if output_dir is None:
            return
        os.makedirs(output_dir, exist_ok=True)

        env_motion_ids_cpu = env_motion_ids.detach().cpu().numpy()
        valid_lengths_cpu = valid_lengths.detach().cpu().numpy()
        motion_keys = getattr(self.env._motion_lib, "_motion_data_keys", [])
        num_unique_motions = getattr(self.env._motion_lib, "_num_unique_motions", len(motion_keys))
        step_dt = self.foot_force_step_dt or 1.0

        for env_idx, records in enumerate(self.foot_force_records):
            if len(records) == 0 or env_idx >= len(env_motion_ids_cpu):
                continue
            motion_idx = int(env_motion_ids_cpu[env_idx])
            if motion_idx >= num_unique_motions:
                continue
            max_steps = (
                int(valid_lengths_cpu[env_idx])
                if env_idx < len(valid_lengths_cpu)
                else len(records)
            )
            selected_records = records[:max_steps]
            if len(selected_records) == 0:
                continue
            forces = np.stack(selected_records, axis=0)
            if forces.size == 0:
                continue

            motion_key = (
                motion_keys[motion_idx]
                if motion_idx < len(motion_keys)
                else f"motion_{motion_idx:06d}"
            )
            basename = self._safe_reward_curve_name(motion_key, motion_idx).replace(
                "_reward_curve", "_foot_forces"
            )
            csv_path = os.path.join(output_dir, f"{basename}.csv")
            npz_path = os.path.join(output_dir, f"{basename}.npz")

            self._write_foot_force_csv(csv_path, forces, step_dt)
            np.savez(
                npz_path,
                forces_fz=forces,
                body_names=np.asarray(self.foot_force_body_names, dtype=object),
                step_dt=np.asarray(step_dt, dtype=np.float64),
                motion_idx=np.asarray(motion_idx, dtype=np.int64),
                motion_key=np.asarray(str(motion_key), dtype=object),
            )
            self.foot_force_files_all.append(
                {
                    "env_idx": int(env_idx),
                    "motion_idx": motion_idx,
                    "motion_key": str(motion_key),
                    "num_steps": int(forces.shape[0]),
                    "step_dt": step_dt,
                    "body_names": list(self.foot_force_body_names),
                    "csv": csv_path,
                    "npz": npz_path,
                }
            )

    def _write_foot_force_manifest(self):
        if not self._foot_force_enabled():
            return
        output_dir = self._foot_force_output_base_dir()
        if output_dir is None:
            return
        os.makedirs(output_dir, exist_ok=True)
        manifest_path = os.path.join(output_dir, "foot_force_manifest.json")
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(self.foot_force_files_all, f, indent=2)

    def _get_reward_manager_for_curves(self):
        base_env = getattr(self.env, "env", self.env)
        return getattr(base_env, "reward_manager", None), base_env

    def _collect_reward_curve_step(self, actor_state):
        if not self.reward_curve_enabled:
            return

        reward_manager, base_env = self._get_reward_manager_for_curves()
        if reward_manager is None or not hasattr(reward_manager, "_step_reward"):
            return

        term_names = list(getattr(reward_manager, "active_terms", []))
        step_reward = getattr(reward_manager, "_step_reward", None)
        if len(term_names) == 0 or step_reward is None:
            return

        step_reward = step_reward.detach()
        if step_reward.ndim != 2 or step_reward.shape[1] < len(term_names):
            return

        step_dt = float(getattr(base_env, "step_dt", getattr(base_env, "dt", 1.0)))
        weighted_step_reward = (
            step_reward[:, : len(term_names)] * step_dt
        ).detach().cpu().numpy()

        if self.reward_curve_term_names is None:
            self.reward_curve_term_names = term_names
            self.reward_curve_step_dt = step_dt

        dones = actor_state["dones"].bool().detach().cpu().numpy()
        time_outs = actor_state["extras"]["time_outs"].bool().detach().cpu().numpy()
        terminated = np.logical_and(dones, np.logical_not(time_outs))

        for env_idx in range(min(self.env.num_envs, weighted_step_reward.shape[0])):
            if self.reward_curve_closed[env_idx]:
                continue
            self.reward_curve_records[env_idx].append(
                weighted_step_reward[env_idx].astype(np.float64)
            )
            if terminated[env_idx]:
                self.reward_curve_closed[env_idx] = True

    def _reward_curve_output_base_dir(self):
        output_dir = self.reward_curve_output_dir
        if output_dir is None:
            if self.output_dir is None:
                return None
            output_dir = os.path.join(self.output_dir, "reward_curves")
        return output_dir

    @staticmethod
    def _safe_reward_curve_name(motion_key, motion_idx):
        key_name = os.path.basename(str(motion_key))
        safe_key = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in key_name)
        safe_key = safe_key.strip("._") or f"motion_{motion_idx:06d}"
        return f"{motion_idx:06d}_{safe_key}_reward_curve"

    def _write_reward_curve_csv(self, csv_path, rewards, term_names, step_dt):
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["step", "time_s", "total_reward", *term_names])
            total_reward = rewards.sum(axis=1)
            for step_idx, term_values in enumerate(rewards):
                writer.writerow(
                    [
                        step_idx,
                        (step_idx + 1) * step_dt,
                        total_reward[step_idx],
                        *term_values.tolist(),
                    ]
                )

    def _write_reward_curve_plot(self, png_path, rewards, term_names, step_dt, motion_key):
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        steps = np.arange(rewards.shape[0])
        time_axis = (steps + 1) * step_dt
        total_reward = rewards.sum(axis=1)
        fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)

        axes[0].plot(time_axis, total_reward, linewidth=1.4, color="black", label="total_reward")
        axes[0].axhline(0.0, color="gray", linewidth=0.8, alpha=0.7)
        axes[0].set_title(f"Reward curve: {motion_key}")
        axes[0].set_ylabel("weighted reward")
        axes[0].legend(loc="best", fontsize=8)
        axes[0].grid(True, alpha=0.25)

        for idx, name in enumerate(term_names):
            axes[1].plot(time_axis, rewards[:, idx], linewidth=1.0, label=name)
        axes[1].axhline(0.0, color="gray", linewidth=0.8, alpha=0.7)
        axes[1].set_xlabel("time (s)")
        axes[1].set_ylabel("term contribution")
        axes[1].grid(True, alpha=0.25)
        axes[1].legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=7)

        fig.tight_layout(rect=(0, 0, 0.82, 1))
        fig.savefig(png_path, dpi=160)
        plt.close(fig)

    def _finalize_reward_curve_batch(self, env_motion_ids, valid_lengths):
        if not self.reward_curve_enabled or self.reward_curve_term_names is None:
            return

        output_dir = self._reward_curve_output_base_dir()
        if output_dir is None:
            return
        os.makedirs(output_dir, exist_ok=True)

        env_motion_ids_cpu = env_motion_ids.detach().cpu().numpy()
        valid_lengths_cpu = valid_lengths.detach().cpu().numpy()
        motion_keys = getattr(self.env._motion_lib, "_motion_data_keys", [])
        num_unique_motions = getattr(self.env._motion_lib, "_num_unique_motions", len(motion_keys))
        term_names = self.reward_curve_term_names
        step_dt = self.reward_curve_step_dt or 1.0

        for env_idx, records in enumerate(self.reward_curve_records):
            if len(records) == 0 or env_idx >= len(env_motion_ids_cpu):
                continue
            motion_idx = int(env_motion_ids_cpu[env_idx])
            if motion_idx >= num_unique_motions:
                continue
            max_steps = (
                int(valid_lengths_cpu[env_idx])
                if env_idx < len(valid_lengths_cpu)
                else len(records)
            )
            selected_records = records[:max_steps]
            if len(selected_records) == 0:
                continue
            rewards = np.stack(selected_records, axis=0)
            if rewards.size == 0:
                continue

            motion_key = (
                motion_keys[motion_idx]
                if motion_idx < len(motion_keys)
                else f"motion_{motion_idx:06d}"
            )
            basename = self._safe_reward_curve_name(motion_key, motion_idx)
            csv_path = os.path.join(output_dir, f"{basename}.csv")
            png_path = os.path.join(output_dir, f"{basename}.png")

            self._write_reward_curve_csv(csv_path, rewards, term_names, step_dt)
            self._write_reward_curve_plot(png_path, rewards, term_names, step_dt, motion_key)
            self.reward_curve_files_all.append(
                {
                    "motion_idx": motion_idx,
                    "motion_key": str(motion_key),
                    "num_steps": int(rewards.shape[0]),
                    "csv": csv_path,
                    "png": png_path,
                }
            )

    def _collect_object_tracking_errors(self):
        """Collect per-step object position and orientation errors (ref vs simulated)."""
        try:
            obj = self.env.env.scene["object"]
            motion_cmd = self.env.motion_command
            current_obj_pos = obj.data.root_pos_w[:, :3]  # (num_envs, 3)
            current_obj_quat = obj.data.root_quat_w  # (num_envs, 4)
            target_obj_pos = motion_cmd.object_root_pos[:, 0, :3]  # (num_envs, 3)
            target_obj_quat = motion_cmd.object_root_quat[:, 0]  # (num_envs, 4)

            pos_error = torch.norm(target_obj_pos - current_obj_pos, dim=-1)  # (num_envs,)

            # Quaternion error: angle between two quaternions
            from isaaclab.utils.math import quat_error_magnitude

            ori_error = quat_error_magnitude(target_obj_quat, current_obj_quat)  # (num_envs,)

            if self.scalable_eval:
                self.obj_pos_error.append(pos_error.detach())
                self.obj_ori_error.append(ori_error.detach())
            else:
                self.obj_pos_error.append(pos_error.cpu())
                self.obj_ori_error.append(ori_error.cpu())
        except (KeyError, AttributeError, IndexError):
            # Gracefully handle missing object, motion data, or shape mismatches
            pass

    def env_step(self, actor_state):
        obs_dict, rewards, dones, extras = self.env.step(actor_state)
        actor_state.update({"obs": obs_dict, "rewards": rewards, "dones": dones, "extras": extras})
        return actor_state

    def _pre_eval_env_step(self, actor_state: dict):
        dones = actor_state.get("dones", torch.zeros(self.env.num_envs, device=self.env.device))
        actions = self.eval_policy(
            obs_dict=actor_state["obs"], cur_dones=dones, skip_episode_attnmask=True
        )
        actor_state.update({"actions": actions})
        return actor_state

    def _finish_scalable_run(self, actor_state):
        self._write_foot_force_manifest()
        self.accelerator.wait_for_everyone()
        aggregated = aggregate_batches(
            self.batch_store.load_all(),
            expected_motion_keys=self.run_manifest.expected_motion_keys,
            allow_partial=self.allow_partial,
        )
        metrics_success = dict(aggregated.metrics_success)
        metrics_success["success_rate"] = aggregated.success_rate
        metrics_success["progress_rate"] = aggregated.progress_rate
        actor_state["metrics_all"] = dict(aggregated.metrics_all)
        actor_state["metrics_success"] = metrics_success
        actor_state["failed_keys"] = aggregated.failed_keys
        actor_state["success_keys"] = aggregated.success_keys
        actor_state["all_metrics_dict"] = dict(aggregated.all_metrics_dict)
        actor_state["failed_metrics_dict"] = dict(aggregated.failed_metrics_dict)
        actor_state["failed_idxes"] = aggregated.failed_idxes
        actor_state["eval_meta"] = {
            "schema_version": 1,
            "partial": aggregated.partial,
            "evaluated_motion_count": aggregated.evaluated_motion_count,
            "expected_motion_count": aggregated.expected_motion_count,
            "missing_motion_count": len(aggregated.missing_motion_keys),
            "missing_motion_keys": list(aggregated.missing_motion_keys),
            "completed_batch_count": aggregated.completed_batch_count,
        }
        if self.reward_curve_enabled:
            actor_state["reward_curves"] = self.reward_curve_files_all
        if self._foot_force_enabled():
            actor_state["foot_force_files"] = self.foot_force_files_all
        if self.accelerator.is_main_process:
            print(f"Success Rate: {aggregated.success_rate:.10f}", flush=True)
            print(f"Progress Rate: {aggregated.progress_rate:.10f}", flush=True)
            print(
                "All: ",
                " \t".join(
                    [f"{key}: {value:.3f}" for key, value in aggregated.metrics_all.items()]
                ),
                flush=True,
            )
            print(
                "Succ: ",
                " \t".join(
                    [
                        f"{key}: {value:.3f}"
                        for key, value in aggregated.metrics_success.items()
                    ]
                ),
                flush=True,
            )
        self._write_scalable_snapshot(force=True)
        actor_state["end_eval"] = True
        if self.pbar.n < self.num_total_env_eval_loops:
            self.pbar.update(self.num_total_env_eval_loops - self.pbar.n)
        self.pbar.refresh()
        return actor_state

    def _finalize_scalable_batch(self, actor_state):
        motion_num_steps = self.env._motion_lib.get_motion_num_steps(self.env.motion_ids)
        env_motion_ids = self.env.start_idx + self.env.motion_ids
        motion_ids_cpu = env_motion_ids.detach().cpu().tolist()
        selected_envs = []
        selected_motion_ids = []
        seen_motion_ids = set()
        num_unique = int(self.env._motion_lib._num_unique_motions)
        for env_idx, motion_idx in enumerate(motion_ids_cpu):
            motion_idx = int(motion_idx)
            if motion_idx >= num_unique or motion_idx in seen_motion_ids:
                continue
            selected_envs.append(env_idx)
            selected_motion_ids.append(motion_idx)
            seen_motion_ids.add(motion_idx)
        if not selected_envs:
            raise RuntimeError(
                f"evaluation batch {self.env_eval_loop_idx} contains no valid unique motions"
            )

        selected = torch.tensor(selected_envs, device=self.env.device, dtype=torch.long)
        selected_motion_idx = torch.tensor(
            selected_motion_ids, device=self.env.device, dtype=torch.long
        )
        pred_pos = torch.stack(self.pred_pos, dim=0).index_select(1, selected)
        gt_pos = torch.stack(self.gt_pos, dim=0).index_select(1, selected)
        selected_lengths = motion_num_steps.index_select(0, selected)
        selected_terminated = self.terminate_state.index_select(0, selected)
        selected_valid_lengths = self.valid_frame_count.index_select(0, selected)
        selected_progress = (
            self.progress_state.index_select(0, selected) / selected_lengths
        )
        sampling_prob = self.env._motion_lib._sampling_prob.index_select(
            0, selected_motion_idx
        )
        obj_pos = obj_ori = None
        if self._has_object and self.obj_pos_error:
            obj_pos = torch.stack(self.obj_pos_error, dim=0).index_select(1, selected)
            obj_ori = torch.stack(self.obj_ori_error, dim=0).index_select(1, selected)
        if not hasattr(self.env, "motion_command") or self.env.motion_command is None:
            raise RuntimeError("scalable evaluation requires env.motion_command body names")

        payload = build_batch_payload(
            pred_pos=pred_pos,
            gt_pos=gt_pos,
            motion_lengths=selected_lengths,
            valid_lengths=selected_valid_lengths,
            motion_idx=selected_motion_idx,
            terminated=selected_terminated,
            progress=selected_progress,
            body_names=self.env.motion_command.cmd_body_names,
            run_id=self.run_manifest.run_id,
            rank=self.args.global_rank,
            batch_idx=self.env_eval_loop_idx,
            sampling_prob=sampling_prob,
            obj_pos_err=obj_pos,
            obj_ori_err=obj_ori,
        )
        batch_path = self.batch_store.write(
            rank=self.args.global_rank,
            batch_idx=self.env_eval_loop_idx,
            payload=payload,
        )
        print(
            f"[scalable_eval] rank={self.args.global_rank} batch="
            f"{self.env_eval_loop_idx} motions={len(selected_motion_ids)} -> {batch_path}",
            flush=True,
        )

        self._finalize_reward_curve_batch(env_motion_ids, self.valid_frame_count)
        self._finalize_foot_force_batch(env_motion_ids, self.valid_frame_count)
        terminated_np = np.asarray(payload["terminated"], dtype=bool)
        progress_np = np.asarray(payload["progress"], dtype=np.float64).copy()
        progress_np[~terminated_np] = 1.0
        self.scalable_evaluated_count += int(terminated_np.size)
        self.scalable_terminated_count += int(terminated_np.sum())
        self.scalable_progress_sum += float(progress_np.sum())
        self.env_eval_loop_idx += 1
        self.success_rate = 1.0 - (
            self.scalable_terminated_count / self.scalable_evaluated_count
        )
        self.progress_rate = self.scalable_progress_sum / self.scalable_evaluated_count
        self._write_scalable_snapshot()

        if self.env_eval_loop_idx >= self.num_total_env_eval_loops:
            return self._finish_scalable_run(actor_state)

        self.env.forward_motion_samples(self.args.global_rank, self.args.world_size)
        self.terminate_state = torch.zeros(self.env.num_envs, device=self.device)
        self.progress_state = torch.zeros(self.env.num_envs, device=self.env.device)
        self.valid_frame_count = torch.zeros(
            self.env.num_envs, device=self.env.device, dtype=torch.long
        )
        self.curr_steps = 0
        actor_state.pop("_eval_step_masks", None)
        self.pbar.update(1)
        self.pbar.refresh()
        self.pbar.set_description(
            f"Scalable eval batch {self.env_eval_loop_idx}/{self.num_total_env_eval_loops} "
            f"| motions {self.scalable_evaluated_count} | succ {self.success_rate:.3f}"
        )
        self._reset_scalable_batch_buffers()
        return actor_state

    def _eval_step_masks(self, actor_state):
        cached = actor_state.get("_eval_step_masks")
        if cached is not None and cached[0] == self.curr_steps:
            return cached[1], cached[2]
        motion_num_steps = self.env._motion_lib.get_motion_num_steps(self.env.motion_ids)
        dones = actor_state["dones"].bool()
        time_outs = actor_state["extras"]["time_outs"].bool()
        died = dones & ~time_outs
        terminating_now = (self.curr_steps <= motion_num_steps - 1) & died
        valid_frame = valid_eval_frame_mask(
            self.curr_steps,
            motion_num_steps,
            self.terminate_state,
            terminating_now,
        )
        cached = (self.curr_steps, terminating_now, valid_frame)
        actor_state["_eval_step_masks"] = cached
        return terminating_now, valid_frame

    def _post_eval_env_step(self, actor_state):
        actor_state["end_eval"] = False
        termination_state, valid_frame = self._eval_step_masks(actor_state)

        if "ref_body_pos_extend" in self.env.extras:
            gt_pos = self.env.extras["ref_body_pos_extend"]
            pred_pos = self.env.extras["rigid_body_pos_extend"]
            mpjpe = self.env.dif_global_body_pos.norm(dim=-1) * 1000
        else:
            gt_pos = self.env.get_env_data("ref_body_pos_extend")
            pred_pos = self.env.get_env_data("rigid_body_pos_extend")
            mpjpe = (gt_pos - pred_pos).norm(dim=-1) * 1000
        if self.scalable_eval:
            self.gt_pos.append(gt_pos.detach())
            self.pred_pos.append(pred_pos.detach())
            self.mpjpe.append(mpjpe.detach())
        else:
            self.gt_pos.append(gt_pos.cpu().numpy())
            self.pred_pos.append(pred_pos.cpu().numpy())
            self.mpjpe.append(mpjpe.cpu())

        # Collect object tracking errors if object exists in scene
        if self._has_object:
            self._collect_object_tracking_errors()

        self._collect_reward_curve_step(actor_state)
        self._collect_foot_force_step(actor_state)

        # self.gt_rot.append(self.env.extras['ref_body_rot_extend'].cpu().numpy())
        # self.pred_rot.append(self.env._rigid_body_rot_extend.cpu().numpy())

        self.valid_frame_count += valid_frame.to(self.valid_frame_count.dtype)
        self.terminate_state = torch.logical_or(termination_state, self.terminate_state)

        self.progress_state[~self.terminate_state] += 1

        if (~self.terminate_state).sum() > 0:
            max_possible_id = self.env._motion_lib._num_unique_motions - 1
            curr_ids = self.env._motion_lib._curr_motion_ids
            if (max_possible_id == curr_ids).sum() > 0:  # When you are running out of motions.
                bound = (max_possible_id == curr_ids).nonzero()[0] + 1
                if (~self.terminate_state[:bound]).sum() > 0:
                    curr_max = (
                        self.env._motion_lib.get_motion_num_steps(self.env.motion_ids)[:bound][
                            ~self.terminate_state[:bound]
                        ]
                        .max()
                        .item()
                    )
                else:
                    curr_max = self.curr_steps - 1  # the ones that should be counted have teimrated
            else:
                curr_max = (
                    self.env._motion_lib.get_motion_num_steps(self.env.motion_ids)[
                        ~self.terminate_state
                    ]
                    .max()
                    .item()
                )

            if self.curr_steps >= curr_max:
                curr_max = self.curr_steps + 1  # For matching up the current steps and max steps.
        else:
            curr_max = self.env._motion_lib.get_motion_num_steps(self.env.motion_ids).max().item()

        if self.steps_pbar is None and (~self.terminate_state).sum() > 0:
            self.steps_pbar = tqdm(
                total=int(curr_max),
                desc="Sequence progress",
                leave=False,
                disable=self.scalable_eval,
            )

        if self.steps_pbar is not None:
            self.steps_pbar.update(1)
            if self.steps_pbar.total != int(curr_max):
                self.steps_pbar.total = int(curr_max)
                self.steps_pbar.refresh()

        self.curr_steps += 1
        if self.curr_steps >= curr_max or self.terminate_state.sum() == self.env.num_envs:
            if self.steps_pbar is not None:
                self.steps_pbar.close()
                self.steps_pbar = None

            if self.scalable_eval:
                return self._finalize_scalable_batch(actor_state)

            self.terminate_memory.append(self.terminate_state.cpu().numpy())
            self.progress_memory.append(
                (
                    self.progress_state
                    / self.env._motion_lib.get_motion_num_steps(self.env.motion_ids)
                )
                .cpu()
                .numpy()
            )

            self.success_rate = (
                1
                - np.concatenate(self.terminate_memory)[
                    : self.env._motion_lib._num_unique_motions
                ].mean()
            )
            self.progress_rate = np.concatenate(self.progress_memory)[
                : self.env._motion_lib._num_unique_motions
            ].mean()

            # MPJPE
            all_mpjpe = torch.stack(self.mpjpe)
            try:
                assert (
                    all_mpjpe.shape[0] == curr_max
                    or self.terminate_state.sum() == self.env.num_envs
                )  # Max should be the same as the number of frames in the motion.
            except AssertionError:
                print(
                    f"Warning: MPJPE shape mismatch: {all_mpjpe.shape[0]} vs curr_max={curr_max}, terminated={self.terminate_state.sum()}/{self.env.num_envs}"
                )

            all_body_pos_pred = np.stack(self.pred_pos)
            all_body_pos_gt = np.stack(self.gt_pos)
            # all_body_rot_pred = np.stack(self.pred_rot)
            # all_body_rot_gt = np.stack(self.gt_rot)

            effective_lengths = self.valid_frame_count.detach().cpu().tolist()
            all_mpjpe = [
                all_mpjpe[:length, idx].mean()
                if length
                else all_mpjpe.new_tensor(float("nan"))
                for idx, length in enumerate(effective_lengths)
            ]
            all_body_pos_pred = [
                all_body_pos_pred[:length, idx]
                for idx, length in enumerate(effective_lengths)
            ]
            all_body_pos_gt = [
                all_body_pos_gt[:length, idx]
                for idx, length in enumerate(effective_lengths)
            ]
            # all_body_rot_pred = [all_body_rot_pred[: (i - 1), idx] for idx, i in enumerate(self.env._motion_lib.get_motion_num_steps())]
            # all_body_rot_gt = [all_body_rot_gt[: (i - 1), idx] for idx, i in enumerate(self.env._motion_lib.get_motion_num_steps())]

            self.mpjpe_all.append(all_mpjpe)
            self.pred_pos_all += all_body_pos_pred
            self.gt_pos_all += all_body_pos_gt
            # self.pred_rot_all += all_body_rot_pred
            # self.gt_rot_all += all_body_rot_gt

            # Aggregate object tracking errors for this batch
            if self._has_object and len(self.obj_pos_error) > 0:
                all_obj_pos_err = torch.stack(self.obj_pos_error)  # (T, num_envs)
                all_obj_ori_err = torch.stack(self.obj_ori_error)  # (T, num_envs)
                per_env_obj_pos_err = [
                    all_obj_pos_err[:length, idx].mean().item() if length else float("nan")
                    for idx, length in enumerate(effective_lengths)
                ]
                per_env_obj_ori_err = [
                    all_obj_ori_err[:length, idx].mean().item() if length else float("nan")
                    for idx, length in enumerate(effective_lengths)
                ]
                self.obj_pos_error_all.append(per_env_obj_pos_err)
                self.obj_ori_error_all.append(per_env_obj_ori_err)

            env_motion_ids = self.env.start_idx + self.env.motion_ids
            self.sampled_motion_idx.append(env_motion_ids)
            self._finalize_reward_curve_batch(
                env_motion_ids, self.valid_frame_count
            )
            self._finalize_foot_force_batch(
                env_motion_ids, self.valid_frame_count
            )
            self.env_eval_loop_idx += 1

            if self.env_eval_loop_idx >= self.num_total_env_eval_loops:
                self._write_foot_force_manifest()
                if self.render_only:
                    print("Rendering only. Reached the end of the evaluation loop.")
                    self.env.end_render_results()
                    actor_state["end_eval"] = True
                    return actor_state

                terminate_hist = np.concatenate(self.terminate_memory)
                progress_hist = np.concatenate(self.progress_memory)
                succ_idxes = np.nonzero(
                    ~terminate_hist[: self.env._motion_lib._num_unique_motions]
                )[0].tolist()
                self.accelerator.wait_for_everyone()
                # metrics_all = compute_metrics_lite(self.pred_pos_all, self.gt_pos_all, self.pred_rot_all, self.gt_rot_all, concatenate = False) # OOM

                print(
                    f"!!!!!!! {len(self.pred_pos_all)} {len(self.gt_pos_all)} {self.env.start_idx} {self.args.global_rank} Time: {datetime.now().strftime('%H:%M:%S')}"
                )

                if hasattr(self.env, "motion_command"):
                    body_names = self.env.motion_command.cmd_body_names
                else:
                    print("No self.env.motion_command.cmd_body_names found!!!!")
                    exit()

                """
                # gear_sonic/config/manager_env/commands/terms/motion.yaml
                body_names: [
                    "pelvis",
                    "left_hip_roll_link",
                    "left_knee_link",
                    "left_ankle_roll_link",
                    "right_hip_roll_link",
                    "right_knee_link",
                    "right_ankle_roll_link",
                    "torso_link",
                    "left_shoulder_roll_link",
                    "left_elbow_link",
                    "left_wrist_yaw_link",
                    "right_shoulder_roll_link",
                    "right_elbow_link",
                    "right_wrist_yaw_link",
                ]
                """

                # Define subsets
                # 6 + 3 + 5 = 14
                legs_subset_names = [
                    "left_hip_roll_link",
                    "left_knee_link",
                    "left_ankle_roll_link",
                    "right_hip_roll_link",
                    "right_knee_link",
                    "right_ankle_roll_link",
                ]
                # NOTE use torso_link instead of head for vr_3points_subset_names
                vr_3points_subset_names = [
                    "torso_link",
                    "left_wrist_yaw_link",
                    "right_wrist_yaw_link",
                ]
                other_upper_bodies_subset_names = [
                    "pelvis",
                    "left_shoulder_roll_link",
                    "left_elbow_link",
                    "right_shoulder_roll_link",
                    "right_elbow_link",
                ]

                foot_subset_names = ["left_ankle_roll_link", "right_ankle_roll_link"]

                # Case-insensitive match (A3 uses "_Link" while G1 uses "_link";
                # A3 also adds "_link" suffix to pelvis which G1 leaves bare)
                _body_name_lookup = {n.lower(): n for n in body_names}

                def _lookup_idx(name):
                    key = name.lower()
                    # 1. exact case-insensitive match
                    if key in _body_name_lookup:
                        return body_names.index(_body_name_lookup[key])
                    # 2. try adding "_link" suffix (G1 "pelvis" -> A3 "pelvis_link")
                    if (key + "_link") in _body_name_lookup:
                        return body_names.index(_body_name_lookup[key + "_link"])
                    # 3. try removing "_link" suffix (reverse direction)
                    if key.endswith("_link") and key[:-5] in _body_name_lookup:
                        return body_names.index(_body_name_lookup[key[:-5]])
                    # 4. fallback: any body name ending with "_" + requested
                    for bn in body_names:
                        if bn.lower().endswith("_" + key):
                            return body_names.index(bn)
                    raise ValueError(
                        f"'{name}' not found in body_names (case-insensitive). "
                        f"Available: {body_names}"
                    )

                # Get indices for subsets
                legs_indices = [_lookup_idx(name) for name in legs_subset_names]
                vr_3points_indices = [_lookup_idx(name) for name in vr_3points_subset_names]
                other_upper_bodies_indices = [
                    _lookup_idx(name) for name in other_upper_bodies_subset_names
                ]
                foot_indices = [_lookup_idx(name) for name in foot_subset_names]
                pelvis_idx = _lookup_idx("pelvis")

                metrics_all = compute_metrics_lite_with_empty(
                    self.pred_pos_all, self.gt_pos_all
                )  # list of length N_env
                metrics_all.update(compute_joint_mse_metrics(self.pred_pos_all, self.gt_pos_all))
                metrics_legs = compute_subset_metrics_lite(
                    self.pred_pos_all,
                    self.gt_pos_all,
                    subset_indices=legs_indices,
                    root_idx=pelvis_idx,
                )
                metrics_vr_3points = compute_subset_metrics_lite(
                    self.pred_pos_all,
                    self.gt_pos_all,
                    subset_indices=vr_3points_indices,
                    root_idx=pelvis_idx,
                )
                metrics_other_upper_bodies = compute_subset_metrics_lite(
                    self.pred_pos_all,
                    self.gt_pos_all,
                    subset_indices=other_upper_bodies_indices,
                    root_idx=pelvis_idx,
                )
                metrics_foot = compute_subset_metrics_lite(
                    self.pred_pos_all,
                    self.gt_pos_all,
                    subset_indices=foot_indices,
                    root_idx=pelvis_idx,
                )

                # Rename keys for subset metrics
                metrics_legs = {f"{k}_legs": v for k, v in metrics_legs.items()}
                metrics_vr_3points = {f"{k}_vr_3points": v for k, v in metrics_vr_3points.items()}
                metrics_other_upper_bodies = {
                    f"{k}_other_upper_bodies": v for k, v in metrics_other_upper_bodies.items()
                }
                metrics_foot = {f"{k}_foot": v for k, v in metrics_foot.items()}

                metrics_all.update(metrics_legs)
                metrics_all.update(metrics_vr_3points)
                metrics_all.update(metrics_other_upper_bodies)
                metrics_all.update(metrics_foot)

                metric_sums_np, metric_counts_np = compute_metric_sums_and_counts(metrics_all)
                metrics_all_sum = {
                    key: torch.as_tensor(value, device=self.env.device)
                    for key, value in metric_sums_np.items()
                }
                metrics_all_count = {
                    key: torch.as_tensor(value, device=self.env.device)
                    for key, value in metric_counts_np.items()
                }

                metrics_all_contactnate = torch.stack(
                    list(metrics_all_sum.values()) + list(metrics_all_count.values()), dim=-1
                )
                terminate_hist_concatenate = torch.tensor(terminate_hist).to(self.env.device)
                progress_hist_concatenate = torch.tensor(progress_hist).to(self.env.device)
                all_motion_idxes = torch.cat(self.sampled_motion_idx).to(self.env.device)

                # Prepare object tracking metrics for gathering
                has_obj_metrics = self._has_object and len(self.obj_pos_error_all) > 0
                if has_obj_metrics:
                    obj_pos_err_flat = torch.tensor(
                        [v for batch in self.obj_pos_error_all for v in batch]
                    ).to(self.env.device)
                    obj_ori_err_flat = torch.tensor(
                        [v for batch in self.obj_ori_error_all for v in batch]
                    ).to(self.env.device)

                # Tensor layout: [metric sums..., metric counts..., terminate,
                # progress, (obj_pos_err, obj_ori_err,), motion_idx]
                tail_tensors = [
                    terminate_hist_concatenate[:, None],
                    progress_hist_concatenate[:, None],
                ]
                if has_obj_metrics:
                    tail_tensors.append(obj_pos_err_flat[:, None])
                    tail_tensors.append(obj_ori_err_flat[:, None])
                tail_tensors.append(all_motion_idxes[:, None])

                all_tensors = torch.cat(
                    [metrics_all_contactnate] + tail_tensors,
                    dim=-1,
                )
                print("Gathering eval tensors", all_tensors.shape, self.accelerator.process_index)

                chunk_size = 1024  # Chunk gathering since it's 4096 is too large.
                chunks = all_tensors.split(chunk_size)
                gathered_chunks = [
                    self.accelerator.gather(chunk).reshape(-1, *chunk.shape) for chunk in chunks
                ]  # each with shape (num_processes x 1024 (chunked_num_env), D_metrics)
                all_metrics = torch.cat(gathered_chunks, dim=1)

                metric_size = all_metrics.shape[-1]
                gathered_metrics_stack = (
                    all_metrics.reshape(
                        self.accelerator.num_processes, -1, self.env.num_envs, metric_size
                    )
                    .transpose(0, 1)
                    .reshape(-1, metric_size)[: self.env._motion_lib._num_unique_motions]
                )  # make sure that we are selecting the correct ones.

                # Extract tail columns: terminate, progress, (obj_pos_err, obj_ori_err,) motion_idx
                num_tail = 3 + (
                    2 if has_obj_metrics else 0
                )  # terminate + progress + (obj*2) + motion_idx
                num_body_metrics = metric_size - num_tail
                num_metrics = len(metrics_all_sum)
                assert num_body_metrics == num_metrics * 2

                gathered_terminate_hist_stack = gathered_metrics_stack[:, num_body_metrics].bool()
                gathered_progress_hist_stack = gathered_metrics_stack[:, num_body_metrics + 1]
                if has_obj_metrics:
                    gathered_obj_pos_err = gathered_metrics_stack[:, num_body_metrics + 2]
                    gathered_obj_ori_err = gathered_metrics_stack[:, num_body_metrics + 3]
                    gathered_motion_idxes = gathered_metrics_stack[:, num_body_metrics + 4].long()
                else:
                    gathered_motion_idxes = gathered_metrics_stack[:, num_body_metrics + 2].long()
                gathered_progress_hist_stack[~gathered_terminate_hist_stack] = 1

                assert (gathered_motion_idxes.diff(dim=0) == 1).all()

                # Micro-average: sum all frame-level sums, divide by total frames
                # (each timestep weighted equally, longer motions contribute more)
                metric_sums = gathered_metrics_stack[:, :num_metrics]
                metric_counts = gathered_metrics_stack[:, num_metrics:num_body_metrics]

                success_mask = ~gathered_terminate_hist_stack
                success_metric_sums = metric_sums[success_mask].sum(dim=0)
                success_metric_counts = metric_counts[success_mask].sum(dim=0)
                success_metrics_mean = torch.where(
                    success_metric_counts > 0,
                    success_metric_sums / success_metric_counts,
                    torch.nan,
                )
                all_metric_sums = metric_sums.sum(dim=0)
                all_metric_counts = metric_counts.sum(dim=0)
                all_metrics_mean = torch.where(
                    all_metric_counts > 0,
                    all_metric_sums / all_metric_counts,
                    torch.nan,
                )

                # Also keep per-motion metrics for downstream use
                all_metrics = torch.where(
                    metric_counts > 0,
                    metric_sums / metric_counts,
                    torch.nan,
                )
                metrics_all_print = {
                    k: all_metrics_mean[idx].cpu().numpy()
                    for idx, (k, v) in enumerate(metrics_all_sum.items())
                }
                metrics_succ_print = {
                    k: success_metrics_mean[idx].cpu().numpy()
                    for idx, (k, v) in enumerate(metrics_all_sum.items())
                }

                # Add object tracking metrics to printed summaries
                if has_obj_metrics:
                    obj_pos_err_mean = gathered_obj_pos_err.mean().cpu().numpy()
                    obj_ori_err_mean = gathered_obj_ori_err.mean().cpu().numpy()
                    obj_pos_err_succ = (
                        gathered_obj_pos_err[~gathered_terminate_hist_stack].mean().cpu().numpy()
                        if (~gathered_terminate_hist_stack).any()
                        else 0.0
                    )
                    obj_ori_err_succ = (
                        gathered_obj_ori_err[~gathered_terminate_hist_stack].mean().cpu().numpy()
                        if (~gathered_terminate_hist_stack).any()
                        else 0.0
                    )
                    metrics_all_print["obj_pos_error"] = obj_pos_err_mean
                    metrics_all_print["obj_ori_error"] = obj_ori_err_mean
                    metrics_succ_print["obj_pos_error"] = obj_pos_err_succ
                    metrics_succ_print["obj_ori_error"] = obj_ori_err_succ

                failed_keys = self.env._motion_lib._motion_data_keys[
                    gathered_terminate_hist_stack.cpu().numpy()
                ]
                success_keys = self.env._motion_lib._motion_data_keys[
                    ~gathered_terminate_hist_stack.cpu().numpy()
                ]
                success_rate = 1 - gathered_terminate_hist_stack.cpu().numpy().mean()
                progress_rate = gathered_progress_hist_stack.cpu().numpy().mean()

                all_metrics_dict = {
                    k: all_metrics[:, idx].cpu().numpy()
                    for idx, (k, v) in enumerate(metrics_all_sum.items())
                }
                all_metrics_dict["terminated"] = gathered_terminate_hist_stack.cpu().numpy()
                all_metrics_dict["progress"] = gathered_progress_hist_stack.cpu().numpy()
                all_metrics_dict["motion_keys"] = self.env._motion_lib._motion_data_keys[
                    gathered_motion_idxes.cpu().numpy()
                ]
                all_metrics_dict["sampling_prob"] = (
                    self.env._motion_lib._sampling_prob[gathered_motion_idxes.cpu().numpy()]
                    .cpu()
                    .numpy()
                )
                # Add per-motion object tracking metrics
                if has_obj_metrics:
                    all_metrics_dict["obj_pos_error"] = gathered_obj_pos_err.cpu().numpy()
                    all_metrics_dict["obj_ori_error"] = gathered_obj_ori_err.cpu().numpy()
                    # Save per-env obj_pos_error for threshold-based success rate analysis
                    if self.eval_only and len(self.obj_pos_error_all) > 0:
                        all_metrics_dict["per_env_obj_pos_error"] = [
                            v for batch in self.obj_pos_error_all for v in batch
                        ]
                        all_metrics_dict["per_env_obj_ori_error"] = [
                            v for batch in self.obj_ori_error_all for v in batch
                        ]

                failed_metrics_dict = {
                    k: all_metrics[gathered_terminate_hist_stack, idx].cpu().numpy()
                    for idx, (k, v) in enumerate(metrics_all_sum.items())
                }
                failed_metrics_dict["motion_keys"] = failed_keys
                failed_metrics_dict["sampling_prob"] = (
                    self.env._motion_lib._sampling_prob[gathered_terminate_hist_stack.cpu().numpy()]
                    .cpu()
                    .numpy()
                )
                if has_obj_metrics:
                    failed_metrics_dict["obj_pos_error"] = (
                        gathered_obj_pos_err[gathered_terminate_hist_stack].cpu().numpy()
                    )
                    failed_metrics_dict["obj_ori_error"] = (
                        gathered_obj_ori_err[gathered_terminate_hist_stack].cpu().numpy()
                    )

                if self.accelerator.is_main_process:
                    print(f"Success Rate: {success_rate:.10f}")
                    print(f"Progress Rate: {progress_rate:.10f}")
                    if has_obj_metrics:
                        print(
                            f"Object Pos Error (all): {obj_pos_err_mean:.4f}m | "
                            f"Object Ori Error (all): {obj_ori_err_mean:.4f}rad"
                        )
                    print(
                        "All: ", " \t".join([f"{k}: {v:.3f}" for k, v in metrics_all_print.items()])
                    )
                    print(
                        "Succ: ",
                        " \t".join([f"{k}: {v:.3f}" for k, v in metrics_succ_print.items()]),
                    )

                metrics_succ_print["success_rate"] = success_rate
                metrics_succ_print["progress_rate"] = progress_rate
                actor_state["metrics_all"] = metrics_all_print
                actor_state["metrics_success"] = metrics_succ_print
                actor_state["failed_keys"] = failed_keys
                actor_state["success_keys"] = success_keys
                actor_state["all_metrics_dict"] = all_metrics_dict
                actor_state["failed_metrics_dict"] = failed_metrics_dict
                actor_state["failed_idxes"] = (
                    gathered_terminate_hist_stack.cpu().numpy().nonzero()[0]
                )
                if self.reward_curve_enabled:
                    actor_state["reward_curves"] = self.reward_curve_files_all
                if self._foot_force_enabled():
                    actor_state["foot_force_files"] = self.foot_force_files_all

                if not self.eval_only:
                    del (
                        self.mpjpe,
                        self.mpjpe_all,
                        self.gt_pos,
                        self.gt_pos_all,
                        self.gt_rot,
                        self.gt_rot_all,
                        self.pred_pos,
                        self.pred_pos_all,
                        self.pred_rot,
                        self.pred_rot_all,
                        self.sampled_motion_idx,
                        self.obj_pos_error,
                        self.obj_pos_error_all,
                        self.obj_ori_error,
                        self.obj_ori_error_all,
                    )
                    gc.collect()
                    torch.cuda.empty_cache()

                actor_state["end_eval"] = True
                self.pbar.update(1)
                self.pbar.refresh()
                return actor_state

            self.env.forward_motion_samples(self.args.global_rank, self.args.world_size)
            self.terminate_state = torch.zeros(self.env.num_envs, device=self.device)
            self.progress_state = torch.zeros(self.env.num_envs, device=self.env.device)
            self.valid_frame_count = torch.zeros(
                self.env.num_envs, device=self.env.device, dtype=torch.long
            )

            self.success_rate = 0
            self.curr_steps = 0
            actor_state.pop("_eval_step_masks", None)

            self.pbar.update(1)
            self.pbar.refresh()
            (
                self.mpjpe,
                self.gt_pos,
                self.pred_pos,
                self.obj_pos_error,
                self.obj_ori_error,
            ) = (
                [],
                [],
                [],
                [],
                [],
            )
            self._reset_reward_curve_batch_buffers()
            self._reset_foot_force_batch_buffers()

        if self.scalable_eval:
            return actor_state

        eval_time = (time.time() - self.time_eval_start) / 60  # in minutes
        obj_str = ""
        if self._has_object and len(self.obj_pos_error_all) > 0:
            mean_obj_pos_err = np.mean([v for batch in self.obj_pos_error_all for v in batch])
            obj_str = f" | ObjPosErr: {mean_obj_pos_err:.4f}m"
        update_str = f"Terminated: {self.terminate_state.sum().item()} | max frames: {curr_max} | steps {self.curr_steps} | env_loop: {self.env_eval_loop_idx} | eval_time: {eval_time:.1f}m | Start: {self.env.start_idx} | Succ rate: {self.success_rate:.3f} | Mpjpe: {np.mean(self.mpjpe_all) * 1000:.3f}{obj_str}"
        self.pbar.set_description(update_str)

        return actor_state
