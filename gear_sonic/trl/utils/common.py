"""Miscellaneous training utilities: W&B helpers, dynamic imports, OmegaConf tools, and timers."""

import hashlib
import importlib
import os
import time

from omegaconf import DictConfig, ListConfig, OmegaConf
import wandb


def wandb_run_exists():
    return isinstance(wandb.run, wandb.sdk.wandb_run.Run)


def import_type_from_str(s):
    module_name, type_name = s.rsplit(".", 1)
    module = importlib.import_module(module_name)
    type_to_import = getattr(module, type_name)
    return type_to_import


def recursive_set_struct(cfg, struct_value: bool):
    OmegaConf.set_struct(cfg, struct_value)
    if isinstance(cfg, DictConfig):
        for key in cfg.keys():
            try:
                value = cfg[key]
                if isinstance(value, (DictConfig, ListConfig)):
                    recursive_set_struct(value, struct_value)
            except Exception as e:
                # print(e)
                pass
    elif isinstance(cfg, ListConfig):
        for item in cfg:
            if isinstance(item, (DictConfig, ListConfig)):
                recursive_set_struct(item, struct_value)


def materialize_lazy_params(policy, env):
    """Materialize lazy parameters (nn.LazyLinear, nn.LazyConv2d) with a dummy forward pass.

    Must be called before DDP wrapping, since accelerator.prepare() requires all params initialized.
    Uses env.reset() with default flatten_dict_obs=True to get flat tensors (not sub-dicts).
    """
    import torch
    import torch.nn as nn

    if any(isinstance(m, (nn.LazyLinear, nn.LazyConv2d)) for m in policy.modules()):
        dummy_obs = env.reset()
        with torch.no_grad():
            policy.act(dummy_obs)


def _distributed_rank_context(accelerator=None):
    global_rank = getattr(accelerator, "process_index", os.environ.get("RANK", 0))
    local_rank = getattr(accelerator, "local_process_index", os.environ.get("LOCAL_RANK", 0))
    world_size = getattr(accelerator, "num_processes", os.environ.get("WORLD_SIZE", 1))
    return (
        f"global_rank={global_rank} local_rank={local_rank} "
        f"world_size={world_size} pid={os.getpid()}"
    )


def distributed_stage(stage, function, *, accelerator=None, log_fn=None):
    """Run one startup stage with rank-aware begin/complete/failure markers."""

    def emit(message):
        if log_fn is None:
            print(message, flush=True)
        else:
            log_fn(message)

    context = _distributed_rank_context(accelerator)
    started_at = time.monotonic()
    emit(f"[DistributedStage] begin stage={stage} {context}")
    try:
        result = function()
    except Exception as exc:
        emit(
            f"[DistributedStage] failed stage={stage} {context} "
            f"elapsed_seconds={time.monotonic() - started_at:.3f} "
            f"error_type={type(exc).__name__} error={exc!r}"
        )
        raise
    emit(
        f"[DistributedStage] complete stage={stage} {context} "
        f"elapsed_seconds={time.monotonic() - started_at:.3f}"
    )
    return result


def _tensor_structure_line(module_name, tensor_kind, tensor_name, tensor):
    try:
        shape = tuple(tensor.shape)
        numel = tensor.numel()
    except RuntimeError:
        shape = ("uninitialized",)
        numel = 0
    requires_grad = getattr(tensor, "requires_grad", False)
    return (
        f"{module_name}|{tensor_kind}|{tensor_name}|{shape}|{tensor.dtype}|"
        f"requires_grad={requires_grad}"
    ), numel


def build_module_parameter_signature(modules):
    """Hash parameter and buffer structure without depending on initialized values."""

    digest = hashlib.sha256()
    parameter_tensors = 0
    parameter_numel = 0
    buffer_tensors = 0
    buffer_numel = 0

    for module_name, module in modules.items():
        if module is None:
            digest.update(f"{module_name}|none\n".encode())
            continue
        for parameter_name, parameter in module.named_parameters():
            line, numel = _tensor_structure_line(
                module_name, "parameter", parameter_name, parameter
            )
            digest.update(f"{line}\n".encode())
            parameter_tensors += 1
            parameter_numel += numel
        for buffer_name, buffer in module.named_buffers():
            line, numel = _tensor_structure_line(module_name, "buffer", buffer_name, buffer)
            digest.update(f"{line}\n".encode())
            buffer_tensors += 1
            buffer_numel += numel

    return {
        "digest": digest.hexdigest(),
        "parameter_tensors": parameter_tensors,
        "parameter_numel": parameter_numel,
        "buffer_tensors": buffer_tensors,
        "buffer_numel": buffer_numel,
    }


def verify_distributed_model_signatures(modules, accelerator):
    """Fail before DDP wrapping when ranks constructed different model structures."""

    import torch.distributed as dist

    signature = build_module_parameter_signature(modules)
    print(
        "[DistributedSignature] local "
        f"{_distributed_rank_context(accelerator)} "
        f"digest={signature['digest']} "
        f"parameter_tensors={signature['parameter_tensors']} "
        f"parameter_numel={signature['parameter_numel']} "
        f"buffer_tensors={signature['buffer_tensors']} "
        f"buffer_numel={signature['buffer_numel']}",
        flush=True,
    )

    world_size = int(getattr(accelerator, "num_processes", 1))
    if world_size <= 1:
        return signature
    if not dist.is_available() or not dist.is_initialized():
        raise RuntimeError(
            "Distributed model signature verification requires an initialized process group"
        )

    gathered = [None] * world_size
    dist.all_gather_object(gathered, signature)
    if any(candidate != gathered[0] for candidate in gathered[1:]):
        per_rank = ", ".join(
            f"rank={rank}:{candidate}" for rank, candidate in enumerate(gathered)
        )
        raise RuntimeError(f"Model parameter signatures differ across ranks: {per_rank}")

    if accelerator.is_main_process:
        print(
            "[DistributedSignature] matched "
            f"world_size={world_size} digest={signature['digest']}",
            flush=True,
        )
    return signature


def get_filtered_state_dict(state_dict, state_dict_key):
    """
    Filter state_dict keys that start with the given prefix and remove the prefix.

    Args:
        state_dict: Dictionary of state dict keys and values
        state_dict_key: Prefix string to filter by

    Returns:
        Filtered dictionary with prefix removed from keys
    """
    filtered_dict = {}
    for key, value in state_dict.items():
        if key.startswith(state_dict_key):
            # Remove the prefix from the key
            new_key = key[len(state_dict_key) :].lstrip(".")
            filtered_dict[new_key] = value
    return filtered_dict


def custom_instantiate(d, _resolve=True, _recursive=False, **add_kwargs):
    """
    Recursively instantiate nested configs with _target_ fields.
    """

    def _recursive_instantiate(obj):
        # If it's a dict and has a _target_, instantiate it
        if isinstance(obj, dict) and "_target_" in obj:
            if obj.get("_recursive_", None) == True:
                assert False, "recursive is not supported"
            obj = obj.copy()
            obj.pop("_recursive_", None)
            obj.pop("_convert_", None)
            obj.pop("_partial_", None)
            _type = import_type_from_str(obj.pop("_target_"))
            # Recursively instantiate all dict/list values
            for k, v in list(obj.items()):
                if isinstance(v, (dict, DictConfig)):
                    obj[k] = _recursive_instantiate(v)
                elif isinstance(v, (list, ListConfig)):
                    obj[k] = [_recursive_instantiate(i) for i in v]
            return _type(**obj)
        # If it's a dict, recursively instantiate its values
        elif isinstance(obj, dict):
            return {k: _recursive_instantiate(v) for k, v in obj.items()}
        # If it's a list, recursively instantiate its items
        elif isinstance(obj, list):
            return [_recursive_instantiate(i) for i in obj]
        else:
            return obj

    # Top-level: allow add_kwargs to override
    d = d.copy()
    if isinstance(d, DictConfig):
        if _resolve:
            d = OmegaConf.to_container(d, resolve=_resolve)
        else:
            recursive_set_struct(d, False)
    if d.get("_recursive_", None) == True:
        assert False, "recursive is not supported"
    d.pop("_recursive_", None)
    d.pop("_convert_", None)
    d.pop("_partial_", None)
    _type = import_type_from_str(d.pop("_target_"))
    if _recursive:
        # Recursively instantiate all dict/list values
        for k, v in list(d.items()):
            if isinstance(v, (dict, DictConfig)):
                d[k] = _recursive_instantiate(v)
            elif isinstance(v, (list, ListConfig)):
                d[k] = [_recursive_instantiate(i) for i in v]
    return _type(**d, **add_kwargs)


# Global variable for timing indentation level
timer_indent_level = 0


# Context manager for timing
class Timer:
    def __init__(self, name="", instance_enabled=True):
        self.name = name
        self.start_time = None
        self.enabled = instance_enabled and os.environ.get("TIMER_ENABLED", "0") == "1"
        if "LOCAL_RANK" in os.environ:
            self.rank = int(os.environ["LOCAL_RANK"])
        else:
            self.rank = 0
        self.show_rank = os.environ.get("TIMER_SHOW_RANK", "0") == "1"
        self.rank_zero_only = os.environ.get("TIMER_RANK_ZERO_ONLY", "0") == "1"

    def __enter__(self):
        if (not self.enabled) or (self.rank_zero_only and self.rank != 0):
            return self
        global timer_indent_level
        self.start_time = time.perf_counter()
        self.current_indent = timer_indent_level  # Capture current indent level
        timer_indent_level += 1  # Increment global indent level for next call
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type:
            return False  # Re-raise the exception
        if (not self.enabled) or (self.rank_zero_only and self.rank != 0):
            return self
        global timer_indent_level
        elapsed_time = time.perf_counter() - self.start_time
        indent = "    " * self.current_indent  # 4 spaces per indent level
        rank_str = f"[rank{self.rank}] " if self.show_rank else ""
        print(f"{indent}{rank_str}[{self.name}] time: {elapsed_time:.4f} seconds")
        timer_indent_level -= 1  # Decrement global indent level after finishing
