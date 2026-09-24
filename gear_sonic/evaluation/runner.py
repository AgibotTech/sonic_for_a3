"""High-level orchestration for the scalable Isaac evaluation CLI."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, replace
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Callable, Mapping, Sequence, TextIO

import yaml

from .aggregator import (
    aggregate_batches,
    aggregate_bucket_outputs,
    write_metrics_eval_json,
)
from .batch_store import BatchStore
from .bucketing import (
    BucketAssignment,
    BucketPlan,
    BucketSpec,
    BucketView,
    build_bucket_plan,
    materialize_bucket_views,
    validate_bucket_specs,
)
from .dataset import (
    LongMotionOptions,
    PreparedDataset,
    SortingOptions,
    prepare_dataset,
    resolve_prepared_output,
)
from .schemas import (
    DatasetManifest,
    ManifestMismatchError,
    RunManifest,
    canonical_json_fingerprint,
    checkpoint_fingerprint,
)
from .validation import (
    ValidationReport,
    validate_bucket_plan,
    validate_evaluation,
)


PACKAGE_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = PACKAGE_DIR / "config" / "default.yaml"
REPO_ROOT = PACKAGE_DIR.parents[1]


class ConfigError(ValueError):
    """Raised when a user-facing evaluation config is invalid."""


@dataclass(frozen=True)
class RunContext:
    config: Mapping[str, Any]
    checkpoint: Path
    output: Path
    prepared: PreparedDataset
    manifest: RunManifest
    command: tuple[str, ...]


@dataclass(frozen=True)
class BucketRunContext:
    assignment: BucketAssignment
    view: BucketView
    output: Path
    manifest: RunManifest
    command: tuple[str, ...]


@dataclass(frozen=True)
class BucketedRunContext:
    config: Mapping[str, Any]
    checkpoint: Path
    output: Path
    prepared: PreparedDataset
    plan: BucketPlan
    manifest: RunManifest
    buckets: tuple[BucketRunContext, ...]


@dataclass(frozen=True)
class BucketedPreparation:
    prepared: PreparedDataset
    plan: BucketPlan
    views: tuple[BucketView, ...]
    plan_path: Path


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ConfigError(f"configuration file does not exist: {path}")
    with path.open(encoding="utf-8") as stream:
        payload = yaml.safe_load(stream) or {}
    if not isinstance(payload, dict):
        raise ConfigError(f"configuration root must be a mapping: {path}")
    return payload


def _validate_known_keys(
    candidate: Mapping[str, Any],
    schema: Mapping[str, Any],
    *,
    prefix: str = "",
) -> None:
    for key, value in candidate.items():
        field_path = f"{prefix}.{key}" if prefix else str(key)
        if key not in schema:
            raise ConfigError(f"unknown configuration field: {field_path}")
        expected = schema[key]
        if isinstance(expected, Mapping):
            if not isinstance(value, Mapping):
                raise ConfigError(f"configuration field {field_path} must be a mapping")
            _validate_known_keys(value, expected, prefix=field_path)
        elif isinstance(expected, list):
            if not isinstance(value, list):
                raise ConfigError(f"configuration field {field_path} must be a list")
            if expected:
                item_schema = expected[0]
                for index, item in enumerate(value):
                    item_path = f"{field_path}[{index}]"
                    if isinstance(item_schema, Mapping):
                        if not isinstance(item, Mapping):
                            raise ConfigError(
                                f"configuration field {item_path} must be a mapping"
                            )
                        _validate_known_keys(item, item_schema, prefix=item_path)
                    elif not isinstance(item, type(item_schema)):
                        raise ConfigError(
                            f"configuration field {item_path} must be "
                            f"{type(item_schema).__name__}"
                        )
        elif value is not None and expected is not None:
            expected_type = type(expected)
            if expected_type is bool:
                valid_type = type(value) is bool
            elif expected_type is int:
                valid_type = type(value) is int
            elif expected_type is float:
                valid_type = isinstance(value, (int, float)) and type(value) is not bool
            else:
                valid_type = isinstance(value, expected_type)
            if not valid_type:
                raise ConfigError(
                    f"configuration field {field_path} must be {expected_type.__name__}"
                )


def _deep_merge(base: dict[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = _deep_merge(dict(merged[key]), value)
        else:
            merged[key] = deepcopy(value)
    return merged


def load_user_config(path: str | Path | None = None) -> dict[str, Any]:
    defaults = _read_yaml(DEFAULT_CONFIG_PATH)
    if path is None:
        return defaults
    user_path = Path(path).expanduser().resolve()
    user = _read_yaml(user_path)
    _validate_known_keys(user, defaults)
    return _deep_merge(defaults, user)


def apply_cli_overrides(
    config: Mapping[str, Any],
    *,
    checkpoint: str | Path | None = None,
    dataset: str | Path | None = None,
    output: str | Path | None = None,
    prepared_output: str | Path | None = None,
    num_envs: int | None = None,
    sorting_mode: str | None = None,
) -> dict[str, Any]:
    resolved = deepcopy(dict(config))
    if checkpoint is not None:
        resolved["checkpoint"] = str(Path(checkpoint).expanduser().resolve())
    if dataset is not None:
        resolved["dataset"]["input_path"] = str(Path(dataset).expanduser().resolve())
    if output is not None:
        resolved["output"] = str(Path(output).expanduser().resolve())
    if prepared_output is not None:
        resolved["dataset"]["prepared_output_dir"] = str(
            Path(prepared_output).expanduser().resolve()
        )
    if num_envs is not None:
        if num_envs < 1:
            raise ConfigError("num_envs must be at least 1")
        resolved["runtime"]["num_envs"] = int(num_envs)
    if sorting_mode is not None:
        if sorting_mode not in {"off", "on", "auto"}:
            raise ConfigError("sorting_mode must be off, on, or auto")
        resolved["dataset"]["sorting"]["mode"] = sorting_mode
    return resolved


def _required_path(config: Mapping[str, Any], key: str) -> Path:
    value = config.get(key)
    if not value:
        raise ConfigError(f"configuration field {key!r} is required")
    return Path(str(value)).expanduser().resolve()


def _bucketing_enabled(config: Mapping[str, Any]) -> bool:
    return bool(config["dataset"]["bucketing"]["enabled"])


def _bucket_specs_from_config(config: Mapping[str, Any]) -> tuple[BucketSpec, ...]:
    raw_buckets = config["dataset"]["bucketing"]["buckets"]
    specs: list[BucketSpec] = []
    for index, raw in enumerate(raw_buckets):
        field_path = f"dataset.bucketing.buckets[{index}]"
        if not isinstance(raw, Mapping):
            raise ConfigError(f"configuration field {field_path} must be a mapping")
        for field in ("name", "max_frames", "num_envs"):
            if field not in raw:
                raise ConfigError(f"configuration field {field_path}.{field} is required")
        if not isinstance(raw["name"], str):
            raise ConfigError(f"configuration field {field_path}.name must be str")
        max_frames = raw["max_frames"]
        if max_frames is not None and type(max_frames) is not int:
            raise ConfigError(
                f"configuration field {field_path}.max_frames must be int or null"
            )
        if type(raw["num_envs"]) is not int:
            raise ConfigError(f"configuration field {field_path}.num_envs must be int")
        specs.append(
            BucketSpec(
                name=raw["name"],
                max_frames=max_frames,
                num_envs=raw["num_envs"],
            )
        )
    try:
        validate_bucket_specs(specs)
    except ValueError as exc:
        raise ConfigError(f"invalid dataset.bucketing.buckets: {exc}") from exc
    return tuple(specs)


def _validate_bucket_config(config: Mapping[str, Any]) -> tuple[BucketSpec, ...]:
    specs = _bucket_specs_from_config(config)
    if int(config["runtime"]["world_size"]) != 1:
        raise ConfigError("bucketed evaluation requires runtime.world_size=1")
    if bool(config["dataset"]["long_motion"]["enabled"]):
        raise ConfigError(
            "dataset.long_motion.enabled cannot be combined with dataset.bucketing"
        )
    return specs


def _prepare(config: Mapping[str, Any]) -> PreparedDataset:
    output = _required_path(config, "output")
    dataset_config = config["dataset"]
    input_path = dataset_config.get("input_path")
    if not input_path:
        raise ConfigError("configuration field 'dataset.input_path' is required")
    sorting_config = dict(dataset_config["sorting"])
    if _bucketing_enabled(config):
        _validate_bucket_config(config)
        sorting_config["mode"] = "on"
    long_config = dataset_config["long_motion"]
    prepared_output = resolve_prepared_output(
        dataset_config.get("prepared_output_dir"),
        output,
    )
    return prepare_dataset(
        input_path,
        prepared_output,
        SortingOptions(**sorting_config),
        batch_size=int(config["runtime"]["num_envs"]),
        long_motion=LongMotionOptions(**long_config),
    )


def prepare_from_config(
    config: Mapping[str, Any],
) -> PreparedDataset | BucketedPreparation:
    prepared = _prepare(config)
    if not _bucketing_enabled(config):
        return prepared
    specs = _validate_bucket_config(config)
    plan = build_bucket_plan(
        prepared.manifest,
        specs,
        _config_fingerprint(config),
    )
    views = materialize_bucket_views(
        prepared,
        plan,
        prepared.manifest_path.parent / "buckets",
        str(config["dataset"]["sorting"]["materialization"]),
    )
    output = _required_path(config, "output")
    plan_path = output / "bucket_plan.json"
    if plan_path.is_file():
        existing = BucketPlan.read(plan_path)
        if existing != plan:
            raise ConfigError(f"bucket plan differs from existing output: {output}")
    else:
        output.mkdir(parents=True, exist_ok=True)
        plan.write(plan_path)
    return BucketedPreparation(
        prepared=prepared,
        plan=plan,
        views=views,
        plan_path=plan_path,
    )


def _config_fingerprint(config: Mapping[str, Any]) -> str:
    relevant = {
        "dataset": config["dataset"],
        "runtime": config["runtime"],
        "persistence": config["persistence"],
        "evaluation": config["evaluation"],
    }
    return canonical_json_fingerprint(relevant)


def build_eval_agent_command(
    config: Mapping[str, Any],
    *,
    checkpoint: Path,
    output: Path,
    prepared: PreparedDataset,
) -> tuple[str, ...]:
    runtime = config["runtime"]
    persistence = config["persistence"]
    evaluation = config["evaluation"]
    command = [
        sys.executable,
        "-u",
        "gear_sonic/eval_agent_trl.py",
        f"+checkpoint={checkpoint}",
        f"+headless={str(bool(runtime['headless']))}",
        "++eval_callbacks=im_eval",
        "++run_eval_loop=False",
        f"++use_encoder={runtime['encoder']}",
        f"++num_envs={int(runtime['num_envs'])}",
        f"++eval_output_dir={output / 'metrics'}",
        f"+manager_env/terminations={evaluation['termination_config']}",
        "++manager_env.commands.motion.motion_lib_cfg.motion_file="
        f"{prepared.motion_dir}",
        "++manager_env.commands.motion.motion_lib_cfg.smpl_motion_file="
        f"{evaluation['smpl_motion_file']}",
        "++manager_env.commands.motion.motion_lib_cfg.multi_thread="
        f"{str(bool(runtime['multi_thread_motion_loading']))}",
        f"++callbacks.im_eval.empty_cache_freq={int(runtime['empty_cache_freq'])}",
    ]
    if persistence["enabled"]:
        command.extend(
            [
                f"++callbacks.im_eval.batch_results_dir={output / 'batch_results'}",
                f"++callbacks.im_eval.run_manifest_path={output / 'run_manifest.json'}",
                f"++callbacks.im_eval.dataset_manifest_path={prepared.manifest_path}",
                f"++callbacks.im_eval.snapshot_dir={output / 'snapshots'}",
                "++callbacks.im_eval.snapshot_every_n_batches="
                f"{int(persistence['snapshot_every_n_batches'])}",
                "++callbacks.im_eval.allow_partial="
                f"{str(bool(evaluation['allow_partial']))}",
            ]
        )
    return tuple(command)


def _candidate_manifest(
    *,
    checkpoint: Path,
    prepared: Any,
    config_fingerprint: str,
    world_size: int,
) -> RunManifest:
    return RunManifest.create(
        checkpoint=checkpoint,
        checkpoint_fingerprint=checkpoint_fingerprint(checkpoint),
        dataset_fingerprint=prepared.manifest.source_fingerprint,
        dataset_manifest_path=prepared.manifest_path,
        config_fingerprint=config_fingerprint,
        expected_motion_keys=[entry.canonical_key for entry in prepared.manifest.entries],
        world_size=world_size,
    )


def prepare_run_context(
    config: Mapping[str, Any], *, write_manifest: bool
) -> RunContext | BucketedRunContext:
    if _bucketing_enabled(config):
        return prepare_bucketed_run_context(config, write_manifest=write_manifest)
    checkpoint = _required_path(config, "checkpoint")
    output = _required_path(config, "output")
    if not checkpoint.is_file():
        raise ConfigError(f"checkpoint does not exist: {checkpoint}")
    prepared = _prepare(config)
    candidate = _candidate_manifest(
        checkpoint=checkpoint,
        prepared=prepared,
        config_fingerprint=_config_fingerprint(config),
        world_size=int(config["runtime"]["world_size"]),
    )
    manifest_path = output / "run_manifest.json"
    if manifest_path.exists():
        existing = RunManifest.read(manifest_path)
        existing.assert_compatible(candidate)
        if existing.status == "complete" and write_manifest:
            raise ConfigError(
                f"evaluation output is already complete; choose a new output directory: {output}"
            )
        manifest = existing
    else:
        manifest = candidate
        if write_manifest:
            output.mkdir(parents=True, exist_ok=True)
            manifest.write(manifest_path)
    command = build_eval_agent_command(
        config,
        checkpoint=checkpoint,
        output=output,
        prepared=prepared,
    )
    return RunContext(
        config=config,
        checkpoint=checkpoint,
        output=output,
        prepared=prepared,
        manifest=manifest,
        command=command,
    )


def prepare_bucketed_run_context(
    config: Mapping[str, Any], *, write_manifest: bool
) -> BucketedRunContext:
    specs = _validate_bucket_config(config)
    checkpoint = _required_path(config, "checkpoint")
    output = _required_path(config, "output")
    if not checkpoint.is_file():
        raise ConfigError(f"checkpoint does not exist: {checkpoint}")

    config_fingerprint = _config_fingerprint(config)
    prepared = _prepare(config)
    plan = build_bucket_plan(prepared.manifest, specs, config_fingerprint)
    views = materialize_bucket_views(
        prepared,
        plan,
        prepared.manifest_path.parent / "buckets",
        str(config["dataset"]["sorting"]["materialization"]),
    )
    candidate = _candidate_manifest(
        checkpoint=checkpoint,
        prepared=prepared,
        config_fingerprint=config_fingerprint,
        world_size=1,
    )

    manifest_path = output / "run_manifest.json"
    plan_path = output / "bucket_plan.json"
    if plan_path.exists():
        existing_plan = BucketPlan.read(plan_path)
        if existing_plan != plan:
            raise ConfigError(f"bucket plan differs from existing output: {output}")
    elif write_manifest:
        output.mkdir(parents=True, exist_ok=True)
        plan.write(plan_path)

    if manifest_path.exists():
        existing = RunManifest.read(manifest_path)
        existing.assert_compatible(candidate)
        if existing.status == "complete" and write_manifest:
            raise ConfigError(
                f"evaluation output is already complete; choose a new output directory: {output}"
            )
        manifest = existing
    else:
        manifest = candidate
        if write_manifest:
            output.mkdir(parents=True, exist_ok=True)
            manifest.write(manifest_path)

    bucket_contexts: list[BucketRunContext] = []
    for view in views:
        assignment = view.assignment
        child_output = output / "buckets" / (
            f"{assignment.ordinal:03d}_{_safe_context_name(assignment.name)}"
        )
        child_config = deepcopy(dict(config))
        child_config["runtime"]["num_envs"] = assignment.num_envs
        child_fingerprint = canonical_json_fingerprint(
            {
                "parent_config_fingerprint": config_fingerprint,
                "bucket_plan_fingerprint": plan.fingerprint(),
                "bucket_ordinal": assignment.ordinal,
            }
        )
        child_candidate = _candidate_manifest(
            checkpoint=checkpoint,
            prepared=view,
            config_fingerprint=child_fingerprint,
            world_size=1,
        )
        child_manifest_path = child_output / "run_manifest.json"
        if child_manifest_path.exists():
            child_existing = RunManifest.read(child_manifest_path)
            child_existing.assert_compatible(child_candidate)
            child_manifest = child_existing
        else:
            child_manifest = child_candidate
        command = (
            build_eval_agent_command(
                child_config,
                checkpoint=checkpoint,
                output=child_output,
                prepared=view,
            )
            if assignment.motion_count
            else ()
        )
        bucket_contexts.append(
            BucketRunContext(
                assignment=assignment,
                view=view,
                output=child_output,
                manifest=child_manifest,
                command=command,
            )
        )

    return BucketedRunContext(
        config=config,
        checkpoint=checkpoint,
        output=output,
        prepared=prepared,
        plan=plan,
        manifest=manifest,
        buckets=tuple(bucket_contexts),
    )


def _safe_context_name(name: str) -> str:
    return "".join(
        character if character.isalnum() or character in "._-" else "_"
        for character in name
    )


def context_as_json(context: RunContext | BucketedRunContext) -> dict[str, Any]:
    if isinstance(context, BucketedRunContext):
        buckets = []
        for bucket in context.buckets:
            assignment = bucket.assignment
            padding_ratio = (
                assignment.estimated_padded_frame_count / assignment.real_frame_count
                if assignment.real_frame_count
                else 0.0
            )
            buckets.append(
                {
                    "ordinal": assignment.ordinal,
                    "name": assignment.name,
                    "min_frames_exclusive": assignment.min_frames_exclusive,
                    "max_frames_inclusive": assignment.max_frames_inclusive,
                    "num_envs": assignment.num_envs,
                    "motion_count": assignment.motion_count,
                    "real_frame_count": assignment.real_frame_count,
                    "estimated_padded_frame_count": (
                        assignment.estimated_padded_frame_count
                    ),
                    "estimated_padding_ratio": padding_ratio,
                    "estimated_batch_count": assignment.estimated_batch_count,
                    "estimated_serial_steps": assignment.estimated_serial_steps,
                    "output": str(bucket.output),
                    "dataset_manifest": str(bucket.view.manifest_path),
                    "command": list(bucket.command) if bucket.command else None,
                }
            )
        fixed_num_envs = int(context.config["runtime"]["num_envs"])
        frame_counts = [entry.frame_count for entry in context.prepared.manifest.entries]
        fixed_serial_steps = sum(
            max(frame_counts[start : start + fixed_num_envs])
            for start in range(0, len(frame_counts), fixed_num_envs)
        )
        bucket_serial_steps = sum(
            assignment.estimated_serial_steps
            for assignment in context.plan.assignments
        )
        return {
            "checkpoint": str(context.checkpoint),
            "output": str(context.output),
            "prepared_dataset": str(context.prepared.motion_dir),
            "dataset_manifest": str(context.prepared.manifest_path),
            "run_id": context.manifest.run_id,
            "bucket_plan_fingerprint": context.plan.fingerprint(),
            "estimated_serial_steps": bucket_serial_steps,
            "fixed_num_envs": fixed_num_envs,
            "fixed_estimated_serial_steps": fixed_serial_steps,
            "estimated_serial_step_ratio": (
                bucket_serial_steps / fixed_serial_steps
                if fixed_serial_steps
                else 0.0
            ),
            "buckets": buckets,
        }
    return {
        "checkpoint": str(context.checkpoint),
        "output": str(context.output),
        "prepared_dataset": str(context.prepared.motion_dir),
        "dataset_manifest": str(context.prepared.manifest_path),
        "run_id": context.manifest.run_id,
        "command": list(context.command),
    }


def run_evaluation(config: Mapping[str, Any]) -> int:
    if _bucketing_enabled(config):
        return run_bucketed_evaluation(config)
    started = time.monotonic()
    context = prepare_run_context(config, write_manifest=True)
    context.output.mkdir(parents=True, exist_ok=True)
    log_path = context.output / "eval.log"
    with log_path.open("a", encoding="utf-8", buffering=1) as log_stream:
        log_stream.write("command: " + " ".join(context.command) + "\n")
        process = subprocess.Popen(
            context.command,
            cwd=REPO_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log_stream.write(line)
        return_code = process.wait()

    elapsed = time.monotonic() - started
    status = "complete" if return_code == 0 else "failed"
    completed = context.manifest.with_completion(
        status=status,
        wall_time_seconds=elapsed,
    )
    completed.write(context.output / "run_manifest.json")
    print(f"Evaluation {status}; wall time {format_duration(elapsed)} ({elapsed:.3f}s)")
    return return_code


def _mark_running(manifest: RunManifest) -> RunManifest:
    return replace(
        manifest,
        status="running",
        finished_at=None,
        wall_time_seconds=None,
    )


def _cumulative_wall_time(manifest: RunManifest, elapsed: float) -> float:
    return float(manifest.wall_time_seconds or 0.0) + float(elapsed)


def _child_validation_errors(context: BucketRunContext) -> tuple[str, ...]:
    manifest_path = context.output / "run_manifest.json"
    metrics_path = context.output / "metrics" / "metrics_eval.json"
    errors: list[str] = []
    if not manifest_path.is_file():
        return ("run manifest is missing",)
    manifest = RunManifest.read(manifest_path)
    try:
        manifest.assert_compatible(context.manifest)
    except ManifestMismatchError as exc:
        errors.append(f"run manifest is incompatible: {exc}")
    if manifest.status != "complete":
        errors.append(f"run manifest status is {manifest.status!r}, expected 'complete'")
    if not metrics_path.is_file():
        errors.append("metrics/metrics_eval.json is missing")
    else:
        try:
            with metrics_path.open(encoding="utf-8") as stream:
                metrics = json.load(stream)
            report = validate_evaluation(manifest, metrics)
            errors.extend(report.errors)
            if report.partial:
                errors.append("child metrics are partial")
        except (OSError, ValueError, TypeError, KeyError) as exc:
            errors.append(f"child metrics are invalid: {exc}")
    if not any((context.output / "batch_results").glob("rank_*/batch_*.pt")):
        errors.append("child batch store is empty")
    return tuple(errors)


def _execute_bucket_child(context: BucketRunContext, parent_log: TextIO) -> int:
    context.output.mkdir(parents=True, exist_ok=True)
    log_path = context.output / "eval.log"
    with log_path.open("a", encoding="utf-8", buffering=1) as child_log:
        command_line = "command: " + " ".join(context.command) + "\n"
        parent_log.write(command_line)
        child_log.write(command_line)
        process = subprocess.Popen(
            context.command,
            cwd=REPO_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            parent_log.write(line)
            child_log.write(line)
        return process.wait()


def _bucket_roots(output: Path, plan: BucketPlan) -> dict[int, Path]:
    return {
        assignment.ordinal: output
        / "buckets"
        / f"{assignment.ordinal:03d}_{_safe_context_name(assignment.name)}"
        for assignment in plan.non_empty_assignments()
    }


def run_bucketed_evaluation(
    config: Mapping[str, Any],
    *,
    child_executor: Callable[[BucketRunContext, TextIO], int] | None = None,
) -> int:
    """Run one fixed-size Isaac child process per non-empty bucket."""

    started = time.monotonic()
    context = prepare_bucketed_run_context(config, write_manifest=True)
    executor = child_executor or _execute_bucket_child
    context.output.mkdir(parents=True, exist_ok=True)
    parent_log_path = context.output / "eval.log"

    with parent_log_path.open("a", encoding="utf-8", buffering=1) as parent_log:
        for bucket in context.buckets:
            if not bucket.assignment.motion_count:
                continue
            validation_errors = _child_validation_errors(bucket)
            if not validation_errors:
                message = f"reusing complete bucket {bucket.assignment.name}\n"
                print(message, end="", flush=True)
                parent_log.write(message)
                continue
            if (bucket.output / "run_manifest.json").is_file():
                parent_log.write(
                    f"rerunning bucket {bucket.assignment.name}: "
                    + "; ".join(validation_errors)
                    + "\n"
                )

            bucket.output.mkdir(parents=True, exist_ok=True)
            running_manifest = _mark_running(bucket.manifest)
            running_manifest.write(bucket.output / "run_manifest.json")
            child_started = time.monotonic()
            return_code = executor(bucket, parent_log)
            child_elapsed = time.monotonic() - child_started
            child_status = "complete" if return_code == 0 else "failed"
            child_total_elapsed = _cumulative_wall_time(
                bucket.manifest, child_elapsed
            )
            running_manifest.with_completion(
                status=child_status,
                wall_time_seconds=child_total_elapsed,
            ).write(bucket.output / "run_manifest.json")

            if return_code != 0:
                elapsed = time.monotonic() - started
                total_elapsed = _cumulative_wall_time(context.manifest, elapsed)
                context.manifest.with_completion(
                    status="failed", wall_time_seconds=total_elapsed
                ).write(context.output / "run_manifest.json")
                print(
                    f"Bucket {bucket.assignment.name} failed; wall time "
                    f"{format_duration(total_elapsed)} ({total_elapsed:.3f}s)",
                    flush=True,
                )
                return return_code

            validation_errors = _child_validation_errors(bucket)
            if validation_errors:
                elapsed = time.monotonic() - started
                total_elapsed = _cumulative_wall_time(context.manifest, elapsed)
                completed_child = RunManifest.read(bucket.output / "run_manifest.json")
                replace(completed_child, status="failed").write(
                    bucket.output / "run_manifest.json"
                )
                context.manifest.with_completion(
                    status="failed", wall_time_seconds=total_elapsed
                ).write(context.output / "run_manifest.json")
                parent_log.write(
                    f"bucket {bucket.assignment.name} validation failed: "
                    + "; ".join(validation_errors)
                    + "\n"
                )
                return 2

    roots = _bucket_roots(context.output, context.plan)
    try:
        result = aggregate_bucket_outputs(
            parent_manifest=context.manifest,
            plan=context.plan,
            bucket_roots=roots,
            allow_partial=False,
        )
        metrics_path = context.output / "metrics" / "metrics_eval.json"
        write_metrics_eval_json(result, metrics_path)
        report = validate_bucket_plan(
            context.manifest,
            context.plan,
            json.loads(metrics_path.read_text(encoding="utf-8")),
            roots,
        )
        if not report.valid:
            raise ValueError("; ".join(report.errors))
    except Exception as exc:
        elapsed = time.monotonic() - started
        total_elapsed = _cumulative_wall_time(context.manifest, elapsed)
        context.manifest.with_completion(
            status="failed", wall_time_seconds=total_elapsed
        ).write(context.output / "run_manifest.json")
        with parent_log_path.open("a", encoding="utf-8") as parent_log:
            parent_log.write(f"top-level aggregation failed: {exc}\n")
        return 2

    elapsed = time.monotonic() - started
    total_elapsed = _cumulative_wall_time(context.manifest, elapsed)
    context.manifest.with_completion(
        status="complete", wall_time_seconds=total_elapsed
    ).write(context.output / "run_manifest.json")
    print(
        "Evaluation complete; wall time "
        f"{format_duration(total_elapsed)} ({total_elapsed:.3f}s)",
        flush=True,
    )
    return 0


def format_duration(seconds: float) -> str:
    total = max(0, int(round(seconds)))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def aggregate_output(output: str | Path, *, allow_partial: bool = False) -> Path:
    root = Path(output).expanduser().resolve()
    run_manifest = RunManifest.read(root / "run_manifest.json")
    bucket_plan_path = root / "bucket_plan.json"
    if bucket_plan_path.is_file():
        plan = BucketPlan.read(bucket_plan_path)
        result = aggregate_bucket_outputs(
            parent_manifest=run_manifest,
            plan=plan,
            bucket_roots=_bucket_roots(root, plan),
            allow_partial=allow_partial,
        )
        path = root / "metrics" / "metrics_eval.json"
        write_metrics_eval_json(result, path)
        return path
    dataset_manifest_path = (
        Path(run_manifest.dataset_manifest_path)
        if run_manifest.dataset_manifest_path is not None
        else root / "prepared_dataset" / "dataset_manifest.json"
    )
    dataset_manifest = DatasetManifest.read(dataset_manifest_path)
    store = BatchStore(root / "batch_results", run_id=run_manifest.run_id)
    result = aggregate_batches(
        store.load_all(),
        expected_motion_keys=run_manifest.expected_motion_keys,
        alias_to_canonical=dataset_manifest.alias_to_canonical(),
        allow_partial=allow_partial,
    )
    path = root / "metrics" / "metrics_eval.json"
    write_metrics_eval_json(result, path)
    return path


def validate_output(output: str | Path) -> ValidationReport:
    root = Path(output).expanduser().resolve()
    run_manifest = RunManifest.read(root / "run_manifest.json")
    with (root / "metrics" / "metrics_eval.json").open(encoding="utf-8") as stream:
        metrics = json.load(stream)
    bucket_plan_path = root / "bucket_plan.json"
    if bucket_plan_path.is_file():
        plan = BucketPlan.read(bucket_plan_path)
        return validate_bucket_plan(
            run_manifest,
            plan,
            metrics,
            _bucket_roots(root, plan),
        )
    return validate_evaluation(run_manifest, metrics)
