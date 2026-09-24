#!/usr/bin/env python3
"""检查 SONIC A3 Evaluation Recorder Demo 1 的一个 session。"""

from __future__ import annotations

import argparse
import ctypes
import json
import math
import statistics
import struct
import sys
from collections import Counter, defaultdict
from pathlib import Path

import yaml


NO_TICK = (1 << 64) - 1
DEPLOY_MODE_MOTION = 3
MOTION_PHASE_PLAYING = 2
ROBOT_STATE_VALID = 1 << 0
REFERENCE_VALID = 1 << 5
INFER_TIMING_VALID = 1 << 6
SEND_TIMING_VALID = 1 << 7
EXPECTED_TORQUE_VALID = 1 << 9

VALID_NAMES = {
    1 << 0: "robot_state",
    1 << 1: "pelvis_imu",
    1 << 2: "raw_action",
    1 << 3: "q_des_policy",
    1 << 4: "robot_command",
    1 << 5: "reference",
    1 << 6: "infer_timing",
    1 << 7: "send_timing",
    1 << 8: "torso_imu",
    1 << 9: "expected_torque",
}

QUALITY_NAMES = {
    1 << 0: "sync_complete",
    1 << 1: "sync_aligned",
    1 << 2: "infer_ok",
    1 << 3: "command_sent",
    1 << 4: "watchdog_safe_halt",
    1 << 5: "recorder_gap",
}


_FRAME_PREFIX_FIELDS = [
        ("frame_id", ctypes.c_uint64),
        ("state_tick", ctypes.c_uint64),
        ("policy_tick", ctypes.c_uint64),
        ("reference_tick", ctypes.c_uint64),
        ("episode_id", ctypes.c_uint64),
        ("active_motion_id", ctypes.c_int32),
        ("selected_motion_id", ctypes.c_int32),
        ("control_start_monotonic_ns", ctypes.c_int64),
        ("state_timestamp_ns", ctypes.c_int64),
        ("state_data_ready_ns", ctypes.c_int64),
        ("state_sync_ready_ns", ctypes.c_int64),
        ("infer_start_monotonic_ns", ctypes.c_int64),
        ("infer_end_monotonic_ns", ctypes.c_int64),
        ("send_start_monotonic_ns", ctypes.c_int64),
        ("send_end_monotonic_ns", ctypes.c_int64),
        ("control_end_monotonic_ns", ctypes.c_int64),
        # DeployMode 未显式指定底层类型，C++ 中按 int32 存储。
        ("deploy_mode", ctypes.c_int32),
        ("motion_phase", ctypes.c_uint8),
        ("command_source", ctypes.c_uint8),
        ("gain_profile_id", ctypes.c_uint8),
        ("valid_mask", ctypes.c_uint32),
        ("quality_flags", ctypes.c_uint32),
        ("q", ctypes.c_float * 31),
        ("dq", ctypes.c_float * 31),
        ("tau_est", ctypes.c_float * 31),
        ("pelvis_quat_wxyz", ctypes.c_float * 4),
        ("pelvis_gyro", ctypes.c_float * 3),
]

_FRAME_POLICY_FIELDS = [
        ("raw_action", ctypes.c_float * 29),
        ("q_des_policy", ctypes.c_float * 29),
]

_FRAME_COMMAND_V1_V2_FIELDS = [
        ("command_q_des", ctypes.c_float * 31),
        ("command_dq_des", ctypes.c_float * 31),
        ("command_tau_ff", ctypes.c_float * 31),
]

_FRAME_REFERENCE_FIELDS = [
        ("q_ref", ctypes.c_float * 29),
        ("dq_ref", ctypes.c_float * 29),
]

_FRAME_REFERENCE_PELVIS_FIELDS = [
        ("reference_pelvis_position_m", ctypes.c_float * 3),
        ("reference_pelvis_quat_wxyz", ctypes.c_float * 4),
]


class EvalFrameV1(ctypes.LittleEndianStructure):
    """旧 schema 1：pelvis quaternion/gyro/accel，不含 torso IMU。"""

    _fields_ = _FRAME_PREFIX_FIELDS + [
        ("pelvis_accel", ctypes.c_float * 3),
    ] + _FRAME_POLICY_FIELDS + _FRAME_COMMAND_V1_V2_FIELDS + _FRAME_REFERENCE_FIELDS


class EvalFrameV2(ctypes.LittleEndianStructure):
    """旧 schema 2：双 IMU，并记录三组 command 数组。"""

    _fields_ = _FRAME_PREFIX_FIELDS + [
        ("torso_quat_wxyz", ctypes.c_float * 4),
        ("torso_gyro", ctypes.c_float * 3),
    ] + _FRAME_POLICY_FIELDS + _FRAME_COMMAND_V1_V2_FIELDS + _FRAME_REFERENCE_FIELDS


class EvalFrameV3(ctypes.LittleEndianStructure):
    """旧 schema 3：双 IMU，command 仅保留最终 q_des。"""

    _fields_ = _FRAME_PREFIX_FIELDS + [
        ("torso_quat_wxyz", ctypes.c_float * 4),
        ("torso_gyro", ctypes.c_float * 3),
    ] + _FRAME_POLICY_FIELDS + [
        ("command_q_des", ctypes.c_float * 31),
    ] + _FRAME_REFERENCE_FIELDS


class EvalFrameV4(ctypes.LittleEndianStructure):
    """旧 schema 4：V3 加入由最终 RobotCommand 重建的理论 PD 力矩。"""

    _fields_ = _FRAME_PREFIX_FIELDS + [
        ("torso_quat_wxyz", ctypes.c_float * 4),
        ("torso_gyro", ctypes.c_float * 3),
    ] + _FRAME_POLICY_FIELDS + [
        ("command_q_des", ctypes.c_float * 31),
        ("tau_cmd_expected", ctypes.c_float * 31),
    ] + _FRAME_REFERENCE_FIELDS


class EvalFrameV5(ctypes.LittleEndianStructure):
    """当前 schema 5：不再记录 command q_des 和 policy q_des。"""

    _fields_ = _FRAME_PREFIX_FIELDS + [
        ("torso_quat_wxyz", ctypes.c_float * 4),
        ("torso_gyro", ctypes.c_float * 3),
        ("raw_action", ctypes.c_float * 29),
        ("tau_cmd_expected", ctypes.c_float * 31),
    ] + _FRAME_REFERENCE_FIELDS


class EvalFrameV6(ctypes.LittleEndianStructure):
    """旧式 frames.bin 的 schema 6：原始 RobotCommand 与 writer gain 快照。"""

    _fields_ = _FRAME_PREFIX_FIELDS + [
        ("torso_quat_wxyz", ctypes.c_float * 4),
        ("torso_gyro", ctypes.c_float * 3),
        ("raw_action", ctypes.c_float * 29),
        ("command_q_des", ctypes.c_float * 31),
        ("command_dq_des", ctypes.c_float * 31),
        ("command_tau_ff", ctypes.c_float * 31),
        ("command_kp", ctypes.c_float * 31),
        ("command_kd", ctypes.c_float * 31),
    ] + _FRAME_REFERENCE_FIELDS


class EvalFrameV7(ctypes.LittleEndianStructure):
    """当前 schema 7：V6 加入 yaw-aligned reference pelvis pose。"""

    _fields_ = _FRAME_PREFIX_FIELDS + [
        ("torso_quat_wxyz", ctypes.c_float * 4),
        ("torso_gyro", ctypes.c_float * 3),
        ("raw_action", ctypes.c_float * 29),
        ("command_q_des", ctypes.c_float * 31),
        ("command_dq_des", ctypes.c_float * 31),
        ("command_tau_ff", ctypes.c_float * 31),
        ("command_kp", ctypes.c_float * 31),
        ("command_kd", ctypes.c_float * 31),
    ] + _FRAME_REFERENCE_FIELDS + _FRAME_REFERENCE_PELVIS_FIELDS


def percentile(values: list[float], ratio: float) -> float:
    if not values:
        return math.nan
    ordered = sorted(values)
    index = min(len(ordered) - 1, math.ceil(ratio * len(ordered)) - 1)
    return ordered[max(0, index)]


def timing_line(name: str, values: list[float]) -> str:
    if not values:
        return f"  {name}: 无有效样本"
    return (
        f"  {name}: n={len(values)} mean={statistics.fmean(values):.3f} ms "
        f"p50={statistics.median(values):.3f} ms "
        f"p95={percentile(values, 0.95):.3f} ms "
        f"max={max(values):.3f} ms"
    )


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"{path} 顶层必须是 JSON object")
    return value


def load_events(path: Path) -> list[dict]:
    events = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: {exc}") from exc
    return events


def inspect_mcap_only(session: Path) -> int:
    """检查 complete-only 正式输出：data 内一份 MCAP 对一份 YAML。"""
    data_dir = session / "data"
    session_meta_path = data_dir / "session.meta.yaml"
    if not session_meta_path.is_file():
        raise FileNotFoundError(f"缺少文件: {session_meta_path}")
    session_meta = yaml.safe_load(session_meta_path.read_text(encoding="utf-8"))
    if not isinstance(session_meta, dict):
        raise ValueError("session.meta.yaml 顶层必须是 map")

    errors: list[str] = []
    data_schema = session_meta.get("data_schema")
    if data_schema not in {
        "a3_eval_raw_v6",
        "a3_eval_raw_v7",
        "a3_eval_raw_v8",
    }:
        errors.append(f"不支持的 data_schema: {data_schema}")
    unexpected_root = sorted(path.name for path in session.iterdir()
                             if path.name != "data")
    if unexpected_root:
        errors.append(f"session 根目录存在非 data 输出: {unexpected_root}")
    if (session / "data_read").exists():
        errors.append("不应生成 data_read")
    partials = sorted(path.name for path in data_dir.glob("*.partial"))
    if partials:
        errors.append(f"存在未提交 partial: {partials}")

    mcap_files = sorted(data_dir.glob("*.mcap"))
    episode_meta_files = sorted(
        path for path in data_dir.glob("*.meta.yaml")
        if path.name != "session.meta.yaml"
    )
    if len(mcap_files) != len(episode_meta_files):
        errors.append(
            f"MCAP/episode metadata 数量不一致: "
            f"{len(mcap_files)}/{len(episode_meta_files)}"
        )

    magic = b"\x89MCAP0\r\n"
    total_frames = 0
    for mcap_path in mcap_files:
        expected_meta = data_dir / f"{mcap_path.stem}.meta.yaml"
        if not expected_meta.is_file():
            errors.append(f"缺少同名 metadata: {expected_meta.name}")
            continue
        with mcap_path.open("rb") as handle:
            prefix = handle.read(len(magic))
            handle.seek(-len(magic), 2)
            suffix = handle.read(len(magic))
        if prefix != magic or suffix != magic:
            errors.append(f"MCAP magic/尾标记无效: {mcap_path.name}")

        meta = yaml.safe_load(expected_meta.read_text(encoding="utf-8"))
        if not isinstance(meta, dict):
            errors.append(f"metadata 顶层不是 map: {expected_meta.name}")
            continue
        if meta.get("data_schema") != data_schema:
            errors.append(
                f"data_schema 与 session 不一致: {expected_meta.name}"
            )
        if meta.get("mcap", {}).get("file") != mcap_path.name:
            errors.append(f"metadata 未绑定对应 MCAP: {expected_meta.name}")
        episode = meta.get("episode", {})
        if episode.get("outcome") != "COMPLETED":
            errors.append(f"正式 data 中出现非完整 episode: {mcap_path.name}")
        frame_count = int(episode.get("frame_count", 0) or 0)
        if frame_count <= 0:
            errors.append(f"frame_count 无效: {expected_meta.name}")
        total_frames += frame_count
        profiles = meta.get("gains", {}).get("profiles", {})
        if not isinstance(profiles, dict) or not profiles:
            errors.append(f"缺少 gain profiles: {expected_meta.name}")
        else:
            for profile_id, profile in profiles.items():
                if len(profile.get("kp", [])) != 31 or len(profile.get("kd", [])) != 31:
                    errors.append(
                        f"gain profile {profile_id} 不是31维: {expected_meta.name}"
                    )
        if data_schema in {"a3_eval_raw_v7", "a3_eval_raw_v8"}:
            per_frame_values = meta.get("gains", {}).get("per_frame_values")
            if per_frame_values != (
                "/a3/eval/command.kp[31], /a3/eval/command.kd[31]"
            ):
                errors.append(
                    f"V7+ 缺少 MCAP 每帧 kp/kd 声明: {expected_meta.name}"
                )
        if data_schema == "a3_eval_raw_v8":
            reference = meta.get("signals", {}).get("reference")
            if reference != (
                "q_ref[29], dq_ref[29], pelvis_position_m[3], "
                "pelvis_quat_wxyz[4]"
            ):
                errors.append(
                    f"V8 缺少 reference pelvis pose 声明: "
                    f"{expected_meta.name}"
                )
        derived = meta.get("offline_derivation", {})
        if derived.get("calculated_during_recording") is not False:
            errors.append(f"派生量被标记为实时计算: {expected_meta.name}")

    print(f"session: {session}")
    print("format: complete-motion raw MCAP + sidecar YAML")
    print(f"data_schema: {data_schema}")
    print(
        f"episodes={len(mcap_files)} metadata={len(episode_meta_files)} "
        f"frames={total_frames}"
    )
    if errors:
        print("RESULT: FAIL")
        for error in errors:
            print(f"  - {error}")
        return 1
    print("RESULT: PASS")
    return 0


def inspect(session: Path) -> int:
    manifest_path = session / "manifest.json"
    if not manifest_path.exists() and (session / "data" / "session.meta.yaml").is_file():
        return inspect_mcap_only(session)
    frames_path = session / "frames.bin"
    events_path = session / "events.jsonl"
    summary_path = session / "summary.json"
    for path in (manifest_path, frames_path, events_path, summary_path):
        if not path.is_file():
            raise FileNotFoundError(f"缺少文件: {path}")

    manifest = load_json(manifest_path)
    summary = load_json(summary_path)
    events = load_events(events_path)
    complete_motion_only = bool(manifest.get("complete_motion_only", False))
    motion_num_ticks = {
        int(motion["motion_id"]): int(motion["num_ticks"])
        for motion in manifest.get("motions", [])
    }
    policy_hz = float(manifest.get("policy_hz", 0.0) or 0.0)
    state_hz = float(manifest.get("state_hz", 0.0) or 0.0)
    expected_state_step = None
    if policy_hz > 0.0 and state_hz >= policy_hz:
        ratio = state_hz / policy_hz
        rounded = round(ratio)
        if rounded >= 1 and abs(ratio - rounded) < 1e-6:
            expected_state_step = rounded

    # C++ 文件头：magic[8], schema_version, record_size, endian, reserved。
    header_format = "<8sIIII"
    header_size = struct.calcsize(header_format)
    with frames_path.open("rb") as handle:
        raw_header = handle.read(header_size)
        if len(raw_header) != header_size:
            raise ValueError("frames.bin 文件头不完整")
        magic, schema, record_size, endian, _ = struct.unpack(
            header_format, raw_header
        )
        if magic != b"A3EVAL1\0":
            raise ValueError(f"frames.bin magic 错误: {magic!r}")
        if endian != 0x01020304:
            raise ValueError(f"不支持的 endian marker: 0x{endian:08x}")
        frame_type_by_schema = {
            1: EvalFrameV1,
            2: EvalFrameV2,
            3: EvalFrameV3,
            4: EvalFrameV4,
            5: EvalFrameV5,
            6: EvalFrameV6,
            7: EvalFrameV7,
        }
        frame_type = frame_type_by_schema.get(schema)
        if frame_type is None:
            raise ValueError(f"不支持的 evaluation schema: {schema}")
        python_record_size = ctypes.sizeof(frame_type)
        if record_size != python_record_size:
            raise ValueError(
                f"EvalFrame schema {schema} 布局不匹配: "
                f"文件={record_size}, inspector={python_record_size}"
            )
        if manifest.get("frame_record_size") != record_size:
            raise ValueError("manifest.frame_record_size 与 frames.bin 不一致")
        # schema 5 已删除 q_des_policy / robot_command 数组及其 valid bit；
        # 旧 schema 仍展示这两项，便于检查历史 session。
        if schema <= 4:
            valid_names = VALID_NAMES
        elif schema == 5:
            valid_names = {
                bit: name for bit, name in VALID_NAMES.items()
                if bit not in (1 << 3, 1 << 4)
            }
        else:
            valid_names = {
                bit: name for bit, name in VALID_NAMES.items()
                if bit not in (1 << 3, 1 << 9)
            }

        payload_bytes = frames_path.stat().st_size - header_size
        if payload_bytes < 0 or payload_bytes % record_size != 0:
            raise ValueError("frames.bin 尾部存在不完整 record")
        expected_frames = payload_bytes // record_size

        frame_count = 0
        frame_gaps = 0
        state_duplicates = 0
        state_backwards = 0
        state_step_anomalies = 0
        state_missing_relative_to_rate = 0
        reference_discontinuities = 0
        non_finite_frames = 0
        valid_counts: Counter[int] = Counter()
        quality_counts: Counter[int] = Counter()
        mode_counts: Counter[int] = Counter()
        phase_counts: Counter[int] = Counter()
        episode_frames: Counter[int] = Counter()
        episode_motion_ids: dict[int, set[int]] = defaultdict(set)
        episode_reference_first: dict[int, int] = {}
        episode_reference_last: dict[int, int] = {}
        stored_scope_violations = 0
        infer_ms: list[float] = []
        send_ms: list[float] = []
        control_ms: list[float] = []
        previous_frame_id = None
        previous_state_tick = None
        previous_episode_id = None
        previous_reference_by_episode: dict[int, int] = {}

        while True:
            raw = handle.read(record_size)
            if not raw:
                break
            if len(raw) != record_size:
                raise ValueError(f"读取到不完整 EvalFrame schema {schema}")
            frame = frame_type.from_buffer_copy(raw)
            frame_count += 1

            same_episode = frame.episode_id == previous_episode_id
            if (
                previous_frame_id is not None
                and (not complete_motion_only or same_episode)
                and frame.frame_id != previous_frame_id + 1
            ):
                frame_gaps += 1
            previous_frame_id = frame.frame_id

            if complete_motion_only and not same_episode:
                # 不同完整 episode 之间的 state tick 跳变属于预期间隔。
                previous_state_tick = None

            if frame.valid_mask & ROBOT_STATE_VALID and frame.state_tick != NO_TICK:
                if previous_state_tick is not None:
                    if frame.state_tick == previous_state_tick:
                        state_duplicates += 1
                    elif frame.state_tick < previous_state_tick:
                        state_backwards += 1
                    else:
                        delta = frame.state_tick - previous_state_tick
                        expected = expected_state_step or 1
                        if delta != expected:
                            state_step_anomalies += 1
                        if delta > expected:
                            state_missing_relative_to_rate += delta - expected
                previous_state_tick = frame.state_tick

            for bit in valid_names:
                if frame.valid_mask & bit:
                    valid_counts[bit] += 1
            for bit in QUALITY_NAMES:
                if frame.quality_flags & bit:
                    quality_counts[bit] += 1
            mode_counts[frame.deploy_mode] += 1
            phase_counts[frame.motion_phase] += 1
            if frame.episode_id:
                episode_frames[frame.episode_id] += 1
                episode_motion_ids[frame.episode_id].add(frame.active_motion_id)
                episode_reference_first.setdefault(
                    frame.episode_id, frame.reference_tick
                )
                episode_reference_last[frame.episode_id] = frame.reference_tick
            if complete_motion_only and (
                frame.episode_id == 0
                or frame.deploy_mode != DEPLOY_MODE_MOTION
                or frame.motion_phase != MOTION_PHASE_PLAYING
            ):
                stored_scope_violations += 1

            if frame.valid_mask & REFERENCE_VALID and frame.episode_id:
                previous = previous_reference_by_episode.get(frame.episode_id)
                if previous is not None and frame.reference_tick != previous + 1:
                    reference_discontinuities += 1
                previous_reference_by_episode[frame.episode_id] = frame.reference_tick

            previous_episode_id = frame.episode_id

            if frame.valid_mask & INFER_TIMING_VALID:
                delta = frame.infer_end_monotonic_ns - frame.infer_start_monotonic_ns
                if delta >= 0:
                    infer_ms.append(delta / 1e6)
            if frame.valid_mask & SEND_TIMING_VALID:
                delta = frame.send_end_monotonic_ns - frame.send_start_monotonic_ns
                if delta >= 0:
                    send_ms.append(delta / 1e6)
            delta = frame.control_end_monotonic_ns - frame.control_start_monotonic_ns
            if delta >= 0:
                control_ms.append(delta / 1e6)

            # 只扫描核心连续数组，快速发现 NaN/Inf 污染。旧 schema 没有
            # tau_cmd_expected，因此使用 getattr 保持向后兼容。
            core_arrays = [frame.q, frame.dq, frame.tau_est, frame.raw_action]
            if hasattr(frame, "command_q_des") and schema >= 6:
                core_arrays.extend([
                    frame.command_q_des,
                    frame.command_dq_des,
                    frame.command_tau_ff,
                    frame.command_kp,
                    frame.command_kd,
                ])
            if hasattr(frame, "tau_cmd_expected") and (
                frame.valid_mask & EXPECTED_TORQUE_VALID
            ):
                core_arrays.append(frame.tau_cmd_expected)
            if not all(
                math.isfinite(value)
                for values in core_arrays
                for value in values
            ):
                non_finite_frames += 1

    if frame_count != expected_frames:
        raise ValueError("frames.bin 记录数计算不一致")

    event_types = Counter(str(event.get("event_type")) for event in events)
    starts = defaultdict(int)
    ends = defaultdict(int)
    outcomes = Counter()
    completed_episode_ids: set[int] = set()
    completed_mcap_files: dict[int, str | None] = {}
    completed_readable_mcap_files: dict[int, str | None] = {}
    for event in events:
        episode_id = int(event.get("episode_id", 0))
        if event.get("event_type") == "EPISODE_STARTED":
            starts[episode_id] += 1
        elif event.get("event_type") == "EPISODE_ENDED":
            ends[episode_id] += 1
            outcome = str(event.get("outcome"))
            outcomes[outcome] += 1
            if outcome == "COMPLETED":
                completed_episode_ids.add(episode_id)
                completed_mcap_files[episode_id] = event.get("mcap_file")
                completed_readable_mcap_files[episode_id] = event.get(
                    "mcap_read_file"
                )
    unmatched_started = sorted(
        episode for episode, count in starts.items() if count != 1 or ends[episode] != 1
    )
    unmatched_ended = sorted(episode for episode in ends if episode not in starts)

    # 新版 session 在 data/ 下为每个完整 episode 保存一个 MCAP。这里不依赖
    # Python mcap 包，先验证文件数量、事件映射、头尾 magic 和残留 partial。
    mcap_cfg = manifest.get("episode_mcap", {})
    mcap_enabled = bool(mcap_cfg.get("enabled", False))
    mcap_files: list[Path] = []
    readable_mcap_files: list[Path] = []
    mcap_validation_errors: list[str] = []
    if mcap_enabled:
        data_dir = session / str(mcap_cfg.get("directory", "data"))
        if not data_dir.is_dir():
            mcap_validation_errors.append(f"缺少 MCAP data 目录: {data_dir}")
        else:
            mcap_files = sorted(data_dir.glob("*.mcap"))
            partial_files = sorted(data_dir.glob("*.partial"))
            if partial_files:
                mcap_validation_errors.append(
                    f"存在未提交 MCAP partial: {[p.name for p in partial_files]}"
                )
            magic = b"\x89MCAP0\r\n"
            for path in mcap_files:
                with path.open("rb") as handle:
                    prefix = handle.read(len(magic))
                    handle.seek(-len(magic), 2)
                    suffix = handle.read(len(magic))
                if prefix != magic or suffix != magic:
                    mcap_validation_errors.append(
                        f"MCAP magic/尾标记无效: {path.name}"
                    )
        for episode_id in sorted(completed_episode_ids):
            relative = completed_mcap_files.get(episode_id)
            if not relative:
                mcap_validation_errors.append(
                    f"完整 episode {episode_id} 没有 events.mcap_file"
                )
                continue
            relative_path = Path(relative)
            if relative_path.is_absolute() or ".." in relative_path.parts:
                mcap_validation_errors.append(
                    f"episode {episode_id} 的 mcap_file 不是安全相对路径"
                )
                continue
            target = session / relative_path
            if not target.is_file():
                mcap_validation_errors.append(
                    f"episode {episode_id} 的 MCAP 文件不存在: {relative}"
                )
            if f"_ep{episode_id:06d}_" not in target.name:
                mcap_validation_errors.append(
                    f"episode {episode_id} 的 MCAP 文件名缺少 episode 标识"
                )

        readable_cfg = mcap_cfg.get("readable_copy", {})
        readable_enabled = bool(readable_cfg.get("enabled", False))
        if readable_enabled:
            readable_dir = session / str(
                readable_cfg.get("directory", "data_read")
            )
            if not readable_dir.is_dir():
                mcap_validation_errors.append(
                    f"缺少可读 MCAP data_read 目录: {readable_dir}"
                )
            else:
                readable_mcap_files = sorted(readable_dir.glob("*.mcap"))
                readable_partial_files = sorted(
                    readable_dir.glob("*.partial")
                )
                if readable_partial_files:
                    mcap_validation_errors.append(
                        "存在未提交可读 MCAP partial: "
                        f"{[p.name for p in readable_partial_files]}"
                    )
                magic = b"\x89MCAP0\r\n"
                for path in readable_mcap_files:
                    with path.open("rb") as handle:
                        prefix = handle.read(len(magic))
                        handle.seek(-len(magic), 2)
                        suffix = handle.read(len(magic))
                    if prefix != magic or suffix != magic:
                        mcap_validation_errors.append(
                            f"可读 MCAP magic/尾标记无效: {path.name}"
                        )
            for episode_id in sorted(completed_episode_ids):
                relative = completed_readable_mcap_files.get(episode_id)
                if not relative:
                    mcap_validation_errors.append(
                        f"完整 episode {episode_id} 没有 events.mcap_read_file"
                    )
                    continue
                relative_path = Path(relative)
                if relative_path.is_absolute() or ".." in relative_path.parts:
                    mcap_validation_errors.append(
                        f"episode {episode_id} 的 mcap_read_file 不是安全相对路径"
                    )
                    continue
                target = session / relative_path
                if not target.is_file():
                    mcap_validation_errors.append(
                        f"episode {episode_id} 的可读 MCAP 不存在: {relative}"
                    )
                if f"_ep{episode_id:06d}_" not in target.name:
                    mcap_validation_errors.append(
                        f"episode {episode_id} 的可读 MCAP 文件名缺少 episode 标识"
                    )
        else:
            readable_enabled = False
    else:
        readable_enabled = False

    print(f"session: {session}")
    print(
        f"schema={schema} record_size={record_size} frames={frame_count} "
        f"events={len(events)}"
    )
    print(
        "writer: "
        f"frames accepted/written/dropped="
        f"{summary.get('frames_accepted')}/{summary.get('frames_written')}/"
        f"{summary.get('frames_dropped')}; events="
        f"{summary.get('events_accepted')}/{summary.get('events_written')}/"
        f"{summary.get('events_dropped')}"
    )
    if complete_motion_only:
        print(
            "complete-motion: "
            f"episodes started/committed/discarded="
            f"{summary.get('episodes_started')}/"
            f"{summary.get('episodes_committed')}/"
            f"{summary.get('episodes_discarded')}; "
            f"frames ignored/discarded="
            f"{summary.get('frames_ignored')}/"
            f"{summary.get('frames_discarded')}"
        )
    if "telemetry_accepted" in summary:
        print(
            "telemetry: accepted/published/dropped="
            f"{summary.get('telemetry_accepted')}/"
            f"{summary.get('telemetry_published')}/"
            f"{summary.get('telemetry_dropped')}"
        )
    if mcap_enabled:
        print(
            "episode-mcap: files/failures/bytes="
            f"{len(mcap_files)}/{summary.get('mcap_write_failures')}/"
            f"{summary.get('mcap_bytes_written')}"
        )
        if readable_enabled:
            print(
                "episode-mcap-readable: files/bytes="
                f"{len(readable_mcap_files)}/"
                f"{summary.get('readable_mcap_bytes_written')}"
            )
    print(
        "continuity: "
        f"frame_gap_segments={frame_gaps}, state_duplicates={state_duplicates}, "
        f"state_backward={state_backwards}, "
        f"expected_state_step={expected_state_step or 'unknown'}, "
        f"state_step_anomalies={state_step_anomalies}, "
        f"state_missing_relative_to_rate={state_missing_relative_to_rate}, "
        f"reference_discontinuities={reference_discontinuities}, "
        f"non_finite_frames={non_finite_frames}"
    )
    print("valid coverage:")
    for bit, name in valid_names.items():
        ratio = 100.0 * valid_counts[bit] / frame_count if frame_count else 0.0
        print(f"  {name}: {valid_counts[bit]}/{frame_count} ({ratio:.1f}%)")
    print("quality flags:")
    for bit, name in QUALITY_NAMES.items():
        print(f"  {name}: {quality_counts[bit]}")
    print(f"mode_counts: {dict(sorted(mode_counts.items()))}")
    print(f"motion_phase_counts: {dict(sorted(phase_counts.items()))}")
    print("timing:")
    print(timing_line("control", control_ms))
    print(timing_line("inference", infer_ms))
    print(timing_line("send", send_ms))
    print(f"event_types: {dict(sorted(event_types.items()))}")
    print(f"episode_outcomes: {dict(sorted(outcomes.items()))}")
    print(f"episode_frame_counts: {dict(sorted(episode_frames.items()))}")

    errors = []
    errors.extend(mcap_validation_errors)
    if int(summary.get("frames_written", -1)) != frame_count:
        errors.append("summary.frames_written 与实际帧数不一致")
    if int(summary.get("events_written", -1)) != len(events):
        errors.append("summary.events_written 与实际事件数不一致")
    if not complete_motion_only and int(summary.get("frames_dropped", 0)) != 0:
        errors.append("writer queue 存在丢帧")
    if int(summary.get("events_dropped", 0)) != 0:
        errors.append("writer queue 存在丢事件")
    if frame_gaps or state_backwards or reference_discontinuities:
        errors.append("时间轴存在不连续")
    if non_finite_frames:
        errors.append("连续信号包含 NaN/Inf")
    if unmatched_started or unmatched_ended:
        errors.append(
            f"episode 事件不配对: started={unmatched_started}, ended={unmatched_ended}"
        )
    if complete_motion_only:
        stored_episode_ids = set(episode_frames)
        if stored_scope_violations:
            errors.append(
                f"complete-only 文件包含非 PLAYING/无 episode 帧: "
                f"{stored_scope_violations}"
            )
        if stored_episode_ids != completed_episode_ids:
            errors.append(
                "已存 episode 与 COMPLETED 事件不一致: "
                f"stored={sorted(stored_episode_ids)}, "
                f"completed={sorted(completed_episode_ids)}"
            )
        for episode_id in sorted(stored_episode_ids):
            motion_ids = episode_motion_ids[episode_id]
            if len(motion_ids) != 1:
                errors.append(
                    f"episode {episode_id} 混入多个 motion: {sorted(motion_ids)}"
                )
                continue
            motion_id = next(iter(motion_ids))
            expected_ticks = motion_num_ticks.get(motion_id)
            if expected_ticks is None:
                errors.append(f"episode {episode_id} 的 motion_id 未写入 manifest")
                continue
            if (
                episode_frames[episode_id] != expected_ticks
                or episode_reference_first.get(episode_id) != 0
                or episode_reference_last.get(episode_id) != expected_ticks - 1
            ):
                errors.append(
                    f"episode {episode_id} reference 不完整: "
                    f"frames={episode_frames[episode_id]}, "
                    f"first={episode_reference_first.get(episode_id)}, "
                    f"last={episode_reference_last.get(episode_id)}, "
                    f"expected={expected_ticks}"
                )
        if int(summary.get("episodes_committed", -1)) != len(
            completed_episode_ids
        ):
            errors.append("summary.episodes_committed 与 COMPLETED 事件数不一致")
        if mcap_enabled:
            if int(summary.get("mcap_files_written", -1)) != len(mcap_files):
                errors.append("summary.mcap_files_written 与实际 MCAP 数量不一致")
            if len(mcap_files) != len(completed_episode_ids):
                errors.append("完整 episode 数与 MCAP 文件数不一致")
            if int(summary.get("mcap_write_failures", -1)) != 0:
                errors.append("存在 MCAP 写入失败")
            if readable_enabled:
                if int(
                    summary.get("readable_mcap_files_written", -1)
                ) != len(readable_mcap_files):
                    errors.append(
                        "summary.readable_mcap_files_written "
                        "与实际可读 MCAP 数量不一致"
                    )
                if len(readable_mcap_files) != len(completed_episode_ids):
                    errors.append("完整 episode 数与可读 MCAP 文件数不一致")

    if errors:
        print("RESULT: FAIL")
        for error in errors:
            print(f"  - {error}")
        return 1
    print("RESULT: PASS")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="检查 A3 Evaluation Recorder Demo 1 session"
    )
    parser.add_argument("session", type=Path, help="session_YYYYmmdd_HHMMSS 目录")
    args = parser.parse_args()
    try:
        return inspect(args.session.expanduser().resolve())
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
