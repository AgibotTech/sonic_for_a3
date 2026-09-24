#!/usr/bin/env python3
"""A3 closed-chain MuJoCo sim2sim runner.

This is the A3 sim2sim ground-truth path.  It uses repository-local defaults
for the T2.5 loop MJCF, A3 ankle/waist solver library, and motion CSV. Pass the
policy checkpoint explicitly; the released default artifact is
``checkpoints/024_step100000/model_step_100000.pt``.

Controls:
  Space   - pause/resume
  ./Right - step forward one policy frame while paused
  ,/Left  - rewind one reference frame and reset sim state
  =       - next motion clip
  -       - previous motion clip
  R       - reset current clip to frame 0
"""

from __future__ import annotations

import argparse
import csv
import copy
import ctypes
import json
from collections import defaultdict, deque
from dataclasses import dataclass, replace
import hashlib
import os
import pickle
import queue as queue_lib
import tempfile
import traceback
import xml.etree.ElementTree as ET
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation, Slerp
import torch
from torch import nn

from gear_sonic.utils.playback import (
    AsyncLRUCache,
    CMD_NEXT_CLIP,
    CMD_PREV_CLIP,
    CMD_RESET,
    CMD_STEP_BACKWARD,
    CMD_STEP_FORWARD,
    CMD_TOGGLE_PAUSE,
    PlaybackCommandQueue,
    TerminalProgress,
    shuffled_once,
)
from gear_sonic.utils.a3_motor_params import (
    ARMATURE_PFP_110_75,
    ARMATURE_PFP_41_48,
    ARMATURE_PFP_59_60,
    ARMATURE_PFP_78_58,
    ARMATURE_PFP_93_65,
)
from gear_sonic.utils.terrain_profile import patch_mjcf_with_terrain_profile


# =============================================================================
# Shared Constants And Data Shapes
# =============================================================================

DEFAULT_URDF = (
    REPO_ROOT
    / "gear_sonic/data/assets/robot_description/urdf/a3/model_collision_optimized_passive_foot_twostage_fit_optimized.urdf"
)
AUTO_OUTPUT_VIDEO = Path("__auto_output_video__")

HEAD_JOINTS = ("head_yaw_joint", "head_pitch_joint")
PASSIVE_FOOT_JOINT_NAMES = (
    "left_foot_forefoot_joint",
    "left_foot_toe_joint",
    "right_foot_forefoot_joint",
    "right_foot_toe_joint",
)

NUM_POLICY_DOFS = 29
NUM_FUTURE_FRAMES = 10
TARGET_FPS = 50.0
POLICY_DT = 1.0 / TARGET_FPS
FUTURE_FRAME_SKIP = 5
ACTION_CLIP = 20.0
ACTION_SCALE_COEFF = 0.25
OBS_TERM_ORDER = ("base_ang_vel", "joint_pos", "joint_vel", "actions", "gravity_dir")
OBS_TERM_DIMS = {
    "base_ang_vel": 3,
    "joint_pos": NUM_POLICY_DOFS,
    "joint_vel": NUM_POLICY_DOFS,
    "actions": NUM_POLICY_DOFS,
    "gravity_dir": 3,
}

# 14 tracked bodies matching training-framework eval (commands.motion.body_names),
# used to align mpjpe_g / mpjpe_l with smpl_sim.compute_metrics_lite (root_idx=0).
TRACKED_BODY_NAMES: tuple[str, ...] = (
    "pelvis_link",
    "left_hip_roll_Link",
    "left_knee_Link",
    "left_ankle_roll_Link",
    "right_hip_roll_Link",
    "right_knee_Link",
    "right_ankle_roll_Link",
    "torso_Link",
    "left_shoulder_roll_Link",
    "left_elbow_Link",
    "left_wrist_yaw_Link",
    "right_shoulder_roll_Link",
    "right_elbow_Link",
    "right_wrist_yaw_Link",
)
# Index of the root body (pelvis_link) within TRACKED_BODY_NAMES; matches
# compute_metrics_lite default root_idx=0.
MPJPE_ROOT_IDX = 0

ENCODER_FRAME_DIM = NUM_POLICY_DOFS * 2 + 6
ENCODER_INPUT_DIM = NUM_FUTURE_FRAMES * ENCODER_FRAME_DIM
ENCODER_TERMS = ("command_multi_future_nonflat", "motion_anchor_ori_b_mf_nonflat")

DEFAULT_Q_BY_NAME = {
    "left_hip_pitch_joint": -0.1311,
    "right_hip_pitch_joint": -0.1311,
    "left_hip_roll_joint": 0.0056,
    "right_hip_roll_joint": -0.0056,
    "left_hip_yaw_joint": -0.0348,
    "right_hip_yaw_joint": 0.0348,
    "left_knee_joint": 0.2468,
    "right_knee_joint": 0.2468,
    "left_ankle_pitch_joint": -0.1204,
    "right_ankle_pitch_joint": -0.1204,
    "left_ankle_roll_joint": -0.0078,
    "right_ankle_roll_joint": 0.0078,
    "waist_yaw_joint": 0.0,
    "waist_roll_joint": 0.0,
    "waist_pitch_joint": 0.0,
    "left_shoulder_pitch_joint": 0.3,
    "right_shoulder_pitch_joint": 0.3,
    "left_shoulder_roll_joint": 0.12,
    "right_shoulder_roll_joint": -0.12,
    "left_shoulder_yaw_joint": 0.0,
    "right_shoulder_yaw_joint": 0.0,
    "left_elbow_joint": 0.8,
    "right_elbow_joint": 0.8,
    "left_wrist_roll_joint": 0.0,
    "right_wrist_roll_joint": 0.0,
    "left_wrist_pitch_joint": 0.0,
    "right_wrist_pitch_joint": 0.0,
    "left_wrist_yaw_joint": 0.0,
    "right_wrist_yaw_joint": 0.0,
}


@dataclass(frozen=True)
class JointControl:
    kp: float
    kd: float
    effort: float
    armature: float
    friction: float


KP_SCALE = 1.0
KD_SCALE = 1.0


# =============================================================================
# Core Data Models
# =============================================================================

@dataclass
class A3Mapping:
    il_joint_names: list[str]
    mj_full_joint_names: list[str]
    mj29_joint_names: list[str]
    head_joint_names: list[str]
    full_joint_ids: list[int]
    mj29_full_indices: np.ndarray
    il_to_mj29: np.ndarray
    mj29_to_il: np.ndarray
    default_q_il: np.ndarray
    default_q_mj29: np.ndarray
    action_scale_il: np.ndarray
    action_scale_mj29: np.ndarray
    actuator_id_by_joint: dict[str, int]
    qpos_addr_by_joint: dict[str, int]
    qvel_addr_by_joint: dict[str, int]
    body_id_by_joint: dict[str, int]
    axis_by_joint: dict[str, np.ndarray]
    root_body_id: int
    anchor_body_id: int = 0
    anchor_body_name: str = "pelvis_link"


@dataclass
class MotionReference:
    path: Path
    fps: float
    qpos: np.ndarray
    qvel: np.ndarray
    dof_full: np.ndarray
    dof_mj29: np.ndarray
    dof_il: np.ndarray
    dof_vel_il: np.ndarray
    root_pos: np.ndarray
    root_quat_wxyz: np.ndarray
    anchor_quat_wxyz: np.ndarray

    @property
    def num_frames(self) -> int:
        return int(self.qpos.shape[0])


@dataclass(frozen=True)
class VideoOutputConfig:
    width: int
    height: int
    fps: float
    encoder: str
    preset: str
    crf: int
    frame_stride: int


@dataclass(frozen=True)
class SimConfig:
    checkpoint: Path
    motion: Path
    urdf: Path
    mjcf: Path
    output_video: Path | None
    metrics_out: Path | None
    timeseries_out: Path | None
    max_policy_steps: int | None
    replay_policy_dump: Path | None
    reset_frame: int
    replay_horizon: int | None
    replay_command: str
    video: VideoOutputConfig
    encoder: str
    future_frame_skip: int
    future_history_frames: int
    future_valid_frames: int | None
    future_zero_pad: bool
    start_paused: bool
    batch_once: bool
    random_order: bool
    cache_size: int
    anchor_body: str
    csv_source_fps: float
    csv_frame_stride: int
    action_delay_ms: float = 0.0


# =============================================================================
# Rendering, Video, And Viewer Reference Robot
# =============================================================================

class VideoRecorder:
    def __init__(
        self,
        model: mujoco.MjModel,
        root_body_id: int,
        path: Path,
        video_config: VideoOutputConfig,
    ) -> None:
        width = video_config.width
        height = video_config.height
        fps = video_config.fps
        if width <= 0 or height <= 0:
            raise ValueError("video width and height must be positive")
        if fps <= 0.0:
            raise ValueError("video FPS must be positive")
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg is None:
            raise RuntimeError("ffmpeg is required for H.264 video output")

        self.path = path.expanduser()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_offscreen_size(model, width, height)
        self.renderer = mujoco.Renderer(model, height=height, width=width)
        self.camera = self._build_track_camera(root_body_id)
        self._ffmpeg = subprocess.Popen(
            self._build_ffmpeg_command(ffmpeg, width, height, fps, self.path, video_config),
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        self._ffmpeg_finalized = False
        self._ffmpeg_stderr = ""
        self._closed = False
        self.frames = 0

    @staticmethod
    def _ensure_offscreen_size(model: mujoco.MjModel, width: int, height: int) -> None:
        model.vis.global_.offwidth = max(int(model.vis.global_.offwidth), width)
        model.vis.global_.offheight = max(int(model.vis.global_.offheight), height)

    @staticmethod
    def _build_track_camera(root_body_id: int) -> mujoco.MjvCamera:
        cam = mujoco.MjvCamera()
        mujoco.mjv_defaultCamera(cam)
        cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        cam.trackbodyid = int(root_body_id)
        cam.distance = 4.0
        cam.azimuth = 90.0
        cam.elevation = -20.0
        return cam

    @staticmethod
    def _build_ffmpeg_command(
        ffmpeg: str,
        width: int,
        height: int,
        fps: float,
        path: Path,
        video_config: VideoOutputConfig,
    ) -> list[str]:
        fps_text = f"{fps:.6f}".rstrip("0").rstrip(".")
        return [
            ffmpeg,
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-vcodec",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            f"{width}x{height}",
            "-r",
            fps_text,
            "-i",
            "-",
            "-an",
            "-c:v",
            video_config.encoder,
            "-preset",
            video_config.preset,
            "-crf",
            str(video_config.crf),
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(path),
        ]

    def _finish_ffmpeg(self, timeout: float) -> int:
        if self._ffmpeg_finalized:
            return 0 if self._ffmpeg.returncode is None else int(self._ffmpeg.returncode)
        if self._ffmpeg.stdin is not None:
            try:
                self._ffmpeg.stdin.close()
            except BrokenPipeError:
                pass
            self._ffmpeg.stdin = None
        try:
            _, stderr = self._ffmpeg.communicate(timeout=timeout)
        except subprocess.TimeoutExpired as exc:
            self._ffmpeg.kill()
            _, stderr = self._ffmpeg.communicate()
            self._ffmpeg_finalized = True
            self._ffmpeg_stderr = stderr.decode(errors="replace").strip()
            raise RuntimeError(f"ffmpeg did not finish writing {self.path}") from exc
        self._ffmpeg_finalized = True
        self._ffmpeg_stderr = stderr.decode(errors="replace").strip()
        return 0 if self._ffmpeg.returncode is None else int(self._ffmpeg.returncode)

    def _ffmpeg_failure_message(self, context: str, returncode: int) -> str:
        detail = f": {self._ffmpeg_stderr}" if self._ffmpeg_stderr else ""
        return f"{context}; ffmpeg exited with code {returncode}{detail}"

    def _raise_if_ffmpeg_exited(self) -> None:
        returncode = self._ffmpeg.poll()
        if returncode is None:
            return
        self._finish_ffmpeg(timeout=5.0)
        raise RuntimeError(
            self._ffmpeg_failure_message(
                f"ffmpeg H.264 encoder stopped while writing {self.path}",
                int(returncode),
            )
        )

    def write(self, data: mujoco.MjData, ref_overlay=None) -> None:
        self._raise_if_ffmpeg_exited()
        if ref_overlay is not None and isinstance(self.camera, mujoco.MjvCamera):
            ref_overlay.apply_camera(self.camera)
        self.renderer.update_scene(data, camera=self.camera)
        if ref_overlay is not None:
            ref_overlay.add_to_scene(self.renderer.scene)
        rgb = self.renderer.render()
        frame = np.ascontiguousarray(rgb)
        if frame.dtype != np.uint8:
            frame = np.clip(frame, 0, 255).astype(np.uint8)
        if self._ffmpeg.stdin is None:
            raise RuntimeError(f"ffmpeg stdin is closed for {self.path}")
        try:
            self._ffmpeg.stdin.write(frame.tobytes())
        except (BrokenPipeError, OSError) as exc:
            returncode = self._finish_ffmpeg(timeout=5.0)
            raise RuntimeError(
                self._ffmpeg_failure_message(
                    f"ffmpeg H.264 encoder stopped while writing {self.path}",
                    returncode,
                )
            ) from exc
        self.frames += 1

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            returncode = self._finish_ffmpeg(timeout=60.0)
            if returncode != 0:
                raise RuntimeError(
                    self._ffmpeg_failure_message(
                        f"ffmpeg failed while writing {self.path}",
                        returncode,
                    )
                )
        finally:
            if self._ffmpeg.poll() is None:
                self._ffmpeg.kill()
                self._ffmpeg.communicate()
            self.renderer.close()


class RefRobotOverlay:
    """Render-only transparent reference robot and camera target."""

    def __init__(
        self,
        model: mujoco.MjModel,
        reference: MotionReference,
        camera_distance: float,
        camera_azimuth: float,
        camera_elevation: float,
    ) -> None:
        self.model = model
        self.reference = reference
        self.ref_offset = np.asarray(REF_OFFSET, dtype=np.float64)
        self.alpha = float(np.clip(REF_ALPHA, 0.0, 1.0))
        self.tint = float(np.clip(REF_TINT, 0.0, 1.0))
        self.camera_distance = float(camera_distance)
        self.camera_azimuth = float(camera_azimuth)
        self.camera_elevation = float(camera_elevation)
        self._fixed_lookat_z: float | None = None
        self.data = mujoco.MjData(model)
        self.scene = mujoco.MjvScene(model, 10000)
        self.opt = mujoco.MjvOption()
        self.cam = mujoco.MjvCamera()
        self.pert = mujoco.MjvPerturb()
        self.geom_ids = self._find_overlay_geom_ids(model)
        self.geom_id_set = set(self.geom_ids)
        if not self.geom_ids:
            raise ValueError("No visual robot geoms found for reference overlay")
        self.current_root_pos = np.zeros(3, dtype=np.float64)

    @staticmethod
    def _find_overlay_geom_ids(model: mujoco.MjModel) -> list[int]:
        def is_hidden_linkage(geom_id: int) -> bool:
            geom_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
            body_name = mujoco.mj_id2name(
                model,
                mujoco.mjtObj.mjOBJ_BODY,
                int(model.geom_bodyid[geom_id]),
            )
            return _is_hidden_ref_visual_name(geom_name, body_name)

        visual_ids = [
            geom_id
            for geom_id in range(model.ngeom)
            if model.geom_bodyid[geom_id] != 0 and int(model.geom_group[geom_id]) == 2
            and not is_hidden_linkage(geom_id)
        ]
        if visual_ids:
            return visual_ids
        return [
            geom_id
            for geom_id in range(model.ngeom)
            if model.geom_bodyid[geom_id] != 0 and model.geom_type[geom_id] == mujoco.mjtGeom.mjGEOM_MESH
            and not is_hidden_linkage(geom_id)
        ]

    def set_reference(self, reference: MotionReference) -> None:
        self.reference = reference

    def update(self, ref_frame: int, current_root_pos: np.ndarray) -> None:
        ref_frame = int(np.clip(ref_frame, 0, self.reference.num_frames - 1))
        self.data.time = ref_frame / self.reference.fps
        self.data.qpos[:] = self.reference.qpos[ref_frame]
        self.data.qvel[:] = self.reference.qvel[ref_frame]
        self.data.ctrl[:] = 0.0
        self.data.qpos[:3] += self.ref_offset
        mujoco.mj_forward(self.model, self.data)
        mujoco.mjv_updateScene(
            self.model,
            self.data,
            self.opt,
            self.pert,
            self.cam,
            mujoco.mjtCatBit.mjCAT_ALL,
            self.scene,
        )
        self.current_root_pos = np.asarray(current_root_pos, dtype=np.float64).copy()
        if self._fixed_lookat_z is None:
            self._fixed_lookat_z = float(self.current_root_pos[2])

    @staticmethod
    def _copy_scene_geom(dst, src) -> None:
        for field in (
            "type",
            "dataid",
            "objtype",
            "objid",
            "category",
            "texcoord",
            "segid",
            "matid",
            "transparent",
            "camdist",
            "modelrbound",
            "emission",
            "specular",
            "shininess",
            "reflectance",
        ):
            setattr(dst, field, getattr(src, field))
        dst.pos[:] = src.pos
        dst.mat[:] = src.mat
        dst.size[:] = src.size
        dst.rgba[:] = src.rgba
        dst.label = ""

    def add_to_scene(self, scene: mujoco.MjvScene) -> None:
        for ref_scene_idx in range(self.scene.ngeom):
            ref_geom = self.scene.geoms[ref_scene_idx]
            if ref_geom.objtype != mujoco.mjtObj.mjOBJ_GEOM or ref_geom.objid not in self.geom_id_set:
                continue
            if scene.ngeom >= scene.maxgeom:
                return
            geom = scene.geoms[scene.ngeom]
            self._copy_scene_geom(geom, ref_geom)
            # Preserve the robot's original black/white mesh appearance; only add
            # a small tint so the reference is distinguishable from the simulated robot.
            rgba = geom.rgba.copy()
            rgba[:3] = (1.0 - self.tint) * rgba[:3] + self.tint * np.array([0.1, 0.55, 1.0], dtype=np.float32)
            rgba[3] = self.alpha
            geom.rgba[:] = rgba
            geom.category = mujoco.mjtCatBit.mjCAT_DECOR
            geom.transparent = 1 if self.alpha < 1.0 else 0
            scene.ngeom += 1

    def camera_lookat(self) -> np.ndarray:
        lookat = self.current_root_pos.copy()
        if self._fixed_lookat_z is not None:
            lookat[2] = self._fixed_lookat_z
        return lookat

    def apply_camera(self, cam: mujoco.MjvCamera) -> None:
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.fixedcamid = -1
        cam.trackbodyid = -1
        cam.lookat[:] = self.camera_lookat()
        cam.distance = self.camera_distance
        cam.azimuth = self.camera_azimuth
        cam.elevation = self.camera_elevation

@dataclass
class VideoOverlayConfig:
    references: list[MotionReference]
    camera_distance: float
    camera_azimuth: float
    camera_elevation: float


def _video_worker_main(
    frame_queue,
    status_queue,
    mjcf_path: str,
    output_path: str,
    video_config: VideoOutputConfig,
    overlay_config: VideoOverlayConfig | None,
) -> None:
    recorder = None
    try:
        model = mujoco.MjModel.from_xml_path(mjcf_path)
        root_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis_link")
        if root_body_id < 0:
            raise ValueError("Expected root body 'pelvis_link' in MJCF")
        recorder = VideoRecorder(
            model=model,
            root_body_id=int(root_body_id),
            path=Path(output_path),
            video_config=video_config,
        )
        ref_overlay = None
        if overlay_config is not None:
            ref_overlay = RefRobotOverlay(
                model=model,
                reference=overlay_config.references[0],
                camera_distance=overlay_config.camera_distance,
                camera_azimuth=overlay_config.camera_azimuth,
                camera_elevation=overlay_config.camera_elevation,
            )
        data = mujoco.MjData(model)
        status_queue.put(("ready", None))
        while True:
            packet = frame_queue.get()
            if packet is None:
                break
            time_s, qpos, qvel, ctrl, ref_index, ref_frame = packet
            data.time = float(time_s)
            data.qpos[:] = qpos
            data.qvel[:] = qvel
            if ctrl is not None and len(ctrl) == model.nu:
                data.ctrl[:] = ctrl
            mujoco.mj_forward(model, data)
            if ref_overlay is not None and ref_frame is not None:
                ref_overlay.set_reference(overlay_config.references[int(ref_index) % len(overlay_config.references)])
                ref_overlay.update(int(ref_frame), data.qpos[:3])
            recorder.write(data, ref_overlay)
        recorder.close()
        recorder = None
        status_queue.put(("done", None))
    except BaseException:
        status_queue.put(("error", traceback.format_exc()))
        raise
    finally:
        if recorder is not None:
            recorder.close()


class AsyncVideoRecorder:
    """Record video in a spawned process so passive viewer GL stays isolated."""

    is_async = True

    def __init__(
        self,
        model_path: Path,
        path: Path,
        video_config: VideoOutputConfig,
        overlay_config: VideoOverlayConfig | None,
    ) -> None:
        import multiprocessing as mp

        self.path = path.expanduser()
        self.frames = 0
        self._closed = False
        ctx = mp.get_context("spawn")
        self._frame_queue = ctx.Queue(maxsize=32)
        self._status_queue = ctx.Queue()
        self._process = ctx.Process(
            target=_video_worker_main,
            args=(
                self._frame_queue,
                self._status_queue,
                str(model_path.expanduser()),
                str(self.path),
                video_config,
                overlay_config,
            ),
            name="a3-sim2sim-video",
        )
        self._process.start()
        self._wait_ready()

    def _wait_ready(self) -> None:
        try:
            kind, payload = self._status_queue.get(timeout=20.0)
        except queue_lib.Empty as exc:
            self._process.terminate()
            self._process.join(timeout=2.0)
            raise RuntimeError("video worker did not become ready within 20 seconds") from exc
        if kind == "ready":
            return
        if kind == "error":
            raise RuntimeError(f"video worker failed during startup:\n{payload}")
        raise RuntimeError(f"unexpected video worker status during startup: {kind}")

    def _raise_if_failed(self) -> None:
        while True:
            try:
                kind, payload = self._status_queue.get_nowait()
            except queue_lib.Empty:
                break
            if kind == "error":
                raise RuntimeError(f"video worker failed:\n{payload}")
        if not self._process.is_alive() and self._process.exitcode not in (0, None):
            raise RuntimeError(f"video worker exited with code {self._process.exitcode}")

    def write(self, data: mujoco.MjData, ref_index: int, ref_frame: int | None) -> None:
        if self._closed:
            return
        self._raise_if_failed()
        packet = (
            float(data.time),
            data.qpos.copy(),
            data.qvel.copy(),
            data.ctrl.copy() if data.ctrl.size else None,
            int(ref_index),
            None if ref_frame is None else int(ref_frame),
        )
        self._frame_queue.put(packet)
        self.frames += 1

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self._frame_queue.put(None)
            self._process.join()
        finally:
            if self._process.is_alive():
                self._process.terminate()
                self._process.join(timeout=2.0)
        self._raise_if_failed()


@dataclass
class ViewerMirror:
    model: mujoco.MjModel
    data: mujoco.MjData
    sim_qpos_width: int
    sim_qvel_width: int
    ref_qpos_start: int
    ref_qvel_start: int
    root_body_id: int


def _prepend_name_attributes(elem: ET.Element, prefix: str) -> None:
    if "name" in elem.attrib:
        elem.attrib["name"] = prefix + elem.attrib["name"]
    for child in list(elem):
        _prepend_name_attributes(child, prefix)


def _format_rgba(rgba: np.ndarray) -> str:
    return " ".join(f"{float(x):.6g}" for x in rgba)


def _is_hidden_ref_visual_name(*names: str | None) -> bool:
    lowered = " ".join(name.lower() for name in names if name)
    return any(part in lowered for part in REF_HIDDEN_VISUAL_NAME_PARTS)


def _ref_geom_rgba(elem: ET.Element, alpha: float, tint: float) -> str:
    raw = elem.attrib.get("rgba")
    if raw:
        rgba = np.fromstring(raw, dtype=np.float64, sep=" ")
        if rgba.shape != (4,):
            rgba = np.array([0.45, 0.45, 0.50, 1.0], dtype=np.float64)
    else:
        rgba = np.array([0.45, 0.45, 0.50, 1.0], dtype=np.float64)
    rgba[:3] = (1.0 - tint) * rgba[:3] + tint * np.array([0.1, 0.55, 1.0], dtype=np.float64)
    rgba[3] = alpha
    return _format_rgba(np.clip(rgba, 0.0, 1.0))


def _make_ref_body_visual(elem: ET.Element, alpha: float, tint: float) -> None:
    for child in list(elem):
        if child.tag == "geom":
            is_collision = child.attrib.get("class") == "collision" or child.attrib.get("group") == "3"
            is_hidden_linkage = _is_hidden_ref_visual_name(
                child.attrib.get("name"),
                child.attrib.get("mesh"),
            )
            if is_collision or is_hidden_linkage:
                elem.remove(child)
                continue
        _make_ref_body_visual(child, alpha, tint)
    if elem.tag == "geom":
        elem.attrib["rgba"] = _ref_geom_rgba(elem, alpha, tint)
        elem.attrib["contype"] = "0"
        elem.attrib["conaffinity"] = "0"


def _absolute_meshdir(mjcf_path: Path, compiler: ET.Element | None) -> None:
    if compiler is None or "meshdir" not in compiler.attrib:
        return
    meshdir = Path(compiler.attrib["meshdir"]).expanduser()
    if not meshdir.is_absolute():
        meshdir = (mjcf_path.expanduser().resolve().parent / meshdir).resolve()
    compiler.attrib["meshdir"] = str(meshdir)


def _build_viewer_mirror_xml(mjcf_path: Path, ref_alpha: float, ref_tint: float) -> Path:
    mjcf_path = mjcf_path.expanduser().resolve()
    source = mjcf_path.read_bytes()
    digest = hashlib.sha1(
        source
        + (
            f"|ref_alpha={float(ref_alpha):.6f}|ref_tint={float(ref_tint):.6f}"
            f"|hide={','.join(REF_HIDDEN_VISUAL_NAME_PARTS)}|viewer_mirror_v3"
        ).encode("utf-8")
    ).hexdigest()[:12]
    cache_path = Path(tempfile.gettempdir()) / f"a3_sim2sim_viewer_mirror_{digest}.xml"
    if cache_path.exists():
        return cache_path

    tree = ET.parse(mjcf_path)
    root = tree.getroot()
    _absolute_meshdir(mjcf_path, root.find("compiler"))
    worldbody = root.find("worldbody")
    if worldbody is None:
        raise ValueError(f"{mjcf_path} has no <worldbody>")
    robot_body = None
    for child in list(worldbody):
        if child.tag == "body":
            robot_body = child
            break
    if robot_body is None:
        raise ValueError(f"{mjcf_path} has no root robot <body>")

    ref_body = copy.deepcopy(robot_body)
    _prepend_name_attributes(ref_body, "ref_")
    tint = float(np.clip(ref_tint, 0.0, 1.0))
    alpha = float(np.clip(ref_alpha, 0.0, 1.0))
    _make_ref_body_visual(ref_body, alpha, tint)
    worldbody.append(ref_body)

    root.attrib["model"] = root.attrib.get("model", "a3") + "_viewer_mirror"
    cache_path.write_bytes(ET.tostring(root, encoding="utf-8"))
    return cache_path


def _read_foot_spring_yaml(path: Path) -> dict[str, dict[str, float]]:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required to load --foot-spring-cfg") from exc

    payload = yaml.safe_load(path.read_text()) or {}
    joints = payload.get("joints", {})
    if not isinstance(joints, dict):
        raise ValueError(f"{path} must contain a 'joints' mapping")

    result: dict[str, dict[str, float]] = {}
    for joint_name, values in joints.items():
        if not isinstance(values, dict):
            raise ValueError(f"spring entry for {joint_name!r} must be a mapping")
        result[str(joint_name)] = {
            str(key): float(value) for key, value in values.items()
        }
    return result


def _joint_float_attr(joint: ET.Element, attr: str, default: float) -> float:
    value = joint.attrib.get(attr)
    return default if value is None else float(value)


def build_foot_spring_mjcf(
    mjcf_path: Path,
    spring_cfg_path: Path | None,
    stiffness_scale: float,
    damping_scale: float,
) -> Path:
    mjcf_path = mjcf_path.expanduser().resolve()
    spring_cfg: dict[str, dict[str, float]] = {}
    cfg_bytes = b""
    if spring_cfg_path is not None:
        spring_cfg_path = spring_cfg_path.expanduser().resolve()
        cfg_bytes = spring_cfg_path.read_bytes()
        spring_cfg = _read_foot_spring_yaml(spring_cfg_path)

    source = mjcf_path.read_bytes()
    digest = hashlib.sha1(
        source
        + cfg_bytes
        + (
            f"|foot_springs_v1|stiffness_scale={float(stiffness_scale):.9g}"
            f"|damping_scale={float(damping_scale):.9g}"
        ).encode("utf-8")
    ).hexdigest()[:12]
    cache_path = Path(tempfile.gettempdir()) / f"a3_sim2sim_foot_springs_{digest}.xml"
    if cache_path.exists():
        return cache_path

    tree = ET.parse(mjcf_path)
    root = tree.getroot()
    _absolute_meshdir(mjcf_path, root.find("compiler"))

    joints_by_name = {
        joint.attrib["name"]: joint
        for joint in root.iter("joint")
        if "name" in joint.attrib
    }
    target_names = tuple(spring_cfg) if spring_cfg else FOOT_SPRING_JOINT_NAMES
    missing = [name for name in target_names if name not in joints_by_name]
    if missing:
        raise ValueError(f"foot spring joints missing from MJCF: {missing}")

    for name in target_names:
        joint = joints_by_name[name]
        values = spring_cfg.get(name, {})
        stiffness = values.get("stiffness", _joint_float_attr(joint, "stiffness", 0.0))
        damping = values.get("damping", _joint_float_attr(joint, "damping", 0.0))
        springref = values.get("ref", _joint_float_attr(joint, "springref", 0.0))
        joint.attrib["stiffness"] = f"{stiffness * stiffness_scale:.9g}"
        joint.attrib["damping"] = f"{damping * damping_scale:.9g}"
        joint.attrib["springref"] = f"{springref:.9g}"

    cache_path.write_bytes(ET.tostring(root, encoding="utf-8"))
    return cache_path


def build_viewer_mirror(mjcf_path: Path, sim_model: mujoco.MjModel, ref_alpha: float, ref_tint: float) -> ViewerMirror:
    mirror_xml = _build_viewer_mirror_xml(mjcf_path, ref_alpha, ref_tint)
    viewer_model = mujoco.MjModel.from_xml_path(str(mirror_xml))
    expected_nq = sim_model.nq * 2
    expected_nv = sim_model.nv * 2
    if viewer_model.nq < expected_nq or viewer_model.nv < expected_nv:
        raise ValueError(
            f"viewer mirror model has nq/nv={viewer_model.nq}/{viewer_model.nv}, "
            f"expected at least {expected_nq}/{expected_nv}"
        )
    root_body_id = mujoco.mj_name2id(viewer_model, mujoco.mjtObj.mjOBJ_BODY, "pelvis_link")
    if root_body_id < 0:
        raise ValueError("viewer mirror is missing original pelvis_link body")
    return ViewerMirror(
        model=viewer_model,
        data=mujoco.MjData(viewer_model),
        sim_qpos_width=sim_model.nq,
        sim_qvel_width=sim_model.nv,
        ref_qpos_start=sim_model.nq,
        ref_qvel_start=sim_model.nv,
        root_body_id=int(root_body_id),
    )


# =============================================================================
# Generic Model, Math, Policy, And Observation Helpers
# =============================================================================

def normalize_name(name: str) -> str:
    return name.strip().lower()


def load_urdf_actuated_joints(urdf_path: Path) -> tuple[list[str], dict[str, str]]:
    root = ET.parse(urdf_path).getroot()
    all_links = [elem.attrib["name"] for elem in root.findall("link")]
    child_links = set()
    parent_to_joints: dict[str, list[tuple[str, str, str]]] = defaultdict(list)
    joint_types: dict[str, str] = {}

    for joint in root.findall("joint"):
        joint_name = joint.attrib["name"]
        joint_type = joint.attrib.get("type", "")
        parent_link = joint.find("parent").attrib["link"]
        child_link = joint.find("child").attrib["link"]
        child_links.add(child_link)
        parent_to_joints[parent_link].append((joint_name, joint_type, child_link))
        joint_types[joint_name] = joint_type

    roots = [link for link in all_links if link not in child_links]
    if len(roots) != 1:
        raise ValueError(f"Expected exactly one URDF root link, got {roots}")

    actuated: list[str] = []
    queue: deque[str] = deque([roots[0]])
    while queue:
        link_name = queue.popleft()
        children = sorted(parent_to_joints.get(link_name, []), key=lambda item: item[2].lower())
        for joint_name, joint_type, child_link in children:
            if joint_type != "fixed" and joint_name not in PASSIVE_FOOT_JOINT_NAMES:
                actuated.append(joint_name)
            queue.append(child_link)

    return actuated, joint_types


def model_name(model: mujoco.MjModel, obj_type: mujoco.mjtObj, obj_id: int) -> str:
    name = mujoco.mj_id2name(model, obj_type, obj_id)
    if name is None:
        raise ValueError(f"MuJoCo object {obj_type} id={obj_id} has no name")
    return name


def get_a3_joint_control(joint_name: str) -> JointControl:
    if "_hip_roll_joint" in joint_name:
        return JointControl(120.0 * KP_SCALE, 4.0 * KD_SCALE, 220.0, ARMATURE_PFP_93_65, 0.15)
    if "_hip_yaw_joint" in joint_name:
        return JointControl(80.0 * KP_SCALE, 3.0 * KD_SCALE, 220.0, ARMATURE_PFP_93_65, 0.15)
    if "_hip_pitch_joint" in joint_name:
        return JointControl(80.0 * KP_SCALE, 3.0 * KD_SCALE, 220.0, ARMATURE_PFP_93_65, 0.15)
    if "_knee_joint" in joint_name:
        return JointControl(250.0 * KP_SCALE, 8.0 * KD_SCALE, 320.0, ARMATURE_PFP_110_75, 0.15)
    if "_ankle_pitch_joint" in joint_name:
        return JointControl(50.0 * KP_SCALE, 2.0 * KD_SCALE, 118.2, ARMATURE_PFP_78_58 * 5.333, 0.25)
    if "_ankle_roll_joint" in joint_name:
        return JointControl(50.0 * KP_SCALE, 2.0 * KD_SCALE, 54.75, ARMATURE_PFP_78_58 * 1.66562, 0.25)
    if joint_name == "waist_yaw_joint":
        return JointControl(85.0 * KP_SCALE, 3.0 * KD_SCALE, 220.0, ARMATURE_PFP_93_65, 0.15)
    if joint_name == "waist_roll_joint":
        return JointControl(50.0 * KP_SCALE, 2.0 * KD_SCALE, 46.0, ARMATURE_PFP_78_58 * 1.21, 0.15)
    if joint_name == "waist_pitch_joint":
        return JointControl(50.0 * KP_SCALE, 2.0 * KD_SCALE, 115.0, ARMATURE_PFP_78_58 * 7.3, 0.15)
    if "_shoulder_pitch_joint" in joint_name:
        return JointControl(40.0 * KP_SCALE, 3.0 * KD_SCALE, 60.0, ARMATURE_PFP_78_58, 0.1)
    if "_shoulder_roll_joint" in joint_name:
        return JointControl(40.0 * KP_SCALE, 3.0 * KD_SCALE, 60.0, ARMATURE_PFP_78_58, 0.1)
    if "_shoulder_yaw_joint" in joint_name:
        return JointControl(30.0 * KP_SCALE, 2.0 * KD_SCALE, 24.0, ARMATURE_PFP_59_60, 0.1)
    if "_elbow_joint" in joint_name:
        return JointControl(30.0 * KP_SCALE, 2.0 * KD_SCALE, 24.0, ARMATURE_PFP_59_60, 0.1)
    if "_wrist_roll_joint" in joint_name:
        return JointControl(30.0 * KP_SCALE, 2.0 * KD_SCALE, 24.0, ARMATURE_PFP_59_60, 0.1)
    if "_wrist_pitch_joint" in joint_name:
        return JointControl(20.0 * KP_SCALE, 2.0 * KD_SCALE, 6.0, ARMATURE_PFP_41_48, 0.1)
    if "_wrist_yaw_joint" in joint_name:
        return JointControl(20.0 * KP_SCALE, 2.0 * KD_SCALE, 6.0, ARMATURE_PFP_41_48, 0.1)
    raise KeyError(f"No A3 control parameters for joint {joint_name!r}")


def get_a3_action_scale(joint_name: str) -> float:
    ctrl = get_a3_joint_control(joint_name)
    if KP_SCALE == 0:
        raise ValueError("KP_SCALE must be non-zero to recover the IsaacLab stiffness")
    stiffness = ctrl.kp / KP_SCALE
    return ACTION_SCALE_COEFF * ctrl.effort / stiffness


def resolve_motion_paths(path: Path) -> list[Path]:
    path = path.expanduser()
    if path.is_file() and path.suffix.lower() == ".csv":
        return [path]
    if path.is_file():
        raise ValueError(f"Expected a .csv motion file, got {path}")
    if not path.is_dir():
        raise FileNotFoundError(path)
    files = sorted(p for p in path.iterdir() if p.is_file() and p.suffix.lower() == ".csv")
    if not files:
        raise FileNotFoundError(f"No .csv motion files in {path}")
    return files


def quat_normalize(q: np.ndarray) -> np.ndarray:
    return q / np.linalg.norm(q, axis=-1, keepdims=True).clip(min=1e-12)


def quat_conj_wxyz(q: np.ndarray) -> np.ndarray:
    out = np.array(q, dtype=np.float64, copy=True)
    out[..., 1:] *= -1.0
    return out


def quat_mul_wxyz(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aw, ax, ay, az = np.moveaxis(a, -1, 0)
    bw, bx, by, bz = np.moveaxis(b, -1, 0)
    return np.stack(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        axis=-1,
    )


def quat_rotate_inverse_wxyz(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    q = quat_normalize(np.asarray(q, dtype=np.float64))
    v_quat = np.zeros(q.shape[:-1] + (4,), dtype=np.float64)
    v_quat[..., 1:] = v
    return quat_mul_wxyz(quat_mul_wxyz(quat_conj_wxyz(q), v_quat), q)[..., 1:]


def quat_to_matrix_wxyz(q: np.ndarray) -> np.ndarray:
    q = quat_normalize(np.asarray(q, dtype=np.float64))
    w, x, y, z = np.moveaxis(q, -1, 0)
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    mat = np.empty(q.shape[:-1] + (3, 3), dtype=np.float64)
    mat[..., 0, 0] = 1.0 - 2.0 * (yy + zz)
    mat[..., 0, 1] = 2.0 * (xy - wz)
    mat[..., 0, 2] = 2.0 * (xz + wy)
    mat[..., 1, 0] = 2.0 * (xy + wz)
    mat[..., 1, 1] = 1.0 - 2.0 * (xx + zz)
    mat[..., 1, 2] = 2.0 * (yz - wx)
    mat[..., 2, 0] = 2.0 * (xz - wy)
    mat[..., 2, 1] = 2.0 * (yz + wx)
    mat[..., 2, 2] = 1.0 - 2.0 * (xx + yy)
    return mat


def root_ori_diff_6d(robot_root_quat_wxyz: np.ndarray, ref_root_quat_wxyz: np.ndarray) -> np.ndarray:
    robot = np.broadcast_to(robot_root_quat_wxyz, ref_root_quat_wxyz.shape)
    rel = quat_mul_wxyz(quat_conj_wxyz(robot), ref_root_quat_wxyz)
    mat = quat_to_matrix_wxyz(rel)
    return mat[..., :2].reshape(ref_root_quat_wxyz.shape[0], 6)


def make_mlp(layer_dims: list[int]) -> nn.Sequential:
    layers: list[nn.Module] = []
    for idx in range(len(layer_dims) - 1):
        layers.append(nn.Linear(layer_dims[idx], layer_dims[idx + 1]))
        if idx < len(layer_dims) - 2:
            layers.append(nn.SiLU())
    return nn.Sequential(*layers)


class _CheckpointDummy:
    """Placeholder for training-only objects stored next to tensor weights."""

    def __init__(self, *args, **kwargs) -> None:
        self.args = args
        self.kwargs = kwargs

    def __setstate__(self, state: object) -> None:
        if isinstance(state, dict):
            self.__dict__.update(state)
        else:
            self.state = state


class _NoTrlImportUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str):
        training_only_prefixes = ("trl.",)
        optional_training_prefixes = ("transformers.", "accelerate.")
        if module.startswith(training_only_prefixes):
            return _CheckpointDummy
        try:
            return super().find_class(module, name)
        except (AttributeError, ModuleNotFoundError):
            if module.startswith(optional_training_prefixes):
                return _CheckpointDummy
            raise


class _NoTrlImportPickleModule:
    __name__ = "pickle"
    Pickler = pickle.Pickler
    Unpickler = _NoTrlImportUnpickler
    dump = staticmethod(pickle.dump)
    dumps = staticmethod(pickle.dumps)
    load = staticmethod(pickle.load)
    loads = staticmethod(pickle.loads)


def load_torch_checkpoint_without_trl_imports(checkpoint: Path) -> dict:
    """Load tensors from the training checkpoint without importing TRL modules."""
    return torch.load(
        checkpoint,
        map_location="cpu",
        pickle_module=_NoTrlImportPickleModule,
    )


class A3Policy(nn.Module):
    def __init__(self, checkpoint: Path, device: torch.device, encoder: str = "g1"):
        super().__init__()
        payload = load_torch_checkpoint_without_trl_imports(checkpoint)
        state = payload.get("policy_state_dict", payload.get("actor_model_state_dict"))
        if state is None:
            raise KeyError("Checkpoint does not contain policy_state_dict or actor_model_state_dict")
        self.encoder_name = encoder
        enc_state, dec_state = self.split_policy_state(state, encoder)

        self.encoder_input_dim = int(enc_state["0.weight"].shape[1])
        if self.encoder_input_dim != ENCODER_INPUT_DIM:
            raise ValueError(
                f"Unsupported encoder input dim {self.encoder_input_dim}; "
                f"expected {ENCODER_INPUT_DIM} ({NUM_FUTURE_FRAMES} x {ENCODER_FRAME_DIM})"
            )
        self.encoder_frame_dim = ENCODER_FRAME_DIM
        self.encoder_terms = ENCODER_TERMS
        self.decoder_input_dim = int(dec_state["0.weight"].shape[1])
        self.actor_obs_dim = self.decoder_input_dim - 64
        if self.actor_obs_dim <= 0:
            raise ValueError(f"Decoder input dim {self.decoder_input_dim} is too small for 64 token dims")

        self.encoder = make_mlp([self.encoder_input_dim, 2048, 1024, 512, 512, 64])
        self.decoder = make_mlp([self.decoder_input_dim, 2048, 2048, 1024, 1024, 512, 512, NUM_POLICY_DOFS])
        self.register_buffer("levels", torch.full((32,), 32, dtype=torch.float32))
        self.load_policy_state(state, enc_state, dec_state)
        self.to(device)
        self.eval()

    @staticmethod
    def split_policy_state(
        state: dict[str, torch.Tensor],
        encoder: str = "g1",
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        enc_prefix = f"actor_module.encoders.{encoder}.module."
        dec_prefix = "actor_module.decoders.g1_dyn.module."
        enc_state = {key[len(enc_prefix) :]: value for key, value in state.items() if key.startswith(enc_prefix)}
        dec_state = {key[len(dec_prefix) :]: value for key, value in state.items() if key.startswith(dec_prefix)}
        if "0.weight" not in enc_state:
            raise KeyError(f"Could not find encoder first layer under prefix {enc_prefix!r}")
        if "0.weight" not in dec_state:
            raise KeyError(f"Could not find decoder first layer under prefix {dec_prefix!r}")
        return enc_state, dec_state

    def load_policy_state(
        self,
        state: dict[str, torch.Tensor],
        enc_state: dict[str, torch.Tensor],
        dec_state: dict[str, torch.Tensor],
    ) -> None:
        self.encoder.load_state_dict(enc_state, strict=True)
        self.decoder.load_state_dict(dec_state, strict=True)
        std = state.get("std", state.get("log_std"))
        if std is not None and std.shape != (NUM_POLICY_DOFS,):
            raise ValueError(f"Expected 29D policy std/log_std, got {tuple(std.shape)}")

    def fsq(self, z: torch.Tensor) -> torch.Tensor:
        # Matches vector_quantize_pytorch.FSQ(levels=[32]*32) with default settings.
        levels = self.levels.to(dtype=z.dtype, device=z.device)
        half_l = (levels - 1.0) * (1.0 + 1e-3) / 2.0
        offset = torch.where(
            (levels.to(torch.int64) % 2) == 0,
            torch.tensor(0.5, device=z.device, dtype=z.dtype),
            torch.tensor(0.0, device=z.device, dtype=z.dtype),
        )
        shift = torch.atanh(offset / half_l)
        bounded = torch.tanh(z + shift) * half_l - offset
        half_width = torch.div(levels.to(torch.int64), 2, rounding_mode="floor").to(dtype=z.dtype)
        return torch.round(bounded) / half_width

    @torch.no_grad()
    def act(self, encoder_input: np.ndarray, actor_obs: np.ndarray, device: torch.device) -> np.ndarray:
        enc = torch.as_tensor(encoder_input, dtype=torch.float32, device=device).view(
            1, NUM_FUTURE_FRAMES, self.encoder_frame_dim
        )
        prop = torch.as_tensor(actor_obs, dtype=torch.float32, device=device).view(1, self.actor_obs_dim)
        latent = self.encoder(enc.reshape(1, self.encoder_input_dim)).view(1, 2, 32)
        tokens = self.fsq(latent).reshape(1, 64)
        decoder_input = torch.cat([tokens, prop], dim=-1)
        if decoder_input.shape[-1] != self.decoder_input_dim:
            raise RuntimeError(
                f"Decoder input dim mismatch: got {decoder_input.shape[-1]}, "
                f"checkpoint expects {self.decoder_input_dim}"
            )
        action = self.decoder(decoder_input)
        if action.shape != (1, NUM_POLICY_DOFS):
            raise RuntimeError(f"Unexpected action shape {tuple(action.shape)}")
        if not torch.isfinite(action).all():
            raise RuntimeError("Policy produced non-finite action")
        return action.squeeze(0).cpu().numpy()


def get_root_quat(data: mujoco.MjData) -> np.ndarray:
    return quat_normalize(data.qpos[3:7].copy())


def get_base_ang_vel_b(model: mujoco.MjModel, data: mujoco.MjData, mapping: A3Mapping) -> np.ndarray:
    # base_ang_vel must be expressed in the ANCHOR body's local frame to match
    # training (gear_sonic base_ang_vel = quat_apply_inverse(robot_anchor_quat_w,
    # robot_anchor_ang_vel_w), anchor = cfg.anchor_body). For pelvis-anchor ckpts
    # anchor_body_id == root_body_id so this is a no-op; for torso-anchor (exp007)
    # it fixes a pelvis/torso frame mismatch that caused 100% fall.
    vel = np.zeros(6, dtype=np.float64)
    mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, mapping.anchor_body_id, vel, 1)
    return vel[:3].copy()


def prime_history(current_terms: dict[str, np.ndarray]) -> dict[str, deque[np.ndarray]]:
    return {
        name: deque([current_terms[name].copy() for _ in range(NUM_FUTURE_FRAMES)], maxlen=NUM_FUTURE_FRAMES)
        for name in OBS_TERM_ORDER
    }


def append_history(history: dict[str, deque[np.ndarray]], current_terms: dict[str, np.ndarray]) -> None:
    for name in OBS_TERM_ORDER:
        history[name].append(current_terms[name].copy())


def flatten_history(history: dict[str, deque[np.ndarray]]) -> np.ndarray:
    # IsaacLab flattens each term's CircularBuffer first, then concatenates terms.
    return np.concatenate(
        [np.concatenate(list(history[name]), axis=0) for name in OBS_TERM_ORDER],
        axis=0,
    ).astype(np.float32)


def build_encoder_input(
    reference: MotionReference,
    ref_frame: int,
    robot_anchor_quat_wxyz: np.ndarray,
    on_end: str = "hold_last",
    frame_skip: int = FUTURE_FRAME_SKIP,
    history_frames: int = 0,
    valid_future_frames: int | None = None,
    zero_pad_invalid_frames: bool = False,
) -> np.ndarray:
    command_flat, ori_6d = build_tokenizer_terms(
        reference,
        ref_frame,
        robot_anchor_quat_wxyz,
        on_end,
        frame_skip,
        history_frames,
        valid_future_frames,
        zero_pad_invalid_frames,
    )
    # Match command_multi_future(non_flatten=True) exactly:
    # command = cat([all future q positions, all future q velocities]), then reshape(10, 58).
    command_nonflat = command_flat.reshape(NUM_FUTURE_FRAMES, NUM_POLICY_DOFS * 2)
    per_frame = np.concatenate([command_nonflat, ori_6d], axis=-1)
    if per_frame.shape != (NUM_FUTURE_FRAMES, ENCODER_FRAME_DIM):
        raise RuntimeError(f"Unexpected encoder input per-frame shape {per_frame.shape}")
    return per_frame.astype(np.float32)


def reference_frame_index(reference: MotionReference, raw_frame: int, on_end: str) -> int | None:
    if raw_frame < reference.num_frames:
        return int(raw_frame)
    if on_end == "wrap":
        return int(raw_frame % reference.num_frames)
    if on_end == "hold_last":
        return reference.num_frames - 1
    if on_end == "stop":
        return None
    raise ValueError(f"Unknown reference on_end policy: {on_end}")


def future_reference_indices(
    reference: MotionReference,
    ref_frame: int,
    on_end: str,
    frame_skip: int = FUTURE_FRAME_SKIP,
    history_frames: int = 0,
    valid_future_frames: int | None = None,
) -> np.ndarray:
    slots = np.arange(NUM_FUTURE_FRAMES, dtype=np.int64) - int(history_frames)
    if valid_future_frames is not None:
        valid = int(valid_future_frames)
        if not 1 <= valid <= NUM_FUTURE_FRAMES:
            raise ValueError(
                f"valid_future_frames must be within [1, {NUM_FUTURE_FRAMES}], got {valid}"
            )
        slots = np.minimum(slots, valid - 1)
    offsets = slots * int(frame_skip)
    raw_indices = ref_frame + offsets
    if on_end == "wrap":
        return raw_indices % reference.num_frames
    if on_end in ("hold_last", "stop"):
        return np.clip(raw_indices, 0, reference.num_frames - 1)
    raise ValueError(f"Unknown reference on_end policy: {on_end}")


def build_tokenizer_terms(
    reference: MotionReference,
    ref_frame: int,
    robot_anchor_quat_wxyz: np.ndarray,
    on_end: str,
    frame_skip: int = FUTURE_FRAME_SKIP,
    history_frames: int = 0,
    valid_future_frames: int | None = None,
    zero_pad_invalid_frames: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    indices = future_reference_indices(
        reference,
        ref_frame,
        on_end,
        frame_skip,
        history_frames,
        valid_future_frames,
    )
    ori_6d = root_ori_diff_6d(robot_anchor_quat_wxyz, reference.anchor_quat_wxyz[indices])
    joint_pos = reference.dof_il[indices].copy()
    joint_vel = reference.dof_vel_il[indices].copy()
    if zero_pad_invalid_frames:
        if valid_future_frames is None:
            raise ValueError("zero_pad_invalid_frames requires valid_future_frames")
        valid = int(valid_future_frames)
        joint_pos[valid:] = 0.0
        joint_vel[valid:] = 0.0
        ori_6d[valid:] = 0.0
    command_flat = np.concatenate([joint_pos.reshape(-1), joint_vel.reshape(-1)], axis=0)
    return command_flat.astype(np.float32), ori_6d.astype(np.float32)


def target_il_to_actuator_targets(mapping: A3Mapping, target_il: np.ndarray) -> dict[str, float]:
    target_mj29 = target_il[mapping.il_to_mj29]
    return {joint_name: float(target_mj29[idx]) for idx, joint_name in enumerate(mapping.mj29_joint_names)}


def display_path(path: Path) -> str:
    return str(path.expanduser().resolve())


def sanitize_filename_part(text: str) -> str:
    chars = []
    for char in text:
        if char.isalnum() or char in ("-", "_", "."):
            chars.append(char)
        else:
            chars.append("_")
    return "".join(chars).strip("._") or "unnamed"


def default_video_path(config: SimConfig, motion_paths: list[Path]) -> Path:
    checkpoint = config.checkpoint.expanduser()
    checkpoint_stem = sanitize_filename_part(checkpoint.stem)
    if len(motion_paths) == 1:
        motion_stem = sanitize_filename_part(motion_paths[0].stem)
    else:
        motion_root = config.motion.expanduser()
        motion_stem = f"{sanitize_filename_part(motion_root.stem)}_motions{len(motion_paths)}"
    return checkpoint.parent / f"{checkpoint_stem}_{motion_stem}_sim2sim.mp4"


# =============================================================================
# Loop Sim2Sim Configuration And A3 CSV Layout
# =============================================================================

DEFAULT_LOOP_MJCF = (
    REPO_ROOT
    / "gear_sonic/data/assets/robot_description/mjcf/a3_t2d5_loop_passive_foot_twostage_fit_optimized.xml"
)
DEFAULT_SOLVER_LIB = (
    REPO_ROOT
    / "gear_sonic/data/assets/robot_description/solver/a3_loop/lib/liba3_ankle_waist_solver.a"
)
DEFAULT_SOLVER_INCLUDE = Path(
    REPO_ROOT / "gear_sonic/data/assets/robot_description/solver/a3_loop/include"
)
DEFAULT_BRIDGE_CACHE = Path(os.environ.get("A3_LOOP_BRIDGE_CACHE", "/tmp/a3_loop_solver_bridge"))
DEFAULT_MOTION = (
    REPO_ROOT / "a3_data/agibot_a3/001_walk_front_slow.csv"
)
FOOT_SPRING_JOINT_NAMES = PASSIVE_FOOT_JOINT_NAMES
DEFAULT_REFERENCE_ON_END = "hold_last"
INIT_FRAME = 0
PROGRESS_INTERVAL = 0.25
PROGRESS_BAR_WIDTH = 32
VIDEO_WIDTH = 1280
VIDEO_HEIGHT = 720
VIDEO_FPS = TARGET_FPS
VIDEO_ENCODER = "libx264"
VIDEO_PRESET = "veryfast"
VIDEO_CRF = 20
VIDEO_FRAME_STRIDE = 1
PREVIEW_VIDEO_WIDTH = 854
PREVIEW_VIDEO_HEIGHT = 480
PREVIEW_VIDEO_FPS = TARGET_FPS / 2.0
PREVIEW_VIDEO_PRESET = "ultrafast"
PREVIEW_VIDEO_CRF = 26
PREVIEW_VIDEO_FRAME_STRIDE = 2
REF_OFFSET = (0.0, 1.2, 0.0)
REF_ALPHA = 0.35
REF_TINT = 0.18
REF_HIDDEN_VISUAL_NAME_PARTS = ("rod",)
CAMERA_DISTANCE = 5.0
CAMERA_AZIMUTH = 135.0
CAMERA_ELEVATION = -18.0
# Legacy flat-CSV evaluations sampled historical 120fps source CSVs with every
# fourth row and treated the selected rows as a 30fps reference stream.  Keep
# those defaults for compatibility; native-30fps sources must explicitly pass
# --csv-source-fps 30 --csv-frame-stride 1.
DEFAULT_CSV_SOURCE_FPS = 30.0
DEFAULT_CSV_FRAME_STRIDE = 4

A3_CSV_JOINT_NAMES = (
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "head_yaw_joint",
    "head_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
)
A3_POLICY_TO_SDK_IDX = (
    0,
    1,
    2,
    5,
    6,
    7,
    8,
    9,
    10,
    11,
    12,
    13,
    14,
    15,
    16,
    17,
    18,
    19,
    20,
    21,
    22,
    23,
    24,
    25,
    26,
    27,
    28,
    29,
    30,
)

LEFT_ANKLE_PR = ("left_ankle_pitch_joint", "left_ankle_roll_joint")
RIGHT_ANKLE_PR = ("right_ankle_pitch_joint", "right_ankle_roll_joint")
LEFT_ANKLE_MOTORS = ("left_ankle_motor_up_joint", "left_ankle_motor_down_joint")
RIGHT_ANKLE_MOTORS = ("right_ankle_motor_up_joint", "right_ankle_motor_down_joint")
WAIST_PR = ("waist_pitch_joint", "waist_roll_joint")
WAIST_MOTORS = ("left_waist_motor_joint", "right_waist_motor_joint")
LOOP_PR_JOINTS = {
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_pitch_joint",
    "waist_roll_joint",
}


# =============================================================================
# A3 Loop Solver Bridge
# =============================================================================

# Python cannot call the C++ solver classes from the static .a directly, so this
# tiny C ABI shim is compiled into a cached .so. Keeping it here preserves the
# single-file sim2sim runner.
BRIDGE_CPP = r"""
#include <array>
#include "a3_ankle_waist_solver/ankle_solver.h"
#include "a3_ankle_waist_solver/waist_solver.h"

struct A3LoopSolvers {
  zy::AnkleAnalyticalSolver left_ankle;
  zy::AnkleAnalyticalSolver right_ankle;
  zy::WaistAnalyticalSolver waist;

  A3LoopSolvers() : left_ankle(0), right_ankle(1), waist() {}
};

static zy::AnkleAnalyticalSolver& ankle_solver(A3LoopSolvers* solvers, int leg) {
  return leg == 0 ? solvers->left_ankle : solvers->right_ankle;
}

extern "C" {
A3LoopSolvers* a3_loop_create() {
  return new A3LoopSolvers();
}

void a3_loop_destroy(A3LoopSolvers* solvers) {
  delete solvers;
}

void a3_ankle_ik(A3LoopSolvers* solvers, int leg, const double* in, double* out) {
  zy::ankle_pr value = {in[0], in[1]};
  auto result = ankle_solver(solvers, leg).IK(value);
  out[0] = result[0];
  out[1] = result[1];
}

void a3_ankle_fk(A3LoopSolvers* solvers, int leg, const double* in, double* out) {
  zy::ankle_j5j6 value = {in[0], in[1]};
  auto result = ankle_solver(solvers, leg).FK(value);
  out[0] = result[0];
  out[1] = result[1];
}

void a3_ankle_dik(A3LoopSolvers* solvers, int leg, const double* in, double* out) {
  zy::ankle_pr value = {in[0], in[1]};
  auto result = ankle_solver(solvers, leg).DIK(value);
  out[0] = result[0];
  out[1] = result[1];
}

void a3_ankle_dfk(A3LoopSolvers* solvers, int leg, const double* in, double* out) {
  zy::ankle_j5j6 value = {in[0], in[1]};
  auto result = ankle_solver(solvers, leg).DFK(value);
  out[0] = result[0];
  out[1] = result[1];
}

void a3_ankle_idyn(A3LoopSolvers* solvers, int leg, const double* in, double* out) {
  zy::ankle_pr value = {in[0], in[1]};
  auto result = ankle_solver(solvers, leg).IDyn(value);
  out[0] = result[0];
  out[1] = result[1];
}

void a3_ankle_fdyn(A3LoopSolvers* solvers, int leg, const double* in, double* out) {
  zy::ankle_j5j6 value = {in[0], in[1]};
  auto result = ankle_solver(solvers, leg).FDyn(value);
  out[0] = result[0];
  out[1] = result[1];
}

void a3_ankle_rl_kd(A3LoopSolvers* solvers, int leg, const double* in, double* out) {
  zy::ankle_pr value = {in[0], in[1]};
  auto result = ankle_solver(solvers, leg).RlConvertKd(value);
  out[0] = result[0];
  out[1] = result[1];
}

void a3_ankle_rl_pos(A3LoopSolvers* solvers, int leg, const double* kp, const double* err, double* out) {
  zy::ankle_pr kp_value = {kp[0], kp[1]};
  zy::ankle_pr err_value = {err[0], err[1]};
  auto result = ankle_solver(solvers, leg).RlConvertPos(kp_value, err_value);
  out[0] = result[0];
  out[1] = result[1];
}

void a3_waist_ik(A3LoopSolvers* solvers, const double* in, double* out) {
  zy::waist_pr value = {in[0], in[1]};
  auto result = solvers->waist.IK(value);
  out[0] = result[0];
  out[1] = result[1];
}

void a3_waist_fk(A3LoopSolvers* solvers, const double* in, double* out) {
  zy::waist_j1j2 value = {in[0], in[1]};
  auto result = solvers->waist.FK(value);
  out[0] = result[0];
  out[1] = result[1];
}

void a3_waist_dik(A3LoopSolvers* solvers, const double* in, double* out) {
  zy::waist_pr value = {in[0], in[1]};
  auto result = solvers->waist.DIK(value);
  out[0] = result[0];
  out[1] = result[1];
}

void a3_waist_dfk(A3LoopSolvers* solvers, const double* in, double* out) {
  zy::waist_j1j2 value = {in[0], in[1]};
  auto result = solvers->waist.DFK(value);
  out[0] = result[0];
  out[1] = result[1];
}

void a3_waist_idyn(A3LoopSolvers* solvers, const double* in, double* out) {
  zy::waist_pr value = {in[0], in[1]};
  auto result = solvers->waist.IDyn(value);
  out[0] = result[0];
  out[1] = result[1];
}

void a3_waist_fdyn(A3LoopSolvers* solvers, const double* in, double* out) {
  zy::waist_j1j2 value = {in[0], in[1]};
  auto result = solvers->waist.FDyn(value);
  out[0] = result[0];
  out[1] = result[1];
}

void a3_waist_rl_kd(A3LoopSolvers* solvers, const double* in, double* out) {
  zy::waist_pr value = {in[0], in[1]};
  auto result = solvers->waist.RlConvertKd(value);
  out[0] = result[0];
  out[1] = result[1];
}

void a3_waist_rl_pos(A3LoopSolvers* solvers, const double* kp, const double* err, double* out) {
  zy::waist_pr kp_value = {kp[0], kp[1]};
  zy::waist_pr err_value = {err[0], err[1]};
  auto result = solvers->waist.RlConvertPos(kp_value, err_value);
  out[0] = result[0];
  out[1] = result[1];
}
}
"""


@dataclass(frozen=True)
class LoopRuntime:
    mapping: A3Mapping
    motor_qpos_addr: dict[str, int]
    motor_qvel_addr: dict[str, int]


class A3LoopSolverBridge:
    def __init__(
        self,
        solver_lib: Path,
        include_dir: Path,
        bridge_cache: Path,
        rebuild: bool = False,
    ) -> None:
        self.solver_lib = solver_lib.expanduser()
        self.include_dir = include_dir.expanduser()
        self.bridge_cache = bridge_cache.expanduser()
        bridge_path = self._ensure_bridge(rebuild)
        self.lib = ctypes.CDLL(str(bridge_path))
        self._configure_ctypes()
        self.obj = self.lib.a3_loop_create()
        if not self.obj:
            raise RuntimeError("a3_loop_create returned null")

    def close(self) -> None:
        obj = getattr(self, "obj", None)
        if obj:
            self.lib.a3_loop_destroy(obj)
            self.obj = None

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def _ensure_bridge(self, rebuild: bool) -> Path:
        if not self.solver_lib.exists():
            raise FileNotFoundError(f"solver library not found: {self.solver_lib}")
        if not self.include_dir.exists():
            raise FileNotFoundError(f"solver include dir not found: {self.include_dir}")
        gxx = shutil.which("g++")
        if gxx is None:
            raise RuntimeError("g++ is required to build the ctypes bridge for the external A3 loop solver")

        digest = hashlib.sha256(
            (
                BRIDGE_CPP
                + "\n"
                + str(self.solver_lib.resolve())
                + "\n"
                + str(self.include_dir.resolve())
            ).encode("utf-8")
        ).hexdigest()[:16]
        build_dir = self.bridge_cache / digest
        cpp_path = build_dir / "a3_loop_solver_bridge.cc"
        so_path = build_dir / "liba3_loop_solver_bridge.so"
        if so_path.exists() and not rebuild:
            return so_path

        build_dir.mkdir(parents=True, exist_ok=True)
        cpp_path.write_text(BRIDGE_CPP)
        cmd = [
            gxx,
            "-O2",
            "-std=c++17",
            "-fPIC",
            "-shared",
            str(cpp_path),
            str(self.solver_lib),
            "-I",
            str(self.include_dir),
            "-o",
            str(so_path),
        ]
        if self.solver_lib.suffix == ".so":
            cmd.insert(-2, f"-Wl,-rpath,{self.solver_lib.parent}")
        subprocess.run(cmd, check=True)
        return so_path

    def _configure_ctypes(self) -> None:
        ptr = ctypes.c_void_p
        arr = ctypes.POINTER(ctypes.c_double)
        self.lib.a3_loop_create.argtypes = []
        self.lib.a3_loop_create.restype = ptr
        self.lib.a3_loop_destroy.argtypes = [ptr]
        self.lib.a3_loop_destroy.restype = None

        for name in (
            "a3_ankle_ik",
            "a3_ankle_fk",
            "a3_ankle_dik",
            "a3_ankle_dfk",
            "a3_ankle_idyn",
            "a3_ankle_fdyn",
            "a3_ankle_rl_kd",
        ):
            fn = getattr(self.lib, name)
            fn.argtypes = [ptr, ctypes.c_int, arr, arr]
            fn.restype = None
        self.lib.a3_ankle_rl_pos.argtypes = [ptr, ctypes.c_int, arr, arr, arr]
        self.lib.a3_ankle_rl_pos.restype = None

        for name in (
            "a3_waist_ik",
            "a3_waist_fk",
            "a3_waist_dik",
            "a3_waist_dfk",
            "a3_waist_idyn",
            "a3_waist_fdyn",
            "a3_waist_rl_kd",
        ):
            fn = getattr(self.lib, name)
            fn.argtypes = [ptr, arr, arr]
            fn.restype = None
        self.lib.a3_waist_rl_pos.argtypes = [ptr, arr, arr, arr]
        self.lib.a3_waist_rl_pos.restype = None

    @staticmethod
    def _arr(values: np.ndarray | list[float] | tuple[float, float]) -> ctypes.Array:
        value = np.asarray(values, dtype=np.float64)
        if value.shape != (2,):
            raise ValueError(f"expected a 2D vector, got shape {value.shape}")
        return (ctypes.c_double * 2)(float(value[0]), float(value[1]))

    @staticmethod
    def _out() -> ctypes.Array:
        return (ctypes.c_double * 2)()

    @staticmethod
    def _to_np(out: ctypes.Array) -> np.ndarray:
        return np.array([out[0], out[1]], dtype=np.float64)

    def _ankle_call(self, name: str, leg: int, values: np.ndarray | tuple[float, float]) -> np.ndarray:
        inp = self._arr(values)
        out = self._out()
        getattr(self.lib, name)(self.obj, int(leg), inp, out)
        return self._to_np(out)

    def ankle_ik(self, leg: int, pr: np.ndarray | tuple[float, float]) -> np.ndarray:
        return self._ankle_call("a3_ankle_ik", leg, pr)

    def ankle_fk(self, leg: int, motors: np.ndarray | tuple[float, float]) -> np.ndarray:
        return self._ankle_call("a3_ankle_fk", leg, motors)

    def ankle_dik(self, leg: int, pr_vel: np.ndarray | tuple[float, float]) -> np.ndarray:
        return self._ankle_call("a3_ankle_dik", leg, pr_vel)

    def ankle_dfk(self, leg: int, motor_vel: np.ndarray | tuple[float, float]) -> np.ndarray:
        return self._ankle_call("a3_ankle_dfk", leg, motor_vel)

    def ankle_idyn(self, leg: int, pr_tau: np.ndarray | tuple[float, float]) -> np.ndarray:
        return self._ankle_call("a3_ankle_idyn", leg, pr_tau)

    def ankle_rl_kd(self, leg: int, kd: np.ndarray | tuple[float, float]) -> np.ndarray:
        return self._ankle_call("a3_ankle_rl_kd", leg, kd)

    def ankle_rl_pos(
        self,
        leg: int,
        kp: np.ndarray | tuple[float, float],
        err: np.ndarray | tuple[float, float],
    ) -> np.ndarray:
        kp_arr = self._arr(kp)
        err_arr = self._arr(err)
        out = self._out()
        self.lib.a3_ankle_rl_pos(self.obj, int(leg), kp_arr, err_arr, out)
        return self._to_np(out)

    def _waist_call(self, name: str, values: np.ndarray | tuple[float, float]) -> np.ndarray:
        inp = self._arr(values)
        out = self._out()
        getattr(self.lib, name)(self.obj, inp, out)
        return self._to_np(out)

    def waist_ik(self, pr: np.ndarray | tuple[float, float]) -> np.ndarray:
        return self._waist_call("a3_waist_ik", pr)

    def waist_fk(self, motors: np.ndarray | tuple[float, float]) -> np.ndarray:
        return self._waist_call("a3_waist_fk", motors)

    def waist_dik(self, pr_vel: np.ndarray | tuple[float, float]) -> np.ndarray:
        return self._waist_call("a3_waist_dik", pr_vel)

    def waist_dfk(self, motor_vel: np.ndarray | tuple[float, float]) -> np.ndarray:
        return self._waist_call("a3_waist_dfk", motor_vel)

    def waist_idyn(self, pr_tau: np.ndarray | tuple[float, float]) -> np.ndarray:
        return self._waist_call("a3_waist_idyn", pr_tau)

    def waist_rl_kd(self, kd: np.ndarray | tuple[float, float]) -> np.ndarray:
        return self._waist_call("a3_waist_rl_kd", kd)

    def waist_rl_pos(
        self,
        kp: np.ndarray | tuple[float, float],
        err: np.ndarray | tuple[float, float],
    ) -> np.ndarray:
        kp_arr = self._arr(kp)
        err_arr = self._arr(err)
        out = self._out()
        self.lib.a3_waist_rl_pos(self.obj, kp_arr, err_arr, out)
        return self._to_np(out)


# =============================================================================
# Loop Model Mapping And Runtime Setup
# =============================================================================

def joint_id(model: mujoco.MjModel, joint_name: str) -> int:
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    if jid < 0:
        raise ValueError(f"joint not found in loop MJCF: {joint_name}")
    return int(jid)


def joint_qpos_addr(model: mujoco.MjModel, joint_name: str) -> int:
    return int(model.jnt_qposadr[joint_id(model, joint_name)])


def joint_qvel_addr(model: mujoco.MjModel, joint_name: str) -> int:
    return int(model.jnt_dofadr[joint_id(model, joint_name)])


def build_loop_runtime(model: mujoco.MjModel, urdf_path: Path, anchor_body_name: str = "pelvis_link") -> LoopRuntime:
    il_joint_names, urdf_joint_types = load_urdf_actuated_joints(urdf_path)
    if len(il_joint_names) != NUM_POLICY_DOFS:
        raise ValueError(f"expected {NUM_POLICY_DOFS} policy joints in URDF, got {len(il_joint_names)}")
    for head in HEAD_JOINTS:
        if urdf_joint_types.get(head) != "fixed":
            raise ValueError(f"expected {head} to be fixed or absent from the 29D URDF")

    full_joint_ids: list[int] = []
    full_joint_names: list[str] = []
    for jid in range(model.njnt):
        if model.jnt_type[jid] == mujoco.mjtJoint.mjJNT_FREE:
            continue
        name = model_name(model, mujoco.mjtObj.mjOBJ_JOINT, jid)
        full_joint_ids.append(int(jid))
        full_joint_names.append(name)

    il_name_set = {normalize_name(name) for name in il_joint_names}
    policy_joint_ids: list[int] = []
    mj29_joint_names: list[str] = []
    for jid, name in zip(full_joint_ids, full_joint_names, strict=True):
        if normalize_name(name) in il_name_set:
            policy_joint_ids.append(jid)
            mj29_joint_names.append(name)

    if len(mj29_joint_names) != NUM_POLICY_DOFS:
        raise ValueError(
            f"expected {NUM_POLICY_DOFS} logical policy joints in loop MJCF, "
            f"got {len(mj29_joint_names)}: {mj29_joint_names}"
        )

    il_index = {normalize_name(name): idx for idx, name in enumerate(il_joint_names)}
    mj29_index = {normalize_name(name): idx for idx, name in enumerate(mj29_joint_names)}
    il_to_mj29 = np.array([il_index[normalize_name(name)] for name in mj29_joint_names], dtype=np.int64)
    mj29_to_il = np.array([mj29_index[normalize_name(name)] for name in il_joint_names], dtype=np.int64)
    if not np.array_equal(np.arange(NUM_POLICY_DOFS), il_to_mj29[mj29_to_il]):
        raise ValueError("loop IL<->MJ29 mapping is not bijective")

    actuator_id_by_joint: dict[str, int] = {}
    for actuator_id in range(model.nu):
        name = model_name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_id)
        if normalize_name(name) in il_name_set or name in HEAD_JOINTS:
            actuator_id_by_joint[name] = int(actuator_id)
    missing_actuators = [name for name in mj29_joint_names if name not in actuator_id_by_joint]
    if missing_actuators:
        raise ValueError(f"logical policy actuators missing from loop MJCF: {missing_actuators}")

    qpos_addr_by_joint: dict[str, int] = {}
    qvel_addr_by_joint: dict[str, int] = {}
    body_id_by_joint: dict[str, int] = {}
    axis_by_joint: dict[str, np.ndarray] = {}
    for jid, name in zip(policy_joint_ids, mj29_joint_names, strict=True):
        qpos_addr_by_joint[name] = int(model.jnt_qposadr[jid])
        qvel_addr_by_joint[name] = int(model.jnt_dofadr[jid])
        body_id_by_joint[name] = int(model.jnt_bodyid[jid])
        axis_by_joint[name] = model.jnt_axis[jid].copy()

    for head in HEAD_JOINTS:
        if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, head) >= 0:
            qpos_addr_by_joint[head] = joint_qpos_addr(model, head)
            qvel_addr_by_joint[head] = joint_qvel_addr(model, head)

    default_q_il = np.array([DEFAULT_Q_BY_NAME[name] for name in il_joint_names], dtype=np.float64)
    default_q_mj29 = default_q_il[il_to_mj29]
    action_scale_il = np.array([get_a3_action_scale(name) for name in il_joint_names], dtype=np.float64)
    action_scale_mj29 = action_scale_il[il_to_mj29]
    root_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis_link")
    if root_body_id < 0:
        raise ValueError("expected root body 'pelvis_link' in loop MJCF")
    anchor_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, anchor_body_name)
    if anchor_body_id < 0:
        raise ValueError(f"anchor body '{anchor_body_name}' not found in loop MJCF")

    mapping = A3Mapping(
        il_joint_names=il_joint_names,
        mj_full_joint_names=mj29_joint_names,
        mj29_joint_names=mj29_joint_names,
        head_joint_names=[head for head in HEAD_JOINTS if head in qpos_addr_by_joint],
        full_joint_ids=policy_joint_ids,
        mj29_full_indices=np.arange(NUM_POLICY_DOFS, dtype=np.int64),
        il_to_mj29=il_to_mj29,
        mj29_to_il=mj29_to_il,
        default_q_il=default_q_il,
        default_q_mj29=default_q_mj29,
        action_scale_il=action_scale_il,
        action_scale_mj29=action_scale_mj29,
        actuator_id_by_joint=actuator_id_by_joint,
        qpos_addr_by_joint=qpos_addr_by_joint,
        qvel_addr_by_joint=qvel_addr_by_joint,
        body_id_by_joint=body_id_by_joint,
        axis_by_joint=axis_by_joint,
        root_body_id=int(root_body_id),
        anchor_body_id=int(anchor_body_id),
        anchor_body_name=str(anchor_body_name),
    )

    motor_names = list(LEFT_ANKLE_MOTORS + RIGHT_ANKLE_MOTORS + WAIST_MOTORS)
    motor_qpos_addr = {name: joint_qpos_addr(model, name) for name in motor_names}
    motor_qvel_addr = {name: joint_qvel_addr(model, name) for name in motor_names}
    return LoopRuntime(mapping=mapping, motor_qpos_addr=motor_qpos_addr, motor_qvel_addr=motor_qvel_addr)


def controls_for_loop(mapping: A3Mapping) -> dict[str, JointControl]:
    return {name: get_a3_joint_control(name) for name in mapping.mj29_joint_names}


# Encoder-window presets for SONIC sim2sim. Each maps a --encoder-mode token to the
# (encoder name, frame skip, history depth, valid future frames) tuple the policy
# was trained with. ``None`` means all 10 slots contain distinct real samples.
# Mapping:
#   g1              -> g1 encoder, 0.1s spacing (skip 5 @ 50fps), all-future (hist 0)
#   a3_fast         -> a3_fast encoder, 0.02s spacing (skip 1), all-future (hist 0)
#   a3_fast_100ms   -> a3_fast encoder, first 5 samples then repeat the fifth
#   a3_fast_history -> a3_fast encoder, 0.02s spacing (skip 1), 5 history + 5 future
# Explicit window flags always override the matching preset field.
ENCODER_MODE_PRESETS: dict[str, tuple[str, int, int, int | None, bool]] = {
    "g1": ("g1", 5, 0, None, False),
    "a3_fast": ("a3_fast", 1, 0, None, False),
    "a3_fast_100ms": ("a3_fast", 1, 0, 5, False),
    "a3_fast_100ms_zero": ("a3_fast", 1, 0, 5, True),
    "a3_fast_history": ("a3_fast", 1, 5, None, False),
}


def _resolve_encoder_window(args: argparse.Namespace) -> tuple[str, int, int, int | None, bool]:
    """Resolve encoder and window parameters from explicit flags + preset.

    ``--encoder-mode`` seeds all four values. Explicit ``--encoder`` /
    ``--future-frame-skip`` / ``--future-history-frames`` /
    ``--future-valid-frames`` (detected via sys.argv, since argparse fills
    defaults regardless) override the matching preset field.
    Without a preset the explicit flags (or their argparse defaults: g1 / 5 / 0)
    are used, preserving legacy behavior.
    """
    import sys as _sys

    preset = getattr(args, "encoder_mode", None)
    enc_p, skip_p, hist_p, valid_p, zero_p = ENCODER_MODE_PRESETS.get(
        preset or "g1", ENCODER_MODE_PRESETS["g1"]
    )
    explicit_enc = "--encoder" in _sys.argv
    explicit_skip = "--future-frame-skip" in _sys.argv
    explicit_hist = "--future-history-frames" in _sys.argv
    explicit_valid = "--future-valid-frames" in _sys.argv
    explicit_zero = "--future-zero-pad" in _sys.argv

    if preset is not None:
        encoder = enc_p
        skip = skip_p
        hist = hist_p
        valid = valid_p
        zero_pad = zero_p
    else:
        encoder = str(getattr(args, "encoder", "g1"))
        skip = int(getattr(args, "future_frame_skip", FUTURE_FRAME_SKIP))
        hist = int(getattr(args, "future_history_frames", 0))
        valid = getattr(args, "future_valid_frames", None)
        zero_pad = bool(getattr(args, "future_zero_pad", False))

    if explicit_enc:
        encoder = str(args.encoder)
    if explicit_skip:
        skip = int(args.future_frame_skip)
    if explicit_hist:
        hist = int(args.future_history_frames)
    if explicit_valid:
        valid = args.future_valid_frames
    if explicit_zero:
        zero_pad = bool(args.future_zero_pad)

    if valid is not None and not 1 <= int(valid) <= NUM_FUTURE_FRAMES:
        raise SystemExit(
            f"--future-valid-frames must be within [1, {NUM_FUTURE_FRAMES}]"
        )
    valid = None if valid is None else int(valid)
    if zero_pad and valid is None:
        raise SystemExit("--future-zero-pad requires --future-valid-frames or a matching preset")
    return encoder, int(skip), max(0, min(NUM_FUTURE_FRAMES, int(hist))), valid, zero_pad


def _resolve_encoder_arg(args: argparse.Namespace) -> str:
    return _resolve_encoder_window(args)[0]


def _resolve_frame_skip_arg(args: argparse.Namespace) -> int:
    return _resolve_encoder_window(args)[1]


def _resolve_history_frames_arg(args: argparse.Namespace) -> int:
    return _resolve_encoder_window(args)[2]


def _resolve_valid_frames_arg(args: argparse.Namespace) -> int | None:
    return _resolve_encoder_window(args)[3]


def _resolve_zero_pad_arg(args: argparse.Namespace) -> bool:
    return _resolve_encoder_window(args)[4]


def build_sim_config(args: argparse.Namespace) -> SimConfig:
    urdf = (DEFAULT_URDF if args.urdf is None else Path(args.urdf)).expanduser()
    source_mjcf = (
        DEFAULT_LOOP_MJCF if args.mjcf is None else Path(args.mjcf)
    ).expanduser()
    if bool(getattr(args, "preview_video", False)):
        default_video_width = PREVIEW_VIDEO_WIDTH
        default_video_height = PREVIEW_VIDEO_HEIGHT
        default_video_fps = PREVIEW_VIDEO_FPS
        default_video_preset = PREVIEW_VIDEO_PRESET
        default_video_crf = PREVIEW_VIDEO_CRF
        default_video_frame_stride = PREVIEW_VIDEO_FRAME_STRIDE
    else:
        default_video_width = VIDEO_WIDTH
        default_video_height = VIDEO_HEIGHT
        default_video_fps = VIDEO_FPS
        default_video_preset = VIDEO_PRESET
        default_video_crf = VIDEO_CRF
        default_video_frame_stride = VIDEO_FRAME_STRIDE
    video_config = VideoOutputConfig(
        width=int(default_video_width if args.video_width is None else args.video_width),
        height=int(default_video_height if args.video_height is None else args.video_height),
        fps=float(default_video_fps if args.video_fps is None else args.video_fps),
        encoder=str(args.video_encoder),
        preset=str(default_video_preset if args.video_preset is None else args.video_preset),
        crf=int(default_video_crf if args.video_crf is None else args.video_crf),
        frame_stride=int(
            default_video_frame_stride
            if args.video_frame_stride is None
            else args.video_frame_stride
        ),
    )
    if video_config.width <= 0 or video_config.height <= 0:
        raise SystemExit("--video-width and --video-height must be positive")
    if video_config.fps <= 0.0:
        raise SystemExit("--video-fps must be positive")
    if video_config.frame_stride < 1:
        raise SystemExit("--video-frame-stride must be >= 1")
    if not 0 <= video_config.crf <= 51:
        raise SystemExit("--video-crf must be in [0, 51]")
    config = SimConfig(
        checkpoint=Path(args.checkpoint).expanduser(),
        motion=(DEFAULT_MOTION if args.motion is None else Path(args.motion)).expanduser(),
        urdf=urdf,
        mjcf=source_mjcf,
        output_video=args.output_video,
        metrics_out=None if args.metrics_out is None else Path(args.metrics_out).expanduser(),
        timeseries_out=None if args.timeseries_out is None else Path(args.timeseries_out).expanduser(),
        max_policy_steps=(
            None if args.max_policy_steps is None else int(args.max_policy_steps)
        ),
        action_delay_ms=float(args.action_delay_ms),
        replay_policy_dump=(
            None
            if args.replay_policy_dump is None
            else Path(args.replay_policy_dump).expanduser()
        ),
        reset_frame=int(args.reset_frame),
        replay_horizon=(
            None if args.replay_horizon is None else int(args.replay_horizon)
        ),
        replay_command=str(args.replay_command),
        video=video_config,
        encoder=_resolve_encoder_arg(args),
        future_frame_skip=_resolve_frame_skip_arg(args),
        future_history_frames=_resolve_history_frames_arg(args),
        future_valid_frames=_resolve_valid_frames_arg(args),
        future_zero_pad=_resolve_zero_pad_arg(args),
        start_paused=bool(args.start_paused),
        batch_once=bool(args.batch_once),
        random_order=bool(args.random),
        cache_size=max(1, int(args.cache_size)),
        anchor_body=str(args.anchor_body),
        csv_source_fps=float(args.csv_source_fps),
        csv_frame_stride=int(args.csv_frame_stride),
    )
    if config.max_policy_steps is not None and config.max_policy_steps <= 0:
        raise SystemExit("--max-policy-steps must be positive")
    if config.action_delay_ms < 0.0:
        raise SystemExit("--action-delay-ms must be non-negative")
    if config.csv_source_fps <= 0.0:
        raise SystemExit("--csv-source-fps must be positive")
    if config.csv_frame_stride < 1:
        raise SystemExit("--csv-frame-stride must be >= 1")
    if config.replay_horizon is not None and config.replay_horizon <= 0:
        raise SystemExit("--replay-horizon must be positive")
    if config.reset_frame < 0:
        raise SystemExit("--reset-frame must be >= 0")
    if config.replay_policy_dump is not None and not config.replay_policy_dump.exists():
        raise SystemExit(f"policy dump NPZ not found: {config.replay_policy_dump}")
    if config.replay_policy_dump is None and not config.checkpoint.exists():
        raise SystemExit(f"checkpoint not found: {config.checkpoint}")
    if config.replay_policy_dump is None and not config.motion.exists():
        raise SystemExit(f"motion CSV path not found: {config.motion}")
    if not config.urdf.exists():
        raise SystemExit(f"URDF not found: {config.urdf}")
    if not config.mjcf.exists():
        raise SystemExit(f"loop MJCF not found: {config.mjcf}")

    terrain_profile_arg = getattr(args, "terrain_profile", None)
    terrain_profile = (
        None if terrain_profile_arg is None else Path(terrain_profile_arg).expanduser()
    )
    if terrain_profile is not None:
        if not terrain_profile.exists():
            raise SystemExit(f"terrain profile not found: {terrain_profile}")
        try:
            terrain_mjcf = patch_mjcf_with_terrain_profile(config.mjcf, terrain_profile)
        except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as exc:
            raise SystemExit(f"failed to build terrain MJCF: {exc}") from exc
        config = replace(config, mjcf=terrain_mjcf)

    foot_spring_cfg_arg = getattr(args, "foot_spring_cfg", None)
    foot_spring_cfg = (
        None if foot_spring_cfg_arg is None else Path(foot_spring_cfg_arg)
    )
    stiffness_scale = float(getattr(args, "foot_stiffness_scale", 1.0))
    damping_scale = float(getattr(args, "foot_damping_scale", 1.0))
    use_foot_spring_patch = (
        foot_spring_cfg is not None
        or stiffness_scale != 1.0
        or damping_scale != 1.0
    )
    if stiffness_scale <= 0.0:
        raise SystemExit("--foot-stiffness-scale must be positive")
    if damping_scale < 0.0:
        raise SystemExit("--foot-damping-scale must be non-negative")
    if use_foot_spring_patch:
        if foot_spring_cfg is not None and not foot_spring_cfg.expanduser().exists():
            raise SystemExit(f"foot spring cfg not found: {foot_spring_cfg}")
        try:
            mjcf = build_foot_spring_mjcf(
                config.mjcf,
                foot_spring_cfg,
                stiffness_scale,
                damping_scale,
            )
        except (RuntimeError, ValueError) as exc:
            raise SystemExit(f"failed to build foot-spring MJCF: {exc}") from exc
        config = replace(config, mjcf=mjcf)

    print("[sim2sim-config] using standalone loop sim2sim defaults")
    if config.replay_policy_dump is None:
        print(f"[sim2sim-config] checkpoint: {display_path(config.checkpoint)}")
        print(f"[sim2sim-config] motion CSV: {display_path(config.motion)}")
    else:
        print(f"[sim2sim-config] replay policy dump: {display_path(config.replay_policy_dump)}")
        print(
            "[sim2sim-config] replay: "
            f"reset_frame={config.reset_frame}, "
            f"horizon={'all' if config.replay_horizon is None else config.replay_horizon}, "
            f"command={config.replay_command}"
        )
    if config.random_order:
        print("[sim2sim-config] motion order: random")
    if config.batch_once:
        print("[sim2sim-config] batch mode: run each CSV once without the passive viewer")
    if config.max_policy_steps is not None:
        print(f"[sim2sim-config] max policy steps per clip: {config.max_policy_steps}")
    print(
        "[sim2sim-config] CSV timing: "
        f"source_fps={config.csv_source_fps:.6g}, frame_stride={config.csv_frame_stride}, "
        f"target_fps={TARGET_FPS:.6g}"
    )
    if config.metrics_out is not None:
        print(f"[sim2sim-config] metrics JSON: {display_path(config.metrics_out)}")
    if config.timeseries_out is not None:
        print(f"[sim2sim-config] timeseries JSON: {display_path(config.timeseries_out)}")
    if terrain_profile is not None:
        print(f"[sim2sim-config] terrain profile: {display_path(terrain_profile)}")
        print(f"[sim2sim-config] terrain MJCF: {display_path(config.mjcf)}")
    if use_foot_spring_patch:
        print(f"[sim2sim-config] source loop MJCF: {display_path(source_mjcf)}")
        if foot_spring_cfg is not None:
            print(f"[sim2sim-config] foot spring cfg: {display_path(foot_spring_cfg)}")
        print(
            "[sim2sim-config] foot spring scales: "
            f"stiffness={stiffness_scale:.4g}, damping={damping_scale:.4g}"
        )
        print(f"[sim2sim-config] patched loop MJCF: {display_path(config.mjcf)}")
    else:
        print(f"[sim2sim-config] loop MJCF: {display_path(config.mjcf)}")
    print(
        "[sim2sim-config] policy timing: "
        f"target_fps={TARGET_FPS:.1f}, encoder={config.encoder}, "
        f"future_frame_skip={config.future_frame_skip}, "
        f"future_history_frames={config.future_history_frames}, "
        f"future_valid_frames={config.future_valid_frames}, "
        f"on_end={DEFAULT_REFERENCE_ON_END}"
    )
    return config


def resolve_loop_output_paths(config: SimConfig, motion_paths: list[Path]) -> SimConfig:
    if config.output_video == AUTO_OUTPUT_VIDEO:
        return replace(config, output_video=default_video_path(config, motion_paths))
    return config


# =============================================================================
# CSV Motion Loading
# =============================================================================

def csv_reference_duration_seconds(row_count: int, source_fps: float) -> float:
    """Return the selected-reference duration convention used by sim2sim."""
    if row_count < 0:
        raise ValueError("CSV row count must be non-negative")
    if source_fps <= 0.0:
        raise ValueError("CSV source fps must be positive")
    return float(row_count) / float(source_fps)


def resolve_a3_csv_joint_columns(fieldnames: list[str], csv_path: Path) -> dict[str, str]:
    """Resolve each logical joint to either its bare or ``_dof`` CSV column."""
    fields = set(fieldnames)
    missing: list[str] = []
    ambiguous: list[tuple[str, str]] = []
    resolved: dict[str, str] = {}
    for joint_name in A3_CSV_JOINT_NAMES:
        bare = joint_name in fields
        dof_alias = f"{joint_name}_dof" in fields
        if bare and dof_alias:
            ambiguous.append((joint_name, f"{joint_name}_dof"))
        elif bare:
            resolved[joint_name] = joint_name
        elif dof_alias:
            resolved[joint_name] = f"{joint_name}_dof"
        else:
            missing.append(f"{joint_name} or {joint_name}_dof")
    if ambiguous:
        pairs = ", ".join(f"{bare} / {alias}" for bare, alias in ambiguous)
        raise ValueError(
            f"{csv_path} has ambiguous A3 joint columns ({pairs}); "
            "keep exactly one spelling for each joint"
        )
    if missing:
        raise ValueError(
            f"{csv_path} is missing A3 joint columns; expected one of each: {', '.join(missing)}"
        )
    return resolved


def load_a3_flat_csv(
    csv_path: Path,
    *,
    source_fps: float = DEFAULT_CSV_SOURCE_FPS,
    frame_stride: int = DEFAULT_CSV_FRAME_STRIDE,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Load a flat A3 CSV and return selected reference rows plus raw row count."""
    if source_fps <= 0.0:
        raise ValueError("CSV source fps must be positive")
    if frame_stride < 1:
        raise ValueError("CSV frame stride must be >= 1")
    rows: list[dict[str, str]] = []
    raw_row_count = 0
    with csv_path.open(newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"empty CSV: {csv_path}")
        required_root = (
            "root_translateX",
            "root_translateY",
            "root_translateZ",
            "root_rotateX",
            "root_rotateY",
            "root_rotateZ",
        )
        missing_root = [name for name in required_root if name not in reader.fieldnames]
        if missing_root:
            raise ValueError(f"{csv_path} is missing root CSV columns: {missing_root}")
        joint_columns = resolve_a3_csv_joint_columns(reader.fieldnames, csv_path)
        for raw_idx, row in enumerate(reader):
            raw_row_count += 1
            if raw_idx % frame_stride == 0:
                rows.append(row)
    if not rows:
        raise ValueError(f"CSV has no data rows after stride {frame_stride}: {csv_path}")

    root_pos = np.array(
        [[float(row["root_translateX"]), float(row["root_translateY"]), float(row["root_translateZ"])] for row in rows],
        dtype=np.float64,
    ) * 0.01
    euler_deg = np.array(
        [[float(row["root_rotateX"]), float(row["root_rotateY"]), float(row["root_rotateZ"])] for row in rows],
        dtype=np.float64,
    )
    root_quat_xyzw = Rotation.from_euler("xyz", euler_deg, degrees=True).as_quat()
    root_quat_wxyz = root_quat_xyzw[:, [3, 0, 1, 2]]
    dof_mj29 = np.array(
        [
            [float(row[joint_columns[A3_CSV_JOINT_NAMES[sdk_idx]]]) for sdk_idx in A3_POLICY_TO_SDK_IDX]
            for row in rows
        ],
        dtype=np.float64,
    )
    dof_mj29 = np.deg2rad(dof_mj29)
    return root_pos, root_quat_wxyz, dof_mj29, raw_row_count


def resample_csv_motion(
    root_pos_src: np.ndarray,
    root_quat_wxyz_src: np.ndarray,
    dof_mj29_src: np.ndarray,
    *,
    source_fps: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if root_pos_src.shape[0] != root_quat_wxyz_src.shape[0] or root_pos_src.shape[0] != dof_mj29_src.shape[0]:
        raise ValueError("CSV root position, root orientation, and joint data frame counts differ")
    if root_pos_src.shape[0] < 2:
        return root_pos_src.copy(), root_quat_wxyz_src.copy(), dof_mj29_src.copy()

    if source_fps <= 0.0:
        raise ValueError("CSV source fps must be positive")
    source_times = np.arange(root_pos_src.shape[0], dtype=np.float64) / source_fps
    duration = source_times[-1]
    target_times = np.arange(0.0, max(duration - 1e-12, 0.0), 1.0 / TARGET_FPS)
    if target_times.size == 0:
        target_times = np.array([0.0], dtype=np.float64)

    root_pos = np.empty((target_times.size, 3), dtype=np.float64)
    for dim in range(3):
        root_pos[:, dim] = np.interp(target_times, source_times, root_pos_src[:, dim])

    root_quat_xyzw_src = root_quat_wxyz_src[:, [1, 2, 3, 0]]
    root_quat_xyzw = Slerp(source_times, Rotation.from_quat(root_quat_xyzw_src))(target_times).as_quat()
    root_quat_wxyz = root_quat_xyzw[:, [3, 0, 1, 2]]

    dof_mj29 = np.empty((target_times.size, NUM_POLICY_DOFS), dtype=np.float64)
    for idx in range(NUM_POLICY_DOFS):
        dof_mj29[:, idx] = np.interp(target_times, source_times, dof_mj29_src[:, idx])
    return root_pos, root_quat_wxyz, dof_mj29


def loop_key_qpos(model: mujoco.MjModel) -> np.ndarray:
    if model.nkey > 0:
        return model.key_qpos[0].copy()
    return model.qpos0.copy()


def set_loop_pair_qpos(
    qpos: np.ndarray,
    runtime: LoopRuntime,
    pair: tuple[str, str],
    values: np.ndarray,
) -> None:
    mapping = runtime.mapping
    qpos[mapping.qpos_addr_by_joint[pair[0]]] = float(values[0])
    qpos[mapping.qpos_addr_by_joint[pair[1]]] = float(values[1])


def fill_loop_qpos_motors(
    qpos: np.ndarray,
    runtime: LoopRuntime,
    solver: A3LoopSolverBridge,
    dof_mj29: np.ndarray,
) -> None:
    mapping = runtime.mapping
    by_name = {name: float(dof_mj29[idx]) for idx, name in enumerate(mapping.mj29_joint_names)}
    for idx, name in enumerate(mapping.mj29_joint_names):
        qpos[mapping.qpos_addr_by_joint[name]] = float(dof_mj29[idx])

    left_pr = np.array([by_name[LEFT_ANKLE_PR[0]], by_name[LEFT_ANKLE_PR[1]]], dtype=np.float64)
    right_pr = np.array([by_name[RIGHT_ANKLE_PR[0]], by_name[RIGHT_ANKLE_PR[1]]], dtype=np.float64)
    waist_pr = np.array([by_name["waist_pitch_joint"], by_name["waist_roll_joint"]], dtype=np.float64)
    left_motor = solver.ankle_ik(0, left_pr)
    right_motor = solver.ankle_ik(1, right_pr)
    waist_motor = solver.waist_ik(waist_pr)

    qpos[runtime.motor_qpos_addr[LEFT_ANKLE_MOTORS[0]]] = left_motor[0]
    qpos[runtime.motor_qpos_addr[LEFT_ANKLE_MOTORS[1]]] = left_motor[1]
    qpos[runtime.motor_qpos_addr[RIGHT_ANKLE_MOTORS[0]]] = right_motor[0]
    qpos[runtime.motor_qpos_addr[RIGHT_ANKLE_MOTORS[1]]] = right_motor[1]
    qpos[runtime.motor_qpos_addr[WAIST_MOTORS[0]]] = waist_motor[0]
    qpos[runtime.motor_qpos_addr[WAIST_MOTORS[1]]] = waist_motor[1]


def load_loop_motion_reference(
    model: mujoco.MjModel,
    runtime: LoopRuntime,
    solver: A3LoopSolverBridge,
    motion_path: Path,
    *,
    csv_source_fps: float = DEFAULT_CSV_SOURCE_FPS,
    csv_frame_stride: int = DEFAULT_CSV_FRAME_STRIDE,
    announce: bool = True,
) -> MotionReference:
    resolved_path = motion_path.expanduser()
    if not resolved_path.is_file():
        raise FileNotFoundError(resolved_path)
    root_pos_src, root_quat_wxyz_src, dof_mj29_src, raw_row_count = load_a3_flat_csv(
        resolved_path,
        source_fps=csv_source_fps,
        frame_stride=csv_frame_stride,
    )
    root_pos, root_quat_wxyz, dof_mj29 = resample_csv_motion(
        root_pos_src,
        root_quat_wxyz_src,
        dof_mj29_src,
        source_fps=csv_source_fps,
    )

    mapping = runtime.mapping
    dof_full = np.zeros((dof_mj29.shape[0], len(mapping.mj_full_joint_names)), dtype=np.float64)
    dof_full[:, mapping.mj29_full_indices] = dof_mj29
    dof_il = dof_mj29[:, mapping.mj29_to_il]
    dof_vel_il = np.zeros_like(dof_il)
    if dof_il.shape[0] >= 2:
        dof_vel_il[:-1] = (dof_il[1:] - dof_il[:-1]) * TARGET_FPS
        dof_vel_il[-1] = dof_vel_il[-2]

    key_qpos = loop_key_qpos(model)
    qpos = np.tile(key_qpos, (root_pos.shape[0], 1))
    qpos[:, :3] = root_pos
    qpos[:, 3:7] = root_quat_wxyz
    for frame in range(qpos.shape[0]):
        fill_loop_qpos_motors(qpos[frame], runtime, solver, dof_mj29[frame])
        for head in mapping.head_joint_names:
            qpos[frame, mapping.qpos_addr_by_joint[head]] = 0.0

    qvel = np.zeros((qpos.shape[0], model.nv), dtype=np.float64)
    if qpos.shape[0] >= 2:
        for frame in range(qpos.shape[0] - 1):
            mujoco.mj_differentiatePos(model, qvel[frame], POLICY_DT, qpos[frame], qpos[frame + 1])
        qvel[-1] = qvel[-2]

    # Compute per-frame anchor body world quaternion by forward-dynamics the reference
    # qpos. When anchor == pelvis this is identical to root_quat_wxyz; for torso_Link
    # (exp007) it captures the torso orientation that training used for gravity_dir
    # and encoder ori_6d.
    ref_data = mujoco.MjData(model)
    anchor_quat_wxyz = np.zeros((qpos.shape[0], 4), dtype=np.float64)
    for frame in range(qpos.shape[0]):
        ref_data.qpos[:] = qpos[frame]
        ref_data.qvel[:] = qvel[frame]
        mujoco.mj_forward(model, ref_data)
        anchor_quat_wxyz[frame] = body_quat_wxyz(ref_data, mapping.anchor_body_id)

    if announce:
        print(
            f"[motion] {resolved_path.name}: raw_rows={raw_row_count} "
            f"strided_rows={root_pos_src.shape[0]} frames={qpos.shape[0]} "
            f"csv_stride={csv_frame_stride} source_fps={csv_source_fps:.6g} "
            f"reference_seconds={csv_reference_duration_seconds(root_pos_src.shape[0], csv_source_fps):.6f} "
            f"expected_policy_steps={csv_reference_duration_seconds(root_pos_src.shape[0], csv_source_fps) * TARGET_FPS:.3f} "
            f"target_fps={TARGET_FPS:.2f}"
        )
    return MotionReference(
        path=resolved_path,
        fps=TARGET_FPS,
        qpos=qpos,
        qvel=qvel,
        dof_full=dof_full,
        dof_mj29=dof_mj29,
        dof_il=dof_il,
        dof_vel_il=dof_vel_il,
        root_pos=root_pos,
        root_quat_wxyz=root_quat_wxyz,
        anchor_quat_wxyz=anchor_quat_wxyz,
    )


# =============================================================================
# Loop State, Observation, And Control
# =============================================================================

def loop_state_overrides(
    data: mujoco.MjData,
    runtime: LoopRuntime,
    solver: A3LoopSolverBridge,
) -> tuple[dict[str, float], dict[str, float]]:
    motor_pos = runtime.motor_qpos_addr
    motor_vel = runtime.motor_qvel_addr

    left_motor_q = np.array(
        [data.qpos[motor_pos[LEFT_ANKLE_MOTORS[0]]], data.qpos[motor_pos[LEFT_ANKLE_MOTORS[1]]]],
        dtype=np.float64,
    )
    right_motor_q = np.array(
        [data.qpos[motor_pos[RIGHT_ANKLE_MOTORS[0]]], data.qpos[motor_pos[RIGHT_ANKLE_MOTORS[1]]]],
        dtype=np.float64,
    )
    waist_motor_q = np.array(
        [data.qpos[motor_pos[WAIST_MOTORS[0]]], data.qpos[motor_pos[WAIST_MOTORS[1]]]],
        dtype=np.float64,
    )
    left_motor_dq = np.array(
        [data.qvel[motor_vel[LEFT_ANKLE_MOTORS[0]]], data.qvel[motor_vel[LEFT_ANKLE_MOTORS[1]]]],
        dtype=np.float64,
    )
    right_motor_dq = np.array(
        [data.qvel[motor_vel[RIGHT_ANKLE_MOTORS[0]]], data.qvel[motor_vel[RIGHT_ANKLE_MOTORS[1]]]],
        dtype=np.float64,
    )
    waist_motor_dq = np.array(
        [data.qvel[motor_vel[WAIST_MOTORS[0]]], data.qvel[motor_vel[WAIST_MOTORS[1]]]],
        dtype=np.float64,
    )

    left_pr_q = solver.ankle_fk(0, left_motor_q)
    right_pr_q = solver.ankle_fk(1, right_motor_q)
    waist_pr_q = solver.waist_fk(waist_motor_q)
    left_pr_dq = solver.ankle_dfk(0, left_motor_dq)
    right_pr_dq = solver.ankle_dfk(1, right_motor_dq)
    waist_pr_dq = solver.waist_dfk(waist_motor_dq)

    q = {
        LEFT_ANKLE_PR[0]: float(left_pr_q[0]),
        LEFT_ANKLE_PR[1]: float(left_pr_q[1]),
        RIGHT_ANKLE_PR[0]: float(right_pr_q[0]),
        RIGHT_ANKLE_PR[1]: float(right_pr_q[1]),
        "waist_pitch_joint": float(waist_pr_q[0]),
        "waist_roll_joint": float(waist_pr_q[1]),
    }
    dq = {
        LEFT_ANKLE_PR[0]: float(left_pr_dq[0]),
        LEFT_ANKLE_PR[1]: float(left_pr_dq[1]),
        RIGHT_ANKLE_PR[0]: float(right_pr_dq[0]),
        RIGHT_ANKLE_PR[1]: float(right_pr_dq[1]),
        "waist_pitch_joint": float(waist_pr_dq[0]),
        "waist_roll_joint": float(waist_pr_dq[1]),
    }
    return q, dq


def get_loop_joint_q_mj29(
    data: mujoco.MjData,
    runtime: LoopRuntime,
    solver: A3LoopSolverBridge,
) -> np.ndarray:
    mapping = runtime.mapping
    overrides, _ = loop_state_overrides(data, runtime, solver)
    values = []
    for name in mapping.mj29_joint_names:
        if name in overrides:
            values.append(overrides[name])
        else:
            values.append(float(data.qpos[mapping.qpos_addr_by_joint[name]]))
    return np.asarray(values, dtype=np.float64)


def get_loop_joint_dq_mj29(
    data: mujoco.MjData,
    runtime: LoopRuntime,
    solver: A3LoopSolverBridge,
) -> np.ndarray:
    mapping = runtime.mapping
    _, overrides = loop_state_overrides(data, runtime, solver)
    values = []
    for name in mapping.mj29_joint_names:
        if name in overrides:
            values.append(overrides[name])
        else:
            values.append(float(data.qvel[mapping.qvel_addr_by_joint[name]]))
    return np.asarray(values, dtype=np.float64)


def get_loop_joint_q_il(data: mujoco.MjData, runtime: LoopRuntime, solver: A3LoopSolverBridge) -> np.ndarray:
    return get_loop_joint_q_mj29(data, runtime, solver)[runtime.mapping.mj29_to_il]


def get_loop_joint_dq_il(data: mujoco.MjData, runtime: LoopRuntime, solver: A3LoopSolverBridge) -> np.ndarray:
    return get_loop_joint_dq_mj29(data, runtime, solver)[runtime.mapping.mj29_to_il]


def build_loop_current_obs_terms(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    runtime: LoopRuntime,
    solver: A3LoopSolverBridge,
    last_action_il: np.ndarray,
) -> dict[str, np.ndarray]:
    mapping = runtime.mapping
    base_ang_vel = get_base_ang_vel_b(model, data, mapping)
    joint_pos_rel = get_loop_joint_q_il(data, runtime, solver) - mapping.default_q_il
    joint_vel_rel = get_loop_joint_dq_il(data, runtime, solver)
    anchor_quat = body_quat_wxyz(data, mapping.anchor_body_id)
    gravity_dir = quat_rotate_inverse_wxyz(anchor_quat, np.array([0.0, 0.0, -1.0], dtype=np.float64))
    terms = {
        "base_ang_vel": base_ang_vel.astype(np.float32),
        "joint_pos": joint_pos_rel.astype(np.float32),
        "joint_vel": joint_vel_rel.astype(np.float32),
        "actions": last_action_il.astype(np.float32),
        "gravity_dir": gravity_dir.astype(np.float32),
    }
    for name in OBS_TERM_ORDER:
        if terms[name].shape != (OBS_TERM_DIMS[name],):
            raise RuntimeError(f"observation term {name} has shape {terms[name].shape}")
    return terms


def set_loop_heads_zero(data: mujoco.MjData, runtime: LoopRuntime) -> None:
    mapping = runtime.mapping
    for head in mapping.head_joint_names:
        data.qpos[mapping.qpos_addr_by_joint[head]] = 0.0
        data.qvel[mapping.qvel_addr_by_joint[head]] = 0.0


def reset_loop_to_reference(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    runtime: LoopRuntime,
    reference: MotionReference,
    frame: int,
) -> None:
    data.time = frame / reference.fps
    data.qpos[:] = reference.qpos[frame]
    data.qvel[:] = reference.qvel[frame]
    data.ctrl[:] = 0.0
    set_loop_heads_zero(data, runtime)
    mujoco.mj_forward(model, data)


def direct_pd_torque(q: float, dq: float, target: float, cfg: JointControl) -> float:
    return float(cfg.kp * (target - q) - cfg.kd * dq)


def apply_loop_pd_control(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    runtime: LoopRuntime,
    solver: A3LoopSolverBridge,
    controls: dict[str, JointControl],
    target_by_joint: dict[str, float],
) -> None:
    mapping = runtime.mapping
    data.ctrl[:] = 0.0

    for joint_name, target in target_by_joint.items():
        if joint_name in LOOP_PR_JOINTS:
            continue
        q = float(data.qpos[mapping.qpos_addr_by_joint[joint_name]])
        dq = float(data.qvel[mapping.qvel_addr_by_joint[joint_name]])
        cfg = controls[joint_name]
        tau = direct_pd_torque(q, dq, target, cfg)
        data.ctrl[mapping.actuator_id_by_joint[joint_name]] = tau

    loop_q, _ = loop_state_overrides(data, runtime, solver)

    left_target = np.array(
        [target_by_joint[LEFT_ANKLE_PR[0]], target_by_joint[LEFT_ANKLE_PR[1]]],
        dtype=np.float64,
    )
    right_target = np.array(
        [target_by_joint[RIGHT_ANKLE_PR[0]], target_by_joint[RIGHT_ANKLE_PR[1]]],
        dtype=np.float64,
    )
    waist_target = np.array(
        [target_by_joint["waist_pitch_joint"], target_by_joint["waist_roll_joint"]],
        dtype=np.float64,
    )

    left_actual = np.array([loop_q[LEFT_ANKLE_PR[0]], loop_q[LEFT_ANKLE_PR[1]]], dtype=np.float64)
    right_actual = np.array([loop_q[RIGHT_ANKLE_PR[0]], loop_q[RIGHT_ANKLE_PR[1]]], dtype=np.float64)
    waist_actual = np.array([loop_q["waist_pitch_joint"], loop_q["waist_roll_joint"]], dtype=np.float64)

    left_vel = np.array(
        [
            data.qvel[runtime.motor_qvel_addr[LEFT_ANKLE_MOTORS[0]]],
            data.qvel[runtime.motor_qvel_addr[LEFT_ANKLE_MOTORS[1]]],
        ],
        dtype=np.float64,
    )
    right_vel = np.array(
        [
            data.qvel[runtime.motor_qvel_addr[RIGHT_ANKLE_MOTORS[0]]],
            data.qvel[runtime.motor_qvel_addr[RIGHT_ANKLE_MOTORS[1]]],
        ],
        dtype=np.float64,
    )
    waist_vel = np.array(
        [
            data.qvel[runtime.motor_qvel_addr[WAIST_MOTORS[0]]],
            data.qvel[runtime.motor_qvel_addr[WAIST_MOTORS[1]]],
        ],
        dtype=np.float64,
    )

    zero = np.zeros(2, dtype=np.float64)
    left_kp = np.array([controls[LEFT_ANKLE_PR[0]].kp, controls[LEFT_ANKLE_PR[1]].kp], dtype=np.float64)
    left_kd = np.array([controls[LEFT_ANKLE_PR[0]].kd, controls[LEFT_ANKLE_PR[1]].kd], dtype=np.float64)
    right_kp = np.array([controls[RIGHT_ANKLE_PR[0]].kp, controls[RIGHT_ANKLE_PR[1]].kp], dtype=np.float64)
    right_kd = np.array([controls[RIGHT_ANKLE_PR[0]].kd, controls[RIGHT_ANKLE_PR[1]].kd], dtype=np.float64)
    waist_kp = np.array([controls["waist_pitch_joint"].kp, controls["waist_roll_joint"].kp], dtype=np.float64)
    waist_kd = np.array([controls["waist_pitch_joint"].kd, controls["waist_roll_joint"].kd], dtype=np.float64)

    left_delta = solver.ankle_rl_pos(0, left_kp, left_target - left_actual)
    right_delta = solver.ankle_rl_pos(1, right_kp, right_target - right_actual)
    waist_delta = solver.waist_rl_pos(waist_kp, waist_target - waist_actual)
    left_kd_m = solver.ankle_rl_kd(0, left_kd)
    right_kd_m = solver.ankle_rl_kd(1, right_kd)
    waist_kd_m = solver.waist_rl_kd(waist_kd)
    left_vel_des = solver.ankle_dik(0, zero)
    right_vel_des = solver.ankle_dik(1, zero)
    waist_vel_des = solver.waist_dik(zero)
    left_tau_ff = solver.ankle_idyn(0, zero)
    right_tau_ff = solver.ankle_idyn(1, zero)
    waist_tau_ff = solver.waist_idyn(zero)

    left_ctrl = left_tau_ff + left_kd_m * (left_vel_des - left_vel) + left_kp * left_delta
    right_ctrl = right_tau_ff + right_kd_m * (right_vel_des - right_vel) + right_kp * right_delta
    waist_ctrl = waist_tau_ff + waist_kd_m * (waist_vel_des - waist_vel) + waist_kp * waist_delta

    loop_outputs = [
        (LEFT_ANKLE_PR[0], left_ctrl[0]),
        (LEFT_ANKLE_PR[1], left_ctrl[1]),
        (RIGHT_ANKLE_PR[0], right_ctrl[0]),
        (RIGHT_ANKLE_PR[1], right_ctrl[1]),
        ("waist_roll_joint", waist_ctrl[0]),
        ("waist_pitch_joint", waist_ctrl[1]),
    ]
    for joint_name, tau in loop_outputs:
        data.ctrl[mapping.actuator_id_by_joint[joint_name]] = float(tau)

    for head in mapping.head_joint_names:
        if head in mapping.actuator_id_by_joint:
            data.ctrl[mapping.actuator_id_by_joint[head]] = 0.0


# =============================================================================
# Metrics And Dump Replay Helpers
# =============================================================================

def root_roll_pitch_deg_from_quat_wxyz(quat_wxyz: np.ndarray) -> tuple[float | None, float | None]:
    try:
        quat = quat_normalize(np.asarray(quat_wxyz, dtype=np.float64))
        if quat.shape != (4,) or not np.isfinite(quat).all():
            return None, None
        euler_xyz = Rotation.from_quat(quat[[1, 2, 3, 0]]).as_euler("xyz", degrees=True)
        return float(euler_xyz[0]), float(euler_xyz[1])
    except Exception:
        return None, None


def finite_float_or_none(value: float | np.floating | None) -> float | None:
    if value is None:
        return None
    value = float(value)
    return value if np.isfinite(value) else None


def rms_or_none(values: np.ndarray | None) -> float | None:
    if values is None or values.size == 0:
        return None
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return None
    return finite_float_or_none(np.sqrt(np.mean(np.square(finite))))


def abs_max_or_none(values: np.ndarray | None) -> float | None:
    if values is None or values.size == 0:
        return None
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return None
    return finite_float_or_none(np.max(np.abs(finite)))


def mean_or_none(values: np.ndarray | None) -> float | None:
    if values is None or values.size == 0:
        return None
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return None
    return finite_float_or_none(np.mean(finite))


def min_or_none(values: np.ndarray | None) -> float | None:
    if values is None or values.size == 0:
        return None
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return None
    return finite_float_or_none(np.min(finite))


def array_or_none(rows: list[np.ndarray], width: int | None = None) -> np.ndarray | None:
    if not rows:
        return None
    arr = np.asarray(rows, dtype=np.float64)
    if width is not None and (arr.ndim != 2 or arr.shape[1] != width):
        return None
    return arr


def stacked_or_none(rows: list[np.ndarray]) -> np.ndarray | None:
    if not rows:
        return None
    arr = np.asarray(rows, dtype=np.float64)
    return arr if arr.size else None


def quat_angle_deg_or_none(a_wxyz: np.ndarray | None, b_wxyz: np.ndarray | None) -> float | None:
    if a_wxyz is None or b_wxyz is None:
        return None
    try:
        a = quat_normalize(np.asarray(a_wxyz, dtype=np.float64))
        b = quat_normalize(np.asarray(b_wxyz, dtype=np.float64))
        if a.shape != (4,) or b.shape != (4,) or not np.isfinite(a).all() or not np.isfinite(b).all():
            return None
        rel = quat_mul_wxyz(quat_conj_wxyz(a), b)
        angle = 2.0 * np.arccos(np.clip(abs(float(rel[0])), -1.0, 1.0))
        return finite_float_or_none(np.rad2deg(angle))
    except Exception:
        return None


def body_quat_wxyz(data: mujoco.MjData, body_id: int) -> np.ndarray:
    quat = np.zeros(4, dtype=np.float64)
    mujoco.mju_mat2Quat(quat, data.xmat[int(body_id)].reshape(9))
    return quat_normalize(quat)


def body_positions_anchor_relative(
    body_pos_w: np.ndarray,
    anchor_pos_w: np.ndarray,
    anchor_quat_wxyz: np.ndarray,
) -> np.ndarray:
    return quat_rotate_inverse_wxyz(
        np.broadcast_to(anchor_quat_wxyz, body_pos_w.shape[:-1] + (4,)),
        body_pos_w - anchor_pos_w,
    )


ANCHOR_BODY_CANDIDATES = (
    "pelvis_link",
    "torso_Link",
    "left_wrist_yaw_Link",
    "right_wrist_yaw_Link",
    "left_ankle_roll_Link",
    "right_ankle_roll_Link",
)


def json_write(path: Path, payload: dict) -> None:
    path = path.expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")


@dataclass
class PolicyStepResult:
    raw_action_il: np.ndarray
    q_des_il: np.ndarray
    actor_obs: np.ndarray


class RolloutMetricsCollector:
    def __init__(self, runtime: LoopRuntime, model: mujoco.MjModel | None = None) -> None:
        self.model = model
        self.ref_data = mujoco.MjData(model) if model is not None else None
        names = runtime.mapping.il_joint_names
        self.group_indices = {
            "all_29": np.arange(NUM_POLICY_DOFS, dtype=np.int64),
            "waist": np.array([i for i, n in enumerate(names) if "waist" in n], dtype=np.int64),
            "arms": np.array(
                [
                    i
                    for i, n in enumerate(names)
                    if any(token in n for token in ("shoulder", "elbow", "wrist"))
                ],
                dtype=np.int64,
            ),
            "legs": np.array(
                [
                    i
                    for i, n in enumerate(names)
                    if any(token in n for token in ("hip", "knee", "ankle"))
                ],
                dtype=np.int64,
            ),
        }
        self.body_ids = [int(runtime.mapping.body_id_by_joint[name]) for name in runtime.mapping.mj29_joint_names]
        # Tracked-body set for framework-aligned mpjpe_g/mpjpe_l (14 bodies, pelvis root).
        self.tracked_body_ids: list[int] = []
        if model is not None:
            for _tb_name in TRACKED_BODY_NAMES:
                _tb_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, _tb_name)
                if _tb_id < 0:
                    raise ValueError(
                        f"tracked body '{_tb_name}' not found in MJCF; cannot align mpjpe_g/mpjpe_l"
                    )
                self.tracked_body_ids.append(int(_tb_id))
        self.body_names = [model_name(model, mujoco.mjtObj.mjOBJ_BODY, body_id) if model is not None else str(body_id) for body_id in self.body_ids]
        self.anchor_body_id = int(runtime.mapping.anchor_body_id)
        self.anchor_body_name = runtime.mapping.anchor_body_name
        self.anchor_body_ids: list[int] = []
        self.anchor_body_names: list[str] = []
        if model is not None:
            for name in ANCHOR_BODY_CANDIDATES:
                body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
                if body_id >= 0:
                    self.anchor_body_ids.append(int(body_id))
                    self.anchor_body_names.append(name)
        self.motions: list[dict] = []
        self.current: dict | None = None

    def start_motion(self, motion_name: str) -> None:
        self.finish_current()
        self.current = {
            "motion_name": motion_name,
            "policy_tick": [],
            "time_s": [],
            "root_height": [],
            "root_roll_deg": [],
            "root_pitch_deg": [],
            "q_state_29": [],
            "q_des_29": [],
            "raw_action_29": [],
            "reference_q_29": [],
            "sim_root_pos_w": [],
            "sim_root_quat_wxyz": [],
            "ref_root_pos_w": [],
            "ref_root_quat_wxyz": [],
            "root_pos_error_m": [],
            "root_quat_error_deg": [],
            "sim_anchor_pos_w": [],
            "sim_anchor_quat_wxyz": [],
            "ref_anchor_pos_w": [],
            "ref_anchor_quat_wxyz": [],
            "anchor_pos_error_m": [],
            "anchor_quat_error_deg": [],
            "sim_body_pos_w": [],
            "ref_body_pos_w": [],
            "sim_tracked_body_pos_w": [],
            "ref_tracked_body_pos_w": [],
            "sim_body_pos_anchor": [],
            "ref_body_pos_anchor": [],
            "sim_key_body_pos_w": [],
            "ref_key_body_pos_w": [],
            "fall_tick": None,
            "fall_time_s": None,
        }

    def _pose_snapshot(self, data: mujoco.MjData) -> dict[str, np.ndarray] | None:
        if self.model is None:
            return None
        root_pos = np.asarray(data.qpos[:3], dtype=np.float64).copy()
        root_quat = get_root_quat(data)
        anchor_pos = np.asarray(data.xpos[self.anchor_body_id], dtype=np.float64).copy()
        anchor_quat = body_quat_wxyz(data, self.anchor_body_id)
        body_pos_w = np.asarray(data.xpos[self.body_ids], dtype=np.float64).copy()
        tracked_body_pos_w = (
            np.asarray(data.xpos[self.tracked_body_ids], dtype=np.float64).copy()
            if self.tracked_body_ids
            else np.zeros((0, 3), dtype=np.float64)
        )
        key_body_pos_w = (
            np.asarray(data.xpos[self.anchor_body_ids], dtype=np.float64).copy()
            if self.anchor_body_ids
            else np.zeros((0, 3), dtype=np.float64)
        )
        return {
            "root_pos_w": root_pos,
            "root_quat_wxyz": root_quat,
            "anchor_pos_w": anchor_pos,
            "anchor_quat_wxyz": anchor_quat,
            "body_pos_w": body_pos_w,
            "tracked_body_pos_w": tracked_body_pos_w,
            "body_pos_anchor": body_positions_anchor_relative(body_pos_w, anchor_pos, anchor_quat),
            "key_body_pos_w": key_body_pos_w,
        }

    def _reference_pose_snapshot(
        self,
        reference: MotionReference | None,
        ref_frame: int | None,
    ) -> dict[str, np.ndarray] | None:
        if self.model is None or self.ref_data is None or reference is None or ref_frame is None:
            return None
        frame = int(np.clip(int(ref_frame), 0, reference.num_frames - 1))
        self.ref_data.time = float(frame / reference.fps)
        self.ref_data.qpos[:] = reference.qpos[frame]
        self.ref_data.qvel[:] = reference.qvel[frame]
        mujoco.mj_forward(self.model, self.ref_data)
        return self._pose_snapshot(self.ref_data)

    def record(
        self,
        *,
        policy_tick: int,
        time_s: float,
        data: mujoco.MjData,
        q_state_il: np.ndarray | None,
        q_des_il: np.ndarray | None,
        raw_action_il: np.ndarray | None,
        reference_q_il: np.ndarray | None,
        reference: MotionReference | None = None,
        ref_frame: int | None = None,
    ) -> None:
        if self.current is None:
            return
        root_height = finite_float_or_none(data.qpos[2] if data.qpos.size >= 3 else None)
        roll_deg, pitch_deg = root_roll_pitch_deg_from_quat_wxyz(get_root_quat(data))
        sim_pose = self._pose_snapshot(data)
        ref_pose = self._reference_pose_snapshot(reference, ref_frame)
        self.current["policy_tick"].append(int(policy_tick))
        self.current["time_s"].append(float(time_s))
        self.current["root_height"].append(root_height)
        self.current["root_roll_deg"].append(roll_deg)
        self.current["root_pitch_deg"].append(pitch_deg)
        if q_state_il is not None:
            self.current["q_state_29"].append(np.asarray(q_state_il, dtype=np.float64).copy())
        if q_des_il is not None:
            self.current["q_des_29"].append(np.asarray(q_des_il, dtype=np.float64).copy())
        if raw_action_il is not None:
            self.current["raw_action_29"].append(np.asarray(raw_action_il, dtype=np.float64).copy())
        if reference_q_il is not None:
            self.current["reference_q_29"].append(np.asarray(reference_q_il, dtype=np.float64).copy())

        if sim_pose is not None:
            self.current["sim_root_pos_w"].append(sim_pose["root_pos_w"])
            self.current["sim_root_quat_wxyz"].append(sim_pose["root_quat_wxyz"])
            self.current["sim_anchor_pos_w"].append(sim_pose["anchor_pos_w"])
            self.current["sim_anchor_quat_wxyz"].append(sim_pose["anchor_quat_wxyz"])
            self.current["sim_body_pos_w"].append(sim_pose["body_pos_w"])
            self.current["sim_tracked_body_pos_w"].append(sim_pose["tracked_body_pos_w"])
            self.current["sim_body_pos_anchor"].append(sim_pose["body_pos_anchor"])
            self.current["sim_key_body_pos_w"].append(sim_pose["key_body_pos_w"])
        if ref_pose is not None:
            self.current["ref_root_pos_w"].append(ref_pose["root_pos_w"])
            self.current["ref_root_quat_wxyz"].append(ref_pose["root_quat_wxyz"])
            self.current["ref_anchor_pos_w"].append(ref_pose["anchor_pos_w"])
            self.current["ref_anchor_quat_wxyz"].append(ref_pose["anchor_quat_wxyz"])
            self.current["ref_body_pos_w"].append(ref_pose["body_pos_w"])
            self.current["ref_tracked_body_pos_w"].append(ref_pose["tracked_body_pos_w"])
            self.current["ref_body_pos_anchor"].append(ref_pose["body_pos_anchor"])
            self.current["ref_key_body_pos_w"].append(ref_pose["key_body_pos_w"])
        if sim_pose is not None and ref_pose is not None:
            self.current["root_pos_error_m"].append(
                np.linalg.norm(sim_pose["root_pos_w"] - ref_pose["root_pos_w"])
            )
            self.current["root_quat_error_deg"].append(
                quat_angle_deg_or_none(sim_pose["root_quat_wxyz"], ref_pose["root_quat_wxyz"])
            )
            self.current["anchor_pos_error_m"].append(
                np.linalg.norm(sim_pose["anchor_pos_w"] - ref_pose["anchor_pos_w"])
            )
            self.current["anchor_quat_error_deg"].append(
                quat_angle_deg_or_none(sim_pose["anchor_quat_wxyz"], ref_pose["anchor_quat_wxyz"])
            )

        roll_abs = None if roll_deg is None else abs(roll_deg)
        pitch_abs = None if pitch_deg is None else abs(pitch_deg)
        fall_by_height = root_height is not None and root_height < 0.45
        fall_by_tilt = (
            (roll_abs is not None and roll_abs > 60.0)
            or (pitch_abs is not None and pitch_abs > 60.0)
        )
        if (fall_by_height or fall_by_tilt) and self.current["fall_tick"] is None:
            self.current["fall_tick"] = int(policy_tick)
            self.current["fall_time_s"] = float(time_s)

    def finish_current(self) -> None:
        if self.current is None:
            return
        self.motions.append(self._summarize_current(self.current))
        self.current = None

    def metrics_payload(self) -> dict:
        self.finish_current()
        motions = [self._summary_without_timeseries(motion) for motion in self.motions]
        payload = {
            "schema_version": 1,
            "fps": TARGET_FPS,
            "fall_heuristic": {
                "root_height_below_m": 0.45,
                "abs_roll_or_pitch_above_deg": 60.0,
            },
            "motions": motions,
        }
        if len(motions) == 1:
            payload.update(motions[0])
        return payload

    def timeseries_payload(self) -> dict:
        self.finish_current()
        return {
            "schema_version": 1,
            "fps": TARGET_FPS,
            "motions": [motion["timeseries"] for motion in self.motions],
        }

    @staticmethod
    def _summary_without_timeseries(motion: dict) -> dict:
        return {key: value for key, value in motion.items() if key != "timeseries"}

    def _group_rmse(self, values: np.ndarray | None, reference: np.ndarray | None) -> dict[str, float | None]:
        out: dict[str, float | None] = {}
        if values is None or reference is None or values.shape != reference.shape:
            for name in self.group_indices:
                out[f"{name}_rmse"] = None
            return out
        err = values - reference
        for name, idx in self.group_indices.items():
            out[f"{name}_rmse"] = rms_or_none(err[:, idx] if idx.size else None)
        return out

    def _mpjpe_summary(self, sim_pos: np.ndarray | None, ref_pos: np.ndarray | None) -> dict[str, float | None]:
        if sim_pos is None or ref_pos is None or sim_pos.shape != ref_pos.shape or sim_pos.ndim != 3:
            return {
                "mpjpe_m": None,
                "mpjpe_mm": None,
                "median_mpjpe_mm": None,
                "p95_mpjpe_mm": None,
                "max_error_mm": None,
            }
        per_body = np.linalg.norm(sim_pos - ref_pos, axis=-1)
        mean_m = mean_or_none(per_body)
        return {
            "mpjpe_m": mean_m,
            "mpjpe_mm": None if mean_m is None else 1000.0 * mean_m,
            "median_mpjpe_mm": finite_float_or_none(1000.0 * np.nanmedian(per_body)),
            "p95_mpjpe_mm": finite_float_or_none(1000.0 * np.nanpercentile(per_body, 95.0)),
            "max_error_mm": finite_float_or_none(1000.0 * np.nanmax(per_body)),
        }

    def _body_group_mpjpe(self, sim_pos: np.ndarray | None, ref_pos: np.ndarray | None) -> dict[str, dict[str, float | None]]:
        out: dict[str, dict[str, float | None]] = {}
        if sim_pos is None or ref_pos is None or sim_pos.shape != ref_pos.shape or sim_pos.ndim != 3:
            for name in self.group_indices:
                out[name] = self._mpjpe_summary(None, None)
            return out
        for name, idx in self.group_indices.items():
            out[name] = self._mpjpe_summary(
                sim_pos[:, idx, :] if idx.size else None,
                ref_pos[:, idx, :] if idx.size else None,
            )
        return out

    @staticmethod
    def _numeric_series(values: list[float | None]) -> np.ndarray:
        return np.asarray([np.nan if v is None else v for v in values], dtype=np.float64)

    def _summarize_current(self, current: dict) -> dict:
        root_height = np.asarray(
            [np.nan if v is None else v for v in current["root_height"]],
            dtype=np.float64,
        )
        roll = np.asarray(
            [np.nan if v is None else v for v in current["root_roll_deg"]],
            dtype=np.float64,
        )
        pitch = np.asarray(
            [np.nan if v is None else v for v in current["root_pitch_deg"]],
            dtype=np.float64,
        )
        q_state = array_or_none(current["q_state_29"], NUM_POLICY_DOFS)
        q_des = array_or_none(current["q_des_29"], NUM_POLICY_DOFS)
        raw_action = array_or_none(current["raw_action_29"], NUM_POLICY_DOFS)
        reference_q = array_or_none(current["reference_q_29"], NUM_POLICY_DOFS)
        q_des_step = np.diff(q_des, axis=0) if q_des is not None and q_des.shape[0] >= 2 else None
        sim_root_pos = array_or_none(current["sim_root_pos_w"], 3)
        ref_root_pos = array_or_none(current["ref_root_pos_w"], 3)
        sim_root_quat = array_or_none(current["sim_root_quat_wxyz"], 4)
        ref_root_quat = array_or_none(current["ref_root_quat_wxyz"], 4)
        sim_anchor_pos = array_or_none(current["sim_anchor_pos_w"], 3)
        ref_anchor_pos = array_or_none(current["ref_anchor_pos_w"], 3)
        sim_anchor_quat = array_or_none(current["sim_anchor_quat_wxyz"], 4)
        ref_anchor_quat = array_or_none(current["ref_anchor_quat_wxyz"], 4)
        sim_body_pos_w = stacked_or_none(current["sim_body_pos_w"])
        ref_body_pos_w = stacked_or_none(current["ref_body_pos_w"])
        sim_tracked_body_pos_w = stacked_or_none(current["sim_tracked_body_pos_w"])
        ref_tracked_body_pos_w = stacked_or_none(current["ref_tracked_body_pos_w"])
        sim_body_pos_anchor = stacked_or_none(current["sim_body_pos_anchor"])
        ref_body_pos_anchor = stacked_or_none(current["ref_body_pos_anchor"])
        sim_key_body_pos_w = stacked_or_none(current["sim_key_body_pos_w"])
        ref_key_body_pos_w = stacked_or_none(current["ref_key_body_pos_w"])
        root_pos_error = self._numeric_series(current["root_pos_error_m"])
        root_quat_error = self._numeric_series(current["root_quat_error_deg"])
        anchor_pos_error = self._numeric_series(current["anchor_pos_error_m"])
        anchor_quat_error = self._numeric_series(current["anchor_quat_error_deg"])

        q_state_tracking = self._group_rmse(q_state, reference_q)
        q_des_tracking = self._group_rmse(q_des, reference_q)
        tracking: dict[str, object] = dict(q_state_tracking)
        tracking["q_state_vs_ref"] = q_state_tracking
        tracking["q_des_vs_ref"] = q_des_tracking
        tracking["global_body_mpjpe"] = self._mpjpe_summary(sim_body_pos_w, ref_body_pos_w)
        tracking["anchor_local_body_mpjpe"] = self._mpjpe_summary(
            sim_body_pos_anchor,
            ref_body_pos_anchor,
        )
        tracking["anchor_local_body_mpjpe_by_group"] = self._body_group_mpjpe(
            sim_body_pos_anchor,
            ref_body_pos_anchor,
        )
        # Framework-aligned mpjpe (matches smpl_sim.compute_metrics_lite, root_idx=0).
        # mpjpe_g: world frame, no root subtraction, 14 tracked bodies.
        tracking["mpjpe_g"] = self._mpjpe_summary(sim_tracked_body_pos_w, ref_tracked_body_pos_w)
        # mpjpe_l: subtract root (pelvis_link, idx=0) translation only, no rotation.
        if (
            sim_tracked_body_pos_w is not None
            and ref_tracked_body_pos_w is not None
            and sim_tracked_body_pos_w.ndim == 3
            and ref_tracked_body_pos_w.ndim == 3
        ):
            sim_tracked_l = sim_tracked_body_pos_w - sim_tracked_body_pos_w[:, [MPJPE_ROOT_IDX], :]
            ref_tracked_l = ref_tracked_body_pos_w - ref_tracked_body_pos_w[:, [MPJPE_ROOT_IDX], :]
            tracking["mpjpe_l"] = self._mpjpe_summary(sim_tracked_l, ref_tracked_l)
        else:
            tracking["mpjpe_l"] = self._mpjpe_summary(None, None)
        timeseries = {
            "motion_name": current["motion_name"],
            "policy_tick": current["policy_tick"],
            "time_s": current["time_s"],
            "body_names": self.body_names,
            "anchor_body_name": self.anchor_body_name,
            "key_body_names": self.anchor_body_names,
            "root_height": current["root_height"],
            "root_roll_deg": current["root_roll_deg"],
            "root_pitch_deg": current["root_pitch_deg"],
            "q_state_29": None if q_state is None else q_state.tolist(),
            "q_des_29": None if q_des is None else q_des.tolist(),
            "raw_action_29": None if raw_action is None else raw_action.tolist(),
            "reference_q_29": None if reference_q is None else reference_q.tolist(),
            "sim_root_pos_w": None if sim_root_pos is None else sim_root_pos.tolist(),
            "sim_root_quat_wxyz": None if sim_root_quat is None else sim_root_quat.tolist(),
            "ref_root_pos_w": None if ref_root_pos is None else ref_root_pos.tolist(),
            "ref_root_quat_wxyz": None if ref_root_quat is None else ref_root_quat.tolist(),
            "root_pos_error_m": root_pos_error.tolist(),
            "root_quat_error_deg": root_quat_error.tolist(),
            "sim_anchor_pos_w": None if sim_anchor_pos is None else sim_anchor_pos.tolist(),
            "sim_anchor_quat_wxyz": None if sim_anchor_quat is None else sim_anchor_quat.tolist(),
            "ref_anchor_pos_w": None if ref_anchor_pos is None else ref_anchor_pos.tolist(),
            "ref_anchor_quat_wxyz": None if ref_anchor_quat is None else ref_anchor_quat.tolist(),
            "anchor_pos_error_m": anchor_pos_error.tolist(),
            "anchor_quat_error_deg": anchor_quat_error.tolist(),
            "sim_body_pos_w": None if sim_body_pos_w is None else sim_body_pos_w.tolist(),
            "ref_body_pos_w": None if ref_body_pos_w is None else ref_body_pos_w.tolist(),
            "sim_tracked_body_pos_w": None if sim_tracked_body_pos_w is None else sim_tracked_body_pos_w.tolist(),
            "ref_tracked_body_pos_w": None if ref_tracked_body_pos_w is None else ref_tracked_body_pos_w.tolist(),
            "sim_body_pos_anchor": None if sim_body_pos_anchor is None else sim_body_pos_anchor.tolist(),
            "ref_body_pos_anchor": None if ref_body_pos_anchor is None else ref_body_pos_anchor.tolist(),
            "sim_key_body_pos_w": None if sim_key_body_pos_w is None else sim_key_body_pos_w.tolist(),
            "ref_key_body_pos_w": None if ref_key_body_pos_w is None else ref_key_body_pos_w.tolist(),
        }
        return {
            "motion_name": current["motion_name"],
            "num_policy_steps": len(current["policy_tick"]),
            "fall": current["fall_tick"] is not None,
            "fall_tick": current["fall_tick"],
            "fall_time_s": current["fall_time_s"],
            "root_height": {
                "min": min_or_none(root_height),
                "mean": mean_or_none(root_height),
            },
            "root_pose": {
                "position_rmse_m": rms_or_none(root_pos_error),
                "position_mean_m": mean_or_none(root_pos_error),
                "position_max_m": abs_max_or_none(root_pos_error),
                "orientation_mean_deg": mean_or_none(root_quat_error),
                "orientation_max_deg": abs_max_or_none(root_quat_error),
            },
            "anchor_pose": {
                "body_name": self.anchor_body_name,
                "position_rmse_m": rms_or_none(anchor_pos_error),
                "position_mean_m": mean_or_none(anchor_pos_error),
                "position_max_m": abs_max_or_none(anchor_pos_error),
                "orientation_mean_deg": mean_or_none(anchor_quat_error),
                "orientation_max_deg": abs_max_or_none(anchor_quat_error),
            },
            "root_roll_pitch_abs_max_deg": abs_max_or_none(np.concatenate([roll, pitch])),
            "tracking": tracking,
            "action": {
                "raw_rms": rms_or_none(raw_action),
                "raw_abs_max": abs_max_or_none(raw_action),
                "q_des_step_rms": rms_or_none(q_des_step),
                "q_des_step_abs_max": abs_max_or_none(q_des_step),
            },
            "timeseries": timeseries,
        }


FOOT_NAME_PARTS = ("foot", "sole", "toe", "ankle")
FLOOR_NAME_PARTS = ("floor", "ground", "terrain", "plane", "world")


def q31_to_q29_mj(q31: np.ndarray) -> np.ndarray:
    q31 = np.asarray(q31, dtype=np.float64)
    return q31[np.asarray(A3_POLICY_TO_SDK_IDX, dtype=np.int64)]


def select_foot_geom_ids(model: mujoco.MjModel) -> tuple[list[int], bool]:
    matched: list[int] = []
    fallback: list[int] = []
    for geom_id in range(model.ngeom):
        if int(model.geom_type[geom_id]) == int(mujoco.mjtGeom.mjGEOM_PLANE):
            continue
        geom_name = (model_name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or "").lower()
        body_id = int(model.geom_bodyid[geom_id])
        body_name = (model_name(model, mujoco.mjtObj.mjOBJ_BODY, body_id) or "").lower()
        combined = f"{geom_name} {body_name}"
        if body_id == 0 or any(part in combined for part in FLOOR_NAME_PARTS):
            continue
        fallback.append(geom_id)
        if any(part in combined for part in FOOT_NAME_PARTS):
            matched.append(geom_id)
    return (matched, False) if matched else (fallback, True)


def geom_lowest_world_z(model: mujoco.MjModel, data: mujoco.MjData, geom_id: int) -> float | None:
    geom_type = int(model.geom_type[geom_id])
    if geom_type == int(mujoco.mjtGeom.mjGEOM_PLANE):
        return None
    xpos = np.asarray(data.geom_xpos[geom_id], dtype=np.float64)
    xmat = np.asarray(data.geom_xmat[geom_id], dtype=np.float64).reshape(3, 3)
    size = np.asarray(model.geom_size[geom_id], dtype=np.float64)

    if geom_type == int(mujoco.mjtGeom.mjGEOM_MESH):
        mesh_id = int(model.geom_dataid[geom_id])
        if mesh_id >= 0:
            adr = int(model.mesh_vertadr[mesh_id])
            count = int(model.mesh_vertnum[mesh_id])
            if count > 0:
                verts = np.asarray(model.mesh_vert[adr : adr + count], dtype=np.float64)
                world = xpos + verts @ xmat.T
                return finite_float_or_none(np.min(world[:, 2]))
        rbound = float(model.geom_rbound[geom_id])
        return finite_float_or_none(xpos[2] - rbound)

    if geom_type == int(mujoco.mjtGeom.mjGEOM_BOX):
        corners = np.array(
            [
                [sx * size[0], sy * size[1], sz * size[2]]
                for sx in (-1.0, 1.0)
                for sy in (-1.0, 1.0)
                for sz in (-1.0, 1.0)
            ],
            dtype=np.float64,
        )
        world = xpos + corners @ xmat.T
        return finite_float_or_none(np.min(world[:, 2]))

    if geom_type == int(mujoco.mjtGeom.mjGEOM_SPHERE):
        return finite_float_or_none(xpos[2] - size[0])

    if geom_type in (
        int(mujoco.mjtGeom.mjGEOM_CAPSULE),
        int(mujoco.mjtGeom.mjGEOM_CYLINDER),
    ):
        axis_z = abs(float(xmat[2, 2]))
        radial_z = np.sqrt(max(0.0, 1.0 - axis_z * axis_z))
        return finite_float_or_none(xpos[2] - size[1] * axis_z - size[0] * radial_z)

    rbound = float(model.geom_rbound[geom_id])
    return finite_float_or_none(xpos[2] - rbound)


def align_root_z_to_lowest_foot_geom(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    *,
    sole_clearance: float = 0.0,
) -> tuple[float | None, bool]:
    mujoco.mj_forward(model, data)
    geom_ids, used_fallback = select_foot_geom_ids(model)
    lows = [
        z
        for geom_id in geom_ids
        for z in [geom_lowest_world_z(model, data, geom_id)]
        if z is not None
    ]
    if not lows:
        return None, used_fallback
    lowest = float(np.min(lows))
    data.qpos[2] += float(sole_clearance) - lowest
    mujoco.mj_forward(model, data)
    return lowest, used_fallback


def set_replay_reset_state_from_dump(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    runtime: LoopRuntime,
    solver: A3LoopSolverBridge,
    dump: dict[str, np.ndarray],
    frame: int,
) -> None:
    q_state_29 = np.asarray(dump["q_state_29"][frame], dtype=np.float64)
    dq_state_29 = np.asarray(dump.get("dq_state_29", np.zeros_like(dump["q_state_29"]))[frame], dtype=np.float64)
    q_state_31 = np.asarray(
        dump.get("q_state_31", np.zeros((dump["q_state_29"].shape[0], 31), dtype=np.float64))[frame],
        dtype=np.float64,
    )

    data.time = 0.0
    data.qpos[:] = loop_key_qpos(model)
    data.qvel[:] = 0.0
    data.ctrl[:] = 0.0
    data.qpos[:3] = 0.0

    imu = dump.get("imu")
    if imu is not None and imu.shape[1] >= 4:
        quat = quat_normalize(np.asarray(imu[frame, :4], dtype=np.float64))
        if quat.shape == (4,) and np.isfinite(quat).all():
            data.qpos[3:7] = quat
    else:
        data.qpos[3:7] = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)

    fill_loop_qpos_motors(data.qpos, runtime, solver, q_state_29)
    mapping = runtime.mapping
    if q_state_31.shape[0] >= 5:
        for head_idx, head in enumerate(HEAD_JOINTS):
            if head in mapping.qpos_addr_by_joint:
                data.qpos[mapping.qpos_addr_by_joint[head]] = float(q_state_31[3 + head_idx])

    dq_il = dq_state_29[mapping.mj29_to_il]
    for il_idx, joint_name in enumerate(mapping.il_joint_names):
        if joint_name in mapping.qvel_addr_by_joint:
            data.qvel[mapping.qvel_addr_by_joint[joint_name]] = float(dq_il[il_idx])
    lowest, used_fallback = align_root_z_to_lowest_foot_geom(model, data)
    if used_fallback:
        print("[replay] warning: no named foot/sole/toe geoms found; aligned root z using all non-world geoms")
    if lowest is None:
        print("[replay] warning: could not infer foot geometry low point; root z left at reset value")


def load_policy_dump_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        dump = {name: data[name] for name in data.files}
    if "q_state_29" not in dump:
        if "q_state_31" not in dump:
            raise ValueError("dump NPZ must contain q_state_29 or q_state_31")
        dump["q_state_29"] = np.asarray([q31_to_q29_mj(row) for row in dump["q_state_31"]], dtype=np.float64)
    if "dq_state_29" not in dump and "dq_state_31" in dump:
        dump["dq_state_29"] = np.asarray([q31_to_q29_mj(row) for row in dump["dq_state_31"]], dtype=np.float64)
    if "q_des_29" not in dump and "q_des_31" in dump:
        dump["q_des_29"] = np.asarray([q31_to_q29_mj(row) for row in dump["q_des_31"]], dtype=np.float64)
    return dump


def replay_command_q_des_mj29(
    dump: dict[str, np.ndarray],
    frame: int,
    replay_command: str,
) -> np.ndarray:
    if replay_command == "q_des_31":
        if "q_des_31" not in dump:
            raise ValueError("dump NPZ does not contain q_des_31")
        return q31_to_q29_mj(dump["q_des_31"][frame])
    if "q_des_29" not in dump:
        raise ValueError("dump NPZ does not contain q_des_29")
    return np.asarray(dump["q_des_29"][frame], dtype=np.float64)


def run_policy_dump_replay(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    runtime: LoopRuntime,
    solver: A3LoopSolverBridge,
    controls: dict[str, JointControl],
    config: SimConfig,
) -> None:
    if config.replay_policy_dump is None:
        raise ValueError("config.replay_policy_dump is required")
    dump = load_policy_dump_npz(config.replay_policy_dump)
    num_frames = int(dump["q_state_29"].shape[0])
    if config.reset_frame >= num_frames:
        raise SystemExit(f"--reset-frame {config.reset_frame} outside dump with {num_frames} frames")
    if config.replay_command == "q_des_31" and "q_des_31" not in dump:
        raise SystemExit("--replay-command q_des_31 requested, but dump NPZ lacks q_des_31")
    if config.replay_command == "q_des_29" and "q_des_29" not in dump:
        raise SystemExit("--replay-command q_des_29 requested, but dump NPZ lacks q_des_29")

    max_available = num_frames - config.reset_frame
    horizon = max_available if config.replay_horizon is None else min(config.replay_horizon, max_available)
    if config.max_policy_steps is not None:
        horizon = min(horizon, config.max_policy_steps)
    if horizon <= 0:
        raise SystemExit("replay horizon is empty")

    set_replay_reset_state_from_dump(
        model,
        data,
        runtime,
        solver,
        dump,
        config.reset_frame,
    )
    policy_decimation = max(1, int(round(POLICY_DT / model.opt.timestep)))
    metrics = RolloutMetricsCollector(runtime, model)
    metrics.start_motion(config.replay_policy_dump.name)
    print(
        "[replay] running recorded commands: "
        f"frames={config.reset_frame}..{config.reset_frame + horizon - 1}, "
        f"decimation={policy_decimation}, command={config.replay_command}"
    )

    for step in range(horizon):
        frame = config.reset_frame + step
        q_des_mj29 = replay_command_q_des_mj29(dump, frame, config.replay_command)
        q_des_il = q_des_mj29[runtime.mapping.mj29_to_il]
        target_by_joint = target_il_to_actuator_targets(runtime.mapping, q_des_il)
        for _substep in range(policy_decimation):
            apply_loop_pd_control(
                model,
                data,
                runtime,
                solver,
                controls,
                target_by_joint,
            )
            mujoco.mj_step(model, data)
            set_loop_heads_zero(data, runtime)
        mujoco.mj_forward(model, data)
        q_state_il = get_loop_joint_q_il(data, runtime, solver)
        ref_q_il = np.asarray(dump["q_state_29"][frame], dtype=np.float64)[runtime.mapping.mj29_to_il]
        raw_action = dump["raw_action_29"][frame] if "raw_action_29" in dump else None
        metrics.record(
            policy_tick=step,
            time_s=step * POLICY_DT,
            data=data,
            q_state_il=q_state_il,
            q_des_il=q_des_il,
            raw_action_il=raw_action,
            reference_q_il=ref_q_il,
        )

    payload = metrics.metrics_payload()
    payload["replay"] = {
        "source_npz": str(config.replay_policy_dump),
        "reset_frame": config.reset_frame,
        "replay_horizon": horizon,
        "replay_command": config.replay_command,
    }
    if config.metrics_out is not None:
        json_write(config.metrics_out, payload)
        print(f"[metrics] wrote {display_path(config.metrics_out)}")
    else:
        motion = payload["motions"][0]
        print(
            "[metrics] replay summary: "
            f"steps={motion['num_policy_steps']} fall={motion['fall']} "
            f"root_min={motion['root_height']['min']} "
            f"all_29_rmse={motion['tracking']['all_29_rmse']}"
        )
    if config.timeseries_out is not None:
        timeseries = metrics.timeseries_payload()
        timeseries["replay"] = payload["replay"]
        json_write(config.timeseries_out, timeseries)
        print(f"[metrics] wrote timeseries {display_path(config.timeseries_out)}")


# =============================================================================
# Simulation Runner
# =============================================================================

@dataclass
class ViewerState:
    paused: bool
    pending_policy_forward: int = 0
    reset_realtime_clock: bool = False


class LoopSimRunner:
    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        runtime: LoopRuntime,
        motion_paths: list[Path],
        reference_cache: AsyncLRUCache[int, MotionReference],
        initial_reference: MotionReference,
        solver: A3LoopSolverBridge,
        policy: A3Policy,
        controls: dict[str, JointControl],
        device: torch.device,
        config: SimConfig,
        video_references: list[MotionReference] | None = None,
    ) -> None:
        if not motion_paths:
            raise ValueError("at least one reference motion is required")

        self.model = model
        self.data = data
        self.runtime = runtime
        self.motion_paths = motion_paths
        self.reference_cache = reference_cache
        self.reference = initial_reference
        self.video_references = video_references
        self.solver = solver
        self.policy = policy
        self.controls = controls
        self.device = device
        self.config = config

        self.init_frame = INIT_FRAME
        if self.init_frame < 0 or self.init_frame >= self.reference.num_frames:
            raise ValueError(f"INIT_FRAME must be in [0, {self.reference.num_frames - 1}], got {self.init_frame}")

        self.playlist_mode = len(motion_paths) > 1
        self.policy_decimation = max(1, int(round(POLICY_DT / model.opt.timestep)))
        self.action_delay_substeps = int(round(self.config.action_delay_ms / 1000.0 / self.model.opt.timestep))
        self._delay_buf: deque[dict[str, float]] = deque(maxlen=max(1, self.action_delay_substeps + 1))
        self._reset_action_delay_buffer()
        self._warn_if_policy_timing_drifted()

        self.current_ref_idx = 0
        self.current_ref_frame = self.init_frame
        self.last_action_il = np.zeros(NUM_POLICY_DOFS, dtype=np.float64)
        self.history: dict[str, deque[np.ndarray]] = {}

        self.video_recorder: AsyncVideoRecorder | None = None
        self.video_source_frames = 0
        self.viewer_ctx = None
        self.viewer = None
        self.viewer_threads: list[threading.Thread] = []
        self.viewer_camera_initialized = False
        self.viewer_mirror = build_viewer_mirror(config.mjcf, model, REF_ALPHA, REF_TINT)
        self.viewer_model = self.viewer_mirror.model
        self.viewer_data = self.viewer_mirror.data
        self.viewer_track_body_id = self.viewer_mirror.root_body_id
        self.viewer_state = ViewerState(paused=config.start_paused)
        self.viewer_commands = PlaybackCommandQueue()
        self.next_wall_time: float | None = None
        self.progress = TerminalProgress(interval=PROGRESS_INTERVAL, width=PROGRESS_BAR_WIDTH)
        self.progress_detail = ""
        self.current_motion_policy_steps = 0
        self.metrics = (
            RolloutMetricsCollector(runtime, model)
            if config.metrics_out is not None or config.timeseries_out is not None
            else None
        )

    def run(self) -> None:
        try:
            self._reset_to_current_reference()
            self._start_metrics_motion()
            self._start_video()
            if self.config.batch_once:
                self._prime_batch()
            else:
                self._prime_viewer()
            self._run_policy_steps()
        finally:
            self._close()

    def _warn_if_policy_timing_drifted(self) -> None:
        if np.isclose(self.policy_decimation * self.model.opt.timestep, POLICY_DT, atol=1e-9):
            return
        print(
            f"[timing] warning: decimation={self.policy_decimation} * timestep={self.model.opt.timestep:.6f} "
            f"!= policy_dt={POLICY_DT:.6f}"
        )

    def _reset_action_delay_buffer(self) -> None:
        self._delay_buf.clear()
        if self.action_delay_substeps <= 0:
            return
        default_target = target_il_to_actuator_targets(
            self.runtime.mapping,
            self.runtime.mapping.default_q_il,
        )
        for _ in range(self.action_delay_substeps + 1):
            self._delay_buf.append(default_target)

    def _reset_to_current_reference(self) -> None:
        self.last_action_il = np.zeros(NUM_POLICY_DOFS, dtype=np.float64)
        self._reset_action_delay_buffer()
        reset_loop_to_reference(self.model, self.data, self.runtime, self.reference, self.current_ref_frame)
        self.history = prime_history(
            build_loop_current_obs_terms(self.model, self.data, self.runtime, self.solver, self.last_action_il)
        )
        self.progress_detail = ""
        self.current_motion_policy_steps = 0

    def _start_metrics_motion(self) -> None:
        if self.metrics is not None:
            self.metrics.start_motion(self.reference.path.name)

    def _finish_progress_line(self) -> None:
        self.progress.finish_line()

    def _print_after_progress(self, message: str) -> None:
        self.progress.print(message)

    def _render_progress(self, *, force: bool = False) -> None:
        frame = self._clamp_reference_frame(self.reference, self.current_ref_frame)
        status = "paused" if self.viewer_state.paused else "running"
        self.progress.render(
            clip_index=self.current_ref_idx,
            clip_count=len(self.motion_paths),
            name=self.reference.path.name,
            frame=frame,
            total_frames=self.reference.num_frames,
            status=status,
            detail=self.progress_detail,
            force=force,
        )

    @staticmethod
    def _clamp_reference_frame(reference: MotionReference, frame: int) -> int:
        return min(max(int(frame), 0), reference.num_frames - 1)

    def _prefetch_neighbor_references(self) -> None:
        if len(self.motion_paths) <= 1:
            return
        self.reference_cache.prefetch(
            [
                (self.current_ref_idx + 1) % len(self.motion_paths),
                (self.current_ref_idx - 1) % len(self.motion_paths),
            ]
        )

    def _load_reference(self, ref_idx: int) -> tuple[int, MotionReference]:
        ref_idx = int(ref_idx) % len(self.motion_paths)
        if not self.reference_cache.is_cached(ref_idx):
            state = "waiting for" if self.reference_cache.is_pending(ref_idx) else "loading"
            self._print_after_progress(
                f"[playlist] {state} clip {ref_idx + 1}/{len(self.motion_paths)}: "
                f"{self.motion_paths[ref_idx].name} ..."
            )
        reference = self.reference_cache.get(ref_idx)
        return ref_idx, reference

    def _select_reference(self, ref_idx: int, frame: int = 0, *, announce: bool = True) -> None:
        if self.metrics is not None:
            self.metrics.finish_current()
        self.current_ref_idx, self.reference = self._load_reference(ref_idx)
        self.current_ref_frame = self._clamp_reference_frame(self.reference, frame)
        self._reset_to_current_reference()
        self._start_metrics_motion()
        self.viewer_state.reset_realtime_clock = True
        self._prefetch_neighbor_references()
        if announce:
            self._print_after_progress(
                f"[playlist] clip {self.current_ref_idx + 1}/{len(self.motion_paths)}: "
                f"{self.reference.path.name} frame={self.current_ref_frame + 1}/{self.reference.num_frames}"
            )
        self._render_progress(force=True)

    def _seek_current_reference(self, frame: int, *, announce: bool = True) -> None:
        self.current_ref_frame = self._clamp_reference_frame(self.reference, frame)
        self._reset_to_current_reference()
        self.viewer_state.reset_realtime_clock = True
        if announce:
            self._print_after_progress(
                f"[viewer] {self.reference.path.name} "
                f"frame={self.current_ref_frame + 1}/{self.reference.num_frames} (sim reset)"
            )
        self._render_progress(force=True)

    def _drain_viewer_commands(self) -> None:
        for command in self.viewer_commands.drain():
            self._apply_viewer_command(command.name, command.value)

    def _apply_viewer_command(self, command: str, value: int) -> None:
        if command == CMD_TOGGLE_PAUSE:
            was_paused = self.viewer_state.paused
            self.viewer_state.paused = not was_paused
            if was_paused and not self.viewer_state.paused:
                self.viewer_state.reset_realtime_clock = True
            state = "paused" if self.viewer_state.paused else "running"
            self._print_after_progress(f"[viewer] {state}")
            self._render_progress(force=True)
            return

        if command == CMD_STEP_FORWARD:
            self.viewer_state.paused = True
            self.viewer_state.pending_policy_forward += max(1, int(value))
            self.viewer_state.reset_realtime_clock = True
            self._render_progress(force=True)
            return

        if command == CMD_STEP_BACKWARD:
            self.viewer_state.paused = True
            self._seek_current_reference(self.current_ref_frame - int(value))
            return

        if command == CMD_RESET:
            self._seek_current_reference(0)
            return

        if command == CMD_NEXT_CLIP:
            self._select_reference(self.current_ref_idx + int(value), 0)
            return

        if command == CMD_PREV_CLIP:
            self._select_reference(self.current_ref_idx - int(value), 0)
            return

        self._print_after_progress(f"[viewer] warning: ignored unknown command {command!r}")

    def _start_video(self) -> None:
        if self.config.output_video is None:
            return
        overlay_config = VideoOverlayConfig(
            references=self.video_references if self.video_references is not None else [self.reference],
            camera_distance=CAMERA_DISTANCE,
            camera_azimuth=CAMERA_AZIMUTH,
            camera_elevation=CAMERA_ELEVATION,
        )
        self.video_recorder = AsyncVideoRecorder(
            model_path=self.config.mjcf,
            path=self.config.output_video,
            video_config=self.config.video,
            overlay_config=overlay_config,
        )
        print(
            f"[video] recording {self.config.video.width}x{self.config.video.height} "
            f"at {self.config.video.fps:.1f} FPS "
            f"(frame_stride={self.config.video.frame_stride}) "
            f"as H.264/{self.config.video.encoder} preset={self.config.video.preset} "
            f"crf={self.config.video.crf} "
            f"to {self.video_recorder.path} (isolated process)"
        )

    def _prime_viewer(self) -> None:
        print("[viewer] transparent reference robot uses an embedded viewer-only model")
        initial_ref_frame = reference_frame_index(self.reference, self.init_frame, DEFAULT_REFERENCE_ON_END)
        self._write_video_frame(initial_ref_frame)
        self._sync_viewer_data(initial_ref_frame)
        self._launch_viewer()

    def _prime_batch(self) -> None:
        initial_ref_frame = reference_frame_index(self.reference, self.init_frame, DEFAULT_REFERENCE_ON_END)
        self._write_video_frame(initial_ref_frame)
        print("[batch] running each motion clip once without the passive viewer")

    def _launch_viewer(self) -> None:
        import glfw
        from mujoco import viewer as mujoco_viewer

        def key_callback(keycode: int) -> None:
            self.viewer_commands.enqueue_keycode(
                keycode,
                step_forward_keycodes=(glfw.KEY_RIGHT,),
                step_backward_keycodes=(glfw.KEY_LEFT,),
            )

        # launch_passive() closes asynchronously; keep its thread so teardown
        # can wait before MjModel/MjData wrappers are released.
        threads_before = {thread.ident for thread in threading.enumerate()}
        viewer_ctx = mujoco_viewer.launch_passive(
            self.viewer_model, self.viewer_data, key_callback=key_callback
        )
        viewer = viewer_ctx.__enter__()
        self.viewer_threads = [
            thread
            for thread in threading.enumerate()
            if thread.ident not in threads_before and thread is not threading.current_thread()
        ]
        self.viewer_ctx = viewer_ctx
        self.viewer = viewer
        print(
            "[viewer] controls: Space pause/resume, ./Right +1 policy step, ,/Left rewind+reset, "
            "= next clip, - previous clip, R reset; close the window to stop"
        )
        if self.viewer_state.paused:
            print("[viewer] paused at startup")

    def _run_policy_steps(self) -> None:
        self._prefetch_neighbor_references()
        self._render_progress(force=True)
        policy_step = 0
        while True:
            if not self.config.batch_once:
                self._drain_viewer_commands()
            ref_frame = self._ref_frame_for_step(policy_step)
            if ref_frame is None:
                break
            if not self.config.batch_once:
                if not self._wait_while_paused(ref_frame):
                    break
                self._drain_viewer_commands()
                ref_frame = self._ref_frame_for_step(policy_step)
                if ref_frame is None:
                    break
                if self.viewer is None or not self.viewer.is_running():
                    break
                self._prepare_realtime_clock()

            sim_time_before = float(self.data.time)
            step_result = self._step_policy(ref_frame)
            self._record_metrics_step(policy_step, ref_frame, step_result)
            self.current_motion_policy_steps += 1
            self._update_progress_detail(policy_step, ref_frame, step_result.raw_action_il)
            self._write_video_frame(ref_frame)

            if not self.config.batch_once and not self._sync_viewer(ref_frame):
                break
            self._render_progress()
            if not self.config.batch_once:
                self._pace_after_step(sim_time_before)
            if not self._advance_playlist_if_needed(policy_step):
                break
            policy_step += 1

    def _ref_frame_for_step(self, policy_step: int) -> int | None:
        del policy_step
        return reference_frame_index(self.reference, self.current_ref_frame, DEFAULT_REFERENCE_ON_END)

    def _wait_while_paused(self, ref_frame: int) -> bool:
        while self.viewer_state.paused:
            self._drain_viewer_commands()
            if self.viewer_state.pending_policy_forward > 0:
                self.viewer_state.pending_policy_forward -= 1
                return True
            ref_frame = self._ref_frame_for_step(0)
            if not self._sync_viewer(ref_frame):
                return False
            self._render_progress()
            time.sleep(0.01)
        return True

    def _prepare_realtime_clock(self) -> None:
        if self.viewer_state.paused:
            return
        if self.viewer_state.reset_realtime_clock:
            self.next_wall_time = time.perf_counter()
            self.viewer_state.reset_realtime_clock = False
        if self.next_wall_time is None:
            self.next_wall_time = time.perf_counter()

    def _step_policy(self, ref_frame: int) -> PolicyStepResult:
        anchor_quat = body_quat_wxyz(self.data, self.runtime.mapping.anchor_body_id)
        encoder_input = build_encoder_input(
            self.reference,
            ref_frame,
            anchor_quat,
            DEFAULT_REFERENCE_ON_END,
            self.config.future_frame_skip,
            self.config.future_history_frames,
            self.config.future_valid_frames,
            self.config.future_zero_pad,
        )
        actor_obs = flatten_history(self.history)
        raw_action_il = self.policy.act(encoder_input, actor_obs, self.device).astype(np.float64)
        raw_action_il = np.clip(raw_action_il, -ACTION_CLIP, ACTION_CLIP)
        target_il = self.runtime.mapping.default_q_il + raw_action_il * self.runtime.mapping.action_scale_il
        target_by_joint = target_il_to_actuator_targets(self.runtime.mapping, target_il)

        if self.action_delay_substeps <= 0:
            for _substep in range(self.policy_decimation):
                apply_loop_pd_control(
                    self.model,
                    self.data,
                    self.runtime,
                    self.solver,
                    self.controls,
                    target_by_joint,
                )
                mujoco.mj_step(self.model, self.data)
                set_loop_heads_zero(self.data, self.runtime)
        else:
            for _substep in range(self.policy_decimation):
                self._delay_buf.append(target_by_joint)
                applied_target = self._delay_buf[0]
                apply_loop_pd_control(
                    self.model,
                    self.data,
                    self.runtime,
                    self.solver,
                    self.controls,
                    applied_target,
                )
                mujoco.mj_step(self.model, self.data)
                set_loop_heads_zero(self.data, self.runtime)
        mujoco.mj_forward(self.model, self.data)

        self.last_action_il = raw_action_il.copy()
        append_history(
            self.history,
            build_loop_current_obs_terms(self.model, self.data, self.runtime, self.solver, self.last_action_il),
        )
        return PolicyStepResult(
            raw_action_il=raw_action_il,
            q_des_il=target_il.copy(),
            actor_obs=actor_obs.copy(),
        )

    def _record_metrics_step(
        self,
        policy_step: int,
        ref_frame: int,
        step_result: PolicyStepResult,
    ) -> None:
        if self.metrics is None:
            return
        q_state_il = None
        ref_q_il = None
        try:
            q_state_il = get_loop_joint_q_il(self.data, self.runtime, self.solver)
        except Exception:
            q_state_il = None
        try:
            ref_q_il = self.reference.dof_il[int(ref_frame)]
        except Exception:
            ref_q_il = None
        self.metrics.record(
            policy_tick=self.current_motion_policy_steps,
            time_s=self.current_motion_policy_steps * POLICY_DT,
            data=self.data,
            q_state_il=q_state_il,
            q_des_il=step_result.q_des_il,
            raw_action_il=step_result.raw_action_il,
            reference_q_il=ref_q_il,
            reference=self.reference,
            ref_frame=ref_frame,
        )

    def _update_progress_detail(self, policy_step: int, ref_frame: int, raw_action_il: np.ndarray) -> None:
        joint_q_il = get_loop_joint_q_il(self.data, self.runtime, self.solver)
        root_err = float(np.linalg.norm(self.data.qpos[:3] - self.reference.root_pos[ref_frame]))
        joint_err = float(np.mean(np.abs(joint_q_il - self.reference.dof_il[ref_frame])))
        self.progress_detail = (
            f"step={policy_step:05d} ref={ref_frame:04d} "
            f"root_err={root_err:.4f} joint_l1={joint_err:.4f} "
            f"action_abs_max={np.max(np.abs(raw_action_il)):.3f}"
        )

    def _write_video_frame(self, ref_frame: int | None) -> None:
        if self.video_recorder is not None:
            if self.video_source_frames % self.config.video.frame_stride == 0:
                self.video_recorder.write(self.data, self.current_ref_idx, ref_frame)
            self.video_source_frames += 1

    def _sync_viewer_data(self, ref_frame: int | None) -> None:
        self.viewer_data.time = float(self.data.time)
        self.viewer_data.qpos[: self.viewer_mirror.sim_qpos_width] = self.data.qpos
        self.viewer_data.qvel[: self.viewer_mirror.sim_qvel_width] = self.data.qvel
        if self.data.ctrl.size and self.viewer_data.ctrl.size:
            width = min(self.data.ctrl.size, self.viewer_data.ctrl.size)
            self.viewer_data.ctrl[:width] = self.data.ctrl[:width]
        if ref_frame is not None:
            ref_qpos = self.reference.qpos[int(ref_frame)].copy()
            ref_qpos[:3] += np.asarray(REF_OFFSET, dtype=np.float64)
            ref_qvel = self.reference.qvel[int(ref_frame)]
            qpos_start = self.viewer_mirror.ref_qpos_start
            qvel_start = self.viewer_mirror.ref_qvel_start
            self.viewer_data.qpos[qpos_start : qpos_start + self.viewer_mirror.sim_qpos_width] = ref_qpos
            self.viewer_data.qvel[qvel_start : qvel_start + self.viewer_mirror.sim_qvel_width] = ref_qvel
        mujoco.mj_forward(self.viewer_model, self.viewer_data)

    def _sync_viewer(self, ref_frame: int | None) -> bool:
        if self.viewer is None:
            return False
        if not self.viewer.is_running():
            return False
        with self.viewer.lock():
            self._sync_viewer_data(ref_frame)
            if not self.viewer_camera_initialized:
                self.viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
                self.viewer.cam.fixedcamid = -1
                self.viewer.cam.trackbodyid = self.viewer_track_body_id
                self.viewer.cam.distance = CAMERA_DISTANCE
                self.viewer.cam.azimuth = CAMERA_AZIMUTH
                self.viewer.cam.elevation = CAMERA_ELEVATION
                self.viewer_camera_initialized = True
        self.viewer.sync()
        return True

    def _pace_after_step(self, sim_time_before: float) -> None:
        if self.viewer_state.paused:
            return
        sim_dt = max(float(self.data.time) - sim_time_before, 0.0)
        if sim_dt <= 0.0:
            return
        if self.next_wall_time is None:
            self.next_wall_time = time.perf_counter()
        self.next_wall_time += sim_dt
        sleep_time = self.next_wall_time - time.perf_counter()
        if sleep_time > 0.0:
            time.sleep(sleep_time)
        else:
            self.next_wall_time = time.perf_counter()

    def _advance_playlist_if_needed(self, policy_step: int) -> bool:
        del policy_step
        if (
            self.config.max_policy_steps is not None
            and self.current_motion_policy_steps >= self.config.max_policy_steps
        ):
            if self.config.batch_once:
                next_idx = self.current_ref_idx + 1
                if next_idx >= len(self.motion_paths):
                    return False
                self._select_reference(next_idx, 0)
                return True
            return False
        self.current_ref_frame += 1
        if self.current_ref_frame < self.reference.num_frames:
            return True
        if self.config.batch_once:
            next_idx = self.current_ref_idx + 1
            if next_idx >= len(self.motion_paths):
                return False
            self._select_reference(next_idx, 0)
            return True
        if not self.playlist_mode:
            self.current_ref_frame = 0
            self._reset_to_current_reference()
            return True
        self._select_reference(self.current_ref_idx + 1, 0)
        return True

    def _close(self) -> None:
        self._finish_progress_line()
        if self.viewer_ctx is not None:
            self.viewer_ctx.__exit__(None, None, None)
            self.viewer_ctx = None
            self.viewer = None
            # Handle.close() only asks the viewer to exit. Joining prevents a
            # process-exit race in GLFW/MuJoCo native cleanup.
            for thread in self.viewer_threads:
                thread.join(timeout=2.0)
                if thread.is_alive():
                    print("[viewer] warning: viewer thread did not exit within 2 seconds")
            self.viewer_threads = []
        if self.video_recorder is not None:
            self.video_recorder.close()
            print(f"[video] wrote {self.video_recorder.frames} frames to {self.video_recorder.path}")
        if self.metrics is not None:
            if self.config.metrics_out is not None:
                payload = self.metrics.metrics_payload()
                json_write(self.config.metrics_out, payload)
                print(f"[metrics] wrote {display_path(self.config.metrics_out)}")
            if self.config.timeseries_out is not None:
                payload = self.metrics.timeseries_payload()
                json_write(self.config.timeseries_out, payload)
                print(f"[metrics] wrote timeseries {display_path(self.config.timeseries_out)}")


def run_loop_sim(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    runtime: LoopRuntime,
    motion_paths: list[Path],
    reference_cache: AsyncLRUCache[int, MotionReference],
    initial_reference: MotionReference,
    solver: A3LoopSolverBridge,
    policy: A3Policy,
    controls: dict[str, JointControl],
    device: torch.device,
    config: SimConfig,
    video_references: list[MotionReference] | None = None,
) -> None:
    LoopSimRunner(
        model,
        data,
        runtime,
        motion_paths,
        reference_cache,
        initial_reference,
        solver,
        policy,
        controls,
        device,
        config,
        video_references,
    ).run()


# =============================================================================
# CLI And Entry Point
# =============================================================================

def print_loop_summary(
    config: SimConfig,
    model: mujoco.MjModel,
    runtime: LoopRuntime,
    motion_paths: list[Path],
    initial_reference: MotionReference,
    policy: A3Policy | None,
    device: torch.device | None,
) -> None:
    mapping = runtime.mapping
    decimation = max(1, int(round(POLICY_DT / model.opt.timestep)))
    action_delay_substeps = int(round(config.action_delay_ms / 1000.0 / model.opt.timestep))
    print(f"[loop-run] checkpoint: {display_path(config.checkpoint)}")
    print(f"[loop-run] motion: {', '.join(display_path(path) for path in motion_paths)}")
    print(f"[loop-run] initial reference: {initial_reference.path.name} frames={initial_reference.num_frames}")
    print(f"[loop-run] URDF: {display_path(config.urdf)}")
    print(f"[loop-run] loop MJCF: {display_path(config.mjcf)}")
    print(f"[loop-run] solver lib: {display_path(DEFAULT_SOLVER_LIB)}")
    if config.output_video is not None:
        print(f"[loop-run] video: {display_path(config.output_video)}")
        print(
            "[loop-run] video config: "
            f"{config.video.width}x{config.video.height}, fps={config.video.fps:.1f}, "
            f"frame_stride={config.video.frame_stride}, encoder={config.video.encoder}, "
            f"preset={config.video.preset}, crf={config.video.crf}"
        )
    print(
        "[loop-run] timing: "
        f"target_fps={TARGET_FPS:.1f}, model_timestep={model.opt.timestep:.4f}, "
        f"policy_decimation={decimation}, encoder={config.encoder}, "
        f"future_frame_skip={config.future_frame_skip}, "
        f"future_history_frames={config.future_history_frames}, "
        f"future_valid_frames={config.future_valid_frames}, "
        f"on_end={DEFAULT_REFERENCE_ON_END}, "
        f"action_delay_ms={config.action_delay_ms:.1f} (={action_delay_substeps} substeps)"
    )
    print(f"[loop-run] joints: logical_policy={len(mapping.mj29_joint_names)} total_model_joints={model.njnt}")
    print(f"[loop-run] anchor body: {mapping.anchor_body_name} (id={mapping.anchor_body_id}), root body: pelvis_link (id={mapping.root_body_id})")
    print("[loop-run] loop joints read via solver: waist pitch/roll, left/right ankle pitch/roll")
    if policy is not None and device is not None:
        print(f"[loop-run] device: {device}")
        print(f"[loop-run] actor obs dim: {policy.actor_obs_dim}")
        print(
            f"[loop-run] encoder: {policy.encoder_input_dim}D = "
            f"{NUM_FUTURE_FRAMES} x {policy.encoder_frame_dim} {policy.encoder_terms} "
            f"(source={policy.encoder_name})"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help=(
            "PT checkpoint. Required; pass the released 024 step-100k PT "
            "or a compatible replacement."
        ),
    )
    parser.add_argument(
        "--motion",
        type=Path,
        default=None,
        help=f"A3 flat CSV file or directory of CSV clips. Default: {display_path(DEFAULT_MOTION)}",
    )
    parser.add_argument(
        "--urdf",
        type=Path,
        default=None,
        help=f"URDF used to derive the 29-DOF policy joint mapping. Default: {display_path(DEFAULT_URDF)}",
    )
    parser.add_argument(
        "--mjcf",
        type=Path,
        default=None,
        help=f"Loop MJCF to simulate. Default: {display_path(DEFAULT_LOOP_MJCF)}",
    )
    parser.add_argument(
        "--terrain-profile",
        type=Path,
        default=None,
        help="GRAIL terrain_profile/<stem>.json used to add matching stair collision boxes.",
    )
    parser.add_argument(
        "--foot-spring-cfg",
        type=Path,
        default=None,
        help=(
            "YAML with passive compliant-foot joint stiffness/damping/ref values. "
            "Useful with the compliant-foot draft MJCF."
        ),
    )
    parser.add_argument(
        "--foot-stiffness-scale",
        type=float,
        default=1.0,
        help=(
            "Scale passive compliant-foot joint stiffness values in the MJCF "
            "or --foot-spring-cfg."
        ),
    )
    parser.add_argument(
        "--foot-damping-scale",
        type=float,
        default=1.0,
        help=(
            "Scale passive compliant-foot joint damping values in the MJCF "
            "or --foot-spring-cfg."
        ),
    )
    parser.add_argument(
        "--random",
        action="store_true",
        help="Shuffle CSV clip order at startup when --motion points to a directory.",
    )
    parser.add_argument(
        "--cache-size",
        type=int,
        default=4,
        help="Number of prepared reference clips to keep while browsing a directory.",
    )
    parser.add_argument(
        "--csv-source-fps",
        type=float,
        default=DEFAULT_CSV_SOURCE_FPS,
        help=(
            "Reference FPS after CSV row selection. Default: 30 (legacy 120fps CSVs use "
            "this with --csv-frame-stride 4; native 30fps CSVs require stride 1)."
        ),
    )
    parser.add_argument(
        "--csv-frame-stride",
        type=int,
        default=DEFAULT_CSV_FRAME_STRIDE,
        help=(
            "Keep every Nth CSV row before resampling. Default: 4 for legacy 120fps input; "
            "use 1 for native 30fps input."
        ),
    )
    parser.add_argument(
        "--anchor-body",
        type=str,
        default="pelvis_link",
        help="Anchor body for base_ang_vel/gravity_dir/encoder ori_6d/metrics frame. Use torso_Link for exp007 torso-IMU ckpts.",
    )
    parser.add_argument("--start-paused", action="store_true", help="Start the MuJoCo viewer paused.")
    parser.add_argument(
        "--encoder-mode",
        choices=("g1", "a3_fast", "a3_fast_100ms", "a3_fast_100ms_zero", "a3_fast_history"),
        default=None,
        help=(
            "Convenience preset that sets --encoder/--future-frame-skip/"
            "--future-history-frames together. 'g1' = g1 encoder, 0.1s all-future "
            "(legacy default). 'a3_fast' = a3_fast encoder, 0.02s all-future "
            "(standard low-latency). 'a3_fast_100ms' = five 0.02s future "
            "samples with the fifth repeated into slots 6-10. "
            "'a3_fast_100ms_zero' = the same five samples with slots 6-10 zeroed. "
            "'a3_fast_history' = a3_fast encoder, 0.02s, "
            "5 history + 5 future (no-latency experiments like exp008). "
            "Explicit --encoder/--future-frame-skip/--future-history-frames/"
            "--future-valid-frames override "
            "the preset. Default: None (falls back to individual flags, = g1)."
        ),
    )
    parser.add_argument(
        "--encoder",
        choices=("g1", "a3_fast"),
        default="g1",
        help=(
            "Which backbone encoder to load from the checkpoint. "
            "Default: g1 (10-frame all-future at 0.1s spacing). "
            "Use a3_fast for no-latency experiments trained with the short "
            "0.02s history+future window."
        ),
    )
    parser.add_argument(
        "--future-frame-skip",
        type=int,
        default=FUTURE_FRAME_SKIP,
        help=(
            f"Reference-frame spacing for the encoder token window. "
            f"Default: {FUTURE_FRAME_SKIP} (=0.1s at {TARGET_FPS:.0f}fps, the g1 standard). "
            "Use 1 (0.02s) for the a3_fast no-latency encoder."
        ),
    )
    parser.add_argument(
        "--future-history-frames",
        type=int,
        default=0,
        help=(
            "Number of past reference frames (out of the 10-frame window) shifted "
            "behind the current frame. Default: 0 (all-future). Use 5 for the "
            "a3_fast no-latency encoder (5 history + 5 future)."
        ),
    )
    parser.add_argument(
        "--future-valid-frames",
        type=int,
        default=None,
        help=(
            "Number of distinct future samples placed in the 10-slot encoder window; "
            "remaining slots repeat the last valid sample. Default: all 10. Use 5 "
            "for the distilled a3_fast_100ms encoder."
        ),
    )
    parser.add_argument(
        "--future-zero-pad",
        action="store_true",
        help="Zero encoder slots after --future-valid-frames instead of repeating the last sample.",
    )
    parser.add_argument(
        "--batch-once",
        action="store_true",
        help=(
            "Run each requested CSV clip exactly once without launching the passive viewer. "
            "Use with --output-video for unattended batch recordings."
        ),
    )
    parser.add_argument(
        "--max-policy-steps",
        type=int,
        default=None,
        help="Stop each clip after N policy steps. Default: run the existing full clip/viewer behavior.",
    )
    parser.add_argument(
        "--action-delay-ms",
        type=float,
        default=0.0,
        help=(
            "Actuator transport delay in milliseconds. The policy target computed each control "
            "step is applied after this delay (rounded to whole physics substeps of "
            "model.opt.timestep). 0 = no delay (default, ideal actuation). Typical A3 hardware "
            "value: 10-15ms."
        ),
    )
    parser.add_argument(
        "--metrics-out",
        type=Path,
        default=None,
        help="Write per-motion rollout metrics as JSON.",
    )
    parser.add_argument(
        "--timeseries-out",
        type=Path,
        default=None,
        help="Write per-policy-step rollout timeseries as JSON.",
    )
    parser.add_argument(
        "--replay-policy-dump",
        type=Path,
        default=None,
        help="Replay a converted A3 policy dump NPZ instead of running the policy.",
    )
    parser.add_argument(
        "--reset-frame",
        type=int,
        default=0,
        help="Frame index inside --replay-policy-dump used for the MuJoCo reset. Default: 0.",
    )
    parser.add_argument(
        "--replay-horizon",
        type=int,
        default=None,
        help="Number of dump frames to replay after --reset-frame. Default: until dump end.",
    )
    parser.add_argument(
        "--replay-command",
        choices=("q_des_31", "q_des_29"),
        default="q_des_31",
        help="Recorded command array to replay from dump NPZ. Default: q_des_31.",
    )
    parser.add_argument(
        "--output-video",
        nargs="?",
        default=None,
        const=AUTO_OUTPUT_VIDEO,
        type=Path,
        help="Write an offscreen-rendered H.264 MP4. Omit the path to write next to --checkpoint.",
    )
    parser.add_argument(
        "--preview-video",
        action="store_true",
        help=(
            "Use faster preview recording defaults: "
            f"{PREVIEW_VIDEO_WIDTH}x{PREVIEW_VIDEO_HEIGHT}, "
            f"{PREVIEW_VIDEO_FPS:.0f} FPS, frame stride {PREVIEW_VIDEO_FRAME_STRIDE}, "
            f"preset={PREVIEW_VIDEO_PRESET}, crf={PREVIEW_VIDEO_CRF}."
        ),
    )
    parser.add_argument(
        "--video-width",
        type=int,
        default=None,
        help=f"Output video width. Default: {VIDEO_WIDTH}, or {PREVIEW_VIDEO_WIDTH} with --preview-video.",
    )
    parser.add_argument(
        "--video-height",
        type=int,
        default=None,
        help=f"Output video height. Default: {VIDEO_HEIGHT}, or {PREVIEW_VIDEO_HEIGHT} with --preview-video.",
    )
    parser.add_argument(
        "--video-fps",
        type=float,
        default=None,
        help=f"Output video FPS. Default: {VIDEO_FPS:.0f}, or {PREVIEW_VIDEO_FPS:.0f} with --preview-video.",
    )
    parser.add_argument(
        "--video-frame-stride",
        type=int,
        default=None,
        help=(
            f"Record every Nth policy frame. Default: {VIDEO_FRAME_STRIDE}, "
            f"or {PREVIEW_VIDEO_FRAME_STRIDE} with --preview-video."
        ),
    )
    parser.add_argument(
        "--video-encoder",
        default=VIDEO_ENCODER,
        help=f"ffmpeg video encoder. Default: {VIDEO_ENCODER}.",
    )
    parser.add_argument(
        "--video-preset",
        default=None,
        help=f"ffmpeg encoder preset. Default: {VIDEO_PRESET}, or {PREVIEW_VIDEO_PRESET} with --preview-video.",
    )
    parser.add_argument(
        "--video-crf",
        type=int,
        default=None,
        help=f"ffmpeg CRF quality. Default: {VIDEO_CRF}, or {PREVIEW_VIDEO_CRF} with --preview-video.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    config = build_sim_config(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = mujoco.MjModel.from_xml_path(str(config.mjcf))
    runtime = build_loop_runtime(model, config.urdf, anchor_body_name=config.anchor_body)
    controls = controls_for_loop(runtime.mapping)
    solver = A3LoopSolverBridge(DEFAULT_SOLVER_LIB, DEFAULT_SOLVER_INCLUDE, DEFAULT_BRIDGE_CACHE, False)
    reference_cache = None
    try:
        if config.replay_policy_dump is not None:
            print(f"[loop-run] replay NPZ: {display_path(config.replay_policy_dump)}")
            print(f"[loop-run] URDF: {display_path(config.urdf)}")
            print(f"[loop-run] loop MJCF: {display_path(config.mjcf)}")
            print(f"[loop-run] solver lib: {display_path(DEFAULT_SOLVER_LIB)}")
            print(f"[loop-run] joints: logical_policy={len(runtime.mapping.mj29_joint_names)} total_model_joints={model.njnt}")
            data = mujoco.MjData(model)
            run_policy_dump_replay(model, data, runtime, solver, controls, config)
            return

        motion_paths = resolve_motion_paths(config.motion)
        if config.random_order:
            motion_paths = shuffled_once(motion_paths)
            order_tag = "randomized " if len(motion_paths) > 1 else ""
            print(f"[playlist] {order_tag}CSV order ({len(motion_paths)} clip(s))")
            for idx, path in enumerate(motion_paths[:10]):
                print(f"  [{idx + 1}] {path.name}")
            if len(motion_paths) > 10:
                print(f"  ... and {len(motion_paths) - 10} more")
        config = resolve_loop_output_paths(config, motion_paths)

        def load_reference_for_cache(index: int) -> MotionReference:
            loader_model = mujoco.MjModel.from_xml_path(str(config.mjcf))
            loader_solver = A3LoopSolverBridge(
                DEFAULT_SOLVER_LIB,
                DEFAULT_SOLVER_INCLUDE,
                DEFAULT_BRIDGE_CACHE,
                False,
            )
            try:
                return load_loop_motion_reference(
                    loader_model,
                    runtime,
                    loader_solver,
                    motion_paths[int(index)],
                    csv_source_fps=config.csv_source_fps,
                    csv_frame_stride=config.csv_frame_stride,
                    announce=int(index) == 0,
                )
            finally:
                loader_solver.close()

        reference_cache = AsyncLRUCache(
            load_reference_for_cache,
            max_size=config.cache_size,
            max_workers=1,
        )
        print(f"[playlist] loading initial clip 1/{len(motion_paths)}: {motion_paths[0].name} ...")
        initial_reference = reference_cache.get(0)
        reference_cache.prefetch([1 % len(motion_paths)] if len(motion_paths) > 1 else [])

        video_references = None
        if config.output_video is not None:
            print("[video] preparing all reference clips for overlay recording ...")
            video_references = [reference_cache.get(idx) for idx in range(len(motion_paths))]

        policy = A3Policy(config.checkpoint, device, encoder=config.encoder)
        print_loop_summary(config, model, runtime, motion_paths, initial_reference, policy, device)
        data = mujoco.MjData(model)
        run_loop_sim(
            model,
            data,
            runtime,
            motion_paths,
            reference_cache,
            initial_reference,
            solver,
            policy,
            controls,
            device,
            config,
            video_references,
        )
    finally:
        if reference_cache is not None:
            reference_cache.close()
        solver.close()


if __name__ == "__main__":
    main()
