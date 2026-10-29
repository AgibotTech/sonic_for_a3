"""Pico (XRoboToolkit) full-body tracking source, NPZ recording and replay.

Every source yields :class:`BodyFrame` in the robot world frame (X forward, Y left,
Z up). XRoboToolkit reports 24 SMPL-ordered joints as ``[x, y, z, qx, qy, qz, qw]``
in a Y-up frame; the conversion matches SONIC's ``pico_manager_thread_server``.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import logging
from pathlib import Path
import subprocess
import threading
import time
from typing import Protocol

import numpy as np
from scipy.spatial.transform import Rotation

logger = logging.getLogger(__name__)

SMPL_JOINT_NAMES = (
    "pelvis",
    "left_hip",
    "right_hip",
    "spine1",
    "left_knee",
    "right_knee",
    "spine2",
    "left_ankle",
    "right_ankle",
    "spine3",
    "left_foot",
    "right_foot",
    "neck",
    "left_collar",
    "right_collar",
    "head",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hand",
    "right_hand",
)
SMPL_INDEX = {name: idx for idx, name in enumerate(SMPL_JOINT_NAMES)}
NUM_BODY_JOINTS = len(SMPL_JOINT_NAMES)

UNITY_TO_ROBOT = np.array([[-1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]])

BUTTON_NAMES = (
    "A",
    "B",
    "X",
    "Y",
    "left_menu",
    "right_menu",
    "left_axis_click",
    "right_axis_click",
)
ANALOG_NAMES = ("left_trigger", "right_trigger", "left_grip", "right_grip")


@dataclass(frozen=True)
class ControllerState:
    left_trigger: float = 0.0
    right_trigger: float = 0.0
    left_grip: float = 0.0
    right_grip: float = 0.0
    buttons: frozenset[str] = frozenset()
    # Preserve acquisition timing when button edges wait behind a slow IK step.
    # Timing is live-only metadata, not part of the controller's recorded state.
    sample_time: float | None = field(default=None, compare=False)

    def pressed(self, *names: str) -> bool:
        return all(name in self.buttons for name in names)

    def analog(self) -> np.ndarray:
        return np.array([getattr(self, name) for name in ANALOG_NAMES], dtype=np.float32)

    @classmethod
    def from_arrays(cls, analog: np.ndarray, buttons: np.ndarray) -> ControllerState:
        values = {name: float(analog[idx]) for idx, name in enumerate(ANALOG_NAMES)}
        pressed = frozenset(name for idx, name in enumerate(BUTTON_NAMES) if bool(buttons[idx]))
        return cls(buttons=pressed, **values)

    def button_array(self) -> np.ndarray:
        return np.array([name in self.buttons for name in BUTTON_NAMES], dtype=bool)


@dataclass
class BodyFrame:
    """One body-tracking sample in the robot world frame."""

    timestamp: float
    """Seconds on this process's ``time.monotonic()`` clock."""
    positions: np.ndarray
    """(24, 3) joint positions, metres."""
    rotations: Rotation
    """24 global joint orientations."""
    controller: ControllerState = field(default_factory=ControllerState)


def unity_poses_to_robot(body_poses: np.ndarray) -> tuple[np.ndarray, Rotation]:
    poses = np.asarray(body_poses, dtype=np.float64).reshape(NUM_BODY_JOINTS, 7)
    positions = poses[:, :3] @ UNITY_TO_ROBOT.T
    rot_unity = Rotation.from_quat(poses[:, 3:7]).as_matrix()
    rot_robot = UNITY_TO_ROBOT @ rot_unity @ UNITY_TO_ROBOT.T
    return positions, Rotation.from_matrix(rot_robot)


def robot_to_unity_poses(positions: np.ndarray, rotations: Rotation) -> np.ndarray:
    """Inverse of :func:`unity_poses_to_robot`; used to synthesise device-format data."""
    pos_unity = np.asarray(positions, dtype=np.float64) @ UNITY_TO_ROBOT
    rot_unity = UNITY_TO_ROBOT.T @ rotations.as_matrix() @ UNITY_TO_ROBOT
    quat = Rotation.from_matrix(rot_unity).as_quat()
    return np.concatenate([pos_unity, quat], axis=1)


class BodySource(Protocol):
    def start(self) -> None: ...

    def stop(self) -> None: ...

    def get_latest(self) -> BodyFrame | None: ...

    @property
    def disconnected(self) -> bool: ...


class BodyRecording:
    """In-memory recording of raw device samples, saved as NPZ.

    Arrays: ``body_poses_unity`` (T, 24, 7), ``timestamps`` (T,) seconds from the first
    sample, ``controller_analog`` (T, 4), ``controller_buttons`` (T, 8) and the scalar
    ``calib_index`` (-1 when unknown).
    """

    def __init__(self) -> None:
        self._poses: list[np.ndarray] = []
        self._stamps: list[float] = []
        self._analog: list[np.ndarray] = []
        self._buttons: list[np.ndarray] = []
        self.calib_index = -1
        self._lock = threading.Lock()

    def __len__(self) -> int:
        return len(self._stamps)

    def append(
        self, body_poses_unity: np.ndarray, timestamp: float, controller: ControllerState
    ) -> None:
        with self._lock:
            self._poses.append(
                np.asarray(body_poses_unity, dtype=np.float32).reshape(NUM_BODY_JOINTS, 7)
            )
            self._stamps.append(float(timestamp))
            self._analog.append(controller.analog())
            self._buttons.append(controller.button_array())

    def mark_calibration(self) -> None:
        with self._lock:
            self.calib_index = max(len(self._stamps) - 1, 0)

    def save(self, path: Path) -> None:
        with self._lock:
            if not self._stamps:
                raise ValueError("recording is empty")
            stamps = np.asarray(self._stamps, dtype=np.float64)
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                path,
                body_poses_unity=np.stack(self._poses),
                timestamps=stamps - stamps[0],
                controller_analog=np.stack(self._analog),
                controller_buttons=np.stack(self._buttons),
                calib_index=np.int64(self.calib_index),
            )


@dataclass
class RecordedBody:
    body_poses_unity: np.ndarray
    timestamps: np.ndarray
    controller_analog: np.ndarray
    controller_buttons: np.ndarray
    calib_index: int

    @classmethod
    def load(cls, path: Path) -> RecordedBody:
        with np.load(path) as npz:
            poses = np.asarray(npz["body_poses_unity"], dtype=np.float64)
            count = poses.shape[0]
            analog = (
                npz["controller_analog"]
                if "controller_analog" in npz
                else np.zeros((count, len(ANALOG_NAMES)))
            )
            buttons = (
                npz["controller_buttons"]
                if "controller_buttons" in npz
                else np.zeros((count, len(BUTTON_NAMES)), dtype=bool)
            )
            calib_index = int(npz["calib_index"]) if "calib_index" in npz else -1
            return cls(
                body_poses_unity=poses,
                timestamps=np.asarray(npz["timestamps"], dtype=np.float64),
                controller_analog=np.asarray(analog, dtype=np.float32),
                controller_buttons=np.asarray(buttons, dtype=bool),
                calib_index=calib_index,
            )

    def __len__(self) -> int:
        return int(self.timestamps.shape[0])

    @property
    def duration(self) -> float:
        return float(self.timestamps[-1] - self.timestamps[0]) if len(self) else 0.0

    def frame(self, index: int, timestamp: float | None = None) -> BodyFrame:
        positions, rotations = unity_poses_to_robot(self.body_poses_unity[index])
        controller = ControllerState.from_arrays(
            self.controller_analog[index], self.controller_buttons[index]
        )
        stamp = float(self.timestamps[index]) if timestamp is None else float(timestamp)
        return BodyFrame(
            timestamp=stamp, positions=positions, rotations=rotations, controller=controller
        )


class RecordedBodySource:
    """Replays a :class:`RecordedBody` against the wall clock, like a live device."""

    def __init__(self, recording: RecordedBody, *, loop: bool = False, start_index: int = 0):
        if len(recording) == 0:
            raise ValueError("recording is empty")
        self.recording = recording
        self.loop = loop
        self.start_index = int(start_index)
        self._t0: float | None = None
        self._finished = False

    def start(self) -> None:
        self._t0 = time.monotonic() - float(self.recording.timestamps[self.start_index])

    def stop(self) -> None:
        self._t0 = None

    @property
    def finished(self) -> bool:
        return self._finished

    @property
    def disconnected(self) -> bool:
        return self._finished

    def get_latest(self) -> BodyFrame | None:
        if self._t0 is None:
            return None
        now = time.monotonic()
        elapsed = now - self._t0
        stamps = self.recording.timestamps
        if elapsed > stamps[-1]:
            if not self.loop:
                self._finished = True
                return None
            self._t0 += stamps[-1] - stamps[0]
            elapsed = now - self._t0
        index = int(np.searchsorted(stamps, elapsed, side="right") - 1)
        index = min(max(index, 0), len(self.recording) - 1)
        return self.recording.frame(index, timestamp=self._t0 + float(stamps[index]))


XRT_SERVICE_SCRIPT = Path("/opt/apps/roboticsservice/runService.sh")


class XrtBodySource:
    """Background reader for the XRoboToolkit PC service (Pico 4 + motion trackers).

    Like SONIC's Pico manager, ``start()`` launches the PC service before ``xrt.init()``
    unless ``start_service`` is False (service already managed elsewhere).
    """

    STALE_TIMEOUT_S = 5.0
    CONTROLLER_POLL_INTERVAL_S = 0.01
    READ_ERROR_LOG_INTERVAL_S = 1.0

    def __init__(self, recording: BodyRecording | None = None, *, start_service: bool = True):
        if start_service and not XRT_SERVICE_SCRIPT.is_file():
            raise RuntimeError(
                f"XRoboToolkit PC service not found at {XRT_SERVICE_SCRIPT}; install "
                "the XRoboToolkit PC service or start it separately and pass --no-start-service"
            )
        self._start_service = start_service
        try:
            import xrobotoolkit_sdk as xrt
        except ImportError as exc:
            raise RuntimeError(
                "xrobotoolkit_sdk is not installed; see gear_sonic_deploy/xr_retarget/README.md"
            ) from exc
        self._xrt = xrt
        self._recording = recording
        self._stop = threading.Event()
        self._disconnected = threading.Event()
        self._lock = threading.Lock()
        self._latest: BodyFrame | None = None
        self._controller_states: deque[ControllerState] = deque(maxlen=512)
        self._last_read_error = float("-inf")
        self._thread = threading.Thread(target=self._run, name="xrt-body", daemon=True)
        self.fps = 0.0

    def start(self) -> None:
        if self._start_service:
            subprocess.Popen(["bash", str(XRT_SERVICE_SCRIPT)])
            time.sleep(0.5)
        self._xrt.init()
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout=1.0)
        try:
            self._xrt.close()
        except Exception:
            logger.exception("[xrt] close failed")

    @property
    def disconnected(self) -> bool:
        return self._disconnected.is_set()

    def get_latest(self) -> BodyFrame | None:
        with self._lock:
            return self._latest

    def drain_controller_states(self) -> list[ControllerState]:
        """Consume button changes in order, including the first controller sample.

        Body frames keep only their latest sample. Button edges need a separate
        queue so a slow retarget step cannot hide a press or release between two
        calls to :meth:`get_latest`. Analog values remain available in each body
        frame's latest controller snapshot.
        """
        with self._lock:
            states = list(self._controller_states)
            self._controller_states.clear()
            return states

    def _log_read_error(self) -> None:
        now = time.monotonic()
        if now - self._last_read_error >= self.READ_ERROR_LOG_INTERVAL_S:
            logger.exception("[xrt] read error")
            self._last_read_error = now

    def _read_controller(self) -> ControllerState:
        xrt = self._xrt
        buttons = {
            "A": xrt.get_A_button(),
            "B": xrt.get_B_button(),
            "X": xrt.get_X_button(),
            "Y": xrt.get_Y_button(),
            "left_menu": xrt.get_left_menu_button(),
            "right_menu": xrt.get_right_menu_button(),
            "left_axis_click": xrt.get_left_axis_click(),
            "right_axis_click": xrt.get_right_axis_click(),
        }
        return ControllerState(
            left_trigger=float(xrt.get_left_trigger()),
            right_trigger=float(xrt.get_right_trigger()),
            left_grip=float(xrt.get_left_grip()),
            right_grip=float(xrt.get_right_grip()),
            buttons=frozenset(name for name, value in buttons.items() if value),
            sample_time=time.monotonic(),
        )

    def _run(self) -> None:
        xrt = self._xrt
        last_stamp_ns: int | None = None
        last_new_data = time.monotonic()
        next_controller_poll = 0.0
        controller = ControllerState()
        controller_buttons: frozenset[str] | None = None
        while not self._stop.is_set():
            now = time.monotonic()
            if now >= next_controller_poll:
                next_controller_poll = now + self.CONTROLLER_POLL_INTERVAL_S
                try:
                    controller = self._read_controller()
                except Exception:
                    self._log_read_error()
                else:
                    if controller.buttons != controller_buttons:
                        with self._lock:
                            self._controller_states.append(controller)
                        controller_buttons = controller.buttons
            try:
                stamp_ns = int(xrt.get_time_stamp_ns()) if xrt.is_body_data_available() else None
            except Exception:
                self._log_read_error()
                time.sleep(0.0005)
                continue
            if stamp_ns is None or stamp_ns == last_stamp_ns:
                if (
                    time.monotonic() - last_new_data > self.STALE_TIMEOUT_S
                    and not self._disconnected.is_set()
                ):
                    logger.warning(
                        "[xrt] no new body data for %.1fs; check that the XRoboToolkit PC service is "
                        "running and the PICO app shows WORKING with full-body tracking enabled",
                        self.STALE_TIMEOUT_S,
                    )
                    self._disconnected.set()
                time.sleep(0.0005)
                continue
            now = time.monotonic()
            if last_stamp_ns is not None and stamp_ns > last_stamp_ns:
                inst = 1e9 / (stamp_ns - last_stamp_ns)
                self.fps = inst if self.fps == 0.0 else 0.9 * self.fps + 0.1 * inst
            last_stamp_ns = stamp_ns
            last_new_data = now
            self._disconnected.clear()
            try:
                raw = np.asarray(xrt.get_body_joints_pose(), dtype=np.float64).reshape(
                    NUM_BODY_JOINTS, 7
                )
            except Exception:
                self._log_read_error()
                continue
            positions, rotations = unity_poses_to_robot(raw)
            frame = BodyFrame(
                timestamp=now, positions=positions, rotations=rotations, controller=controller
            )
            with self._lock:
                self._latest = frame
            if self._recording is not None:
                self._recording.append(raw, now, controller)
