"""Reference data and velocity estimation for the XR retarget publisher."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

@dataclass
class A3RefFrame:
    frame_index: int
    stamp_s: float
    dof_il: np.ndarray
    dof_vel_il: np.ndarray
    root_pos: np.ndarray
    root_quat_wxyz: np.ndarray


@dataclass(frozen=True)
class A3ControlState:
    """Publisher session, acquisition and current uninterrupted reference stream.

    ``start_id`` changes on a new acquisition and ``stream_id`` changes when a
    paused stream resumes. A consumer can discard queued references across either
    boundary without changing the reference-frame wire format.
    """

    session_id: int
    start_id: int
    stream_id: int
    started: bool
    send_data: bool
    manual_start: bool = True
    # A resolved standalone-X gesture. The counter persists in heartbeats so a
    # slow consumer cannot miss it; it never resets during a publisher session.
    finish_id: int = 0
    finish_repeat: bool = False

    def __post_init__(self) -> None:
        for name in ("session_id", "start_id", "stream_id", "finish_id"):
            value = getattr(self, name)
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
                raise ValueError(f"{name} must be an integer")
            if not 0 <= value <= np.iinfo(np.int64).max:
                raise ValueError(f"{name} must fit a nonnegative int64")
        for name in ("started", "send_data", "manual_start", "finish_repeat"):
            if not isinstance(getattr(self, name), (bool, np.bool_)):
                raise ValueError(f"{name} must be a boolean")
        if self.send_data and not self.started:
            raise ValueError("send_data requires started=True")


class VelocityEstimator:
    """Finite-difference joint velocity with EMA smoothing, reset on gaps."""

    def __init__(self, alpha: float = 0.5, max_gap_s: float = 0.2):
        self.alpha = alpha
        self.max_gap_s = max_gap_s
        self._prev_q: np.ndarray | None = None
        self._prev_t: float | None = None
        self._vel: np.ndarray | None = None

    def reset(self) -> None:
        self._prev_q = self._prev_t = self._vel = None

    def update(self, q: np.ndarray, t: float) -> np.ndarray:
        if self._prev_q is None or self._prev_t is None or not (0.0 < t - self._prev_t <= self.max_gap_s):
            vel = np.zeros_like(q)
            self._vel = vel
        else:
            raw = (q - self._prev_q) / (t - self._prev_t)
            self._vel = raw if self._vel is None else self._vel + self.alpha * (raw - self._vel)
            vel = self._vel
        self._prev_q = q.copy()
        self._prev_t = t
        return vel.copy()
