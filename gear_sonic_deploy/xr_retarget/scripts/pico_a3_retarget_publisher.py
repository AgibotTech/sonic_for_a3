#!/usr/bin/env python3
"""Pico body tracking -> A3 Ultra retargeting -> AimRT TA ROS 2 channel.

Sources
  xrt        live Pico 4 + motion trackers through the XRoboToolkit PC service
  recording  replay an NPZ recorded with ``--record``
  synthetic  Pico-format body data synthesised from an A3 CSV (no hardware)

Live controls (xrt):
  PICO app       keep Send ON and Switch w/ A Button OFF (PC handles A)
  A+B+X+Y         calibrate while matching the robot's standing pose, then wait with send_data=False
  A              after calibration, start/pause/resume simulation and PKL recording
  X              finish/save; double press within 0.3 s discards and repeats
  hold left Menu  temporarily pause streaming; match the pose before releasing

Example (from the repository root):
  python gear_sonic_deploy/xr_retarget/scripts/pico_a3_retarget_publisher.py --source xrt
"""

from __future__ import annotations

import argparse
import enum
from pathlib import Path
import signal
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sonic_a3.stream.a3_ref_stream import (  # noqa: E402
    A3ControlState,
    A3RefFrame,
    VelocityEstimator,
)
from sonic_a3.stream.ta_ros_publisher import TaRosPublisher  # noqa: E402
from sonic_a3.teleop.a3_motion_io import load_a3_csv  # noqa: E402
from sonic_a3.teleop.a3_retargeter import DEFAULT_CONFIG, A3Retargeter, RetargetConfig  # noqa: E402
from sonic_a3.teleop.pico_body_source import (  # noqa: E402
    BodyFrame,
    BodyRecording,
    ControllerState,
    RecordedBody,
    RecordedBodySource,
    XrtBodySource,
)
from sonic_a3.teleop.synthetic_body import SyntheticOptions, synthesize_body_recording  # noqa: E402

POLL_PERIOD_S = 0.004
CONTROL_HEARTBEAT_S = 0.1
CALIB_COMBO = ("A", "B", "X", "Y")
FINISH_DOUBLE_PRESS_S = 0.3


class Mode(enum.Enum):
    IDLE = "idle"
    CALIBRATING = "calibrating"
    STREAMING = "streaming"
    PAUSED = "paused"


class RetargetPublisher:
    def __init__(
        self, retargeter: A3Retargeter, publisher: TaRosPublisher, recording: BodyRecording | None,
        *, manual_start: bool = True,
    ):
        self.retargeter = retargeter
        self.publisher = publisher
        self.recording = recording
        self.velocity = VelocityEstimator()
        self.mode = Mode.IDLE
        self.frame_index = 0
        self.sent = 0
        self._calib_frames: list[BodyFrame] = []
        self._last_stamp = -1.0
        self._combo_down = False
        self._a_down = False
        self._menu_down = False
        self._x_down = False
        self._x_hold_valid = False
        self._x_hold_second = False
        self._x_pending_until: float | None = None
        self.session_id = time.time_ns()
        self.start_id = 0
        self.stream_id = 0
        self.finish_id = 0
        self.finish_repeat = False
        self.manual_start = manual_start
        self.started = False
        # A controls this latch; Menu can temporarily suppress it. Processing
        # latency must never change the operator's requested send state.
        self.send_data = False
        self._effective_send_data = False
        self._last_control = None
        self._last_control_at = float("-inf")

    def set_mode(self, mode: Mode, *, reason: str | None = None) -> None:
        if mode != self.mode:
            detail = "" if reason is None else f" ({reason}, requested_send_data={self.send_data})"
            print(f"[publisher] {self.mode.value} -> {mode.value}{detail}")
            self.mode = mode

    def start_calibration(self, *, now: float | None = None) -> None:
        self._cancel_x_gesture()
        self.start_id += 1
        self.started = False
        self.send_data = False
        self._calib_frames = []
        self.set_mode(Mode.CALIBRATING)
        self.publish_control(force=True, now=now)

    def calibrate_now(self, frames: list[BodyFrame]) -> None:
        self.retargeter.calibrate(frames)
        self.velocity.reset()
        if self.recording is not None:
            self.recording.mark_calibration()
        print(
            f"[publisher] calibrated on {len(frames)} frame(s); root_scale={self.retargeter.root_scale:.3f}"
        )
        if self.start_id == 0:
            self.start_id = 1
        self.started = True
        # Live calibration only arms collection. The operator starts explicitly
        # with a separate A press; recorded/synthetic sources still auto-start.
        self.send_data = not self.manual_start
        if self.manual_start:
            print("[publisher] calibration ready; send_data=False. Release A+B+X+Y, then press A to start.")
        # The consumer also waits for this session's first reference frame.
        self.publish_control(force=True)

    def publish_control(self, *, force: bool = False, now: float | None = None) -> None:
        """Send state changes immediately and heartbeat even without body tracking."""
        now = time.monotonic() if now is None else now
        self._resolve_x_finish(now)
        # This is control intent, not an IK-processing watchdog. In particular,
        # an expensive retarget() call must not publish a false pause afterwards.
        # The consumer independently detects missing heartbeats/references.
        x_pause = self._x_hold_valid or self._x_pending_until is not None
        effective = self.started and self.send_data and not self._menu_down and not x_pause
        if effective != self._effective_send_data:
            self.stream_id += 1
            if effective:
                self.velocity.reset()
            self._effective_send_data = effective
        if self.started:
            reason = "enabled" if effective else (
                "A pause" if not self.send_data else "X finish pending" if x_pause else "left Menu held"
            )
            self.set_mode(Mode.STREAMING if effective else Mode.PAUSED, reason=reason)
        state = A3ControlState(
            session_id=self.session_id,
            start_id=self.start_id,
            stream_id=self.stream_id,
            started=self.started,
            send_data=effective,
            manual_start=self.manual_start,
            finish_id=self.finish_id,
            finish_repeat=self.finish_repeat,
        )
        if force or state != self._last_control or now - self._last_control_at >= CONTROL_HEARTBEAT_S:
            self.publisher.send_control(state)
            self._last_control = state
            self._last_control_at = now

    def stop_streaming(self) -> None:
        self._cancel_x_gesture()
        self.started = False
        self.send_data = False
        self.set_mode(Mode.IDLE)
        self.publish_control(force=True)

    def handle_buttons(self, frame: BodyFrame) -> None:
        self.handle_controller(frame.controller)

    def _cancel_x_gesture(self) -> None:
        self._x_hold_valid = False
        self._x_hold_second = False
        self._x_pending_until = None

    def _finish_episode(self, *, repeat: bool) -> None:
        self._cancel_x_gesture()
        self.send_data = False
        self.finish_id += 1
        self.finish_repeat = repeat
        action = "discard and repeat" if repeat else "save and finish"
        print(f"[publisher] X: {action}; send_data=False")

    def _resolve_x_finish(self, now: float) -> None:
        # A second X press is accepted by its acquisition time. Wait for its
        # release before deciding: A/B/Y may still join it as a calibration chord.
        if (
            self.manual_start
            and self._x_pending_until is not None
            and not self._x_hold_valid
            and now > self._x_pending_until
        ):
            self._finish_episode(repeat=False)

    def handle_controller(self, controller: ControllerState, *, now: float | None = None) -> None:
        """Consume every queued button change, independently of pose retargeting."""
        if now is None:
            now = controller.sample_time
        now = time.monotonic() if now is None else now
        combo = controller.pressed(*CALIB_COMBO)
        x_down = controller.pressed("X")
        x_other_face = any(controller.pressed(name) for name in ("A", "B", "Y"))
        # Invalidate the whole X hold if another face button ever joins. This
        # includes slowly assembled X-first calibration chords and partial chords.
        if x_other_face or not self.manual_start or not self.started:
            self._cancel_x_gesture()
        else:
            self._resolve_x_finish(now)
            if x_down and not self._x_down:
                self._x_hold_valid = True
                self._x_hold_second = self._x_pending_until is not None
            elif not x_down and self._x_down and self._x_hold_valid:
                if self._x_hold_second:
                    self._finish_episode(repeat=True)
                else:
                    self._x_pending_until = now + FINISH_DOUBLE_PRESS_S
                    self._x_hold_valid = False
        if combo and not self._combo_down:
            self.start_calibration(now=now)
        a_down = controller.pressed("A")
        other_face_button = any(controller.pressed(name) for name in ("B", "X", "Y"))
        if a_down and not self._a_down and not combo and not other_face_button and self.started:
            self.send_data = not self.send_data
            print(f"[publisher] A: send_data={self.send_data}")
        self._combo_down = combo
        self._x_down = x_down
        # Consume A even inside the four-button chord; holding it never toggles twice.
        self._a_down = a_down
        self._menu_down = controller.pressed("left_menu")
        # Never expire a gesture using wall time halfway through a queue of
        # historical edges. The main loop sends a wall-clock heartbeat afterwards.
        self.publish_control(now=now)

    def step(self, frame: BodyFrame) -> None:
        if frame.timestamp <= self._last_stamp:
            return
        self._last_stamp = frame.timestamp

        if self.mode == Mode.CALIBRATING:
            self._calib_frames.append(frame)
            if len(self._calib_frames) >= self.retargeter.config.calibration_frames:
                self.calibrate_now(self._calib_frames)
            return
        # The control transition must precede the first pose of each active span.
        self.publish_control()
        if not self._effective_send_data:
            return

        out = self.retargeter.retarget(frame)
        self.publisher.send(
            A3RefFrame(
                frame_index=self.frame_index,
                stamp_s=out.timestamp,
                dof_il=out.dof_il,
                dof_vel_il=self.velocity.update(out.dof_il, out.timestamp),
                root_pos=out.root_pos,
                root_quat_wxyz=out.root_quat_wxyz,
            )
        )
        self.frame_index += 1
        self.sent += 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--source", choices=("xrt", "recording", "synthetic"), required=True)
    parser.add_argument("--recording", type=Path, help="NPZ for --source recording")
    parser.add_argument("--synthetic-from", type=Path, help="A3 CSV for --source synthetic")
    parser.add_argument("--loop", action="store_true", help="loop recording / synthetic sources")
    parser.add_argument("--record", type=Path, help="save the raw xrt session to this NPZ on exit")
    parser.add_argument(
        "--no-start-service",
        action="store_true",
        help="xrt: do not launch /opt/apps/roboticsservice/runService.sh (already running)",
    )
    parser.add_argument("--topic", default="/ta/whole_body_command", help="AimRT TA topic")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--duration", type=float, default=None, help="stop after this many seconds")
    parser.add_argument(
        "--start-delay",
        type=float,
        default=1.0,
        help="seconds to wait for subscribers before replay",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    retargeter = A3Retargeter(RetargetConfig.load(args.config))
    publisher = TaRosPublisher(args.topic)
    recording = BodyRecording() if args.record is not None else None
    node = RetargetPublisher(retargeter, publisher, recording, manual_start=args.source == "xrt")
    print(f"[publisher] TA channel {args.topic} (ROS 2: {publisher.topic})")

    if args.source == "xrt":
        source = XrtBodySource(recording, start_service=not args.no_start_service)
        print(
            "[publisher] PICO Data & Control: Send=ON, Switch w/ A Button=OFF. "
            "Keep Controller tracking enabled; this publisher handles A pause/resume."
        )
        print("[publisher] waiting for PICO body tracking data ...")
    else:
        if args.source == "recording":
            if args.recording is None:
                raise SystemExit("--source recording needs --recording")
            body = RecordedBody.load(args.recording)
        else:
            if args.synthetic_from is None:
                raise SystemExit("--source synthetic needs --synthetic-from")
            body = synthesize_body_recording(load_a3_csv(args.synthetic_from), SyntheticOptions())
        calib_index = max(body.calib_index, 0)
        n_cal = retargeter.config.calibration_frames
        lo = max(0, calib_index - n_cal // 2)
        node.calibrate_now([body.frame(i) for i in range(lo, min(len(body), lo + n_cal))])
        source = RecordedBodySource(body, loop=args.loop, start_index=calib_index)
        print(
            f"[publisher] replaying {len(body)} frames ({body.duration:.1f}s) from frame {calib_index}"
        )
        start_at = time.monotonic() + args.start_delay
        while (wait := start_at - time.monotonic()) > 0:
            node.publish_control()
            time.sleep(min(wait, 0.01))

    stop = False

    def _stop(*_):
        nonlocal stop
        stop = True

    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)

    source.start()
    t_start = time.monotonic()
    last_report = t_start
    live = isinstance(source, XrtBodySource)
    seen_body = False
    try:
        while not stop:
            if live:
                for controller in source.drain_controller_states():
                    node.handle_controller(controller)
            frame = source.get_latest()
            if frame is not None:
                if live and not seen_body:
                    seen_body = True
                    print(
                        "[publisher] body tracking data received; match the robot's standing pose "
                        "and press A+B+X+Y to calibrate, then release and press A to start"
                    )
                node.step(frame)
            elif isinstance(source, RecordedBodySource) and source.finished:
                print("[publisher] replay finished")
                break
            now = time.monotonic()
            node.publish_control(now=now)
            if args.duration is not None and now - t_start >= args.duration:
                break
            if now - last_report >= 5.0:
                body_fps = f" body_fps={source.fps:.0f}" if live else ""
                print(
                    f"[publisher] mode={node.mode.value} send_data={node._effective_send_data}"
                    f" requested_send_data={node.send_data} menu_pause={node._menu_down}"
                    f" sent={node.sent}{body_fps}"
                )
                last_report = now
            time.sleep(POLL_PERIOD_S)
    finally:
        node.stop_streaming()
        source.stop()
        publisher.close()
        if recording is not None and len(recording):
            recording.save(args.record)
            print(f"[publisher] saved {len(recording)} frames to {args.record}")


if __name__ == "__main__":
    main()
