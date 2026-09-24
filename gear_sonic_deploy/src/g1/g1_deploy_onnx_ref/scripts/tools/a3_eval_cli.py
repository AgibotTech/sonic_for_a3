#!/usr/bin/env python3
"""Small real-robot evaluation TUI backed by AimRT HTTP RPC."""

from __future__ import annotations

import argparse
import curses
import json
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any


RPC_BASE = os.environ.get("A3_MOTION_RPC_URL", "http://127.0.0.1:50901").rstrip("/")
TARGET_REPEATS = max(1, int(os.environ.get("A3_EVAL_TARGET_REPEATS", "3")))
PACKAGE_DIR = Path(__file__).resolve().parent.parent
WORK_DIR = (
    Path.cwd()
    if (Path.cwd() / "motions").is_dir() or (Path.cwd() / "csv").is_dir()
    else PACKAGE_DIR
)
RECORD_DIR = WORK_DIR / "eval_records"
STATE_FILE = RECORD_DIR / ".a3_eval_cli_state.json"
SERVICE = "a3.deploy.rpc.MotionControlService"


@dataclass(frozen=True)
class Motion:
    path: Path
    name: str


def find_motion_dir() -> Path:
    candidates = [
        WORK_DIR / "motions",
        WORK_DIR / "csv",
        PACKAGE_DIR / "motions",
    ]
    for candidate in candidates:
        if candidate.is_dir() and any(candidate.glob("*.csv")):
            return candidate
    raise FileNotFoundError(
        "找不到 motions/*.csv；请在 deploy 包根目录运行 tools/a3_eval_cli.py"
    )


def load_motions() -> list[Motion]:
    return [
        Motion(path=path.resolve(), name=path.stem)
        for path in sorted(find_motion_dir().glob("*.csv"))
    ]


def find_video(motion: Motion) -> Path | None:
    candidates = [
        WORK_DIR / "videos" / f"{motion.name}.mp4",
        PACKAGE_DIR / "videos" / f"{motion.name}.mp4",
    ]
    return next((path.resolve() for path in candidates if path.is_file()), None)


def open_video(motion: Motion) -> str:
    video = find_video(motion)
    if video is None:
        return f"没有对应视频: {motion.name}.mp4"
    player = shutil.which("mpv")
    command = (
        [player, "--force-window=yes", "--keep-open=no", str(video)]
        if player
        else ["xdg-open", str(video)]
    )
    try:
        subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
    except OSError as exc:
        return f"打开视频失败: {exc}"
    return f"视频: {video.name}"


def _yaml_scalar(raw: str) -> str:
    value = raw.strip()
    if value.startswith('"') and value.endswith('"'):
        try:
            return str(json.loads(value))
        except json.JSONDecodeError:
            pass
    if value.startswith("'") and value.endswith("'"):
        return value[1:-1].replace("''", "'")
    return value


def _sidecar_fields(path: Path) -> dict[str, str]:
    fields: dict[str, str] = {}
    section = ""
    wanted = {"file", "motion_source_path", "motion_name", "outcome"}
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            if line and not line[0].isspace() and line.endswith(":"):
                section = line[:-1]
                continue
            match = re.match(r"^\s+([A-Za-z0-9_]+):\s*(.*?)\s*$", line)
            if not match or match.group(1) not in wanted:
                continue
            key = match.group(1)
            if key == "file":
                key = f"{section}.file"
            fields[key] = _yaml_scalar(match.group(2))
    except OSError:
        return {}
    return fields


def scan_completed_recordings(record_dir: Path = RECORD_DIR) -> dict[str, int]:
    counts: dict[str, int] = {}
    if not record_dir.is_dir():
        return counts
    for sidecar in record_dir.glob("session_*/data/*.meta.yaml"):
        if sidecar.name == "session.meta.yaml":
            continue
        fields = _sidecar_fields(sidecar)
        if fields.get("outcome") != "COMPLETED":
            continue
        mcap_name = fields.get("mcap.file", "")
        mcap_path = sidecar.parent / mcap_name
        if not mcap_name or not mcap_path.is_file() or mcap_path.stat().st_size <= 0:
            continue
        source = fields.get("motion_source_path", "")
        name = Path(source).name if source else fields.get("motion_name", "")
        if name:
            counts[name] = counts.get(name, 0) + 1
    return counts


def load_attempts(state_file: Path = STATE_FILE) -> dict[str, int]:
    try:
        data = json.loads(state_file.read_text(encoding="utf-8"))
        attempts = data.get("attempts", {})
        return {
            str(key): max(0, int(value))
            for key, value in attempts.items()
            if isinstance(key, str)
        }
    except (OSError, ValueError, TypeError):
        return {}


def save_attempts(attempts: dict[str, int], state_file: Path = STATE_FILE) -> None:
    state_file.parent.mkdir(parents=True, exist_ok=True)
    temporary = state_file.with_suffix(".tmp")
    temporary.write_text(
        json.dumps({"attempts": attempts}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(state_file)


def rpc_call(method: str, payload: dict[str, Any]) -> dict[str, Any]:
    url = f"{RPC_BASE}/rpc/{SERVICE}/{method}"
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"content-type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=2.0) as response:
            body = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"RPC 连接失败: {exc.reason}") from exc
    decoded = json.loads(body or "{}")
    if not isinstance(decoded, dict):
        raise RuntimeError("RPC 返回不是 JSON object")
    return decoded


def _draw(stdscr: Any, motions: list[Motion]) -> None:
    try:
        curses.curs_set(0)
    except curses.error:
        pass
    stdscr.nodelay(True)
    stdscr.timeout(200)
    if curses.has_colors():
        curses.start_color()
        curses.use_default_colors()
        curses.init_pair(1, curses.COLOR_GREEN, -1)
        curses.init_pair(2, curses.COLOR_YELLOW, -1)
        curses.init_pair(3, curses.COLOR_RED, -1)
        curses.init_pair(4, curses.COLOR_CYAN, -1)

    selected = 0
    show_incomplete = True
    target = TARGET_REPEATS
    attempts = load_attempts()
    completed = scan_completed_recordings()
    message = "就绪"
    playback: dict[str, Any] = {}
    next_refresh = 0.0
    preview_only = find_motion_dir().name == "csv" and (WORK_DIR / "videos").is_dir()

    while True:
        now = time.monotonic()
        if not preview_only and now >= next_refresh:
            completed = scan_completed_recordings()
            try:
                playback = rpc_call("GetPlaybackStatus", {})
            except RuntimeError as exc:
                playback = {}
                message = str(exc)
            next_refresh = now + 1.0

        visible = [
            motion
            for motion in motions
            if not show_incomplete or completed.get(motion.path.name, 0) < target
        ]
        if not visible:
            visible = motions
            message = f"所有动作均已达到 {target} 次；已显示全量"
        selected = min(selected, max(0, len(visible) - 1))

        stdscr.erase()
        height, width = stdscr.getmaxyx()
        mode = "未达标" if show_incomplete else "全部"
        header = (
            f"A3 动作视频挑选  共 {len(motions)} 条"
            if preview_only
            else (
                f"A3 动作评测  [{mode}]  目标={target}次  "
                f"成功={sum(completed.values())}  记录={RECORD_DIR}"
            )
        )
        stdscr.addnstr(0, 0, header, max(1, width - 1), curses.A_BOLD)
        stdscr.addnstr(
            1,
            0,
            (
                "↑↓选择  Enter/v查看视频  q退出"
                if preview_only
                else "↑↓选择  Enter播放动作  v查看视频  f切换全部/未达标  "
                "+/-目标次数  x停止  q退出"
            ),
            max(1, width - 1),
        )
        state = "preview" if preview_only else playback.get("state", "offline")
        active_name = playback.get("motionName", "")
        tick = playback.get("tick", 0)
        total_ticks = playback.get("totalTicks", 0)
        stdscr.addnstr(
            2,
            0,
            f"Policy: {state}  {active_name}  {tick}/{total_ticks}    {message}",
            max(1, width - 1),
            curses.color_pair(4) if curses.has_colors() else 0,
        )

        list_height = max(1, height - 5)
        top = max(0, selected - list_height + 1)
        for row, motion in enumerate(visible[top : top + list_height], start=4):
            index = top + row - 4
            ok = completed.get(motion.path.name, 0)
            played = attempts.get(motion.path.name, 0)
            marker = "✓" if ok >= target else "!"
            text = f"{marker} {motion.name:<52} 播放 {played:2d}  录制成功 {ok:2d}/{target}"
            attr = curses.A_REVERSE if index == selected else 0
            if curses.has_colors():
                attr |= curses.color_pair(1 if ok >= target else 2 if ok else 3)
            stdscr.addnstr(row, 0, text, max(1, width - 1), attr)
        stdscr.refresh()

        key = stdscr.getch()
        if key in (ord("q"), ord("Q")):
            return
        if key in (curses.KEY_UP, ord("k")):
            selected = (selected - 1) % len(visible)
        elif key in (curses.KEY_DOWN, ord("j")):
            selected = (selected + 1) % len(visible)
        elif key in (ord("f"), ord("F")):
            show_incomplete = not show_incomplete
            selected = 0
        elif key in (ord("+"), ord("=")):
            target += 1
        elif key == ord("-"):
            target = max(1, target - 1)
        elif key in (ord("x"), ord("X")):
            try:
                reply = rpc_call("StopMotion", {})
                message = reply.get("message", "停止请求已发送")
            except RuntimeError as exc:
                message = str(exc)
        elif key in (ord("v"), ord("V")) and visible:
            message = open_video(visible[selected])
        elif key in (curses.KEY_ENTER, 10, 13) and visible and preview_only:
            message = open_video(visible[selected])
        elif key in (curses.KEY_ENTER, 10, 13) and visible:
            motion = visible[selected]
            try:
                reply = rpc_call(
                    "PlayMotion", {"motionPath": str(motion.path)}
                )
                if reply.get("accepted"):
                    attempts[motion.path.name] = (
                        attempts.get(motion.path.name, 0) + 1
                    )
                    save_attempts(attempts)
                message = reply.get("message", str(reply))
                playback = reply
            except RuntimeError as exc:
                message = str(exc)
            next_refresh = 0.0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="A3 real-robot evaluation motion selector"
    )
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--status", action="store_true", help="print RPC status")
    action.add_argument("--stop", action="store_true", help="stop current motion")
    action.add_argument("--play", metavar="CSV", help="play one loaded motion path")
    action.add_argument("--list", action="store_true", help="list motions and counts")
    args = parser.parse_args()

    if args.status:
        print(json.dumps(rpc_call("GetPlaybackStatus", {}), ensure_ascii=False))
        return 0
    if args.stop:
        print(json.dumps(rpc_call("StopMotion", {}), ensure_ascii=False))
        return 0
    if args.play:
        path = Path(args.play).resolve()
        print(
            json.dumps(
                rpc_call("PlayMotion", {"motionPath": str(path)}),
                ensure_ascii=False,
            )
        )
        return 0

    motions = load_motions()
    if args.list:
        completed = scan_completed_recordings()
        attempts = load_attempts()
        for motion in motions:
            print(
                f"{motion.name}\t播放={attempts.get(motion.path.name, 0)}"
                f"\t录制成功={completed.get(motion.path.name, 0)}"
            )
        return 0
    curses.wrapper(_draw, motions)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"a3_eval_cli: {exc}", file=sys.stderr)
        raise SystemExit(1)
