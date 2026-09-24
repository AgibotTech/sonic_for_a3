"""Shared playback helpers for motion viewers and sim2sim scripts."""

from __future__ import annotations

from collections import OrderedDict
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
import queue
import random
import sys
import threading
import time
from typing import Callable, Generic, Hashable, Iterable, TypeVar


T = TypeVar("T")
K = TypeVar("K", bound=Hashable)

CMD_TOGGLE_PAUSE = "toggle_pause"
CMD_STEP_FORWARD = "step_forward"
CMD_STEP_BACKWARD = "step_backward"
CMD_NEXT_CLIP = "next_clip"
CMD_PREV_CLIP = "prev_clip"
CMD_RESET = "reset"


@dataclass(frozen=True)
class PlaybackCommand:
    name: str
    value: int = 1


def shuffled_once(items: Iterable[T]) -> list[T]:
    """Shuffle with system entropy and avoid returning the original order when possible."""
    randomized = list(items)
    if len(randomized) <= 1:
        return randomized

    original = list(randomized)
    rng = random.SystemRandom()
    for _ in range(8):
        rng.shuffle(randomized)
        if randomized != original:
            break
    if randomized == original:
        randomized = original[1:] + original[:1]
    return randomized


def format_progress_bar(progress: float, width: int = 32) -> str:
    progress = min(1.0, max(0.0, progress))
    filled = int(round(progress * width))
    return "[" + "#" * filled + "-" * (width - filled) + "]"


class PlaybackCommandQueue:
    def __init__(self) -> None:
        self._queue: queue.Queue[PlaybackCommand] = queue.Queue()

    def put(self, name: str, value: int = 1) -> None:
        self._queue.put(PlaybackCommand(name, value))

    def drain(self) -> list[PlaybackCommand]:
        commands: list[PlaybackCommand] = []
        while True:
            try:
                commands.append(self._queue.get_nowait())
            except queue.Empty:
                return commands

    def enqueue_keycode(
        self,
        keycode: int,
        *,
        step_forward_keycodes: Iterable[int] = (),
        step_backward_keycodes: Iterable[int] = (),
    ) -> bool:
        if keycode in set(step_forward_keycodes):
            self.put(CMD_STEP_FORWARD)
            return True
        if keycode in set(step_backward_keycodes):
            self.put(CMD_STEP_BACKWARD)
            return True

        try:
            char = chr(keycode)
        except (ValueError, OverflowError):
            return False

        if char == " ":
            self.put(CMD_TOGGLE_PAUSE)
        elif char == ".":
            self.put(CMD_STEP_FORWARD)
        elif char == ",":
            self.put(CMD_STEP_BACKWARD)
        elif char == "=":
            self.put(CMD_NEXT_CLIP)
        elif char == "-":
            self.put(CMD_PREV_CLIP)
        elif char in ("R", "r"):
            self.put(CMD_RESET)
        else:
            return False
        return True


class TerminalProgress:
    def __init__(
        self,
        *,
        interval: float = 0.25,
        width: int = 32,
        stream=None,
    ) -> None:
        self.interval = interval
        self.width = width
        self.stream = sys.stdout if stream is None else stream
        self.last_render = 0.0
        self.line_active = False
        self._lock = threading.RLock()

    def finish_line(self) -> None:
        with self._lock:
            if not self.line_active:
                return
            self.stream.write("\n")
            self.stream.flush()
            self.line_active = False

    def print(self, message: str) -> None:
        with self._lock:
            self.finish_line()
            print(message, file=self.stream)
            self.last_render = 0.0

    def render(
        self,
        *,
        clip_index: int,
        clip_count: int,
        name: str,
        frame: int,
        total_frames: int,
        status: str,
        detail: str = "",
        force: bool = False,
    ) -> None:
        with self._lock:
            now = time.monotonic()
            if not force and now - self.last_render < self.interval:
                return

            total_frames = max(1, int(total_frames))
            frame = min(max(int(frame), 0), total_frames - 1)
            progress = float(frame + 1) / float(total_frames)
            detail_text = f"  {detail}" if detail else ""
            self.stream.write(
                f"\r[{clip_index + 1}/{clip_count}] {name}  "
                f"{format_progress_bar(progress, self.width)} {progress * 100.0:6.2f}%  "
                f"frame {frame + 1}/{total_frames} {status}"
                f"{detail_text}\x1b[K"
            )
            self.stream.flush()
            self.line_active = True
            self.last_render = now


class AsyncLRUCache(Generic[K, T]):
    """Threaded LRU cache with explicit prefetch for expensive clip loading."""

    def __init__(
        self,
        loader: Callable[[K], T],
        *,
        max_size: int = 8,
        max_workers: int = 1,
    ) -> None:
        self.loader = loader
        self.max_size = max(1, int(max_size))
        self._executor = ThreadPoolExecutor(max_workers=max(1, int(max_workers)))
        self._cache: OrderedDict[K, T] = OrderedDict()
        self._futures: dict[K, Future[T]] = {}
        self._lock = threading.Lock()

    def preload(self, key: K, value: T) -> None:
        with self._lock:
            self._cache[key] = value
            self._cache.move_to_end(key)
            self._evict_locked()

    def is_cached(self, key: K) -> bool:
        with self._lock:
            return key in self._cache

    def is_pending(self, key: K) -> bool:
        with self._lock:
            future = self._futures.get(key)
            return future is not None and not future.done()

    def prefetch(self, keys: Iterable[K]) -> None:
        for key in keys:
            self._submit(key)

    def get(self, key: K) -> T:
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                return self._cache[key]

        future = self._submit(key)
        try:
            value = future.result()
        except BaseException:
            with self._lock:
                self._futures.pop(key, None)
            raise

        with self._lock:
            self._futures.pop(key, None)
            self._cache[key] = value
            self._cache.move_to_end(key)
            self._evict_locked()
            return self._cache[key]

    def close(self) -> None:
        self._executor.shutdown(wait=True, cancel_futures=True)

    def _submit(self, key: K) -> Future[T]:
        with self._lock:
            if key in self._cache:
                done: Future[T] = Future()
                done.set_result(self._cache[key])
                return done
            future = self._futures.get(key)
            if future is None:
                future = self._executor.submit(self.loader, key)
                self._futures[key] = future
            return future

    def _evict_locked(self) -> None:
        while len(self._cache) > self.max_size:
            self._cache.popitem(last=False)
