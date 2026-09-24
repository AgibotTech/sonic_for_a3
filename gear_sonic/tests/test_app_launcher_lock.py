"""Tests for bounded, rank-aware Isaac AppLauncher serialization."""

from filelock import FileLock
import pytest

from gear_sonic.train_agent_trl import (
    _get_app_launcher_lock_timeout_seconds,
    launch_isaac_app_with_lock,
)


class _FakeLauncher:
    def __init__(self, args_cli):
        self.args_cli = args_cli


def test_app_launcher_lock_reports_rank_and_returns_launcher(tmp_path):
    messages = []
    args_cli = object()

    launcher = launch_isaac_app_with_lock(
        _FakeLauncher,
        args_cli,
        global_rank=19,
        local_rank=3,
        lock_path=str(tmp_path / "app.lock"),
        timeout_seconds=1.0,
        log_fn=messages.append,
    )

    assert launcher.args_cli is args_cli
    assert [message.split()[1] for message in messages] == ["waiting", "acquired", "initialized"]
    assert all("global_rank=19" in message for message in messages)
    assert all("local_rank=3" in message for message in messages)


def test_app_launcher_lock_times_out_instead_of_waiting_forever(tmp_path):
    lock_path = str(tmp_path / "app.lock")
    messages = []
    holder = FileLock(lock_path)

    with holder:
        with pytest.raises(RuntimeError, match=r"global_rank=27 local_rank=3.*waited_seconds="):
            launch_isaac_app_with_lock(
                _FakeLauncher,
                object(),
                global_rank=27,
                local_rank=3,
                lock_path=lock_path,
                timeout_seconds=0.05,
                log_fn=messages.append,
            )

    assert messages[0].startswith("[AppLauncherLock] waiting")
    assert messages[-1].startswith("[AppLauncherLock] timeout")


@pytest.mark.parametrize("value", ["0", "-1", "nan", "not-a-number"])
def test_app_launcher_lock_timeout_env_must_be_positive_finite(monkeypatch, value):
    monkeypatch.setenv("ISAACLAB_APP_LAUNCHER_LOCK_TIMEOUT_SECONDS", value)

    with pytest.raises(ValueError, match="positive"):
        _get_app_launcher_lock_timeout_seconds()
