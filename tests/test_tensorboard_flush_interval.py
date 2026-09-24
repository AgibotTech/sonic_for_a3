from types import SimpleNamespace

import pytest

from gear_sonic.trl.callbacks.tensorboard_callback import SonicTensorBoardCallback


class _Writer:
    def __init__(self):
        self.scalars = []
        self.flush_count = 0
        self.close_count = 0

    def add_scalar(self, tag, value, step):
        self.scalars.append((tag, value, step))

    def flush(self):
        self.flush_count += 1

    def close(self):
        self.close_count += 1


def test_tensorboard_batches_flushes_but_keeps_every_scalar():
    callback = SonicTensorBoardCallback(flush_every_n_steps=10)
    callback.writer = _Writer()
    state = SimpleNamespace(is_world_process_zero=True, global_step=1)

    callback.on_log(None, state, None, logs={"fps": 123})
    state.global_step = 10
    callback.on_log(None, state, None, logs={"fps": 456})

    assert callback.writer.scalars == [
        ("Train/fps", 123.0, 1),
        ("Train/fps", 456.0, 10),
    ]
    assert callback.writer.flush_count == 1


def test_tensorboard_flushes_and_closes_at_train_end():
    callback = SonicTensorBoardCallback(flush_every_n_steps=10)
    writer = _Writer()
    callback.writer = writer

    callback.on_train_end(None, None, None)

    assert writer.flush_count == 1
    assert writer.close_count == 1
    assert callback.writer is None


def test_tensorboard_flush_interval_must_be_positive():
    with pytest.raises(ValueError, match="at least 1"):
        SonicTensorBoardCallback(flush_every_n_steps=0)
