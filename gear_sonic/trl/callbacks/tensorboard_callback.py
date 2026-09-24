from transformers import TrainerCallback

from gear_sonic.trl.utils import logger_utils


class SonicTensorBoardCallback(TrainerCallback):
    """TensorBoard logger that preserves SONIC/W&B-style metric groups."""

    def __init__(self, log_dir=None, flush_every_n_steps=1):
        self.log_dir = log_dir
        self.flush_every_n_steps = int(flush_every_n_steps)
        if self.flush_every_n_steps < 1:
            raise ValueError("flush_every_n_steps must be at least 1")
        self.writer = None

    def _ensure_writer(self, args):
        if self.writer is None:
            log_dir = self.log_dir or args.logging_dir
            self.writer = logger_utils.create_summary_writer(log_dir)

    def on_train_begin(self, args, state, control, **kwargs):
        if not state.is_world_process_zero:
            return
        self._ensure_writer(args)
        self.writer.add_text("args", args.to_json_string())
        self.writer.flush()

    def on_log(self, args, state, control, logs=None, **kwargs):  # noqa: ARG002
        if not state.is_world_process_zero:
            return
        self._ensure_writer(args)
        for tag, value in logger_utils.tensorboard_train_logs(logs or {}).items():
            self.writer.add_scalar(tag, value, state.global_step)
        if state.global_step % self.flush_every_n_steps == 0:
            self.writer.flush()

    def on_train_end(self, args, state, control, **kwargs):  # noqa: ARG002
        if self.writer is not None:
            self.writer.flush()
            self.writer.close()
            self.writer = None
