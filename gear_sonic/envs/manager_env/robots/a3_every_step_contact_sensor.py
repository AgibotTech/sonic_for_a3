"""Exact eager-update fast path for A3's period-zero history sensors.

All environments are due after update(dt>=0) when update_period is zero.
Use that known contract instead of selecting a CUDA nonzero index tensor.
Partial resets and lazy reads retain the upstream selective-update path.
"""
from isaaclab.sensors import ContactSensor


class EveryStepContactSensor(ContactSensor):
    def _initialize_impl(self):
        self._a3_buffers_fresh = False
        super()._initialize_impl()

    def reset(self, env_ids=None):
        self._a3_buffers_fresh = False
        super().reset(env_ids)

    def update(self, dt, force_recompute=False):
        self._a3_buffers_fresh = False
        if self.cfg.update_period != 0.0 or self.cfg.history_length <= 0 or dt < 0:
            return super().update(dt, force_recompute)
        self._timestamp += dt
        # ContactSensor recognizes a full-length sequence and uses slice(None).
        self._update_buffers_impl(range(self._num_envs))
        self._timestamp_last_update.copy_(self._timestamp)
        self._is_outdated.fill_(False)
        self._a3_buffers_fresh = True

    def _update_outdated_buffers(self):
        if getattr(self, '_a3_buffers_fresh', False):
            return
        super()._update_outdated_buffers()
        self._a3_buffers_fresh = True
