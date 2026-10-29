"""Batched A3 ankle inertia projection; independent of Isaac Sim imports.

The HAL table maps motor-output velocity to serial joint velocity (J).
Gamma = inv(J), so reflected inertia is Gamma.T @ I_motor @ Gamma.
All derivatives below are analytic derivatives of the bilinear J interpolant.
"""

from __future__ import annotations

import numpy as np
import torch

from gear_sonic.utils.a3_motor_model import ParallelJacobianTable

A3_ANKLE_MOTOR_ARMATURE = 30.118e-6 * 20.03**2


def _inverse2(x: torch.Tensor) -> torch.Tensor:
    a, b, c, d = x[..., 0, 0], x[..., 0, 1], x[..., 1, 0], x[..., 1, 1]
    return torch.stack((d, -b, -c, a), -1).reshape(x.shape) / (a * d - b * c)[..., None, None]


class A3DynamicArmature:
    """Two ankle pairs, in (left pitch, left roll, right pitch, right roll) order.

    Returns diagonal armature, nonlinear bias h_a, and off-diagonal inertia.
    Outside the table the Jacobian is held at its boundary and derivatives in
    the clamped direction are zero. ``out_of_bounds`` is exposed for monitoring.
    """

    def __init__(
        self, *, device="cpu", dtype=torch.float32, table_dir=None, motor_armature=A3_ANKLE_MOTOR_ARMATURE
    ):
        if not np.isfinite(motor_armature) or motor_armature <= 0:
            raise ValueError("motor_armature must be positive and finite")
        tables = [
            ParallelJacobianTable.load(name, device=device, dtype=dtype, table_dir=table_dir)
            for name in ("left_ankle", "right_ankle")
        ]
        # Validate offline, never synchronize the GPU in the simulation hot path.
        for t in tables:
            j = t.jacobian.cpu().numpy()
            if not t.reachable.all().item() or not np.isfinite(j).all():
                raise ValueError("Dynamic armature requires a finite, fully reachable table")
            if np.min(abs(np.linalg.det(j))) < 1e-5:
                raise ValueError("Dynamic armature Jacobian table is singular")
        for t in tables[1:]:
            if not torch.equal(t.pitch_axis, tables[0].pitch_axis) or not torch.equal(
                t.roll_axis, tables[0].roll_axis
            ):
                raise ValueError("Left and right ankle grids must match")
        self.pitch = tables[0].pitch_axis
        self.roll = tables[0].roll_axis
        self.j = torch.stack([t.jacobian for t in tables])
        self.side = torch.arange(2, device=device)
        self.motor_armature = motor_armature
        self.out_of_bounds = None

    def terms(self, q: torch.Tensor, v: torch.Tensor, motor_scales: torch.Tensor | None = None):
        q = q.reshape(-1, 2, 2)
        v = v.reshape(-1, 2, 2)
        p, r = q[..., 0].contiguous(), q[..., 1].contiguous()
        ip = torch.searchsorted(self.pitch, p).clamp(1, self.pitch.numel() - 1)
        ir = torch.searchsorted(self.roll, r).clamp(1, self.roll.numel() - 1)
        dp, dr = self.pitch[ip] - self.pitch[ip - 1], self.roll[ir] - self.roll[ir - 1]
        wp = ((p - self.pitch[ip - 1]) / dp).clamp(0, 1)[..., None, None]
        wr = ((r - self.roll[ir - 1]) / dr).clamp(0, 1)[..., None, None]
        j00, j01 = self.j[self.side, ip - 1, ir - 1], self.j[self.side, ip - 1, ir]
        j10, j11 = self.j[self.side, ip, ir - 1], self.j[self.side, ip, ir]
        j = (j00 * (1 - wr) + j01 * wr) * (1 - wp) + (j10 * (1 - wr) + j11 * wr) * wp
        inside_p = (p >= self.pitch[0]) & (p <= self.pitch[-1])
        inside_r = (r >= self.roll[0]) & (r <= self.roll[-1])
        self.out_of_bounds = ~(inside_p & inside_r)
        jp = ((j10 - j00) * (1 - wr) + (j11 - j01) * wr) / dp[..., None, None]
        jr = ((j01 - j00) * (1 - wp) + (j11 - j10) * wp) / dr[..., None, None]
        jdot = jp * (v[..., 0] * inside_p)[..., None, None] + jr * (v[..., 1] * inside_r)[..., None, None]
        gamma = _inverse2(j)
        weighted_gamma = gamma if motor_scales is None else motor_scales.reshape(-1, 2, 2, 1) * gamma
        mass = self.motor_armature * (gamma.transpose(-1, -2) @ weighted_gamma)
        # Gamma_dot = -Gamma J_dot Gamma; h_a = Gamma.T I Gamma_dot v.
        h = -(mass @ jdot @ gamma @ v[..., None]).squeeze(-1)
        diag = mass.diagonal(dim1=-2, dim2=-1)
        offdiag = mass - torch.diag_embed(diag)
        return diag.reshape(-1, 4), h.reshape(-1, 4), offdiag

    @staticmethod
    def correction(h, offdiag, previous_acceleration):
        coupled = (offdiag @ previous_acceleration.reshape(-1, 2, 2, 1)).reshape(-1, 4)
        return -h - coupled
