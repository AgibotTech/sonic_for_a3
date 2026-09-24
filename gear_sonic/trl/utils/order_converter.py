"""
IsaacLab <-> MuJoCo ordering conversion utilities.

This module provides utilities for converting both DOF and body ordering
between IsaacLab and MuJoCo conventions for humanoid robots.

qpos format: [root_trans(3), root_quat(4), dof_angles(N)]
"""

from abc import ABC

import numpy as np
import torch


class IsaacLabMuJoCoConverter(ABC):
    """Abstract base class for DOF/body order conversion between IsaacLab and MuJoCo.

    Subclasses must define DOF_MAPPINGS for their specific robot.
    """

    ROOT_QPOS_OFFSET = 7  # root_trans(3) + root_quat(4)
    DOF_MAPPINGS: dict = {}  # Must be overridden by subclasses
    VALID_DOF_ORDERS = ("mujoco", "isaaclab")

    def convert(self, data: torch.Tensor, from_order: str, to_order: str) -> torch.Tensor:
        """Convert DOF order between conventions.

        Auto-detects input format:
            - qpos:           [..., 7 + num_dof]           -> reorder DOF portion, keep root
            - dof angles:     [..., num_dof]                -> reorder directly
            - body transforms [..., num_dof + 1, D]         -> keep root body, reorder DOF bodies
            - body transforms [..., num_dof, D]             -> reorder directly
            - rotation mats   [..., num_dof (+ 1), 3, 3]   -> same as above on dim -3

        Note: Body reordering for per-body transforms uses DOF_MAPPINGS (not
        BODY_MAPPINGS) because each DOF link has exactly one parent body in G1,
        so DOF order and DOF-body order are consistent by construction.  The
        separate BODY_MAPPINGS include the root (pelvis) at index 0 and are used
        only when the full 30-body ordering is needed.
        """
        if from_order == to_order:
            return data

        mapping = self.DOF_MAPPINGS[(from_order, to_order)]
        last_dim = data.shape[-1]

        if last_dim == self.ROOT_QPOS_OFFSET + self.num_dof:
            # qpos: [..., 7 + num_dof]
            return torch.cat(
                [
                    data[..., : self.ROOT_QPOS_OFFSET],
                    data[..., self.ROOT_QPOS_OFFSET :][..., mapping],
                ],
                dim=-1,
            )
        elif last_dim == self.num_dof:
            # Raw DOF angles: [..., num_dof]
            return data[..., mapping]
        else:
            # Per-body transforms: body dim is -3 for [..., J, 3, 3], else -2 for [..., J, D]
            body_dim = -3 if (data.ndim >= 3 and data.shape[-1] == 3 and data.shape[-2] == 3) else -2
            num_bodies = data.shape[body_dim]

            if num_bodies == self.num_dof + 1:
                # Includes root at index 0 — keep root, reorder DOF bodies
                body_order = [0] + [m + 1 for m in mapping]
            elif num_bodies == self.num_dof:
                body_order = mapping
            else:
                raise ValueError(
                    f"Cannot detect format: last_dim={last_dim}, body_dim size={num_bodies}, "
                    f"expected num_dof+1={self.num_dof + 1} or num_dof={self.num_dof}"
                )

            if body_dim == -2:
                return data[..., body_order, :]
            else:  # body_dim == -3
                return data[..., body_order, :, :]

    def to_mujoco(self, data: torch.Tensor) -> torch.Tensor:
        """Convert from IsaacLab to MuJoCo DOF order."""
        return self.convert(data, from_order="isaaclab", to_order="mujoco")

    def to_isaaclab(self, data: torch.Tensor) -> torch.Tensor:
        """Convert from MuJoCo to IsaacLab DOF order."""
        return self.convert(data, from_order="mujoco", to_order="isaaclab")

    @property
    def num_dof(self) -> int:
        """Number of actuated DOFs (excluding root)."""
        return len(self.DOF_MAPPINGS[(self.VALID_DOF_ORDERS[0], self.VALID_DOF_ORDERS[1])])


class G1Converter(IsaacLabMuJoCoConverter):
    """Legacy ordering converter retained for the A3-024 checkpoint ABI."""

    def __init__(self):
        # The compatibility module contains static tables only, so it remains
        # available without a G1 simulator model or IsaacLab asset import.
        from gear_sonic.envs.manager_env.robots.g1 import (
            G1_ISAACLAB_JOINTS,
            G1_ISAACLAB_TO_MUJOCO_BODY,
            G1_ISAACLAB_TO_MUJOCO_DOF,
            G1_MUJOCO_TO_ISAACLAB_BODY,
            G1_MUJOCO_TO_ISAACLAB_DOF,
        )

        self.JOINT_NAMES = G1_ISAACLAB_JOINTS
        self.DOF_MAPPINGS = {
            ("isaaclab", "mujoco"): G1_ISAACLAB_TO_MUJOCO_DOF,
            ("mujoco", "isaaclab"): G1_MUJOCO_TO_ISAACLAB_DOF,
        }
        self.BODY_MAPPINGS = {
            ("isaaclab", "mujoco"): G1_ISAACLAB_TO_MUJOCO_BODY,
            ("mujoco", "isaaclab"): G1_MUJOCO_TO_ISAACLAB_BODY,
        }

    # Body subset names for MPJPE metrics (used by reconstruction_trainer)
    VR_3POINTS_BODY_NAMES = ["torso_link", "left_wrist_yaw_link", "right_wrist_yaw_link"]
    FOOT_BODY_NAMES = ["left_ankle_roll_link", "right_ankle_roll_link"]

    @property
    def vr_3points_mujoco_indices(self):
        """VR 3-point body indices in full (30-body) MuJoCo body order.

        These index into the full body array after isaaclab_to_mujoco_body
        reordering, NOT the 14-body motion.yaml body_names subset.
        """
        mj_names = [self.JOINT_NAMES[i] for i in self.isaaclab_to_mujoco_body]
        return [mj_names.index(n) for n in self.VR_3POINTS_BODY_NAMES]

    @property
    def foot_mujoco_indices(self):
        """Foot body indices in full (30-body) MuJoCo body order.

        These index into the full body array after isaaclab_to_mujoco_body
        reordering, NOT the 14-body motion.yaml body_names subset.
        """
        mj_names = [self.JOINT_NAMES[i] for i in self.isaaclab_to_mujoco_body]
        return [mj_names.index(n) for n in self.FOOT_BODY_NAMES]

    @property
    def isaaclab_to_mujoco_dof(self):
        """DOF reorder indices: IsaacLab -> MuJoCo."""
        return self.DOF_MAPPINGS[("isaaclab", "mujoco")]

    @property
    def mujoco_to_isaaclab_dof(self):
        """DOF reorder indices: MuJoCo -> IsaacLab."""
        return self.DOF_MAPPINGS[("mujoco", "isaaclab")]

    @property
    def isaaclab_to_mujoco_body(self):
        """Body reorder indices: IsaacLab -> MuJoCo."""
        return self.BODY_MAPPINGS[("isaaclab", "mujoco")]

    @property
    def mujoco_to_isaaclab_body(self):
        """Body reorder indices: MuJoCo -> IsaacLab."""
        return self.BODY_MAPPINGS[("mujoco", "isaaclab")]

    def get_isaaclab_to_mujoco_mapping(self):
        """Return the full mapping dict for body/DOF reordering."""
        return {
            "isaaclab_joints": self.JOINT_NAMES,
            "isaaclab_to_mujoco_dof": self.isaaclab_to_mujoco_dof,
            "mujoco_to_isaaclab_dof": self.mujoco_to_isaaclab_dof,
            "isaaclab_to_mujoco_body": self.isaaclab_to_mujoco_body,
            "mujoco_to_isaaclab_body": self.mujoco_to_isaaclab_body,
        }


class A3Converter(IsaacLabMuJoCoConverter):
    """Convert the compact 29-DOF A3 policy order to compact MuJoCo order.

    A3's MJCF has two non-actuated head joints, so its full FK output is 31D
    while the policy ABI is 29D.  ``convert()`` deliberately operates on the
    compact 29D MuJoCo order (with those head joints removed), which makes the
    two directions a true round trip.  The full 31D gather selectors remain
    available through :meth:`get_isaaclab_to_mujoco_mapping` for
    ``MotionLibRobot`` to consume its FK output.
    """

    def __init__(self):
        from gear_sonic.envs.manager_env.robots.a3 import (
            A3_ISAACLAB_JOINTS,
            A3_ISAACLAB_TO_MUJOCO_BODY,
            A3_ISAACLAB_TO_MUJOCO_DOF,
            A3_MUJOCO_TO_ISAACLAB_BODY,
            A3_MUJOCO_TO_ISAACLAB_DOF,
        )

        self.JOINT_NAMES = A3_ISAACLAB_JOINTS

        # ``A3_MUJOCO_TO_ISAACLAB_DOF`` indexes the *full* 31-DOF MuJoCo
        # output, including the two head slots.  It cannot be the inverse used
        # by ``convert`` because that helper receives/returns compact 29D
        # vectors.  Build the compact inverse from the bijective 29D mapping.
        compact_mujoco_to_isaaclab_dof = [0] * len(A3_ISAACLAB_TO_MUJOCO_DOF)
        for isaaclab_idx, mujoco_idx in enumerate(A3_ISAACLAB_TO_MUJOCO_DOF):
            compact_mujoco_to_isaaclab_dof[mujoco_idx] = isaaclab_idx
        self.DOF_MAPPINGS = {
            ("isaaclab", "mujoco"): A3_ISAACLAB_TO_MUJOCO_DOF,
            ("mujoco", "isaaclab"): compact_mujoco_to_isaaclab_dof,
        }
        self.BODY_MAPPINGS = {
            ("isaaclab", "mujoco"): A3_ISAACLAB_TO_MUJOCO_BODY,
            ("mujoco", "isaaclab"): A3_MUJOCO_TO_ISAACLAB_BODY,
        }
        self._motion_lib_mujoco_to_isaaclab_dof = A3_MUJOCO_TO_ISAACLAB_DOF

    @property
    def isaaclab_to_mujoco_dof(self):
        """29D IsaacLab policy order -> compact 29D MuJoCo order."""
        return self.DOF_MAPPINGS[("isaaclab", "mujoco")]

    @property
    def mujoco_to_isaaclab_dof(self):
        """Full 31D MuJoCo FK output -> 29D IsaacLab policy order."""
        return self._motion_lib_mujoco_to_isaaclab_dof

    @property
    def isaaclab_to_mujoco_body(self):
        """30-body IsaacLab order -> compact 30-body MuJoCo order."""
        return self.BODY_MAPPINGS[("isaaclab", "mujoco")]

    @property
    def mujoco_to_isaaclab_body(self):
        """Full 32-body MuJoCo FK output -> 30-body IsaacLab order."""
        return self.BODY_MAPPINGS[("mujoco", "isaaclab")]

    VR_3POINTS_BODY_NAMES = ["torso_Link", "left_wrist_yaw_Link", "right_wrist_yaw_Link"]
    FOOT_BODY_NAMES = ["left_ankle_roll_Link", "right_ankle_roll_Link"]


def load_qpos_from_csv(csv_path: str) -> torch.Tensor:
    """Load qpos [T, D] from CSV."""
    import pandas as pd

    return torch.from_numpy(pd.read_csv(csv_path).values.astype(np.float32))


def save_qpos_to_csv(qpos: torch.Tensor, csv_path: str):
    """Save qpos to CSV."""
    import pandas as pd

    data = qpos[0].cpu().numpy() if qpos.dim() == 3 else qpos.cpu().numpy()
    pd.DataFrame(data).to_csv(csv_path, index=False)


if __name__ == "__main__":
    import argparse
    import os
    import sys

    from isaacsim import SimulationApp

    _sim_app = SimulationApp({"headless": True})

    # isaaclab's package layout requires path bootstrapping
    import isaaclab

    _src = os.path.join(os.path.dirname(isaaclab.__file__), "source")
    for _sub in ["isaaclab", "isaaclab_contrib", "isaaclab_assets", "isaaclab_tasks", "isaaclab_rl"]:
        _p = os.path.join(_src, _sub)
        if os.path.isdir(_p):
            sys.path.insert(0, _p)
    isaaclab.__path__.insert(0, os.path.join(_src, "isaaclab", "isaaclab"))

    parser = argparse.ArgumentParser(description="Test IsaacLab <-> MuJoCo order conversion.")
    parser.add_argument(
        "--robot",
        type=str,
        default="a3",
        choices=["a3", "g1"],
        help="Order mapping to test; A3 is the public-release default.",
    )
    args = parser.parse_args()

    converter = A3Converter() if args.robot == "a3" else G1Converter()

    robot_name = args.robot.upper()
    T = 30
    num_dof = converter.num_dof

    qpos = torch.cat(
        [
            torch.tensor([[0.0, 0.0, 1.09]]).expand(T, 3),
            torch.tensor([[1.0, 0.0, 0.0, 0.0]]).expand(T, 4),
            torch.randn(T, num_dof) * 0.3,
        ],
        dim=-1,
    )

    qpos_mujoco = converter.to_mujoco(qpos)
    qpos_back = converter.to_isaaclab(qpos_mujoco)
    print(f"{robot_name} qpos round-trip error: {(qpos - qpos_back).abs().max():.2e}")

    dof_only = torch.randn(T, num_dof)
    dof_mj = converter.to_mujoco(dof_only)
    dof_back = converter.to_isaaclab(dof_mj)
    print(f"{robot_name} DOF-only round-trip error: {(dof_only - dof_back).abs().max():.2e}")

    qpos_mujoco2 = converter.convert(qpos, from_order="isaaclab", to_order="mujoco")
    print(f"to_mujoco vs convert match: {torch.allclose(qpos_mujoco, qpos_mujoco2)}")
