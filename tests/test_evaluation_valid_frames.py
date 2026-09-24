import sys
import types

import numpy as np
import torch

from gear_sonic.evaluation.metrics import build_batch_payload
from gear_sonic.evaluation.valid_frames import valid_eval_frame_mask
from gear_sonic.trl.callbacks.im_eval_callback import ImEvalCallback


def test_valid_frame_mask_uses_motion_length_minus_one():
    motion_steps = torch.tensor([4])

    masks = [
        valid_eval_frame_mask(
            step,
            motion_steps,
            torch.tensor([False]),
            torch.tensor([False]),
        ).item()
        for step in range(4)
    ]

    assert masks == [True, True, True, False]


def test_valid_frame_mask_excludes_current_and_previous_termination():
    mask = valid_eval_frame_mask(
        3,
        torch.tensor([10, 10, 10]),
        torch.tensor([False, True, False]),
        torch.tensor([True, False, False]),
    )

    assert mask.tolist() == [False, False, True]


def test_valid_frame_count_freezes_at_first_termination():
    motion_steps = torch.tensor([10])
    terminated = torch.tensor([False])
    valid_count = torch.tensor([0])

    for step in range(6):
        terminating_now = torch.tensor([step == 2])
        valid_count += valid_eval_frame_mask(
            step, motion_steps, terminated, terminating_now
        ).long()
        terminated |= terminating_now

    assert valid_count.tolist() == [2]


def test_callback_uses_termination_aware_valid_frame_mask():
    callback = ImEvalCallback.__new__(ImEvalCallback)
    callback.curr_steps = 2
    callback.terminate_state = torch.tensor([False, True, False])
    callback.env = types.SimpleNamespace(
        motion_ids=torch.tensor([0, 1, 2]),
        _motion_lib=types.SimpleNamespace(
            get_motion_num_steps=lambda _ids: torch.tensor([10, 10, 3])
        ),
    )
    actor_state = {
        "dones": torch.tensor([True, False, False]),
        "extras": {"time_outs": torch.tensor([False, False, False])},
    }

    terminating_now, valid_frame = callback._eval_step_masks(actor_state)

    assert terminating_now.tolist() == [True, False, False]
    assert valid_frame.tolist() == [False, False, False]


def test_scalable_payload_preserves_zero_length_motion():
    smpl_eval = types.ModuleType("smpl_sim.smpllib.smpl_eval")

    def fake_compute(pred, gt, concatenate=False, use_tqdm=False):
        assert all(len(trajectory) > 0 for trajectory in pred)
        return {
            "mpjpe_g": [np.linalg.norm(p - g, axis=-1) for p, g in zip(pred, gt)],
            "mpjpe_l": [np.linalg.norm(p - g, axis=-1) for p, g in zip(pred, gt)],
            "vel_dist": [np.ones(max(len(p) - 1, 0)) for p in pred],
        }

    smpl_eval.compute_metrics_lite = fake_compute
    old_modules = {
        name: sys.modules.get(name)
        for name in ("smpl_sim", "smpl_sim.smpllib", "smpl_sim.smpllib.smpl_eval")
    }
    sys.modules["smpl_sim"] = types.ModuleType("smpl_sim")
    sys.modules["smpl_sim.smpllib"] = types.ModuleType("smpl_sim.smpllib")
    sys.modules["smpl_sim.smpllib.smpl_eval"] = smpl_eval
    try:
        body_names = [
            "pelvis",
            "left_hip_roll_link",
            "left_knee_link",
            "left_ankle_roll_link",
            "right_hip_roll_link",
            "right_knee_link",
            "right_ankle_roll_link",
            "torso_link",
            "left_wrist_yaw_link",
            "right_wrist_yaw_link",
            "left_shoulder_roll_link",
            "left_elbow_link",
            "right_shoulder_roll_link",
            "right_elbow_link",
        ]
        pred = np.zeros((3, 2, len(body_names), 3), dtype=np.float32)
        payload = build_batch_payload(
            pred_pos=pred,
            gt_pos=np.zeros_like(pred),
            motion_lengths=np.array([4, 4]),
            valid_lengths=np.array([0, 2]),
            motion_idx=np.array([0, 1]),
            terminated=np.array([True, False]),
            progress=np.array([0.0, 0.5]),
            body_names=body_names,
            run_id="test",
            rank=0,
            batch_idx=0,
        )
    finally:
        for name, module in old_modules.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module

    assert payload["length_per_motion"].tolist() == [0, 2]
    assert payload["counts_by_metric"]["mpjpe_g"].tolist() == [0, 2]
    assert payload["counts_by_metric"]["vel_dist"].tolist() == [0, 1]
