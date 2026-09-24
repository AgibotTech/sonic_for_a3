import sys
import types

import numpy as np

from gear_sonic.evaluation.metrics import compute_subset_metrics_lite


def test_subset_mpjpe_l_uses_full_body_pelvis_root():
    smpl_eval = types.ModuleType("smpl_sim.smpllib.smpl_eval")

    def fake_compute(pred, gt, concatenate=False, use_tqdm=False):
        return {
            "mpjpe_g": [np.linalg.norm(p - g, axis=-1) * 1000 for p, g in zip(pred, gt)],
            "mpjpe_l": [np.zeros(p.shape[:2]) for p in pred],
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
        # Bodies are pelvis, left foot, right foot. Both predicted feet are
        # shifted by +1 m while pelvis remains correct.
        gt = np.array([[[0.0, 0, 0], [1.0, 0, 0], [-1.0, 0, 0]]])
        pred = np.array([[[0.0, 0, 0], [2.0, 0, 0], [0.0, 0, 0]]])
        metrics = compute_subset_metrics_lite(
            [pred],
            [gt],
            subset_indices=[1, 2],
            root_idx=0,
        )
    finally:
        for name, module in old_modules.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module

    np.testing.assert_allclose(metrics["mpjpe_l"][0], [[1000.0, 1000.0]])


def test_subset_global_metrics_keep_original_world_positions():
    smpl_eval = types.ModuleType("smpl_sim.smpllib.smpl_eval")

    def fake_compute(pred, gt, concatenate=False, use_tqdm=False):
        return {
            "mpjpe_g": [np.linalg.norm(p - g, axis=-1) * 1000 for p, g in zip(pred, gt)],
            "mpjpe_l": [np.zeros(p.shape[:2]) for p in pred],
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
        gt = np.zeros((1, 3, 3))
        pred = np.ones((1, 3, 3))
        metrics = compute_subset_metrics_lite(
            [pred],
            [gt],
            subset_indices=[1, 2],
            root_idx=0,
        )
    finally:
        for name, module in old_modules.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module

    np.testing.assert_allclose(metrics["mpjpe_g"][0], np.sqrt(3) * 1000)
