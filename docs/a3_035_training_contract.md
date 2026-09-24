# A3 035 Training Contract

This document records the public, direct 035 configuration shipped in this
focused release. It also records the reserved step-200,000 checkpoint location;
large private training archives and all other historical checkpoints remain out
of scope.

## Public configuration

The only public 035 training leaf is:

```text
gear_sonic/config/exp/manager/universal_token/all_modes/sonic_a3_035.yaml
```

It directly inherits the general A3 configuration and explicitly selects the
A3-fast encoder/tokenizer, passive-foot A3 asset, 035-compatible actuator settings, and
full motor-constraint reward bundle. It does not depend on a historical resume
launcher or an internal motion-corpus path.

| Item | 035 value |
| --- | --- |
| Robot configuration | `a3_024`, fitted two-stage passive foot |
| Policy joints | 29 active A3 joints; passive foot joints are physics-only |
| Encoders | `g1`, `a3_fast` |
| Decoders | `g1_dyn`, `g1_kin` |
| Auxiliary losses | `g1_recon=0.01`, `a3_fast_g1_latent=1.0` |
| A3-fast reference window | 10 frames at 20 ms spacing (0–180 ms) |
| Ankle gains | Kp 60, Kd 1.2 |
| Default rollout | 4096 environments/rank, 24 steps/environment |
| PPO | 5 epochs, 4 minibatches, clip 0.2 |
| Actor / critic LR | `2e-5` / `1e-3` |
| Discount / GAE | 0.99 / 0.95 |

The names containing `g1` are checkpoint/ONNX ABI names. Do not rename them
without migrating the checkpoint and export pipeline together.

## Reward bundle

| Reward term | Weight | Purpose |
| --- | ---: | --- |
| `motor_tn_serial` | -1.0 | Penalize serial-joint requests beyond motor T–N capability |
| `motor_tn_parallel` | -1.0 | Apply the T–N check after mapping ankle/waist joint demands to physical motors |
| `ankle_motor_speed` | -2.0 | Penalize ankle motor speed above 80% utilization |
| `ankle_motor_power` | -2.0 | Penalize ankle motor mechanical power above 80% utilization |
| `ankle_motor_fixed_torque` | -1.0 | Penalize ankle motor torque above 58 Nm |
| `parallel_ankle_cross` | -0.1 | Low-weight shaping for simultaneous high ankle-axis load |
| `parallel_waist_cross` | -0.05 | Low-weight shaping for simultaneous high waist-axis load |
| `ankle_position_85` | -10.0 | Quadratic protection near 85% of ankle soft limits |

Reward implementation is in
`gear_sonic/envs/manager_env/mdp/rewards.py`; motor curves and the three
parallel Jacobian tables are in `gear_sonic/utils/a3_motor_model.py` and
`gear_sonic/data/a3_motor_model/`. The rationale and the limits of the
approximation are documented in [T–N constraints and passive foot](a3_024_sim2real_materials/README.md).

## Included selected20 data

The release includes 20 selected raw A3 flat CSVs under `a3_data/agibot_a3/`.
These 20 examples are not part of the dataset used to train the 035 checkpoint.
They are enough to run the documented conversion, from-scratch smoke run,
warm-start fine-tune, sim2sim example, and normal deploy playback. They are not
claimed to be a substitute for a large training corpus.

The release includes the 20 selected CSV examples under `a3_data/agibot_a3/`;
the source snapshot and derived motion-library files are not included.

## Training modes

`train_a3_035_fromscratch.sh` takes an explicit `MOTION_FILE` motion-lib input.

- With no `CHECKPOINT`, it runs 035 from random initialization:
  `checkpoint=null`, `resume=false`.
- With the released 035 `.pt` (or another user-supplied compatible PT) as
  `CHECKPOINT`, it loads policy/value weights but keeps `resume=false`;
  optimizer, scheduler, sampler, environment state, and training step are
  reset. This is fine-tuning/warm-start, not full-state resume.

The released ONNX/RKNN pair cannot be used to continue training. The
step-200k PT and the export/evaluation-required sibling `config.yaml` are
downloaded to `checkpoints/035_step200000/` with:

```bash
python download_from_hf.py --component pt
```

The downloader verifies the published files against `a3_hf_manifest.json`.

The released config is a path-scrubbed derivative of the original resolved
training config; its detailed provenance and verification command are in
[the checkpoint README](../checkpoints/035_step200000/README.md). The public
workflow deliberately does not offer exact optimizer/data-stream resume.

## Released inference artifacts

The paired G1 and A3-fast ONNX/RKNN artifacts are downloaded to
`gear_sonic_deploy/assets/a3_runtime/`. Their exact checksums and I/O contract
are in the [runtime asset manifest](../gear_sonic_deploy/assets/a3_runtime/README.md).

Use the Hugging Face PT for training/sim2sim. Fetch ONNX/RKNN with
`python download_from_hf.py --component onnx rknn` before deploy. Do not
mix models merely because their tensor shapes match. Detailed commands are in
[the end-to-end guide](a3_training2sim2deploy.md).
