# A3 089 Training Contract

This document records the public training configuration for the A3 089
step-136,000 checkpoint. The public configuration uses the later rigid-foot
collision asset selected for release. The original motion corpus and exact
training state are not included, so this is a reproducible configuration and
launch recipe, not an exact reproduction of the original run.

## Public configuration

The training leaf is
[`sonic_a3_089.yaml`](../gear_sonic/config/exp/manager/universal_token/all_modes/sonic_a3_089.yaml),
and the launcher is [`train_a3_089.sh`](../train_a3_089.sh). The launcher
requires an explicit motion library and does not depend on an internal data
path.

| Item | 089 value |
| --- | --- |
| Robot | `a3_exp023`, later 089 rigid-foot collision asset |
| Policy joints | 29 active A3 joints; no passive foot joints |
| Active encoders | `g1`, `a3_fast` |
| Active decoders | `g1_dyn`, `g1_kin` |
| Auxiliary losses | `g1_recon=0.01`, `a3_fast_g1_latent=1.0` |
| A3-fast reference | 10 frames at 20 ms spacing, covering 0–180 ms |
| Simulation / policy step | 5 ms / 20 ms (decimation 4) |
| Default rollout | 4096 environments per process, 24 steps per environment |
| PPO | 5 epochs, 4 minibatches, clip 0.2 |
| Actor / critic LR | `2e-5` / `1e-3` |
| Discount / GAE | 0.99 / 0.95 |

The configuration contains definitions for other encoders, but `smpl` and
`teleop` have zero sampling weight and are not active for this checkpoint.
Names containing `g1` are checkpoint and ONNX interface names; changing them
requires a corresponding checkpoint/export migration.

## Robot, collision and actuation

The training URDF is
[`a3_089_rigid_foot.urdf`](../gear_sonic/data/assets/robot_description/urdf/a3/a3_089_rigid_foot.urdf),
SHA-256 `a85eef849d65105f5c222961589445f94b7d26e1726eba24b4781212423e441a`.
It uses the later collision geometry, with rigidly merged feet and self-collision
enabled. The [reviewed collision manifest](../gear_sonic/config/collision/a3_089_rigid_foot_pairs.json)
excludes pelvis/torso, left wrist roll/yaw, and right wrist roll/yaw; self-contact
sensors and the self-contact penalty remain enabled for the other contacts.

Actuator delay is enabled with a configured 0–2 step range. The dynamic ankle
model is enabled in nominal mode, with shared action delay and randomized motor
inertia scale `0.7–2.0`. The configuration also randomizes actuator gains,
joint friction/armature, mass, contact material, joint offsets and pushes. The
complete ranges and update modes are in the training YAML.

The matching sim2sim asset is
[`a3_089_rigid_foot_loop.xml`](../gear_sonic/data/assets/robot_description/mjcf/a3_089_rigid_foot_loop.xml).
It carries the corresponding collision geometry and exclusions, but its MuJoCo
closed-loop simulation does not reproduce all Isaac Lab dynamics or training
randomization.

## Reward bundle

The table lists the A3 motor constraints and 089-specific additions; tracking,
action-rate, joint-limit and contact terms are also enabled in the YAML.

| Reward term | Weight | Purpose |
| --- | ---: | --- |
| `motor_tn_serial` | -1.0 | Penalize serial joint demands beyond motor T–N capability |
| `motor_tn_parallel` | -1.0 | Evaluate parallel ankle/waist demands in motor space |
| `ankle_motor_speed` | -2.0 | Penalize ankle motor speed above 80% utilization |
| `ankle_motor_power` | -2.0 | Penalize ankle motor power above 80% utilization |
| `ankle_motor_fixed_torque` | -1.0 | Penalize ankle motor torque above 58 Nm |
| `parallel_ankle_cross` | -0.1 | Shape simultaneous ankle-axis load |
| `parallel_waist_cross` | -0.05 | Shape simultaneous waist-axis load |
| `ankle_position_85` | -10.0 | Penalize ankle positions near 85% of soft limits |
| `feet_acc` | -2.5e-6 | Penalize ankle joint acceleration |
| `tracking_wrist_joint_angle` | 0.5 | Track the six wrist joints |
| `self_collision_contacts` | -0.1 | Penalize self-contact detected by the configured sensors |

Reward implementations are in
[`rewards.py`](../gear_sonic/envs/manager_env/mdp/rewards.py) and
[`a3_self_contact_reward.py`](../gear_sonic/envs/manager_env/robots/a3_self_contact_reward.py).
Motor curves and parallel Jacobian tables are in
[`a3_motor_model.py`](../gear_sonic/utils/a3_motor_model.py) and
[`a3_motor_model/`](../gear_sonic/data/a3_motor_model/).

## Included selected20 data

The repository includes 20 example A3 flat CSV motions in
[`a3_data/agibot_a3/`](../a3_data/agibot_a3/). Convert them to a motion
library before training; they are sufficient for the documented software smoke
run, but are not a substitute for the original training corpus. The original
corpus and derived motion library are not included.

## Training modes and artifacts

`train_a3_089.sh` accepts a `MOTION_FILE` pointing to a motion-library PKL, a
directory, or a comma-separated list of inputs.

- Without `CHECKPOINT`, it starts from random initialization with `resume=false`.
- With a compatible PT passed as `CHECKPOINT` or `--checkpoint`, it loads
  policy/value weights with `resume=false`; optimizer, scheduler, sampler,
  environment and training step restart. This is a warm start, not full-state
  resume.

The checkpoint's optimizer and scheduler tensors are preserved, but the release
omits internal training arguments and motion-library state. Its PT and adjacent
configuration belong in [`checkpoints/089_step136000/`](../checkpoints/089_step136000/);
the checkpoint README records the SHA-256 verification command and release
status. The PT is required for training and sim2sim. ONNX and RKNN are
inference artifacts and cannot resume training; use the matching 089 exports.

See the [training-to-deployment guide](a3_training2sim2deploy.md) for motion
conversion, training, export and deployment commands.
