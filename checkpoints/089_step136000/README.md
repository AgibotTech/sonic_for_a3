# A3 089 step-136,000 checkpoint

Download the PT and matching inference artifacts from [Hugging Face](https://huggingface.co/sonic-for-a3/sonic/tree/main/089_step136000), from the source repository root:

```bash
python -m pip install huggingface_hub
python download_from_hf.py --model 089 --component pt onnx rknn
(cd checkpoints/089_step136000 && sha256sum -c SHA256SUMS)
```

The PT is ignored by Git; the downloader verifies its size and SHA-256.

The checkpoint preserves the original policy, value, optimizer and scheduler tensors, and the training step. Internal training arguments and motion-library state are omitted; it is not an exact training-resume snapshot. The adjacent path-scrubbed configuration includes the later rigid-foot collision model, self-contact terms and dynamic ankle model. The public training example uses a different motion dataset and is not an exact original-environment resume.

The checkpoint provides G1 and A3-fast encoders, not an SMPL encoder. The matching ONNX and RKNN artifacts are organized alongside the PT in the Hub model directory.

## Training configuration compared with 035

The comparison below uses the released, path-scrubbed checkpoint configurations
and the public training contracts. It describes configuration differences, not
an ablation study: the effect of each change has not been isolated. The 089
configuration selects the later rigid-foot collision asset used for this release.

| Area | 035 | 089 |
| --- | --- | --- |
| Robot and feet | `a3_024` with fitted two-stage passive feet; passive joints are simulated but not policy actions. | `a3_exp023` with rigidly merged feet and no passive foot joints. Both use 29 active policy joints. |
| Ankle dynamics and delay | Passive-foot model with fitted gains; no dynamic ankle or actuator-delay settings specified in the released configuration. | Nominal dynamic ankle model with shared action delay; actuator delay is enabled with a 0–2 simulation-step range. |
| Self-collision | No self-collision or self-contact reward configuration specified in the released configuration. | Self-collision and self-contact sensors enabled; pelvis/torso and the left and right wrist roll/yaw pairs are excluded. Remaining detected self-contacts are penalized. |
| Wrist tracking | Wrist tracking points are at the wrist yaw link origins; local five-point tracking reward weight is 3.0. | Wrist tracking points have approximately 11 cm forward offsets; local five-point reward weight is 2.0. Adds a six-wrist-joint angle tracking reward (weight 0.5, std 0.15). |
| Joint and actuator randomization | Passive-foot stiffness is scaled 0.6–1.6 and damping ratio 0.75–1.25 at startup. At reset, ankle/waist armature scale is 0.8–1.2 and other joint armature scale is 0.9–1.1. | At reset, active-joint stiffness is scaled 0.8–1.2 and damping ratio 0.5–2.0. Waist and regular-joint armature scale is 0.5–1.5; ankle motor inertia scale is 0.7–2.0, with separate ankle friction randomization. |
| Upper-body pose augmentation | `cat_upper_body_poses` enabled with probability 0.5. | Disabled. |

The 089 self-contact penalty has weight -0.1 and a contact threshold of 1.0.
Both configurations retain the motor torque–speed, ankle speed/power/torque,
parallel-joint cross-load and ankle-position constraints. They also both include
the ankle acceleration penalty (`feet_acc`, weight -2.5e-6).

The policy and optimization setup is otherwise largely shared: both use the
G1 and A3-fast encoders, G1 dynamic and kinematic decoders, and the same
reconstruction and latent matching loss weights (0.01 and 1.0). A3-fast uses
10 reference frames at 20 ms intervals, spanning 0–180 ms. Both public launchers
default to 4,096 environments per process, 24 rollout steps, PPO with 5 epochs
and 4 minibatches, clip 0.2, actor/critic learning rates of 2e-5/1e-3, and
discount/GAE values of 0.99/0.95. The released checkpoints are at steps
200,000 and 136,000 respectively; this does not imply that their optimization
histories or exact resume states are interchangeable.

The release checkpoint and configuration are provided under the [Apache License 2.0](LICENSE); the binary is distributed separately from Git.

## Public configuration

The training leaf is
[`sonic_a3_089.yaml`](../../gear_sonic/config/exp/manager/universal_token/all_modes/sonic_a3_089.yaml),
and the launcher is [`train_a3_089.sh`](../../train_a3_089.sh). The launcher
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
[`a3_089_rigid_foot.urdf`](../../gear_sonic/data/assets/robot_description/urdf/a3/a3_089_rigid_foot.urdf),
SHA-256 `a85eef849d65105f5c222961589445f94b7d26e1726eba24b4781212423e441a`.
It uses the later collision geometry, with rigidly merged feet and self-collision
enabled. The [reviewed collision manifest](../../gear_sonic/config/collision/a3_089_rigid_foot_pairs.json)
excludes pelvis/torso, left wrist roll/yaw, and right wrist roll/yaw; self-contact
sensors and the self-contact penalty remain enabled for the other contacts.

Actuator delay is enabled with a configured 0–2 step range. The dynamic ankle
model is enabled in nominal mode, with shared action delay and randomized motor
inertia scale `0.7–2.0`. The configuration also randomizes actuator gains,
joint friction/armature, mass, contact material, joint offsets and pushes. The
complete ranges and update modes are in the training YAML.

The matching sim2sim asset is
[`a3_089_rigid_foot_loop.xml`](../../gear_sonic/data/assets/robot_description/mjcf/a3_089_rigid_foot_loop.xml).
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
[`rewards.py`](../../gear_sonic/envs/manager_env/mdp/rewards.py) and
[`a3_self_contact_reward.py`](../../gear_sonic/envs/manager_env/robots/a3_self_contact_reward.py).
Motor curves and parallel Jacobian tables are in
[`a3_motor_model.py`](../../gear_sonic/utils/a3_motor_model.py) and
[`a3_motor_model/`](../../gear_sonic/data/a3_motor_model/).

## Included selected20 data

The repository includes 20 example A3 flat CSV motions in
[`a3_data/agibot_a3/`](../../a3_data/agibot_a3/). Convert them to a motion
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

The PT is required for training and sim2sim. ONNX and RKNN are inference
artifacts and cannot resume training; use the matching 089 exports.

See the [training-to-deployment guide](../../docs/a3_training2sim2deploy.md) for motion
conversion, training, export and deployment commands.
