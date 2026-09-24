# A3 035: Training → Sim2Sim → Sim2Deploy

This is the single end-to-end guide for the A3 035 line in this repository. It
assumes that readers are already familiar with the SONIC paper and the
[NVIDIA GR00T-WholeBodyControl repository](https://github.com/NVlabs/GR00T-WholeBodyControl)
and its [project documentation](https://nvlabs.github.io/GR00T-WholeBodyControl/).
The focus here is therefore the A3-specific body, simulation, training, and
deployment details rather than a reintroduction to the general SONIC or
GR00T workflow.

It uses the 20 A3 flat CSV files included in the repository to:

1. convert CSV files into a training motion library;
2. train from scratch or fine-tune from 035 PT weights;
3. run closed-loop MuJoCo sim2sim with the same PT checkpoint;
4. export and validate ONNX, convert it to RKNN, and build an A3 deploy package;
5. bring up the package in stages with `latest_cache` state handling.

For the complete training parameters, see the [035 training contract](a3_035_training_contract.md).
For T–N curves, Jacobian tables, and passive-foot fitting material, see
[T–N / passive foot](a3_024_sim2real_materials/README.md).

## 0. Scope, A3 model, and environment

This repository retains the 035 code, configuration, A3 assets, 20 publishable
CSV examples, the 035 step-200k PT checkpoint download manifest
with its matching side-by-side configuration, and the ONNX/RKNN deploy paths.
The PT, ONNX/RKNN and Rockchip sysroot binaries are hosted on
[Hugging Face](https://huggingface.co/sonic-for-a3/sonic); download the required groups before use. Training and sim2sim
use the PT; an ONNX artifact cannot be converted back into a trainable
checkpoint.

From the repository root, download the groups required by your workflow:

```bash
python -m pip install huggingface_hub
python download_from_hf.py --component pt       # training / MuJoCo: PT + matching configs
python download_from_hf.py --component onnx     # full policies + encoder / decoder
python download_from_hf.py --component rknn     # RK3588 models + sidecars
python download_from_hf.py --component sysroot  # Rockchip cross-build input
# Or fetch everything:
python download_from_hf.py --component all
```

The A3 URDF/MJCF derives from the public
[AgibotTech/A3-A3U-robot-model](https://github.com/AgibotTech/A3-A3U-robot-model).

The policy action space contains 29 actuated A3 joints; the two head joints are
not part of the training action view. `g1`, `g1_dyn`, `g1_kin`, and `a3_fast`
are historical checkpoint/ONNX ABI names. They **do not** mean that this
repository targets G1 hardware and must not be renamed casually.

For environment setup, follow the upstream SONIC documentation:

- [Installation (Training)](https://nvlabs.github.io/GR00T-WholeBodyControl/getting_started/installation_training.html)
- [Installation (Deployment)](https://nvlabs.github.io/GR00T-WholeBodyControl/getting_started/installation_deploy.html)

Once the environment is ready, continue with the A3-specific workflow below.

## 1. Three sim2real design choices in 035

### 1.1 Two-stage passive feet: suppress toe-standing and forward lean through physical cost

With the earlier rigid, single-piece foot model, a policy could treat the
forefoot or toe as a low-cost support point, leading to toe-standing and a
forward-leaning body. On a real A3, that regime causes forefoot roll, toe
deformation, and redistribution of contact load. 035 therefore models each
foot as three segments—rear foot → forefoot → toe—with two physics-only passive
hinges per foot (four in total). They are not part of the 29-dimensional policy
action.

| Passive hinge | Start from rear foot | Stiffness | Damping | Travel |
| --- | ---: | ---: | ---: | ---: |
| forefoot | 90 mm | 2396.666 Nm/rad | 1.51683722 | ±0.06606093 rad |
| toe | 140 mm | 27.58407 Nm/rad | 0.0686125693 | ±0.23058250 rad |

Consequently, forefoot loading and roll carry more realistic spring-restoring,
joint-limit, contact-geometry, and dynamics costs, reducing the opportunity to
exploit a rigid-foot loophole. This "penalty" is an **indirect physical cost**;
it is not a standalone `foot_end_load` reward. The current 035 configuration
does not enable an explicit toe-load reward. Adding one would change the
published 035 training contract.

The training URDF and the sim2sim MJCF must be used as a matched pair:

```text
gear_sonic/data/assets/robot_description/urdf/a3/
  model_collision_optimized_passive_foot_twostage_fit_optimized.urdf
gear_sonic/data/assets/robot_description/mjcf/
  a3_t2d5_loop_passive_foot_twostage_fit_optimized.xml
```

#### Original bench data and its boundary

This release also includes the **original quasi-static force–displacement CSV
exports** for five A3 metal-shoe toe-cantilever locations (A–E; source labels
X90, 117, 140, 160, and 182 mm), under
[`docs/a3_024_sim2real_materials/03_passive_foot/raw/`](a3_024_sim2real_materials/03_passive_foot/raw/README.md).
They retain their original GB18030 bytes with no resampling, filtering, unit
conversion, or encoding conversion; the five files total only 2,612,163 bytes.
Each contains the instrument-exported summary table and the raw Result Table 2
with time, displacement, force, and compressive strain. The complete source
archive SHA-256, original member names, and per-file checksums are in
[`raw_manifest.csv`](a3_024_sim2real_materials/03_passive_foot/raw/raw_manifest.csv).

The data can be inspected alongside the
passive-foot stiffness/limit choice, but they are quasi-static curves: they do
not independently identify damping, dynamic contact behavior, full
fixture/calibration details, or a sim2real gain.

Although some passive-foot parameters were identified from physical measurements,
the model still differs substantially from real sole deformation. Its main
practical benefit so far is that the relatively soft toe discourages the policy
from supporting the robot's full weight on its toes. This reduces the tendency
to stand on tiptoe and improves standing and walking stability.

The passive joints also introduce drawbacks, including simulation instability
and excessive bending that the policy can sometimes exploit in simulation.
We are still exploring simpler ways to reduce the sim2real gap. **Use passive
feet cautiously when training your own policies.** They are included here to
fully document the training recipe used for the 035 checkpoint, rather than as
a general recommendation for new training runs.

### 1.2 T–N rewards for ankle/waist parallel joints: designed for real motor capability

The ankle and waist pitch/roll axes are parallel mechanisms coupled by two
motors. An implicit PD controller in simulation can request joint-space
torque/speed combinations that the two physical motors cannot jointly deliver.
035 therefore maps requests into motor coordinates with a position-dependent
Jacobian, then checks every motor against its torque–speed (T–N) capability:

$$
\dot q_{joint}=J(q)\dot q_{motor},\qquad
\dot q_{motor}=J(q)^{-1}\dot q_{joint},\qquad
\tau_{motor}=J(q)^T\tau_{joint}.
$$

For each motor, utilization is

$$
u=\frac{|\tau_{motor}|}{\tau_{max}(|\dot q_{motor}|)}.
$$

The 035 PFP78 reward curve remains at 58 Nm up to 16.755161 rad/s, then drops
linearly to zero at 23.561945 rad/s. `motor_tn_parallel` covers the paired
motors in both ankles and the waist; `motor_tn_serial` covers serial joints.
The central T–N penalty is:

$$
p_{TN}=0.05[\min(\operatorname{clip}(u,0,2),1)]^8+
5[\max(\operatorname{clip}(u,0,2)-1,0)]^2.
$$

035 also includes an 80%-threshold ankle motor speed/power term, a fixed 58 Nm
ankle-torque term, high-load shaping on both ankle/waist axes, and 85% ankle
joint-position protection. These read `computed_torque` before implicit-PD
effort clipping. They **only change the training reward**: they do not hard-limit
PhysX actuator effort and are not a deploy-time hardware protection mechanism.
The real robot still depends on current, thermal, position-limit, watchdog, and
operational safety systems.

### 1.3 Armature: a reflected-inertia starting point, not complete closed-chain dynamics

The nominal armature for a serial joint approximates motor-side effective
inertia reflected through the reduction ratio to the output shaft:

$$
I_{armature}=I_{effective,motor}N^2.
$$

For example, the training-side PFP78 uses
`30.118e-6 × 20.03² = 0.01208337` kg·m². The parallel ankles/waist then multiply
this by empirical factors in the current 035 training configuration:

| Joint coordinate | Current expression | Nominal armature |
| --- | --- | ---: |
| ankle pitch | `PFP78 × 5.333` | 0.0644406 |
| ankle roll | `PFP78 × 1.66562` | 0.0201263 |
| waist pitch | `PFP78 × 7.3` | 0.0882086 |
| waist roll | `PFP78 × 1.21` | 0.0146209 |

These four factors did not come from a sim2real rollout search. They are a
historical output-speed-matching heuristic. Starting from a PFP78 output-side
speed reference of $\omega_{ref}\approx25$ rad/s, use the following for a
parallel output joint with a speed limit $\omega_{joint,max}$. Here $r$ is
an **effective transmission-ratio multiplier relative to the PFP78 output
side**, not the actual reduction ratio obtained from CAD or a Jacobian:

$$
r\approx\frac{\omega_{ref}}{\omega_{joint,max}},\qquad
k\approx r^2=\left(\frac{25}{\omega_{joint,max}}\right)^2,\qquad
I_{joint}\approx I_{PFP78,out}k.
$$

Reflected inertia scales with the square of the effective transmission ratio,
so $\sqrt{k}\omega_{joint,max}$ returns to approximately 25 rad/s:

| Joint coordinate | Configured $k$ | Joint speed limit (rad/s) | $\sqrt{k}\omega_{joint,max}$ (rad/s) |
| --- | ---: | ---: | ---: |
| ankle pitch | 5.333 | 10.8 | 24.941 |
| ankle roll | 1.66562 | 19.37 | 24.999 |
| waist pitch | 7.3 | 9.24785 | 24.986 |
| waist roll | 1.21 | 22.7 | 24.970 |

Treat these values as a speed-normalization/effective-transmission
approximation, not as a full closed-chain Jacobian inertia matrix or physical
parameters identified on a test bench. The 25 rad/s is the historical
output-side/simulation speed reference used to select armature; it is **not**
the 23.561945 rad/s zero-torque speed of the 035 T–N reward curve. Training
also randomizes armature by 0.9–1.1× for ordinary joints and 0.8–1.2× for the
ankles/waist. PhysX `armature` can only specify diagonal joint terms, so the
closed-chain, compliance, and transmission limits discussed below remain
incompletely represented.

Sim2sim and training use the same canonical armature values from
`gear_sonic/utils/a3_motor_params.py`; sim2sim must not introduce a separate
planetary reflected-inertia approximation. For PFP78 the shared serial-joint
value is `30.118e-6 × 20.03² = 0.01208337` kg·m², and the parallel ankle/waist
coordinates use the same configured multipliers listed above. This keeps the
training and sim2sim physics conventions aligned.

A more complete parallel-mechanism model would use the configuration-dependent
closed-chain inertia matrix:

$$
\Gamma(q)=J(q)^{-1},\qquad M_{joint}(q)=\Gamma(q)^T\operatorname{diag}(I_m)\Gamma(q).
$$

However, PhysX `armature` can only express diagonal joint terms. The current
model therefore omits configuration-dependent off-diagonal coupling as well as
transmission compliance, backlash, friction, current loops, temperature,
latency, and contact load.

## 2. Convert the repository's selected20 CSV data into a training motion library

`a3_data/agibot_a3/` contains 20 A3 flat CSV files at 120 Hz with 38 columns.
They are the common examples for training, sim2sim, and normal deploy playback.
The training loader accepts only `.pkl` motion-library data, so convert them first:

```bash
python gear_sonic/data_process/convert_soma_csv_to_motion_lib.py \
  --input a3_data/agibot_a3 \
  --output .cache/motionlib_035_selected20 \
  --individual \
  --robot a3_29 \
  --fps 30 \
  --fps_source 120 \
  --num_workers 8

export MOTION_FILE="$PWD/.cache/motionlib_035_selected20"
```

`a3_29` removes the two head joints from the 31-DOF A3 CSV view. The following
preflight creates no logs and allocates no environments:

```bash
ISAAC_PYTHON="$(command -v python)" \
MOTION_FILE="$MOTION_FILE" \
bash train_a3_035_fromscratch.sh --preflight
```

## 3. Two 035 training commands

The two commands below use the repository's `selected20` set only to demonstrate
how to start 035 training and to provide a quick-start regression check. Training
`selected20` on top of 035 is not the recommended recipe for broad motion
tracking. For a large tracking run, replace `MOTION_FILE` with a large motion
dataset; for a particular user's requested behaviors, use the desired motion set
for targeted fine-tuning on top of 035.

For large-scale sources, start with [HiPHI](https://github.com/noitom-robotics/hiphi)
and the [BONES-SEED dataset](https://bones.studio/datasets/seed). HiPHI provides
large-scale human motion and synchronized interaction data, while BONES-SEED is
a broad everyday embodied-motion source. The data license, release format, and
retargeting quality still need to be checked for the intended experiment.

[SOMA Retargeter](https://github.com/NVIDIA/soma-retargeter) now includes an
`agibot-a3t3` robot configuration, so BONES-SEED or other human-motion sources
can be retargeted into A3-compatible robot motion as a fast data-preparation
route. HiPHI's BVH release can likewise be used as the upstream motion source:
convert its skeleton/joint naming into the SOMA-X skeleton format expected by
SOMA Retargeter, then run the A3 retargeting configuration. The exact BVH-to-
SOMA-X adapter should be validated against the released HiPHI joint names,
coordinate conventions, and contacts before using the result for training.

**Known issue in SOMA Retargeter v0.2.0:** A3 retargeting is generally good for
motions dominated by two-foot ground contact, such as standing and walking. It
currently produces more ground penetration for motions with richer contact
patterns, including sitting on the ground, kneeling, crawling, falling, and
standing up after a fall. Treat those motions as requiring contact-aware
post-processing and manual visual validation before adding them to a training
set; do not assume that a successful two-foot retarget guarantees valid
multi-contact geometry.

### 3.1 Train from scratch on selected20

This is the smallest reproducible mainline in the repository. The example uses
small smoke-test parameters. Remove the `NUM_ENVS` and
`NUM_LEARNING_ITERATIONS` overrides to use the script defaults of 4,096
environments and 200,000 iterations.

```bash
ISAAC_PYTHON="$(command -v python)" \
MOTION_FILE="$PWD/.cache/motionlib_035_selected20" \
NUM_ENVS=16 \
NUM_LEARNING_ITERATIONS=2 \
bash train_a3_035_fromscratch.sh --num-processes 1
```

### 3.2 Fine-tune the 035 PT on selected20

Download the released step-200k PT with `python download_from_hf.py --component pt`.
It is installed at `checkpoints/035_step200000/model_step_200000.pt` and verified against the release SHA-256. The side-by-side `config.yaml` is the matching release-portable configuration used
for export/evaluation. The PT is not an ONNX/RKNN artifact.

```bash
export MODEL_035_PT="$PWD/checkpoints/035_step200000/model_step_200000.pt"
printf '%s  %s\n' \
  '9cf33be2f4e602858b68ce31d5824113ab1dda250acaab5842b6d3bf88b70f2d' \
  "$MODEL_035_PT" | sha256sum -c -
test -f "${MODEL_035_PT%/*}/config.yaml"

ISAAC_PYTHON="$(command -v python)" \
MOTION_FILE="$PWD/.cache/motionlib_035_selected20" \
CHECKPOINT="$MODEL_035_PT" \
NUM_ENVS=16 \
NUM_LEARNING_ITERATIONS=2 \
bash train_a3_035_fromscratch.sh --num-processes 1
```

This mode loads policy/value weights while fixing `resume=false`: optimizer,
scheduler, sampler, environment state, and training step all restart. It is a
selected20 fine-tune/warm start, not a full-state resume of an older experiment.
Append `--preflight` first to validate inputs only.

## 4. PT sim2sim and ONNX export

Sim2sim uses the PT, the `a3_fast` encoder, the same two-stage passive-foot
MJCF, and selected20 CSV data. It is a regression gate, not proof of hardware
safety:

```bash
export MODEL_035_PT="$PWD/checkpoints/035_step200000/model_step_200000.pt"
python gear_sonic/scripts/sim2sim_a3_mujoco.py \
  --checkpoint "$MODEL_035_PT" \
  --motion a3_data/agibot_a3/001_walk_front_slow.csv \
  --encoder-mode a3_fast \
  --mjcf gear_sonic/data/assets/robot_description/mjcf/a3_t2d5_loop_passive_foot_twostage_fit_optimized.xml \
  --batch-once \
  --metrics-out /tmp/a3_035_sim2sim_metrics.json \
  --output-video /tmp/a3_035_sim2sim.mp4
```

For a quick smoke test, append `--max-policy-steps 10` and omit the video.
The example uses the pelvis anchor, 120 Hz CSV input (stride 4 to 30 Hz),
a 50 Hz policy clock, and zero actuator delay. It tests the software pipeline;
for a hardware-delay comparison, set `--action-delay-ms 10` explicitly and
record it as a different protocol. Any run that changes the foot asset, motion
delay, CSV sampling rate, encoder window, or checkpoint must be recorded as a new experiment.

Export/evaluation reads the `config.yaml` beside the checkpoint; a bare `.pt`
cannot reliably reconstruct the network. The released step-200k PT already has
its path-scrubbed matching configuration, which defaults to the selected20
motion-lib produced above. Override it explicitly if you write the motion-lib
elsewhere. Both 035 ONNX artifacts must satisfy
`obs_dict[1,1570] → action[1,29]`:

```bash
python gear_sonic/eval_agent_trl.py \
  +checkpoint="$MODEL_035_PT" \
  +headless=True \
  ++num_envs=1 \
  "++manager_env.commands.motion.motion_lib_cfg.motion_file=[$PWD/.cache/motionlib_035_selected20]" \
  +export_onnx_only=true
```

## 5. ONNX → RKNN and the deploy package

The default runtime configuration is already bound to the repository's 035
step-200k G1/A3-fast ONNX and RKNN paths. Fetch them with
`python download_from_hf.py --component onnx rknn` before use:

```text
gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/config/a3_runtime_config.yaml
```

The export command above writes the four ONNX files to
`checkpoints/035_step200000/exported/` and regenerates the adjacent
`model_config.yaml`. Only the full G1 and A3-fast policies are supported by
the RKNN converter; do not pass the standalone encoder or decoder.

Use a separate Python 3.10/3.11 environment for RKNN conversion: Toolkit 2.3.2
requires PyTorch ≤2.4.0 and would downgrade a newer Isaac Lab training environment.
CPU PyTorch is sufficient for conversion. Run from the repository root:

```bash
python -m venv .venv_rknn
.venv_rknn/bin/python -m pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cpu
.venv_rknn/bin/python -m pip install -r gear_sonic_deploy/requirements-rknn.txt
.venv_rknn/bin/python gear_sonic_deploy/scripts/convert_a3_onnx_to_rknn.py \
  --onnx \
    checkpoints/035_step200000/exported/model_step_200000_g1.onnx \
    checkpoints/035_step200000/exported/model_step_200000_a3_fast.onnx \
  --out-dir assets/a3_runtime/rknn_models/035_reexported \
  --target-platform rk3588 \
  --overwrite
```

A relative `--out-dir` is resolved beneath `gear_sonic_deploy/`. This example
keeps re-exported RKNN files separate from the downloaded release. The package
commands below still use the verified Hugging Face release artifacts. To package
your own exports, place both ONNX files in a new runtime model directory and
update the two ONNX paths and two RKNN paths in a copy of
`a3_runtime_config.yaml`; pass that copy with `--runtime-cfg`. Keep the generated
`.rknn.json` checksum sidecars beside the RKNN files.

Build the x86_64 package first. The Rockchip AArch64 sysroot is distributed
through Hugging Face; fetch and verify it first:

```bash
python download_from_hf.py --component sysroot
(cd gear_sonic_deploy/thirdparty/rockchip_sysroot && \
  sha256sum -c rockchip-1.0-aarch64-sysroot.tar.gz.sha256)
```

Rockchip still needs one local build input that is **not distributed with this
repository**: obtain or build a target-compatible aarch64 ONNX Runtime package from the
[official ONNX Runtime v1.19.2 release](https://github.com/microsoft/onnxruntime/releases/tag/v1.19.2),
and archive it with
`<install-prefix>/lib/cmake/onnxruntime/onnxruntimeConfig.cmake`.
The archive must match the target ABI. Set its path independently of the
optional sysroot-regeneration step:

```bash
export A3_ONNXRUNTIME_AARCH64_TARBALL='/absolute/path/to/onnxruntime-aarch64-1.19.2.tar.gz'
```

If the target BSP, OS, ROS ABI, or userspace does not match the bundled
sysroot, obtain a matching AArch64 NVIDIA Thor base image through NVIDIA's
official [Thor Docker setup guide](https://docs.nvidia.com/jetson/agx-thor-devkit/user-guide/latest/setup_docker.html),
load/tag it in the local Docker daemon, then regenerate the sysroot. This
repository intentionally does not prescribe an NVIDIA image tag or pull from
an implicit registry:

```bash
export A3_THOR_BASE_IMAGE='<local-nvidia-thor-image>'
gear_sonic_deploy/scripts/export_rockchip_sysroot.sh \
  --image "$A3_THOR_BASE_IMAGE"
```

For the A3 package, the native x86 build requires ROS 2 development packages
(Humble was tested) and ONNX Runtime C++ headers/libraries. The build script
sources `ROS_DISTRO` when set, otherwise an installed Jazzy or Humble environment.
Rockchip cross-compilation requires Docker and the two archives described above.
These package commands disable TensorRT; their inference backends are ONNX
Runtime CPU and RKNN respectively.

Then build:

```bash
gear_sonic_deploy/scripts/build_a3_deploy_pkg.sh \
  --arch x86_64 \
  --jobs 20 \
  --runtime-cfg gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/config/a3_runtime_config.yaml

gear_sonic_deploy/scripts/build_a3_deploy_pkg.sh \
  --arch rockchip \
  --jobs 20 \
  --onnxruntime-aarch64-tarball "$A3_ONNXRUNTIME_AARCH64_TARBALL" \
  --runtime-cfg gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/config/a3_runtime_config.yaml
```

The default sysroot output is
`gear_sonic_deploy/thirdparty/rockchip_sysroot/rockchip-1.0-aarch64-sysroot.tar.gz`.
It is downloaded with `python download_from_hf.py --component sysroot` as a build-only sysroot, not as a complete
Thor OS image. The ONNX Runtime archive remains a `.gitignore`d local input.
Missing files, mismatched architecture, or a wrong archive layout cause an
actionable error; the build never silently falls back to an incidental host
dependency.

The normal playback list points directly to the 20 CSV files in
`a3_data/agibot_a3/` to avoid a duplicate source-tree copy. The package builder
copies them into package-local `motions/`. The four directional
`remote_motions/` clips and the standing/idle `teleop_motions/` clips are
independent and do not enter the normal playback list.

## 6. Default runtime and staged bring-up

The default source-runtime semantics are:

| Setting | Default |
| --- | --- |
| policy clock | 50 Hz |
| `phase_align_to_sync` | `false` |
| `backend.sync_mode` | `latest_cache` |
| per-channel freshness | 10 ms |
| frame-age watchdog | 50 ms |
| x86 backend | ONNX Runtime CPU |
| packaged Rockchip backend | RKNN |

`latest_cache` updates each state topic independently and lets the fixed 50 Hz
policy loop read the newest valid sample at every tick. Timestamp skew is only a
diagnostic; it does not require six topics to form a 200 Hz coherent pair. The
10 ms gate is per-channel freshness. Only when a new state cannot continuously
be formed does the 50 ms frame-age watchdog execute safe halt. Do not switch to
a different synchronization mode on a robot without revalidating timing and
safety behavior.

### 6.1 MDU physical-robot bring-up

Run this only on an authorized target with a safety tether, emergency stop,
on-site supervision, and completed bench/sim2sim validation. The development
machine only builds and transfers packages; the HDU is only an SSH/rsync jump
host. Perform the service configuration, probe, and formal launch **on the
MDU**. Never allow SONIC and the system-provided `motion_control` to publish
body-drive commands concurrently.

1. On the MDU, make the Motion function group start only the agent. Back up the
   vendor configuration and edit it manually. Do not edit `sm_config.yaml.dump`
   or perform an unverified bulk `sed` rewrite.

   ```bash
   cd /agibot/software/v0/config/sm
   BACKUP="sm_config.yaml.before_sonic_$(date +%Y%m%d_%H%M%S)"
   sudo cp -a sm_config.yaml "$BACKUP"
   sudoedit sm_config.yaml
   ```

   In the existing `"Motion"` function group, change only the function list
   that originally contains `"mc"` to:

   ```text
   [ "agent" ]
   ```

   Preserve all other YAML content, indentation, and vendor configuration.

2. Let the service manager bring the agent path back up:

   ```bash
   sudo systemctl restart agibot_pm
   ```

   Wait about three minutes, then check:

   ```bash
   systemctl is-active --quiet agibot_pm
   ps aux | grep '[m]otion_control'
   ```

   The first command must succeed and the second must produce no output. That
   proves the stock motion control was not automatically restarted. Otherwise,
   stop here, restore the backup, and restart the service; do not start SONIC:

   ```bash
   sudo cp -a "$BACKUP" sm_config.yaml
   sudo systemctl restart agibot_pm
   ```

   This path is mutually exclusive with the legacy `systemctl stop agibot_pm`
   plus manual `start_hal_ethercat.sh` path. Do not mix them.

3. Enter the Rockchip package copied to the MDU and perform a read-only package
   preflight. Confirm that the transferred artifact is an aarch64/RKNN package,
   not source code or an x86 package:

   ```bash
   cd /agibot/<sonic-package-root>
   file ./a3_deploy_onnx_ref | grep -F 'ARM aarch64'
   grep -Eq '^[[:space:]]*backend:[[:space:]]*rknn[[:space:]]*$' \
     config/a3_runtime_config.yaml
   test -f models/model_step_200000_g1.rknn
   test -f models/model_step_200000_a3_fast.rknn
   ```

4. Run the **no-command-publish** policy probe first. 035 has no SMPL ONNX, so
   the probe must select `g1`; `run_a3_probe.sh` now defaults to that value.

   ```bash
   export A3_ROBOT_ENV=/agibot/software/v0/entry/env/env.sh
   test -r "$A3_ROBOT_ENV"

   A3_TRANSPORT=iceoryx \
   A3_PROBE_SOURCE=g1 \
   A3_LATENCY_LOG=verbose \
   taskset -c 4-5 ./run_a3_probe.sh
   ```

   `--probe` loads the 035 model, receives state, and measures inference
   latency, but forcibly disables the command publisher. Confirm that all six
   body-drive state streams are ready, freshness/watchdog behavior is normal,
   the model backend is correct, and there is no continuous stale state or safe
   halt before proceeding. To isolate a transport issue, run
   `taskset -c 4-5 ./run_a3.sh --dry-run` with the same CPU binding; it does not
   load a model.

5. Exit the probe and formally start the program with the same CPU binding:

   ```bash
   taskset -c 4-5 ./run_a3.sh
   ```

   `taskset -c 4-5` sets the initial CPU affinity inherited by ordinary
   threads. The Rockchip policy RT thread explicitly sets its own affinity to
   CPU 6 (`policy_driver.rt.cpu_affinity: 6`) and uses FIFO priority 80
   (`policy_driver.rt.sched_fifo_priority: 80`). These are separate settings:
   keep the launch command on CPUs 4–5 and the policy RT thread on CPU 6.
   Do not use `--auto-start`. The program begins in `IDLE` and sends no
   commands. Follow the terminal sequence in §6.2 before playing any motion,
   and use only short, small-amplitude motion for the first run.

### 6.2 Terminal operation and PD-stand check

Perform this sequence with the robot suspended by a safety gantry/sling or another
protective support:

1. While the robot is still suspended, press **`S`** to enter `PD_STAND`
   position control.
2. Lower the robot to the ground and let it settle. This position-control
   action is intentionally designed to let the robot stand stably on the
   ground. If the robot cannot stand after the external sling force is removed,
   stop the procedure and treat it as evidence of a hardware or calibration
   problem, such as an incorrect joint zero; do not continue to motion playback.
3. Once the robot is standing stably, press **`M`** to enter motion mode.
   With no motion command, the deploy program selects its built-in standing
   motion by default. Use **`[`** and **`]`** to switch the selected motion,
   then press **`R`** to play the selected action.
4. At any time, press **`P`** to enter `passive`. The whole body then relaxes
   into the passive state. Use this key whenever the robot becomes unstable or
   behaves abnormally.

Keep the robot on a safety gantry or equivalent protective setup during
debugging as far as practical. Emergency-stop coverage and on-site supervision remain required;
`P` is a software fallback and does not replace the physical emergency stop.
