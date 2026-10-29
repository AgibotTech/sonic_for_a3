# A3 089: Training → Sim2Sim → Sim2Deploy

This guide uses the 089 / step-136,000 A3 policy and its later rigid-foot
collision contract. The release has G1 and A3-fast encoders; it has no SMPL
encoder. The PT is required for training and sim2sim; ONNX and RKNN files are
inference artifacts.

The 089 binaries have not yet been publicly uploaded. Until a release location
is available, obtain the artifact bundle separately and check the PT using
`checkpoints/089_step136000/SHA256SUMS`. The examples below assume execution from the repository root
with the SONIC/Isaac Lab environment activated; see the upstream
[training installation](https://nvlabs.github.io/GR00T-WholeBodyControl/getting_started/installation_training.html)
and [deployment installation](https://nvlabs.github.io/GR00T-WholeBodyControl/getting_started/installation_deploy.html).

## 1. Training contract

The launch configuration is
[`sonic_a3_089.yaml`](../gear_sonic/config/exp/manager/universal_token/all_modes/sonic_a3_089.yaml).
Its robot asset is the **later**
[`a3_089_rigid_foot.urdf`](../gear_sonic/data/assets/robot_description/urdf/a3/a3_089_rigid_foot.urdf)
(SHA-256 `a85eef849d65105f5c222961589445f94b7d26e1726eba24b4781212423e441a`).
The foot geometry is rigidly merged; the 29 policy joints do not include
passive foot joints. Self-collision is enabled with three reviewed filters:
pelvis/torso, left wrist roll/yaw, and right wrist roll/yaw. The filters and
asset hash are in [`a3_089_rigid_foot_pairs.json`](../gear_sonic/config/collision/a3_089_rigid_foot_pairs.json).
Self-contact sensing and its reward, wrist-joint tracking, delayed actuation,
and the configuration-dependent ankle inertia model are enabled by the
configuration. The previous passive-foot asset is not used for this training.

The original training corpus is not included. The public example below is a
small software/recipe example, not a reproduction of the original dataset,
training trajectory, or checkpoint quality. Motion conversion and retargeting
must preserve A3 joint order and coordinate conventions.

## 2. Prepare a motion library and check inputs

```bash
python gear_sonic/data_process/convert_soma_csv_to_motion_lib.py \
  --input a3_data/agibot_a3 \
  --output .cache/motionlib_089_selected20 \
  --individual --robot a3_29 --fps 30 --fps_source 120 --num_workers 8

export MOTION_FILE="$PWD/.cache/motionlib_089_selected20"
ISAAC_PYTHON="$(command -v python)" MOTION_FILE="$MOTION_FILE" \
  bash train_a3_089.sh --preflight
```

Preflight checks the motion input, URDF hash, collision mesh hashes and the
three-filter manifest. Supply a larger compatible motion library for an actual
tracking experiment; a directory, a PKL file, or a comma-separated list is
accepted. The converted example does not include the original motion metadata.

## 3. Train or fine-tune

For a small functional run from random initialization:

```bash
ISAAC_PYTHON="$(command -v python)" MOTION_FILE="$MOTION_FILE" \
  NUM_ENVS=16 NUM_LEARNING_ITERATIONS=2 \
  bash train_a3_089.sh --num-processes 1
```

For a warm start from the release checkpoint:

```bash
export MODEL_089_PT="$PWD/checkpoints/089_step136000/model_step_136000.pt"
(cd checkpoints/089_step136000 && sha256sum -c SHA256SUMS)
ISAAC_PYTHON="$(command -v python)" MOTION_FILE="$MOTION_FILE" \
  CHECKPOINT="$MODEL_089_PT" NUM_ENVS=16 NUM_LEARNING_ITERATIONS=2 \
  bash train_a3_089.sh --num-processes 1
```

The launcher loads policy/value weights with `resume=false`; optimizer,
scheduler, sampler, environment state and step restart. This is fine-tuning,
not continuation of the original run. Omit the small-run overrides for the
launcher defaults; inspect `bash train_a3_089.sh --help` for multi-process
options. Set a unique `EXP_VAR` and `EXPERIMENT_DIR` for each run. Check the
motion data and GPU capacity before increasing the workload.

## 4. Sim2sim and export

The closed-loop MuJoCo asset
[`a3_089_rigid_foot_loop.xml`](../gear_sonic/data/assets/robot_description/mjcf/a3_089_rigid_foot_loop.xml)
uses collision shapes and poses from the later 089 URDF and the same three
explicit body-pair exclusions. Its ankle/waist linkages and contact solver are
MuJoCo's closed-loop model; it does not implement the Isaac Lab dynamic-ankle
correction or training domain randomization, so it is a sim2sim approximation,
not identical training physics. The runner uses the 089 policy control gains
and action scales. Run the full selected motion headlessly:

```bash
export MODEL_089_PT="$PWD/checkpoints/089_step136000/model_step_136000.pt"
python gear_sonic/scripts/sim2sim_a3_mujoco.py \
  --checkpoint "$MODEL_089_PT" \
  --motion a3_data/agibot_a3/001_walk_front_slow.csv \
  --encoder-mode a3_fast --batch-once \
  --metrics-out /tmp/a3_089_sim2sim_metrics.json
```

For a short smoke check, add `--max-policy-steps 50`. The full command was
exercised on one 33-second walk clip; it ran without a detected fall, but
position tracking error was substantial and should be evaluated on the
intended motions before drawing performance or hardware conclusions.

To regenerate the MJCF collision geometry after changing the source URDF,
run `python gear_sonic/data_process/build_a3_089_mjcf.py` and the contract test
`python tests/test_089_training_contract.py -v`.

Isaac Lab scene initialization and ONNX export use the 089 URDF:

```bash
export MODEL_089_PT="$PWD/checkpoints/089_step136000/model_step_136000.pt"
python gear_sonic/eval_agent_trl.py \
  +checkpoint="$MODEL_089_PT" +headless=True ++num_envs=1 \
  "++manager_env.commands.motion.motion_lib_cfg.motion_file=[$MOTION_FILE]" \
  ++manager_env.commands.motion.motion_lib_cfg.motion_metadata_files=null \
  +export_onnx_only=true
```

The command reads the adjacent `config.yaml` and writes G1, A3-fast, encoder
and decoder ONNX files in `checkpoints/089_step136000/exported/`. The two full
policies use `obs_dict[1,1570] → action[1,29]`. An export smoke check does not
establish closed-loop stability.

## 5. Prepare ONNX/RKNN and build deploy packages

Given the 089 bundle containing `onnx/` and `rknn/`, stage its verified full
policies and RKNN sidecars:

```bash
python gear_sonic_deploy/scripts/prepare_a3_089_runtime.py \
  --source /path/to/089_136k_ckpt_onnx_rknn --include-rknn

bash gear_sonic_deploy/scripts/build_a3_deploy_pkg.sh --arch x86_64 --jobs 20 \
  --runtime-cfg gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/config/a3_runtime_config_089.yaml
```

The native package requires ROS 2 development packages and ONNX Runtime C++.
For RK3588 cross-compilation, obtain the Rockchip sysroot with
`python download_from_hf.py --component sysroot`, provide an ABI-compatible
AArch64 ONNX Runtime archive, and run:

```bash
export A3_ONNXRUNTIME_AARCH64_TARBALL=/absolute/path/to/onnxruntime-aarch64.tar.gz
bash gear_sonic_deploy/scripts/build_a3_deploy_pkg.sh --arch rockchip --jobs 20 \
  --onnxruntime-aarch64-tarball "$A3_ONNXRUNTIME_AARCH64_TARBALL" \
  --runtime-cfg gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/config/a3_runtime_config_089.yaml
```

To convert newly exported full policies, use a separate Python 3.10/3.11
RKNN environment and `gear_sonic_deploy/scripts/convert_a3_onnx_to_rknn.py`;
update the ONNX and RKNN paths in a copy of the runtime YAML. Keep the generated
sidecars with the RKNN models. Do not pass the standalone encoder/decoder to the
full-policy RKNN converter.

## 6. Simulation teleoperation and hardware bring-up

PICO body tracking is retargeted on the HDU and sent over ROS 2 to the
089 A3-fast policy on the MDU. The publisher uses the existing
`/ta/whole_body_command` interface. Prepare and start it as described in §6.3.

Before any real-robot deployment, validate the matching package and simulated
behavior, use an authorized target with a safety support and emergency stop,
and run the package's no-command-publish `run_a3_probe.sh` first. The probe
must show fresh body-drive state and the intended model/backend before a formal
run. Do not run SONIC concurrently with another body-drive command publisher.
A successful simulation or probe is not proof of hardware safety.

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
   test -f models/model_step_136000_g1.rknn
   test -f models/model_step_136000_a3_fast.rknn
   ```

4. Run the **no-command-publish** policy probe first. 089 has no SMPL ONNX, so
   the probe must select `g1`; `run_a3_probe.sh` now defaults to that value.

   ```bash
   export A3_ROBOT_ENV=/agibot/software/v0/entry/env/env.sh
   test -r "$A3_ROBOT_ENV"

   A3_TRANSPORT=iceoryx \
   A3_PROBE_SOURCE=g1 \
   A3_LATENCY_LOG=verbose \
   taskset -c 4-5 ./run_a3_probe.sh
   ```

   `--probe` loads the 089 model, receives state, and measures inference
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

### 6.3 XR teleoperation on the HDU

The HDU receives PICO tracking data and publishes A3 references to the MDU
through `/ta/whole_body_command`. The following setup uses Python 3.11 and
ROS 2 Jazzy on an AArch64 HDU.

**1. Upload the files.** From the repository root on the development machine:

```bash
HDU='agi@<hdu-address>'
tar -czf /tmp/xr_retarget.tar.gz \
  gear_sonic_deploy/xr_retarget \
  gear_sonic_deploy/assets/a3_runtime/teleop_motions/BMD_0319_a3_filtered_20260319_164305__stand_Skeleton0.csv \
  external_dependencies/XRoboToolkit-PC-Service-Pybind_X86_and_ARM64
ssh "$HDU" 'mkdir -p /agibot/sonic_a3_xr_retarget'
scp /tmp/xr_retarget.tar.gz "$HDU:/agibot/sonic_a3_xr_retarget/"
```

The AArch64 XRoboToolkit PC service package is provided separately. Copy it to
`/agibot/sonic_a3_xr_retarget/roboticsservice_arm64.deb` on the HDU. Ensure the
SDK directory contains `lib/aarch64/libPXREARobotSDK.so`.

**2. Prepare the HDU environment once.** The HDU needs `colcon`, CMake, a C++
compiler and Python venv support:

```bash
cd /agibot/sonic_a3_xr_retarget
tar -xzf xr_retarget.tar.gz
dpkg-deb -x roboticsservice_arm64.deb service
python3 -m venv --system-site-packages .venv
.venv/bin/python -m pip install numpy scipy pyyaml mujoco mink daqp 'protobuf>=6.30,<7' pybind11 wheel
SDK="$PWD/external_dependencies/XRoboToolkit-PC-Service-Pybind_X86_and_ARM64"
export LD_LIBRARY_PATH="$SDK/lib/aarch64:${LD_LIBRARY_PATH:-}"
export CMAKE_PREFIX_PATH="$(.venv/bin/python -m pybind11 --cmakedir)"
.venv/bin/python -m pip install --no-build-isolation -e "$SDK"
source /agibot/software/v0/entry/env/env.sh
source /opt/ros/jazzy/setup.bash
colcon --log-base .cache/log build --base-paths gear_sonic_deploy/xr_retarget/ros2 \
  --build-base .cache/build --install-base .cache/install \
  --cmake-args -DPython3_EXECUTABLE=/usr/bin/python3
```

On Debian 12, the PC service used in the verified setup also needs ICU 70. If
it is missing, extract an AArch64 `libicu70` package into `service_deps` with
`dpkg-deb -x /path/to/libicu70_arm64.deb service_deps`.

**3. Start the publisher on the HDU.** The script loads the robot environment
before ROS 2, which is required for communication with deploy, and starts the
PC service if needed:

```bash
cd /agibot/sonic_a3_xr_retarget
./gear_sonic_deploy/xr_retarget/run_xr_retarget.sh
```

In the PICO tracker app, complete body-tracker calibration, enable full-body tracking and **Send**,
and disable **Switch w/ A Button**. Match the robot's standing pose, hold
**A+B+X+Y** to calibrate retargeting, release the buttons, and press **A** to
start or pause streaming. Holding left Menu temporarily pauses the stream.

On the MDU, after the probe and standing checks in §6.1–6.2, use the deploy
terminal to switch modes before teleoperation:

1. Press **T** to enter teleoperation mode.
2. Press **3** to select the **A3-fast** teleoperation source.

The logs should show `first whole_body_command received` and
`active source -> a3_fast`. Press **P** to return to passive. If the topic is
publishing but the reference does not move, check the PICO tracker app
calibration.
