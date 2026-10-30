#!/usr/bin/env bash
set -euo pipefail

# Generic public entry point for the 035 reward/robot contract.  It intentionally
# accepts an explicit motion-lib input instead of assuming an internal corpus.
REPO_DIR="${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
ISAAC_PYTHON="${ISAAC_PYTHON:-${PYTHON_BIN:-python}}"
MOTION_FILE="${MOTION_FILE:-}"
SMPL_MOTION_FILE="${SMPL_MOTION_FILE:-dummy}"
CHECKPOINT="${CHECKPOINT:-}"
EXP_CONFIG="${EXP_CONFIG:-manager/universal_token/all_modes/sonic_a3_035}"
PROJECT_NAME="${PROJECT_NAME:-TRL_A3_035}"
EXP_VAR="${EXP_VAR:-}"
NUM_ENVS="${NUM_ENVS:-4096}"
NUM_MINI_BATCHES="${NUM_MINI_BATCHES:-4}"
NUM_LEARNING_ITERATIONS="${NUM_LEARNING_ITERATIONS:-200000}"
NUM_PROCESSES="${NUM_PROCESSES:-1}"
NUM_MACHINES="${NUM_MACHINES:-1}"
MACHINE_RANK="${MACHINE_RANK:-0}"
MAIN_PROCESS_IP="${MAIN_PROCESS_IP:-}"
MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-29500}"
SAME_NETWORK="${SAME_NETWORK:-false}"
TIMESTAMP_INPUT="${TIMESTAMP:-}"
EXPERIMENT_DIR_INPUT="${EXPERIMENT_DIR:-}"
PREFLIGHT_ONLY=false
EXTRA_ARGS=()

export OMNI_KIT_ACCEPT_EULA="${OMNI_KIT_ACCEPT_EULA:-YES}"
export PRIVACY_CONSENT="${PRIVACY_CONSENT:-Y}"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
export TORCH_NCCL_BLOCKING_WAIT="${TORCH_NCCL_BLOCKING_WAIT:-1}"

usage() {
  cat <<USAGE
Usage: MOTION_FILE=/absolute/path/to/a3_motionlib \\
  bash $(basename "$0") [options] [Hydra overrides...]

Required environment:
  MOTION_FILE            One .pkl file, a directory of A3 motion-lib .pkl
                         files, or a comma-separated list of those paths.

Options:
  --num-processes N      Total accelerate process count (default: ${NUM_PROCESSES})
  --num-machines N       Number of machines (default: ${NUM_MACHINES})
  --machine-rank N       This machine's zero-based rank (default: ${MACHINE_RANK})
  --main-process-ip IP   Rank-0 reachable IP/hostname; required for N>1 machines
  --main-process-port P  Rendezvous TCP port for N>1 machines (default: ${MAIN_PROCESS_PORT})
  --same-network         Tell accelerate all machines are on one local network
  --isaac-python PATH    Isaac Lab Python; aliases --python-bin
  --checkpoint PATH      Load 035 PT policy/value weights, but reset optimizer,
                         scheduler, sampler, and trainer state for fine-tuning
  --preflight            Validate motion inputs and exit without creating logs
  -h, --help             Show this help

With no checkpoint, the launcher starts the 035 motor-constraint contract from
random initialization. With CHECKPOINT or --checkpoint, it warm-starts from a
release-provided or user-supplied 035 PT while keeping resume=false; this is
fine-tuning, not an exact full-state resume. It is suitable for a compact smoke
run after converting the public flat CSVs, or for a larger user-supplied corpus.

For more than one machine, use the same TIMESTAMP on every host (or explicitly
set a per-host EXPERIMENT_DIR), expose exactly NUM_PROCESSES/NUM_MACHINES GPUs
per host, and use a rank-0-reachable --main-process-ip/port pair.
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --num-processes)
      NUM_PROCESSES="$2"
      shift 2
      ;;
    --num-machines)
      NUM_MACHINES="$2"
      shift 2
      ;;
    --machine-rank)
      MACHINE_RANK="$2"
      shift 2
      ;;
    --main-process-ip)
      MAIN_PROCESS_IP="$2"
      shift 2
      ;;
    --main-process-port)
      MAIN_PROCESS_PORT="$2"
      shift 2
      ;;
    --same-network)
      SAME_NETWORK=true
      shift
      ;;
    --python-bin|--isaac-python)
      ISAAC_PYTHON="$2"
      shift 2
      ;;
    --checkpoint)
      CHECKPOINT="$2"
      shift 2
      ;;
    --preflight)
      PREFLIGHT_ONLY=true
      shift
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      EXTRA_ARGS+=("$1")
      shift
      ;;
  esac
done

[[ -n "${MOTION_FILE}" ]] || {
  echo "ERROR: set MOTION_FILE to an A3 motion-lib .pkl file/directory" >&2
  exit 1
}
[[ "${NUM_PROCESSES}" =~ ^[1-9][0-9]*$ ]] || {
  echo "ERROR: --num-processes must be a positive integer" >&2
  exit 1
}
[[ "${NUM_MACHINES}" =~ ^[1-9][0-9]*$ ]] || {
  echo "ERROR: --num-machines must be a positive integer" >&2
  exit 1
}
[[ "${MACHINE_RANK}" =~ ^[0-9]+$ ]] || {
  echo "ERROR: --machine-rank must be a non-negative integer" >&2
  exit 1
}
[[ "${MAIN_PROCESS_PORT}" =~ ^[1-9][0-9]*$ ]] && (( MAIN_PROCESS_PORT <= 65535 )) || {
  echo "ERROR: --main-process-port must be in 1..65535" >&2
  exit 1
}
if (( NUM_MACHINES == 1 )); then
  (( MACHINE_RANK == 0 )) || {
    echo "ERROR: --machine-rank must be 0 when --num-machines is 1" >&2
    exit 1
  }
else
  (( MACHINE_RANK < NUM_MACHINES )) || {
    echo "ERROR: --machine-rank must be smaller than --num-machines" >&2
    exit 1
  }
  (( NUM_PROCESSES >= NUM_MACHINES && NUM_PROCESSES % NUM_MACHINES == 0 )) || {
    echo "ERROR: --num-processes must be a multiple of --num-machines" >&2
    exit 1
  }
  [[ -n "${MAIN_PROCESS_IP}" ]] || {
    echo "ERROR: --main-process-ip is required when --num-machines is greater than 1" >&2
    exit 1
  }
  [[ -n "${TIMESTAMP_INPUT}" || -n "${EXPERIMENT_DIR_INPUT}" ]] || {
    echo "ERROR: multi-machine launch requires one shared TIMESTAMP or EXPERIMENT_DIR on every machine" >&2
    exit 1
  }
fi

LOCAL_PROCESSES=$((NUM_PROCESSES / NUM_MACHINES))

if [[ -n "${CHECKPOINT}" ]]; then
  [[ -f "${CHECKPOINT}" ]] || {
    echo "ERROR: --checkpoint/CHECKPOINT must name an existing 035 .pt file" >&2
    exit 1
  }
  CHECKPOINT="$(realpath "${CHECKPOINT}")"
  TRAIN_MODE="fine_tune"
  EXP_VAR="${EXP_VAR:-a3_035_finetune_selected20}"
else
  TRAIN_MODE="from_scratch"
  EXP_VAR="${EXP_VAR:-a3_035_fromscratch}"
fi

if [[ "${ISAAC_PYTHON}" != /* ]]; then
  ISAAC_PYTHON="$(command -v "${ISAAC_PYTHON}" || true)"
fi
[[ -n "${ISAAC_PYTHON}" && -x "${ISAAC_PYTHON}" ]] || {
  echo "ERROR: Isaac Python missing; activate the Isaac Lab environment or pass --isaac-python PATH" >&2
  exit 1
}

"${ISAAC_PYTHON}" - "${MOTION_FILE}" <<'PY'
from pathlib import Path
import sys

entries = [item.strip() for item in sys.argv[1].split(",") if item.strip()]
if not entries:
    raise SystemExit("ERROR: MOTION_FILE has no usable entries")

keys: set[str] = set()
count = 0
for raw in entries:
    path = Path(raw).expanduser()
    if path.is_file():
        candidates = [path]
    elif path.is_dir():
        candidates = sorted(path.rglob("*.pkl"))
    else:
        raise SystemExit(f"ERROR: motion input does not exist: {path}")
    for candidate in candidates:
        if candidate.name == "metadata.pkl":
            continue
        if candidate.suffix != ".pkl":
            continue
        key = candidate.stem
        if key in keys:
            raise SystemExit(f"ERROR: duplicate motion key in MOTION_FILE inputs: {key}")
        keys.add(key)
        count += 1
if count == 0:
    raise SystemExit("ERROR: no motion-lib .pkl files found")
print(f"preflight motion_files={count}")
PY

# Accelerate derives the local process count by dividing the global world size
# by the machine count.  Catch a common multi-node mistake before Isaac Sim
# starts: a different visible-GPU count on one host causes a deadlock or an
# invalid rank-to-device mapping later in torchrun.  A single node may expose
# more GPUs than requested and deliberately let Accelerate choose a subset.
if (( NUM_MACHINES > 1 )); then
  VISIBLE_GPU_COUNT="$("${ISAAC_PYTHON}" - <<'PY'
import torch
print(torch.cuda.device_count())
PY
)"
  [[ "${VISIBLE_GPU_COUNT}" =~ ^[0-9]+$ ]] || {
    echo "ERROR: could not determine visible CUDA GPU count" >&2
    exit 1
  }
  (( VISIBLE_GPU_COUNT == LOCAL_PROCESSES )) || {
    echo "ERROR: this machine exposes ${VISIBLE_GPU_COUNT} CUDA GPUs, but topology needs ${LOCAL_PROCESSES}; set CUDA_VISIBLE_DEVICES consistently" >&2
    exit 1
  }
fi

ACCEL_ARGS=(
  --num_processes="${NUM_PROCESSES}"
  --num_machines="${NUM_MACHINES}"
  --mixed_precision=no
  --dynamo_backend=no
)
if (( NUM_MACHINES > 1 )); then
  ACCEL_ARGS+=(
    --machine_rank="${MACHINE_RANK}"
    --main_process_ip="${MAIN_PROCESS_IP}"
    --main_process_port="${MAIN_PROCESS_PORT}"
  )
  if [[ "${SAME_NETWORK}" == true ]]; then
    ACCEL_ARGS+=(--same_network)
  fi
fi
if [[ "${NUM_PROCESSES}" -gt 1 ]]; then
  ACCEL_ARGS=(--multi_gpu "${ACCEL_ARGS[@]}")
fi

if [[ "${PREFLIGHT_ONLY}" == true ]]; then
  echo "035 ${TRAIN_MODE} preflight passed; exiting before logs or training are created."
  echo "Launch topology: total_processes=${NUM_PROCESSES} num_machines=${NUM_MACHINES} machine_rank=${MACHINE_RANK}"
  exit 0
fi

MOTION_FILE_LIST_OVERRIDE="++manager_env.commands.motion.motion_lib_cfg.motion_file=[${MOTION_FILE}]"
TIMESTAMP="${TIMESTAMP_INPUT:-$(date +%Y%m%d_%H%M%S)}"
EXPERIMENT_DIR="${EXPERIMENT_DIR_INPUT:-${REPO_DIR}/logs_rl/${PROJECT_NAME}/${EXP_CONFIG}_${EXP_VAR}-${TIMESTAMP}}"

cd "${REPO_DIR}"
mkdir -p "${EXPERIMENT_DIR}"
if (( NUM_MACHINES > 1 )); then
  LOG_FILE="${EXPERIMENT_DIR}/train.launcher.machine${MACHINE_RANK}.log"
else
  LOG_FILE="${EXPERIMENT_DIR}/train.rank0.log"
fi
exec > >(tee -a "${LOG_FILE}") 2>&1

echo "Launch contract: mode=${TRAIN_MODE} exp=${EXP_CONFIG} checkpoint=${CHECKPOINT:-null} resume=false"
echo "Launch contract: motion ListConfig=${MOTION_FILE_LIST_OVERRIDE}"
echo "Launch contract: num_envs=${NUM_ENVS} num_processes=${NUM_PROCESSES} local_processes=${LOCAL_PROCESSES} num_mini_batches=${NUM_MINI_BATCHES}"
echo "Launch topology: num_machines=${NUM_MACHINES} machine_rank=${MACHINE_RANK} main_process_ip=${MAIN_PROCESS_IP:-local} main_process_port=${MAIN_PROCESS_PORT}"

"${ISAAC_PYTHON}" -m accelerate.commands.launch "${ACCEL_ARGS[@]}" "${REPO_DIR}/gear_sonic/train_agent_trl.py" \
  +exp="${EXP_CONFIG}" \
  checkpoint="${CHECKPOINT:-null}" \
  +resume=false \
  experiment_dir="${EXPERIMENT_DIR}" \
  headless=true \
  use_wandb=false \
  num_envs="${NUM_ENVS}" \
  algo.config.num_mini_batches="${NUM_MINI_BATCHES}" \
  algo.config.num_learning_iterations="${NUM_LEARNING_ITERATIONS}" \
  algo.trl.report_to=tensorboard \
  "${MOTION_FILE_LIST_OVERRIDE}" \
  ++manager_env.commands.motion.motion_lib_cfg.smpl_motion_file="${SMPL_MOTION_FILE}" \
  exp_var="${EXP_VAR}" \
  "${EXTRA_ARGS[@]}"
