#!/usr/bin/env bash
# install_pico_minimal.sh
# Minimal venv environment for gear_sonic/scripts/pico_pose_zmq_minimal.py.
# Usage: bash install_scripts/install_pico_minimal.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python3.10}"
VENV_DIR="${VENV_DIR:-.venv_pico_minimal}"
XRT_DIR="$REPO_ROOT/external_dependencies/XRoboToolkit-PC-Service-Pybind_X86_and_ARM64"
WHEELHOUSE_DIR="${WHEELHOUSE_DIR:-$REPO_ROOT/wheelhouse}"
ARCH="$(uname -m)"
if [[ "$ARCH" == "aarch64" ]]; then
    XRT_LIB_DIR="$XRT_DIR/lib/aarch64"
else
    XRT_LIB_DIR="$XRT_DIR/lib"
fi

echo "[OK] Architecture: $ARCH"
echo "[INFO] Repo root: $REPO_ROOT"

if [[ ! -f "$REPO_ROOT/gear_sonic/scripts/pico_pose_zmq_minimal.py" ]]; then
    echo "[ERROR] Missing gear_sonic/scripts/pico_pose_zmq_minimal.py" >&2
    exit 66
fi
if [[ ! -f "$REPO_ROOT/gear_sonic/data/human/human_joints_info.npz" ]]; then
    echo "[ERROR] Missing gear_sonic/data/human/human_joints_info.npz" >&2
    exit 66
fi
if [[ ! -d "$XRT_DIR" ]]; then
    echo "[ERROR] Missing XRoboToolkit SDK source: $XRT_DIR" >&2
    exit 66
fi

if ! command -v c++ >/dev/null 2>&1 && ! command -v g++ >/dev/null 2>&1; then
    echo "[ERROR] C++ compiler not found. Install build-essential first." >&2
    exit 69
fi

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    echo "[ERROR] $PYTHON_BIN not found. Install python3.10/python3.10-venv or set PYTHON_BIN." >&2
    exit 69
fi
echo "[OK] Using Python: $($PYTHON_BIN --version)"

cd "$REPO_ROOT"
PIP_INDEX_ARGS=()
if [[ -d "$WHEELHOUSE_DIR" ]]; then
    echo "[INFO] Using local wheelhouse: $WHEELHOUSE_DIR"
    PIP_INDEX_ARGS=(--no-index --find-links "$WHEELHOUSE_DIR")
fi

echo "[INFO] Recreating $VENV_DIR..."
rm -rf "$VENV_DIR"
"$PYTHON_BIN" -m venv "$VENV_DIR"
# shellcheck disable=SC1090
source "$VENV_DIR/bin/activate"

python -m pip install "${PIP_INDEX_ARGS[@]}" --upgrade pip setuptools wheel

echo "[INFO] Installing minimal Python deps..."
python -m pip install "${PIP_INDEX_ARGS[@]}" \
    numpy==1.26.4 \
    scipy==1.15.3 \
    pyzmq \
    cmake \
    pybind11 \
    setuptools \
    wheel

if [[ "$ARCH" == "aarch64" && ! -f "$XRT_DIR/lib/aarch64/libPXREARobotSDK.so" ]]; then
    echo "[INFO] Building PXREARobotSDK native library for aarch64..."
    XRT_TMP="$XRT_DIR/tmp"
    mkdir -p "$XRT_TMP"
    if [[ ! -d "$XRT_TMP/XRoboToolkit-PC-Service" ]]; then
        git clone -b orin https://github.com/XR-Robotics/XRoboToolkit-PC-Service.git \
            "$XRT_TMP/XRoboToolkit-PC-Service"
    fi
    pushd "$XRT_TMP/XRoboToolkit-PC-Service/RoboticsService/PXREARobotSDK" >/dev/null
    bash build.sh
    popd >/dev/null
    mkdir -p "$XRT_DIR/lib/aarch64" "$XRT_DIR/include/aarch64"
    cp "$XRT_TMP/XRoboToolkit-PC-Service/RoboticsService/PXREARobotSDK/PXREARobotSDK.h" \
        "$XRT_DIR/include/aarch64/"
    cp -r "$XRT_TMP/XRoboToolkit-PC-Service/RoboticsService/PXREARobotSDK/nlohmann" \
        "$XRT_DIR/include/aarch64/nlohmann/"
    cp "$XRT_TMP/XRoboToolkit-PC-Service/RoboticsService/PXREARobotSDK/build/libPXREARobotSDK.so" \
        "$XRT_DIR/lib/aarch64/"
    rm -rf "$XRT_TMP"
fi

echo "[INFO] Installing xrobotoolkit_sdk..."
export CMAKE_PREFIX_PATH="$(python -m pybind11 --cmakedir)"
export LD_LIBRARY_PATH="$XRT_LIB_DIR:${LD_LIBRARY_PATH:-}"
python -m pip install "${PIP_INDEX_ARGS[@]}" --no-build-isolation -e "$XRT_DIR"

python - <<'PY'
import numpy
import scipy
import zmq
import xrobotoolkit_sdk

print("pico minimal imports ok")
PY

cat <<EOF

Setup complete. Start the minimal PICO ZMQ sender with:

  cd $REPO_ROOT
  source $VENV_DIR/bin/activate
  export LD_LIBRARY_PATH="$XRT_LIB_DIR:\${LD_LIBRARY_PATH:-}"
  python gear_sonic/scripts/pico_pose_zmq_minimal.py --port 5556 --target_fps 50 --num_frames_to_send 10

EOF
