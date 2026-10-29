#!/usr/bin/env bash
set -eo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
cd "${ROOT}"

source /agibot/software/v0/entry/env/env.sh
source /opt/ros/jazzy/setup.bash
source "${ROOT}/.cache/install/setup.bash"

SDK="${ROOT}/external_dependencies/XRoboToolkit-PC-Service-Pybind_X86_and_ARM64"
export LD_LIBRARY_PATH="${ROOT}/service_deps/usr/lib/aarch64-linux-gnu:${SDK}/lib/aarch64:${LD_LIBRARY_PATH:-}"
SERVICE="${ROOT}/service/opt/apps/roboticsservice/runService.sh"
if [[ "${1:-}" != "--help" && "${1:-}" != "-h" ]] && ! pgrep -f '[R]oboticsServiceProcess' >/dev/null; then
  if [[ ! -f "${SERVICE}" ]]; then
    echo "XRoboToolkit PC service not found: ${SERVICE}" >&2
    exit 1
  fi
  bash "${SERVICE}"
  sleep 1
fi

exec "${ROOT}/.venv/bin/python" "${SCRIPT_DIR}/scripts/pico_a3_retarget_publisher.py" \
  --source xrt --no-start-service "$@"
