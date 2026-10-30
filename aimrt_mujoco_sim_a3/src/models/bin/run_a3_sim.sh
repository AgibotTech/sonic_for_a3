#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
model="${1:-a3_t3d0}"
case "$model" in
  a3_t3d0|a3u_t3d0) ;;
  *) echo "Usage: $0 [a3_t3d0|a3u_t3d0]" >&2; exit 2 ;;
esac
set +u
source /opt/ros/humble/setup.bash
set -u
export LD_LIBRARY_PATH="$PWD:$PWD/../lib:${LD_LIBRARY_PATH:-}"
if ! pgrep -x iox-roudi >/dev/null; then
  ./iox-roudi &
fi
exec ./aimrt_main --cfg_file_path="./cfg/${model}.yaml"
