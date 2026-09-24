#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  gear_sonic_deploy/scripts/export_rockchip_sysroot.sh [--image IMAGE] [--output PATH]

Options:
  --image IMAGE    Locally available, matching AArch64 NVIDIA Thor base image.
                   Required; defaults to A3_THOR_BASE_IMAGE when set.
  --output PATH    Output tarball.
                   Default: gear_sonic_deploy/thirdparty/rockchip_sysroot/rockchip-1.0-aarch64-sysroot.tar.gz
  -h, --help       Show this help message.

Obtain and locally tag/load the matching NVIDIA Thor base image before running
this script. The script never pulls from a project-specific registry. It
creates a container from the local arm64 image and exports the target sysroot
pieces needed by Dockerfile.a3-rockchip-builder. It does not execute inside the
arm64 container, so qemu is not required.
USAGE
}

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
GEAR_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

IMAGE="${A3_THOR_BASE_IMAGE:-}"
OUTPUT="${GEAR_ROOT}/thirdparty/rockchip_sysroot/rockchip-1.0-aarch64-sysroot.tar.gz"
PLATFORM="linux/arm64"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --image)
      IMAGE="${2:-}"
      shift 2
      ;;
    --output)
      OUTPUT="${2:-}"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "unknown argument: $1" >&2
      usage >&2
      exit 64
      ;;
  esac
done

if [[ -z "${IMAGE}" ]]; then
  cat >&2 <<'EOF'
--image is required (or set A3_THOR_BASE_IMAGE).

First obtain a local, matching AArch64 NVIDIA Thor base image from NVIDIA's
official Thor/JetPack distribution, then pass its local Docker image name.
This project deliberately does not prescribe an NVIDIA image tag because the
required image must match the target BSP, OS, and ROS ABI.
EOF
  exit 64
fi
if [[ -z "${OUTPUT}" ]]; then
  echo "--output cannot be empty" >&2
  exit 64
fi
if ! command -v docker >/dev/null 2>&1; then
  echo "docker is required" >&2
  exit 69
fi

mkdir -p "$(dirname "${OUTPUT}")"
tmp_dir="$(mktemp -d "${TMPDIR:-/tmp}/rockchip-sysroot-export.XXXXXX")"
container_id=""

cleanup() {
  if [[ -n "${container_id}" ]]; then
    docker rm -f "${container_id}" >/dev/null 2>&1 || true
  fi
  rm -rf "${tmp_dir}"
}
trap cleanup EXIT

if ! docker image inspect "${IMAGE}" >/dev/null 2>&1; then
  cat >&2 <<EOF
local Thor base image not found: ${IMAGE}

Obtain the matching AArch64 NVIDIA Thor base image from NVIDIA's official
distribution, load or tag it in the local Docker daemon, then rerun with
--image <local-image> (or A3_THOR_BASE_IMAGE=<local-image>). This script does
not pull images from an implicit registry.
EOF
  exit 66
fi

container_id="$(docker create --platform "${PLATFORM}" "${IMAGE}" /bin/true)"

mkdir -p "${tmp_dir}/sysroot"

echo "extracting target sysroot paths from ${IMAGE}"
docker export "${container_id}" | tar --no-same-owner -C "${tmp_dir}/sysroot" -xf - \
  opt/ros/jazzy \
  usr/include \
  usr/share/eigen3 \
  usr/lib/aarch64-linux-gnu

tmp_output="${OUTPUT}.tmp"
rm -f "${tmp_output}"
tar --format=posix --sort=name --owner=0 --group=0 --numeric-owner \
  --mtime='@0' --pax-option=delete=atime,delete=ctime \
  -C "${tmp_dir}/sysroot" -cf - \
  opt/ros/jazzy \
  usr/include \
  usr/share/eigen3 \
  usr/lib/aarch64-linux-gnu | gzip -n > "${tmp_output}"
mv -f "${tmp_output}" "${OUTPUT}"

if command -v sha256sum >/dev/null 2>&1; then
  (
    cd "$(dirname "${OUTPUT}")"
    sha256sum "$(basename "${OUTPUT}")" > "$(basename "${OUTPUT}").sha256"
  )
fi

echo "Rockchip sysroot bundle ready: ${OUTPUT}"
