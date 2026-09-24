# Rockchip AArch64 Sysroot Bundle

Rockchip deployment builds use an x86_64 builder image plus this separately downloaded,
metadata-sanitized AArch64 target sysroot:

```text
gear_sonic_deploy/thirdparty/rockchip_sysroot/rockchip-1.0-aarch64-sysroot.tar.gz
```

It is hosted on [Hugging Face](https://huggingface.co/sonic-for-a3/sonic/tree/main/rockchip_sysroot), outside the source tree.
From the repository root, fetch and verify it before a Rockchip build:

```bash
python -m pip install huggingface_hub
python download_from_hf.py --component sysroot
(cd gear_sonic_deploy/thirdparty/rockchip_sysroot && \
  sha256sum -c rockchip-1.0-aarch64-sysroot.tar.gz.sha256)
```

The tarball is a build input, not a full Thor OS image. It contains the
directories that `Dockerfile.a3-rockchip-builder` restores into the builder:

```text
opt/ros/jazzy
usr/include
usr/share/eigen3
usr/lib/aarch64-linux-gnu
```

If the target BSP, OS, ROS ABI, or userspace changes, regenerate the sysroot
from a matching locally available NVIDIA Thor base image. Follow NVIDIA's
[Thor Docker setup guide](https://docs.nvidia.com/jetson/agx-thor-devkit/user-guide/latest/setup_docker.html)
to obtain that image; this repository deliberately does not prescribe an image
tag or pull from an implicit registry:

```bash
export A3_THOR_BASE_IMAGE='<local-nvidia-thor-image>'
gear_sonic_deploy/scripts/export_rockchip_sysroot.sh \
  --image "$A3_THOR_BASE_IMAGE"
```

The exporter normalizes archive owner/group and timestamp metadata and writes a
new adjacent checksum. Retain third-party notices carried in the extracted
sysroot when redistributing a refreshed artifact.
