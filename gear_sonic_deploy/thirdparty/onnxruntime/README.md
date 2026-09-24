# ONNX Runtime aarch64

This directory is the conventional location for a local prebuilt ONNX Runtime
package used for RK3588 cross/arm64 builds of A3 deployment. The package is a
**local build prerequisite**, not a release artifact: it depends on the target
ABI and is ignored by Git. x86 builds use the system `find_package(onnxruntime)`
result by default.

Obtain or build a target-compatible ONNX Runtime 1.19.2 package from the
[official ONNX Runtime v1.19.2 release](https://github.com/microsoft/onnxruntime/releases/tag/v1.19.2).
No single binary tag is prescribed here because the package must match the
target's architecture, C library, CUDA/JetPack stack, and enabled execution
providers.

For the Rockchip cross-build, the archive must contain exactly one install
prefix with:

```text
<install-prefix>/lib/cmake/onnxruntime/onnxruntimeConfig.cmake
```

Pass the archive explicitly when building the Rockchip package:

```bash
gear_sonic_deploy/scripts/build_a3_deploy_pkg.sh \
  --arch rockchip \
  --onnxruntime-aarch64-tarball /absolute/path/to/onnxruntime-aarch64-1.19.2.tar.gz \
  --jobs 20
```

Equivalently, set `A3_ONNXRUNTIME_AARCH64_TARBALL` to the absolute archive
path. The build fails instead of silently falling back if the file is missing
or has the wrong layout.
