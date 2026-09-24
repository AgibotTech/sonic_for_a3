#!/usr/bin/env python3
"""Pre-flight environment check for GR00T-WholeBodyControl.

Run this before training or deployment to verify all prerequisites are met.

Usage:
    python check_environment.py              # Check everything
    python check_environment.py --training   # Training checks only
    python check_environment.py --deploy     # Deployment checks + ONNX integrity
    python check_environment.py --training --components pt  # Fine-tune weights
    python check_environment.py --deploy --components rknn sysroot  # RKNN assets
"""

import argparse
import importlib
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path


def check(name, passed, msg_pass="", msg_fail=""):
    status = "PASS" if passed else "FAIL"
    symbol = "[+]" if passed else "[X]"
    detail = msg_pass if passed else msg_fail
    print(f"  {symbol} {name}: {detail}" if detail else f"  {symbol} {name}")
    return passed


def check_python(training=False):
    v = sys.version_info
    version_str = f"{v.major}.{v.minor}.{v.micro}"
    if training:
        # Isaac Sim 4.5 / Isaac Lab 2.1 (the public Forge c5862bd6 path) uses
        # Python 3.10, while newer Isaac runtimes use Python 3.11.  Version
        # compatibility with the simulator is checked by the environment guide;
        # this generic preflight must not reject the supported 4.5 stack.
        ok = v.major == 3 and v.minor in (10, 11)
        return check(
            "Python version",
            ok,
            msg_pass=version_str,
            msg_fail=f"{version_str} (training needs Python 3.10 or 3.11, matching Isaac Sim/Lab)",
        )
    else:
        ok = v.major == 3 and v.minor >= 10
        return check(
            "Python version",
            ok,
            msg_pass=version_str,
            msg_fail=f"{version_str} (need 3.10+)",
        )


ROOT = Path(__file__).resolve().parent


def check_git_lfs(root=ROOT):
    if shutil.which("git-lfs") is None:
        return check("Git LFS", False, msg_fail="not installed (sudo apt install git-lfs)")
    result = subprocess.run(
        ["git", "lfs", "ls-files", "--name-only"], cwd=root,
        capture_output=True, text=True,
    )
    if result.returncode:
        return check("Git LFS", False, msg_fail="cannot list LFS assets; run from a Git checkout")
    invalid = []
    for name in result.stdout.splitlines():
        path = Path(root) / name
        if not path.is_file() or path.stat().st_size == 0:
            invalid.append(name)
            continue
        with path.open("rb") as stream:
            if stream.read(43).startswith(b"version https://git-lfs.github.com/spec/v1"):
                invalid.append(name)
    return check("Git LFS", not invalid, msg_pass="all listed assets materialized",
                 msg_fail="missing or unresolved assets: " + ", ".join(invalid[:5]) + "; run git lfs pull")


def check_hf_artifacts(components, root=ROOT):
    from download_from_hf import matches

    if not components:
        print("  [i] From-scratch training needs no pretrained weights; for fine-tuning, "
              "add --components pt (download with python download_from_hf.py --component pt).")
        return True
    manifest = json.loads((Path(root) / "a3_hf_manifest.json").read_text())
    selected = [entry for entry in manifest["files"] if entry["group"] in components]
    invalid = [entry["destination"] for entry in selected
               if not matches(Path(root) / entry["destination"], entry)]
    return check("Hugging Face artifacts", bool(selected) and not invalid,
                 msg_pass="selected artifacts pass size/SHA-256 checks",
                 msg_fail="missing or corrupt: " + ", ".join(invalid[:5])
                 + "; run python download_from_hf.py --component " + " ".join(components))


def check_cuda():
    try:
        import torch

        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            cuda_version = torch.version.cuda
            return check("CUDA", True, msg_pass=f"{device_name} (CUDA {cuda_version})")
        else:
            return check("CUDA", False, msg_fail="torch.cuda.is_available() = False")
    except ImportError:
        return check("CUDA", False, msg_fail="PyTorch not installed")


def check_torch():
    try:
        import torch

        return check("PyTorch", True, msg_pass=torch.__version__)
    except ImportError:
        return check(
            "PyTorch",
            False,
            msg_fail="not installed (pip install torch)",
        )


def check_isaaclab():
    try:
        import isaaclab

        version = getattr(isaaclab, "__version__", "unknown")
        return check("Isaac Lab", True, msg_pass=version)
    except ImportError:
        return check(
            "Isaac Lab",
            False,
            msg_fail="not installed — see https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html",
        )


def check_gear_sonic():
    try:
        from importlib.metadata import version as get_version
        ver = get_version("gear_sonic")
        return check("gear_sonic", True, msg_pass=f"installed ({ver})")
    except ImportError:
        return check(
            "gear_sonic",
            False,
            msg_fail="not installed (pip install -e 'gear_sonic/[training]')",
        )


def check_training_deps():
    results = []
    for pkg, pip_name in [
        ("hydra", "hydra-core"),
        ("trl", "trl"),
        ("transformers", "transformers"),
        ("accelerate", "accelerate"),
        ("wandb", "wandb"),
    ]:
        try:
            mod = importlib.import_module(pkg)
            version = getattr(mod, "__version__", "ok")
            results.append(check(pip_name, True, msg_pass=version))
        except ImportError:
            results.append(
                check(pip_name, False, msg_fail=f"not installed (pip install {pip_name})")
            )
    return all(results)


def check_tensorrt():
    trt_root = os.environ.get("TensorRT_ROOT", "")
    if not trt_root:
        return check(
            "TensorRT",
            False,
            msg_fail="TensorRT_ROOT not set (export TensorRT_ROOT=$HOME/TensorRT)",
        )
    if not os.path.isdir(trt_root):
        return check("TensorRT", False, msg_fail=f"TensorRT_ROOT={trt_root} does not exist")

    # Check for the library
    lib_dir = os.path.join(trt_root, "lib")
    if os.path.isdir(lib_dir):
        libs = [f for f in os.listdir(lib_dir) if "nvinfer" in f and f.endswith(".so")]
        if libs:
            # Try to extract version from filename
            for lib in libs:
                if "nvinfer.so." in lib:
                    version = lib.split("nvinfer.so.")[-1]
                    return check("TensorRT", True, msg_pass=f"{version} at {trt_root}")
            return check("TensorRT", True, msg_pass=f"found at {trt_root}")

    return check("TensorRT", False, msg_fail=f"libnvinfer not found in {lib_dir}")


def check_disk_space():
    stat = os.statvfs(".")
    free_gb = (stat.f_bavail * stat.f_frsize) / (1024**3)
    ok = free_gb > 10
    return check(
        "Disk space",
        ok,
        msg_pass=f"{free_gb:.0f} GB free",
        msg_fail=f"{free_gb:.1f} GB free (recommend 10+ GB)",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--training", action="store_true")
    modes.add_argument("--deploy", action="store_true")
    parser.add_argument("--components", nargs="+", choices=["pt", "onnx", "rknn", "sysroot"],
                        help="HF artifacts to verify; training defaults to none, deploy to onnx, all to pt+onnx")
    args = parser.parse_args()
    mode = "training" if args.training else "deploy" if args.deploy else "all"
    components = args.components if args.components is not None else {
        "training": [], "deploy": ["onnx"], "all": ["pt", "onnx"]
    }[mode]

    print(f"GR00T-WholeBodyControl Environment Check")
    print(f"Platform: {platform.system()} {platform.machine()}")
    print(f"Python:   {sys.executable}")
    print()

    all_pass = True

    # Basic checks (always run)
    print("Basic:")
    all_pass &= check_python(training=(mode in ("all", "training")))
    all_pass &= check_git_lfs()
    all_pass &= check_hf_artifacts(components)
    all_pass &= check_cuda()
    all_pass &= check_torch()
    all_pass &= check_disk_space()
    print()

    if mode in ("all", "training"):
        print("Training:")
        all_pass &= check_isaaclab()
        all_pass &= check_gear_sonic()
        all_pass &= check_training_deps()
        print()

    if mode in ("all", "deploy"):
        print("Deployment:")
        all_pass &= check_tensorrt()
        print()

    if all_pass:
        print("All checks passed.")
    else:
        print("Some checks failed. See above for details.")
        sys.exit(1)


if __name__ == "__main__":
    main()
