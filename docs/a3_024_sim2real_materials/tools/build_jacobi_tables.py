#!/usr/bin/env python3
"""Package public A3 Jacobian CSV grids into runtime-compatible NPZ files.

The input CSV files are the compact source evidence.  The generated NPZ files
are duplicates of the runtime tables, so this tool requires a separate output
directory and never writes generated files into the release tree by default.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


JOINTS = ("left_ankle", "right_ankle", "waist")
DEFAULT_INPUT_DIR = Path(__file__).resolve().parents[1] / "tables"


def load_grid(input_dir: Path, name: str) -> dict[str, np.ndarray]:
    """Load one CSV grid into the NPZ structure used by the reward."""
    raw = np.genfromtxt(input_dir / f"{name}_jacobi.csv", delimiter=",", names=True)
    pitch = np.unique(raw["pitch"])
    roll = np.unique(raw["roll"])
    rows, cols = pitch.size, roll.size
    if raw.size != rows * cols:
        raise ValueError(f"{name}: {raw.size} rows do not form a {rows}x{cols} grid")

    def grid(column: str) -> np.ndarray:
        # The source is pitch-major, roll-minor.
        return raw[column].reshape(rows, cols)

    jacobian = np.empty((rows, cols, 2, 2))
    jacobian[..., 0, 0] = grid("J00")
    jacobian[..., 0, 1] = grid("J01")
    jacobian[..., 1, 0] = grid("J10")
    jacobian[..., 1, 1] = grid("J11")
    return {
        "pitch_axis": pitch,
        "roll_axis": roll,
        "J": jacobian,
        "m": np.stack([grid("m0"), grid("m1")], axis=-1),
        "det": grid("det"),
        "err": grid("err"),
        "reachable": grid("reachable").astype(np.int8),
        "near_singular": grid("near_singular").astype(np.int8),
    }


def save_npz(output_dir: Path, name: str, grid: dict[str, np.ndarray]) -> Path:
    output = output_dir / f"{name}_jacobi.npz"
    np.savez_compressed(output, **grid)
    return output


def save_plot(output_dir: Path, name: str, grid: dict[str, np.ndarray]) -> Path:
    """Optionally save diagnostics; Matplotlib is intentionally not required."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("--plots requires matplotlib") from exc

    pitch, roll = grid["pitch_axis"], grid["roll_axis"]
    extent = [np.degrees(roll[0]), np.degrees(roll[-1]),
              np.degrees(pitch[0]), np.degrees(pitch[-1])]
    panels = (
        ("J00", grid["J"][..., 0, 0], "viridis"),
        ("J01", grid["J"][..., 0, 1], "viridis"),
        ("J10", grid["J"][..., 1, 0], "viridis"),
        ("J11", grid["J"][..., 1, 1], "viridis"),
        ("det(J)", grid["det"], "coolwarm"),
        ("round-trip error [rad]", grid["err"], "magma"),
    )
    figure, axes = plt.subplots(2, 3, figsize=(15, 9))
    for axis, (title, values, cmap) in zip(axes.ravel(), panels):
        image = axis.imshow(values, origin="lower", extent=extent, aspect="auto", cmap=cmap)
        axis.set(title=title, xlabel="roll [deg]", ylabel="pitch [deg]")
        figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    figure.suptitle(f"{name}: parallel Jacobian over (pitch, roll)")
    figure.tight_layout()
    output = output_dir / f"{name}_jacobi.png"
    figure.savefig(output, dpi=110)
    plt.close(figure)
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR,
                        help="directory containing *_jacobi.csv source grids")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="separate directory for generated NPZ files")
    parser.add_argument("--plots", action="store_true",
                        help="also write optional PNG diagnostics to --output-dir")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name in JOINTS:
        grid = load_grid(args.input_dir, name)
        output = save_npz(args.output_dir, name, grid)
        rows, cols = grid["J"].shape[:2]
        unreachable = int((grid["reachable"] == 0).sum())
        message = f"{name}: {rows}x{cols}, unreachable={unreachable} -> {output.name}"
        if args.plots:
            message += f", {save_plot(args.output_dir, name, grid).name}"
        print(message)


if __name__ == "__main__":
    main()
