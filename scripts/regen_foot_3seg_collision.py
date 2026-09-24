#!/usr/bin/env python3
"""Regenerate A3 hard-sole 3-segment collision meshes from clipped sole hulls."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import trimesh


DEFAULT_BASE_DIR = Path("gear_sonic/data/assets/robot_description/urdf/a3/meshes_collision_optimized")
SEGMENTS = (
    ("rear", None, -0.077),
    ("mid", -0.077, 0.161),
    ("front", 0.161, None),
)


def _clip_slab(mesh: trimesh.Trimesh, xlo: float | None, xhi: float | None) -> trimesh.Trimesh:
    clipped = mesh
    if xlo is not None:
        clipped = clipped.slice_plane([xlo, 0.0, 0.0], [1.0, 0.0, 0.0])
    if xhi is not None:
        clipped = clipped.slice_plane([xhi, 0.0, 0.0], [-1.0, 0.0, 0.0])
    if clipped is None or len(clipped.vertices) == 0:
        raise RuntimeError(f"empty clipped slab for xlo={xlo}, xhi={xhi}")
    return clipped


def _stats(mesh: trimesh.Trimesh) -> str:
    bounds_mm = mesh.bounds * 1000.0
    dims_mm = (mesh.bounds[1] - mesh.bounds[0]) * 1000.0
    min_s = ", ".join(f"{v:.3f}" for v in bounds_mm[0])
    max_s = ", ".join(f"{v:.3f}" for v in bounds_mm[1])
    dim_s = ", ".join(f"{v:.3f}" for v in dims_mm)
    return (
        f"verts={len(mesh.vertices)} faces={len(mesh.faces)} "
        f"volume_cm3={mesh.volume * 1e6:.3f} watertight={mesh.is_watertight} convex={mesh.is_convex} "
        f"bounds_min_mm=[{min_s}] bounds_max_mm=[{max_s}] dims_mm=[{dim_s}]"
    )


def regenerate(base_dir: Path) -> None:
    for side in ("left", "right"):
        source = base_dir / f"{side}_ankle_roll_Link_whole_sole_collision_convex.STL"
        mesh = trimesh.load(source, force="mesh")
        if not isinstance(mesh, trimesh.Trimesh):
            raise TypeError(f"{source} did not load as a mesh")
        if not mesh.is_watertight or mesh.volume <= 0.0:
            raise ValueError(f"{source} is not a valid closed source mesh: {_stats(mesh)}")

        print(f"{side} source {_stats(mesh)}")
        for name, xlo, xhi in SEGMENTS:
            clipped = _clip_slab(mesh, xlo, xhi)
            hull = clipped.convex_hull
            hull.update_faces(hull.unique_faces())
            hull.remove_unreferenced_vertices()

            if hull.volume <= 0.0 or len(hull.vertices) < 4:
                raise ValueError(f"{side} {name} invalid convex hull: {_stats(hull)}")
            if not np.isfinite(hull.vertices).all():
                raise ValueError(f"{side} {name} has non-finite vertices")

            output = base_dir / f"{side}_foot_{name}_collision_3seg.STL"
            hull.export(output)
            print(f"{side} {name} -> {output.name} {_stats(hull)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-dir", type=Path, default=DEFAULT_BASE_DIR)
    args = parser.parse_args()
    regenerate(args.base_dir)


if __name__ == "__main__":
    main()
