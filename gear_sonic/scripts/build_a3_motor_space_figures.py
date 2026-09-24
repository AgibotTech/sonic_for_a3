#!/usr/bin/env python3
"""Build the A3 serial torque-space and Experiment 024 T-N figures."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull

ROOT = Path(__file__).resolve().parents[2]
ACT = {
    "PFP41": ((0,20.943951,26.179939),(6,6,0)),
    "PFP59": ((0,13.613568,14.660766,15.707963,16.755161,20.943951),(34.8,34.8,24.053,20.918,17.073,0)),
    "PFP78": ((0,16.755161,17.802358,18.849556,23.561945),(60,60,50.104,44.310,0)),
    "PFP93": ((0,10.471976,11.519173,12.566371,13.613568,14.660766,15.707963,16.755161,20.943951),(220,220,193.135,160.961,133.026,104.350,75.010,54.401,0)),
    "PFP110": ((0,9.424778,10.471976,11.519173,12.566371,13.613568,14.660766,15.707963,19.634954),(305,305,264.297,224.922,182.896,143.348,105.184,62.183,0)),
}
REWARD = {"PFP41": ((0,20.943951,26.179939),(5.8,5.8,0)), "PFP59": ((0,13.613568,17),(34,34,0)), "PFP78": ((0,16.755161,23.561945),(58,58,0)), "PFP93": ((0,10.471976,18.5),(215,215,0)), "PFP110": ((0,9.424778,17.3),(300,300,0))}

def torque_space(out: Path) -> None:
    corners = np.array([[-60,-60],[-60,60],[60,60],[60,-60]])
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), constrained_layout=True)
    for ax, (name, file) in zip(axes, [("Left ankle", "left_ankle_jacobi.npz"), ("Right ankle", "right_ankle_jacobi.npz"), ("Waist", "waist_jacobi.npz")]):
        data = np.load(ROOT / "gear_sonic/data/a3_motor_model" / file)
        vertices = []
        for jac in data["J"].reshape(-1, 2, 2):
            if np.all(np.isfinite(jac)) and abs(np.linalg.det(jac)) > 1e-8:
                vertices.append(np.linalg.solve(jac.T, corners.T).T)
        points = np.concatenate(vertices)
        hull = ConvexHull(points); envelope = points[hull.vertices]
        ax.fill(envelope[:, 0], envelope[:, 1], color="#4C78A8", alpha=.18, label="pose envelope")
        ax.plot(*np.vstack([envelope, envelope[0]]).T, color="#2F5D8A", lw=2)
        idx = np.argmin(np.sum(np.square(np.stack(np.meshgrid(data["pitch_axis"], data["roll_axis"], indexing="ij"), axis=-1)), axis=-1))
        neutral = np.linalg.solve(data["J"].reshape(-1,2,2)[idx].T, corners.T).T
        ax.fill(neutral[:, 0], neutral[:, 1], color="#E45756", alpha=.28, label="q≈0 polygon")
        ax.plot(*np.vstack([neutral, neutral[0]]).T, color="#B33C3C", lw=1.5)
        ax.axhline(0, color="0.75", lw=.8); ax.axvline(0, color="0.75", lw=.8)
        ax.set_title(name); ax.set_xlabel(r"$\tau_{pitch}$ [N m]"); ax.set_ylabel(r"$\tau_{roll}$ [N m]")
        ax.set_aspect("equal", "box"); ax.grid(alpha=.2)
    axes[0].legend(loc="upper right", fontsize=8)
    fig.suptitle(r"Physical PFP78 limits mapped into serial torque space: $\tau_m=J^T\tau_q$; ±60 N m")
    fig.savefig(out / "a3_serial_joint_torque_space.png", dpi=220); plt.close(fig)

def tn(out: Path) -> None:
    colors = ["#4C78A8", "#F58518", "#54A24B", "#E45756", "#B279A2"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.4), constrained_layout=True)
    for ax, title, profiles, style in [(axes[0], "Actuator envelope (existing)", ACT, "--"), (axes[1], "Experiment 024 reward profile", REWARD, "-")]:
        for (family, (speed, torque)), color in zip(profiles.items(), colors):
            ax.plot(speed, torque, style, color=color, lw=2.4, marker="o" if ax is axes[1] else None, label=family)
        ax.set_title(title); ax.set_xlabel("|motor speed| [rad/s]"); ax.set_ylabel("torque limit [N m]")
        ax.set_xlim(left=0); ax.set_ylim(bottom=0); ax.grid(alpha=.22); ax.legend(fontsize=8)
    fig.suptitle("A3 T-N curves: physical envelope versus 024 conservative two-segment fit")
    fig.savefig(out / "a3_024_torque_speed_reward.png", dpi=220); plt.close(fig)

def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--output-dir", type=Path, default=ROOT / "gear_sonic/docs/figures")
    args = parser.parse_args(); args.output_dir.mkdir(parents=True, exist_ok=True); torque_space(args.output_dir); tn(args.output_dir)

if __name__ == "__main__": main()
