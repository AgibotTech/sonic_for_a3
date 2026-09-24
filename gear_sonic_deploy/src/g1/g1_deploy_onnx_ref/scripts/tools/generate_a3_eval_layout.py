#!/usr/bin/env python3
"""生成直接读取 A3 evaluation 原始 MCAP 的 Foxglove Layout。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


JOINT_NAMES_31 = [
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "head_yaw_joint",
    "head_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
]

# 29 维 policy/reference 不含 neck；该表必须与 kA3PolicyToSdkIdx 一致。
POLICY_TO_SDK_31 = [
    0,
    1,
    2,
    5,
    6,
    7,
    8,
    9,
    10,
    11,
    12,
    13,
    14,
    15,
    16,
    17,
    18,
    19,
    20,
    21,
    22,
    23,
    24,
    25,
    26,
    27,
    28,
    29,
    30,
]
SDK_TO_POLICY_29 = {
    sdk_index: policy_index
    for policy_index, sdk_index in enumerate(POLICY_TO_SDK_31)
}

COLORS = {
    "actual": "#4E79A7",
    "reference": "#F28E2B",
    "command": "#59A14F",
    "policy": "#E15759",
    "action": "#B07AA1",
    "x": "#4E79A7",
    "y": "#F28E2B",
    "z": "#59A14F",
    "w": "#E15759",
}


def series(value: str, label: str, color: str) -> dict[str, Any]:
    return {
        "color": color,
        "enabled": True,
        "label": label,
        "lineSize": 1.5,
        "showLine": True,
        "timestampMethod": "receiveTime",
        "value": value,
    }


def plot(title: str, y_axis_label: str, paths: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "foxglovePanelTitle": title,
        "isSynced": True,
        "legendDisplay": "floating",
        "paths": paths,
        "showLegend": True,
        "showPlotValuesInLegend": True,
        "showXAxisLabels": True,
        "showYAxisLabels": True,
        "sidebarDimension": 240,
        "title": title,
        "xAxisVal": "timestamp",
        "yAxisLabel": y_axis_label,
    }


def stack(panel_ids: list[str]) -> str | dict[str, Any]:
    if len(panel_ids) == 1:
        return panel_ids[0]
    return {
        "direction": "column",
        "first": panel_ids[0],
        "second": stack(panel_ids[1:]),
        "splitPercentage": round(100.0 / len(panel_ids), 3),
    }


def grid(panel_ids: list[str], columns: int = 2) -> str | dict[str, Any]:
    if columns <= 1 or len(panel_ids) == 1:
        return stack(panel_ids)
    left_count = (len(panel_ids) + 1) // 2
    return {
        "direction": "row",
        "first": stack(panel_ids[:left_count]),
        "second": stack(panel_ids[left_count:]),
        "splitPercentage": 50,
    }


def position_paths(sdk_index: int) -> list[dict[str, Any]]:
    policy_index = SDK_TO_POLICY_29[sdk_index]
    return [
        series(
            f"/a3/eval/state.q[{sdk_index}]",
            "actual q",
            COLORS["actual"],
        ),
        series(
            f"/a3/eval/reference.q_ref[{policy_index}]",
            "reference q",
            COLORS["reference"],
        ),
    ]


def velocity_paths(sdk_index: int) -> list[dict[str, Any]]:
    policy_index = SDK_TO_POLICY_29[sdk_index]
    return [
        series(
            f"/a3/eval/state.dq[{sdk_index}]",
            "actual dq",
            COLORS["actual"],
        ),
        series(
            f"/a3/eval/reference.dq_ref[{policy_index}]",
            "reference dq",
            COLORS["reference"],
        ),
    ]


def torque_paths(sdk_index: int) -> list[dict[str, Any]]:
    return [
        series(
            f"/a3/eval/state.tau_est[{sdk_index}]",
            "actual tau_est",
            COLORS["actual"],
        ),
    ]


def make_layout() -> dict[str, Any]:
    configs: dict[str, Any] = {}
    tabs: list[dict[str, Any]] = []

    def add_tab(title: str, panel_specs: list[tuple[str, dict[str, Any]]]) -> None:
        panel_ids = []
        for panel_id, config in panel_specs:
            configs[panel_id] = config
            panel_ids.append(panel_id)
        tabs.append({"title": title, "layout": grid(panel_ids)})

    key_leg_indices = [19, 22, 23, 25, 28, 29]
    add_tab(
        "关键下肢位置",
        [
            (
                f"Plot!eval-key-pos-{sdk}",
                plot(JOINT_NAMES_31[sdk], "position [rad]", position_paths(sdk)),
            )
            for sdk in key_leg_indices
        ],
    )

    leg_indices = list(range(19, 31))
    add_tab(
        "下肢位置",
        [
            (
                f"Plot!eval-leg-pos-{sdk}",
                plot(JOINT_NAMES_31[sdk], "position [rad]", position_paths(sdk)),
            )
            for sdk in leg_indices
        ],
    )
    add_tab(
        "下肢速度",
        [
            (
                f"Plot!eval-leg-vel-{sdk}",
                plot(JOINT_NAMES_31[sdk], "velocity [rad/s]", velocity_paths(sdk)),
            )
            for sdk in leg_indices
        ],
    )
    add_tab(
        "下肢力矩",
        [
            (
                f"Plot!eval-leg-tau-{sdk}",
                plot(
                    JOINT_NAMES_31[sdk],
                    "torque [Nm]",
                    torque_paths(sdk),
                ),
            )
            for sdk in leg_indices
        ],
    )

    knee_specs: list[tuple[str, dict[str, Any]]] = []
    for side, sdk in (("left", 22), ("right", 28)):
        policy = SDK_TO_POLICY_29[sdk]
        knee_specs.extend(
            [
                (
                    f"Plot!eval-knee-{side}-pos",
                    plot(f"{side} knee position", "position [rad]", position_paths(sdk)),
                ),
                (
                    f"Plot!eval-knee-{side}-vel",
                    plot(f"{side} knee velocity", "velocity [rad/s]", velocity_paths(sdk)),
                ),
                (
                    f"Plot!eval-knee-{side}-tau",
                    plot(
                        f"{side} knee torque",
                        "torque [Nm]",
                        torque_paths(sdk),
                    ),
                ),
                (
                    f"Plot!eval-knee-{side}-action",
                    plot(
                        f"{side} knee raw action",
                        "normalized action",
                        [
                            series(
                                f"/a3/eval/policy.raw_action[{policy}]",
                                "raw_action",
                                COLORS["action"],
                            )
                        ],
                    ),
                ),
            ]
        )
    add_tab("膝关节诊断", knee_specs)

    waist_indices = [0, 1, 2]
    waist_specs: list[tuple[str, dict[str, Any]]] = []
    for sdk in waist_indices:
        waist_specs.extend(
            [
                (
                    f"Plot!eval-waist-pos-{sdk}",
                    plot(JOINT_NAMES_31[sdk] + " position", "position [rad]", position_paths(sdk)),
                ),
                (
                    f"Plot!eval-waist-vel-{sdk}",
                    plot(JOINT_NAMES_31[sdk] + " velocity", "velocity [rad/s]", velocity_paths(sdk)),
                ),
            ]
        )
    add_tab("腰部跟踪", waist_specs)

    upper_indices = list(range(5, 19))
    add_tab(
        "上肢位置",
        [
            (
                f"Plot!eval-upper-pos-{sdk}",
                plot(JOINT_NAMES_31[sdk], "position [rad]", position_paths(sdk)),
            )
            for sdk in upper_indices
        ],
    )

    imu_specs: list[tuple[str, dict[str, Any]]] = []
    for body in ("pelvis", "torso"):
        imu_specs.append(
            (
                f"Plot!eval-{body}-quat",
                plot(
                    f"{body} quaternion WXYZ",
                    "quaternion",
                    [
                        series(
                            f"/a3/eval/{body}_imu.quat_wxyz[{index}]",
                            axis,
                            COLORS[axis],
                        )
                        for index, axis in enumerate(("w", "x", "y", "z"))
                    ],
                ),
            )
        )
        imu_specs.append(
            (
                f"Plot!eval-{body}-gyro",
                plot(
                    f"{body} gyro",
                    "angular velocity [rad/s]",
                    [
                        series(
                            f"/a3/eval/{body}_imu.gyro[{index}]",
                            axis,
                            COLORS[axis],
                        )
                        for index, axis in enumerate(("x", "y", "z"))
                    ],
                ),
            )
        )
    add_tab("IMU", imu_specs)

    timing_specs = [
        (
            "Plot!eval-timing-inference",
            plot(
                "Inference latency",
                "time [ms]",
                [series("/a3/eval/timing.inference_ms", "inference", COLORS["policy"])],
            ),
        ),
        (
            "Plot!eval-timing-send",
            plot(
                "Command send latency",
                "time [ms]",
                [series("/a3/eval/timing.send_ms", "send", COLORS["command"])],
            ),
        ),
        (
            "Plot!eval-timing-control",
            plot(
                "Control-loop latency",
                "time [ms]",
                [series("/a3/eval/timing.control_ms", "control", COLORS["actual"])],
            ),
        ),
        (
            "Plot!eval-reference-tick",
            plot(
                "Episode progress",
                "tick",
                [
                    series(
                        "/a3/eval/status.reference_tick",
                        "reference_tick",
                        COLORS["reference"],
                    ),
                    series(
                        "/a3/eval/status.policy_tick",
                        "policy_tick",
                        COLORS["policy"],
                    ),
                ],
            ),
        ),
    ]
    add_tab("时序与进度", timing_specs)

    tab_id = "Tab!a3-eval-raw"
    configs[tab_id] = {"activeTabIdx": 0, "tabs": tabs}
    return {
        "configById": configs,
        "globalVariables": {},
        "layout": tab_id,
        "playbackConfig": {"speed": 1},
        "savedProps": configs,
        "userNodes": {},
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    layout = make_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(layout, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"written: {args.output}")
    print(f"panels: {len(layout['configById']) - 1}")
    print(f"tabs: {len(layout['configById']['Tab!a3-eval-raw']['tabs'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
