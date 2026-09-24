// Copyright (c) 2026, AgiBot Inc. All rights reserved.
// 对应 notes/a3_backend_plan.md §PR 8 Task 8.2
//
// ExpandToBackend: glue from the 29-DOF policy output (MuJoCo/flat body-joint
// order, first 29 entries of MakeA3Layout31()) to a 31-DOF RobotCommand that
// A3AimrtBackend::SendCommand can consume. Head/neck slots [3..4] are not
// controlled by the policy; deployment fills a safe zero target by default,
// and teleop can overlay /ta/whole_body_command head targets after expansion.
#pragma once

#include "robot_io/robot_io_backend.hpp"

#include <array>

namespace a3_deploy {

inline constexpr double kA3HeadTargetPositionRad = 0.0;
inline constexpr double kA3HeadKp = 40.0;
inline constexpr double kA3HeadKd = 2.0;

// Expand a 29-DOF policy output + per-joint Kp/Kd arrays into a 31-DOF
// RobotCommand. All five Eigen vectors of `out` are resized to 31; body
// [0..28] filled from inputs; head/neck [3..4] get fixed zero-position PD
// control by default; dq_des / tau_ff zeroed everywhere.
void ExpandToBackend(
    const std::array<double, 29>& q_des_29,
    const std::array<double, 29>& kps,
    const std::array<double, 29>& kds,
    robot_io::RobotCommand& out);

// Overlay teleop-avatar head targets onto the native A3 neck slots [3..4].
// This mirrors aimrt_motion_control_a3 FoundationPolicy: position-only PD with
// zero velocity and zero torque feedforward.
void ApplyA3HeadCommand(const std::array<double, 2>& head_q_rad,
                        robot_io::RobotCommand& out);

}  // namespace a3_deploy
