// Copyright (c) 2026, AgiBot Inc. All rights reserved.
// 对应 notes/a3_backend_plan.md §PR 8
#include "a3_deploy/safe_halt.hpp"

#include "robot_io/a3_layout_extra.hpp"

namespace a3_deploy {

void BuildSafeHaltCommand(const Eigen::VectorXd& q_current,
                          robot_io::RobotCommand& out) {
  constexpr int N = robot_io::kA3Dof;  // 31
  if (out.q_des.size()  != N) out.q_des.resize(N);
  if (out.dq_des.size() != N) out.dq_des.resize(N);
  if (out.tau_ff.size() != N) out.tau_ff.resize(N);
  if (out.kp.size()     != N) out.kp.resize(N);
  if (out.kd.size()     != N) out.kd.resize(N);

  if (q_current.size() == N) {
    out.q_des = q_current;
  } else {
    out.q_des.setZero();
  }
  out.dq_des.setZero();
  out.tau_ff.setZero();
  out.kp.setZero();
  out.kd.setZero();
}

}  // namespace a3_deploy
