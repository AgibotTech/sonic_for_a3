// Copyright (c) 2026, AgiBot Inc. All rights reserved.
// 对应 notes/a3_backend_plan.md §? / PR 5
#include "a3_io/subscribers/neck_state_sub.hpp"

#include "a3_io/convert_helpers.hpp"

namespace a3_io {
namespace {

bool IsNeckJointNameMatch(const std::string& input,
                          const std::string& expected) {
  if (expected == "head_pitch_joint") {
    return input == expected || input == "ultra_head_pitch_joint";
  }
  return input == expected;
}

bool ReorderNeckByName(const std::array<std::string, NeckStateSub::kDof>& names_in,
                       const std::array<double, NeckStateSub::kDof>& pos_in,
                       const std::array<double, NeckStateSub::kDof>& vel_in,
                       const std::array<double, NeckStateSub::kDof>& eff_in,
                       std::array<double, NeckStateSub::kDof>& out_pos,
                       std::array<double, NeckStateSub::kDof>& out_vel,
                       std::array<double, NeckStateSub::kDof>& out_eff) {
  for (std::size_t j = 0; j < NeckStateSub::kDof; ++j) {
    bool found = false;
    for (std::size_t i = 0; i < NeckStateSub::kDof; ++i) {
      if (IsNeckJointNameMatch(names_in[i], robot_io::kA3NeckJointNames[j])) {
        out_pos[j] = pos_in[i];
        out_vel[j] = vel_in[i];
        out_eff[j] = eff_in[i];
        found = true;
      }
    }
    if (!found) return false;
  }
  return true;
}

}  // namespace

bool NeckStateSub::ConvertMessage(
    const std::array<std::string, kDof>& names_in,
    const std::array<double, kDof>& pos_in,
    const std::array<double, kDof>& vel_in,
    const std::array<double, kDof>& eff_in,
    std::int64_t stamp_ns,
    a3_sync::NeckSample& out) {
  if (!ReorderNeckByName(names_in, pos_in, vel_in, eff_in,
                         out.pos, out.vel, out.eff)) {
    return false;
  }
  out.stamp.value = stamp_ns;
  return true;
}

#ifdef HAS_A3_ROS_MSGS
void NeckStateSub::OnMessage(
    const std::shared_ptr<const joint_msgs::msg::JointState>& msg) {
  if (!msg) return;
  if (msg->joints.size() != kDof) return;

  std::array<std::string, kDof> names_in{};
  std::array<double, kDof> pos_in{}, vel_in{}, eff_in{};
  for (std::size_t i = 0; i < kDof; ++i) {
    names_in[i] = msg->joints[i].name;
    pos_in[i]   = msg->joints[i].position;
    vel_in[i]   = msg->joints[i].velocity;
    eff_in[i]   = msg->joints[i].effort;
  }

  const std::int64_t stamp_ns =
      StampToNs(msg->header.stamp.sec, msg->header.stamp.nanosec);

  a3_sync::NeckSample sample{};
  if (!ConvertMessage(names_in, pos_in, vel_in, eff_in, stamp_ns, sample)) {
    return;
  }
  sample.source_sequence = msg->joints[0].sequence;
  sample.source_sequence_valid = true;
  StampReceiptTime(sample);
  ring_.write([&](a3_sync::NeckSample& slot) { slot = sample; });
  MarkReady();
  NotifySample(sample);
}
#endif

}  // namespace a3_io
