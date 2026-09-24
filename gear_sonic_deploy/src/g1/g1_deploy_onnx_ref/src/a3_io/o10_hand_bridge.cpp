// Copyright (c) 2026, AgiBot Inc. All rights reserved.
#include "a3_io/o10_hand_bridge.hpp"

#include <array>
#include <cstdint>
#include <string>

namespace a3_io {

#if defined(HAS_A3_ROS_MSGS) && defined(HAS_A3_HAND_PROTO)

namespace {

using O10Pos = aimdk::protocol::O10FingerPos;
using O10Torque = aimdk::protocol::FingerTorque;

constexpr std::array<const char*, kO10HandTotalDof> kO10JointNames = {
    "left_thumb_roration_0",  "left_thumb_wiggles_1",
    "left_thumb_bent_2",      "left_index_wiggles_0",
    "left_index_bent_1",      "left_middle_bent",
    "left_ring_wiggles_0",    "left_ring_bent_1",
    "left_pinky_wiggles_0",   "left_pinky_bent_1",
    "right_thumb_rotation_0", "right_thumb_wiggles_1",
    "right_thumb_bent_2",     "right_index_wiggles_0",
    "right_index_bent_1",     "right_middle_bent",
    "right_ring_wiggles_0",   "right_ring_bent_1",
    "right_pinky_wiggles_0",  "right_pinky_bent_1",
};

void SetError(std::string* error, const std::string& value) {
  if (error) *error = value;
}

void FillPbHeader(aimdk::protocol::Header* header,
                  std::uint32_t seq,
                  std::int64_t stamp_ns,
                  const char* frame_id) {
  if (!header) return;
  header->set_seq(seq);
  header->set_frame_id(frame_id);
  auto* ts = header->mutable_timestamp();
  ts->set_seconds(stamp_ns / 1'000'000'000LL);
  ts->set_nanos(static_cast<std::int32_t>(stamp_ns % 1'000'000'000LL));
  ts->set_ms_since_epoch(stamp_ns / 1'000'000LL);
}

void FillRosStamp(std::int64_t stamp_ns, sensor_msgs::msg::JointState& out) {
  out.header.stamp.sec =
      static_cast<std::int32_t>(stamp_ns / 1'000'000'000LL);
  out.header.stamp.nanosec =
      static_cast<std::uint32_t>(stamp_ns % 1'000'000'000LL);
}

void FillO10PosFromArray(const std::array<std::int32_t, kO10HandDofPerSide>& v,
                         O10Pos* pos) {
  if (!pos) return;
  pos->set_thumb_roration_pos_0(v[0]);
  pos->set_thumb_wiggles_pos_1(v[1]);
  pos->set_thumb_bent_pos_2(v[2]);
  pos->set_index_wiggles_pos_0(v[3]);
  pos->set_index_bent_pos_1(v[4]);
  pos->set_middle_bent_pos(v[5]);
  pos->set_ring_wiggles_pos_0(v[6]);
  pos->set_ring_bent_pos_1(v[7]);
  pos->set_pinky_wiggles_pos_0(v[8]);
  pos->set_pinky_bent_pos_1(v[9]);
}

void FillO10TorqueFromArray(
    const std::array<std::int32_t, kO10HandDofPerSide>& v,
    O10Torque* toq) {
  if (!toq) return;
  toq->set_thumb_rotate_torque(v[0]);
  toq->set_thumb_swing_torque(v[1]);
  toq->set_thumb_bend_torque(v[2]);
  toq->set_index_swing_torque(v[3]);
  toq->set_index_bend_torque(v[4]);
  toq->set_middle_bend_torque(v[5]);
  toq->set_ring_swing_torque(v[6]);
  toq->set_ring_bend_torque(v[7]);
  toq->set_pinky_swing_torque(v[8]);
  toq->set_pinky_bend_torque(v[9]);
}

std::array<std::int32_t, kO10HandDofPerSide> ReadO10Pos(
    const O10Pos& pos) {
  return {pos.thumb_roration_pos_0(), pos.thumb_wiggles_pos_1(),
          pos.thumb_bent_pos_2(),     pos.index_wiggles_pos_0(),
          pos.index_bent_pos_1(),     pos.middle_bent_pos(),
          pos.ring_wiggles_pos_0(),   pos.ring_bent_pos_1(),
          pos.pinky_wiggles_pos_0(),  pos.pinky_bent_pos_1()};
}

std::array<std::int32_t, kO10HandDofPerSide> ReadO10Torque(
    const O10Torque& toq) {
  return {toq.thumb_rotate_torque(), toq.thumb_swing_torque(),
          toq.thumb_bend_torque(),   toq.index_swing_torque(),
          toq.index_bend_torque(),   toq.middle_bend_torque(),
          toq.ring_swing_torque(),   toq.ring_bend_torque(),
          toq.pinky_swing_torque(),  toq.pinky_bend_torque()};
}

void FillOneHandCommand(
    const std::array<std::int32_t, kO10HandDofPerSide>& pos,
    const std::array<std::int32_t, kO10HandDofPerSide>& torque,
    aimdk::protocol::SingleHandCommand* out) {
  auto* finger = out->mutable_agi_o10_hand()->mutable_finger();
  FillO10PosFromArray(pos, finger->mutable_pos());
  FillO10TorqueFromArray(torque, finger->mutable_toq());
}

}  // namespace

bool ConvertO10JointStateToHandCommandChannel(
    const sensor_msgs::msg::JointState& in,
    std::uint32_t seq,
    std::int64_t stamp_ns,
    aimdk::protocol::HandCommandChannel& out,
    std::string* error) {
  if (in.header.frame_id != "O10Hand") {
    SetError(error, "unsupported hand frame_id: " + in.header.frame_id);
    return false;
  }
  if (in.position.size() != kO10HandTotalDof) {
    SetError(error, "O10Hand position size must be 20");
    return false;
  }

  std::array<std::int32_t, kO10HandDofPerSide> left_pos{};
  std::array<std::int32_t, kO10HandDofPerSide> right_pos{};
  std::array<std::int32_t, kO10HandDofPerSide> left_toq{};
  std::array<std::int32_t, kO10HandDofPerSide> right_toq{};
  left_toq.fill(kO10DefaultTorque);
  right_toq.fill(kO10DefaultTorque);

  for (int i = 0; i < kO10HandDofPerSide; ++i) {
    left_pos[i] = static_cast<std::int32_t>(in.position[i]);
    right_pos[i] =
        static_cast<std::int32_t>(in.position[kO10HandDofPerSide + i]);
  }
  if (in.effort.size() == kO10HandTotalDof) {
    for (int i = 0; i < kO10HandDofPerSide; ++i) {
      left_toq[i] = static_cast<std::int32_t>(in.effort[i]);
      right_toq[i] =
          static_cast<std::int32_t>(in.effort[kO10HandDofPerSide + i]);
    }
  }

  out.Clear();
  FillPbHeader(out.mutable_header(), seq, stamp_ns, "O10Hand");
  auto* data = out.mutable_data();
  FillOneHandCommand(left_pos, left_toq, data->mutable_left());
  FillOneHandCommand(right_pos, right_toq, data->mutable_right());
  return true;
}

bool ConvertO10HandStateChannelToJointState(
    const aimdk::protocol::HandStateChannel& in,
    std::int64_t stamp_ns,
    sensor_msgs::msg::JointState& out,
    std::string* error) {
  if (!in.data().left().has_agi_o10_hand() ||
      !in.data().right().has_agi_o10_hand()) {
    SetError(error, "HandStateChannel is not bilateral O10Hand");
    return false;
  }

  const auto left_pos =
      ReadO10Pos(in.data().left().agi_o10_hand().finger().pos());
  const auto right_pos =
      ReadO10Pos(in.data().right().agi_o10_hand().finger().pos());
  const auto left_toq =
      ReadO10Torque(in.data().left().agi_o10_hand().finger().toq());
  const auto right_toq =
      ReadO10Torque(in.data().right().agi_o10_hand().finger().toq());

  out = sensor_msgs::msg::JointState{};
  FillRosStamp(stamp_ns, out);
  out.header.frame_id = "O10Hand";
  out.name.resize(kO10HandTotalDof);
  out.position.resize(kO10HandTotalDof);
  out.velocity.resize(kO10HandTotalDof);
  for (int i = 0; i < kO10HandDofPerSide; ++i) {
    out.name[i] = kO10JointNames[i];
    out.name[kO10HandDofPerSide + i] = kO10JointNames[kO10HandDofPerSide + i];
    out.position[i] = static_cast<double>(left_pos[i]);
    out.position[kO10HandDofPerSide + i] = static_cast<double>(right_pos[i]);
    out.velocity[i] = static_cast<double>(left_toq[i]);
    out.velocity[kO10HandDofPerSide + i] = static_cast<double>(right_toq[i]);
  }
  return true;
}

#endif

}  // namespace a3_io
