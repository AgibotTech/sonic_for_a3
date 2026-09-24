// Copyright (c) 2026, AgiBot Inc. All rights reserved.
//
// O10 hand bridge helpers for the A3 AimRT backend.
//
// Teleop Avatar emits hand commands as sensor_msgs/JointState on
// /motion/control/hand_joint_command. Motion Control translates that into the
// protobuf /body_drive/hand_joint_command path. These helpers mirror the O10
// subset of that conversion and keep it independent from the 31-DOF policy
// command path.
#pragma once

#include <cstdint>
#include <string>

#ifdef HAS_A3_HAND_PROTO
#include "aimdk/protocol/hal/hand/hand_channel.pb.h"
#endif

#ifdef HAS_A3_ROS_MSGS
#include "sensor_msgs/msg/joint_state.hpp"
#endif

namespace a3_io {

constexpr int kO10HandDofPerSide = 10;
constexpr int kO10HandTotalDof = 20;
constexpr int kO10DefaultTorque = 100;

#if defined(HAS_A3_ROS_MSGS) && defined(HAS_A3_HAND_PROTO)

bool ConvertO10JointStateToHandCommandChannel(
    const sensor_msgs::msg::JointState& in,
    std::uint32_t seq,
    std::int64_t stamp_ns,
    aimdk::protocol::HandCommandChannel& out,
    std::string* error = nullptr);

bool ConvertO10HandStateChannelToJointState(
    const aimdk::protocol::HandStateChannel& in,
    std::int64_t stamp_ns,
    sensor_msgs::msg::JointState& out,
    std::string* error = nullptr);

#endif

}  // namespace a3_io
