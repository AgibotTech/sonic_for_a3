// Copyright (c) 2026, AgiBot Inc. All rights reserved.
#include <gtest/gtest.h>

#if defined(HAS_A3_ROS_MSGS) && defined(HAS_A3_HAND_PROTO)

#include <cstdint>
#include <set>
#include <vector>

#include "a3_io/o10_hand_bridge.hpp"
#include "robot_io/a3_layout_extra.hpp"
#include "robot_io/a3_aimrt_backend.hpp"

namespace {

sensor_msgs::msg::JointState MakeO10CommandMsg() {
  sensor_msgs::msg::JointState msg;
  msg.header.frame_id = "O10Hand";
  msg.position.resize(a3_io::kO10HandTotalDof);
  for (int i = 0; i < a3_io::kO10HandTotalDof; ++i) {
    msg.position[i] = 1000.0 + static_cast<double>(i);
  }
  return msg;
}

void FillO10Pos(aimdk::protocol::O10FingerPos* pos, int base) {
  pos->set_thumb_roration_pos_0(base + 0);
  pos->set_thumb_wiggles_pos_1(base + 1);
  pos->set_thumb_bent_pos_2(base + 2);
  pos->set_index_wiggles_pos_0(base + 3);
  pos->set_index_bent_pos_1(base + 4);
  pos->set_middle_bent_pos(base + 5);
  pos->set_ring_wiggles_pos_0(base + 6);
  pos->set_ring_bent_pos_1(base + 7);
  pos->set_pinky_wiggles_pos_0(base + 8);
  pos->set_pinky_bent_pos_1(base + 9);
}

void FillO10Torque(aimdk::protocol::FingerTorque* toq, int base) {
  toq->set_thumb_rotate_torque(base + 0);
  toq->set_thumb_swing_torque(base + 1);
  toq->set_thumb_bend_torque(base + 2);
  toq->set_index_swing_torque(base + 3);
  toq->set_index_bend_torque(base + 4);
  toq->set_middle_bend_torque(base + 5);
  toq->set_ring_swing_torque(base + 6);
  toq->set_ring_bend_torque(base + 7);
  toq->set_pinky_swing_torque(base + 8);
  toq->set_pinky_bend_torque(base + 9);
}

void FillO10StateHand(aimdk::protocol::SingleHandState* hand,
                      int pos_base,
                      int torque_base) {
  auto* finger = hand->mutable_agi_o10_hand()->mutable_finger();
  FillO10Pos(finger->mutable_pos(), pos_base);
  FillO10Torque(finger->mutable_toq(), torque_base);
}

robot_io::RobotCommand ZeroCmd31() {
  robot_io::RobotCommand cmd;
  cmd.q_des = Eigen::VectorXd::Zero(robot_io::kA3Dof);
  cmd.dq_des = Eigen::VectorXd::Zero(robot_io::kA3Dof);
  cmd.tau_ff = Eigen::VectorXd::Zero(robot_io::kA3Dof);
  cmd.kp = Eigen::VectorXd::Zero(robot_io::kA3Dof);
  cmd.kd = Eigen::VectorXd::Zero(robot_io::kA3Dof);
  return cmd;
}

}  // namespace

TEST(A3O10HandBridge, JointStateCommandMapsToHandCommandChannel) {
  const auto msg = MakeO10CommandMsg();
  aimdk::protocol::HandCommandChannel out;
  std::string error;
  ASSERT_TRUE(a3_io::ConvertO10JointStateToHandCommandChannel(
      msg, /*seq=*/7, /*stamp_ns=*/1'234'567'890LL, out, &error))
      << error;

  EXPECT_EQ(out.header().seq(), 7u);
  EXPECT_EQ(out.header().frame_id(), "O10Hand");
  EXPECT_EQ(out.header().timestamp().seconds(), 1);
  EXPECT_EQ(out.header().timestamp().nanos(), 234'567'890);

  const auto& left_pos = out.data().left().agi_o10_hand().finger().pos();
  const auto& right_pos = out.data().right().agi_o10_hand().finger().pos();
  EXPECT_EQ(left_pos.thumb_roration_pos_0(), 1000);
  EXPECT_EQ(left_pos.pinky_bent_pos_1(), 1009);
  EXPECT_EQ(right_pos.thumb_roration_pos_0(), 1010);
  EXPECT_EQ(right_pos.pinky_bent_pos_1(), 1019);
}

TEST(A3O10HandBridge, JointStateCommandDefaultsTorqueTo100) {
  const auto msg = MakeO10CommandMsg();
  aimdk::protocol::HandCommandChannel out;
  ASSERT_TRUE(a3_io::ConvertO10JointStateToHandCommandChannel(
      msg, /*seq=*/1, /*stamp_ns=*/2'000'000'000LL, out));

  const auto& left_toq = out.data().left().agi_o10_hand().finger().toq();
  const auto& right_toq = out.data().right().agi_o10_hand().finger().toq();
  EXPECT_EQ(left_toq.thumb_rotate_torque(), a3_io::kO10DefaultTorque);
  EXPECT_EQ(left_toq.pinky_bend_torque(), a3_io::kO10DefaultTorque);
  EXPECT_EQ(right_toq.thumb_rotate_torque(), a3_io::kO10DefaultTorque);
  EXPECT_EQ(right_toq.pinky_bend_torque(), a3_io::kO10DefaultTorque);
}

TEST(A3O10HandBridge, JointStateCommandMapsEffortToTorque) {
  auto msg = MakeO10CommandMsg();
  msg.effort.resize(a3_io::kO10HandTotalDof);
  for (int i = 0; i < a3_io::kO10HandTotalDof; ++i) {
    msg.effort[i] = 200.0 + static_cast<double>(i);
  }

  aimdk::protocol::HandCommandChannel out;
  ASSERT_TRUE(a3_io::ConvertO10JointStateToHandCommandChannel(
      msg, /*seq=*/1, /*stamp_ns=*/2'000'000'000LL, out));

  const auto& left_toq = out.data().left().agi_o10_hand().finger().toq();
  const auto& right_toq = out.data().right().agi_o10_hand().finger().toq();
  EXPECT_EQ(left_toq.thumb_rotate_torque(), 200);
  EXPECT_EQ(left_toq.pinky_bend_torque(), 209);
  EXPECT_EQ(right_toq.thumb_rotate_torque(), 210);
  EXPECT_EQ(right_toq.pinky_bend_torque(), 219);
}

TEST(A3O10HandBridge, JointStateCommandRejectsNonO10OrBadSize) {
  auto msg = MakeO10CommandMsg();
  aimdk::protocol::HandCommandChannel out;

  msg.header.frame_id = "AgiClaw";
  EXPECT_FALSE(a3_io::ConvertO10JointStateToHandCommandChannel(
      msg, /*seq=*/1, /*stamp_ns=*/1, out));

  msg.header.frame_id = "O10Hand";
  msg.position.resize(a3_io::kO10HandTotalDof - 1);
  EXPECT_FALSE(a3_io::ConvertO10JointStateToHandCommandChannel(
      msg, /*seq=*/1, /*stamp_ns=*/1, out));
}

TEST(A3O10HandBridge, HandStateChannelMapsToJointState) {
  aimdk::protocol::HandStateChannel in;
  FillO10StateHand(in.mutable_data()->mutable_left(), 10, 110);
  FillO10StateHand(in.mutable_data()->mutable_right(), 20, 120);

  sensor_msgs::msg::JointState out;
  std::string error;
  ASSERT_TRUE(a3_io::ConvertO10HandStateChannelToJointState(
      in, /*stamp_ns=*/3'000'000'004LL, out, &error))
      << error;

  EXPECT_EQ(out.header.frame_id, "O10Hand");
  EXPECT_EQ(out.header.stamp.sec, 3);
  EXPECT_EQ(out.header.stamp.nanosec, 4u);
  ASSERT_EQ(out.position.size(), static_cast<std::size_t>(a3_io::kO10HandTotalDof));
  ASSERT_EQ(out.velocity.size(), static_cast<std::size_t>(a3_io::kO10HandTotalDof));
  EXPECT_EQ(out.name[0], "left_thumb_roration_0");
  EXPECT_DOUBLE_EQ(out.position[0], 10.0);
  EXPECT_DOUBLE_EQ(out.position[9], 19.0);
  EXPECT_DOUBLE_EQ(out.position[10], 20.0);
  EXPECT_DOUBLE_EQ(out.position[19], 29.0);
  EXPECT_DOUBLE_EQ(out.velocity[0], 110.0);
  EXPECT_DOUBLE_EQ(out.velocity[19], 129.0);
}

TEST(A3O10HandBridge, HandStateChannelRejectsNonO10) {
  aimdk::protocol::HandStateChannel in;
  in.mutable_data()->mutable_left()->mutable_agi_o10_hand();
  in.mutable_data()->mutable_right()->mutable_agi_claw_state();

  sensor_msgs::msg::JointState out;
  EXPECT_FALSE(a3_io::ConvertO10HandStateChannelToJointState(
      in, /*stamp_ns=*/1, out));
}

TEST(A3AimrtBackend, HandBridgeTestHooksConvertBothDirections) {
  robot_io::A3AimrtBackend backend;
  std::vector<aimdk::protocol::HandCommandChannel> captured_cmds;
  std::vector<sensor_msgs::msg::JointState> captured_states;
  backend.SetHandCommandCaptureFn_ForTest(
      [&](std::int64_t, std::uint32_t,
          const aimdk::protocol::HandCommandChannel& msg) {
        captured_cmds.push_back(msg);
      });
  backend.SetHandStateCaptureFn_ForTest(
      [&](std::int64_t, const sensor_msgs::msg::JointState& msg) {
        captured_states.push_back(msg);
      });

  ASSERT_TRUE(backend.InjectMotionHandCommand_ForTest(MakeO10CommandMsg()));
  ASSERT_EQ(captured_cmds.size(), 1u);
  EXPECT_EQ(captured_cmds[0].data()
                .right()
                .agi_o10_hand()
                .finger()
                .pos()
                .thumb_roration_pos_0(),
            1010);

  aimdk::protocol::HandStateChannel state;
  FillO10StateHand(state.mutable_data()->mutable_left(), 30, 130);
  FillO10StateHand(state.mutable_data()->mutable_right(), 40, 140);
  ASSERT_TRUE(backend.InjectBodyDriveHandState_ForTest(state));
  ASSERT_EQ(captured_states.size(), 1u);
  EXPECT_EQ(captured_states[0].header.frame_id, "O10Hand");
  EXPECT_DOUBLE_EQ(captured_states[0].position[10], 40.0);
}

TEST(A3AimrtBackend, HandBridgeCanBeDisabledByBackendConfig) {
  robot_io::A3AimrtBackend backend;
#ifdef ENABLE_A3_AIMRT_BACKEND
  ASSERT_TRUE(backend.Init(
      "cfg_file_path=gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/config/"
      "a3_aimrt_config.yaml,"
      "hand_bridge_enabled=false"));
#else
  ASSERT_TRUE(backend.Init("hand_bridge_enabled=false"));
#endif

  int captured_commands = 0;
  int captured_states = 0;
  backend.SetHandCommandCaptureFn_ForTest(
      [&](std::int64_t, std::uint32_t,
          const aimdk::protocol::HandCommandChannel&) {
        ++captured_commands;
      });
  backend.SetHandStateCaptureFn_ForTest(
      [&](std::int64_t, const sensor_msgs::msg::JointState&) {
        ++captured_states;
      });

  EXPECT_FALSE(backend.InjectMotionHandCommand_ForTest(MakeO10CommandMsg()));

  aimdk::protocol::HandStateChannel state;
  FillO10StateHand(state.mutable_data()->mutable_left(), 30, 130);
  FillO10StateHand(state.mutable_data()->mutable_right(), 40, 140);
  EXPECT_FALSE(backend.InjectBodyDriveHandState_ForTest(state));

  EXPECT_EQ(captured_commands, 0);
  EXPECT_EQ(captured_states, 0);
}

TEST(A3AimrtBackend, HandBridgeDoesNotChangeBodyCommandCaptureTopics) {
  robot_io::A3AimrtBackend backend;
  std::vector<std::string> topics;
  backend.SetTestCaptureFn_ForTest(
      [&](const std::string& topic, std::int64_t, std::uint32_t,
          const robot_io::RobotCommand&) { topics.push_back(topic); });

  ASSERT_TRUE(backend.SendCommand(ZeroCmd31()));
  ASSERT_EQ(topics.size(), 4u);
  const std::set<std::string> unique(topics.begin(), topics.end());
  EXPECT_TRUE(unique.count("waist"));
  EXPECT_TRUE(unique.count("leg"));
  EXPECT_TRUE(unique.count("arm"));
  EXPECT_TRUE(unique.count("neck"));
  EXPECT_FALSE(unique.count("hand"));
}

#else

TEST(A3O10HandBridge, SkipsWithoutRosAndHandProtoDeps) {
  GTEST_SKIP() << "requires HAS_A3_ROS_MSGS and HAS_A3_HAND_PROTO";
}

#endif
