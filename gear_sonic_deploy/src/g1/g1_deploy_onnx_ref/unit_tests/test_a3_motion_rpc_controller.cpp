// Copyright (c) 2026, AgiBot Inc. All rights reserved.

#include "a3_deploy/a3_motion_rpc_controller.hpp"

#include <gtest/gtest.h>

#include <filesystem>
#include <vector>

namespace a3_deploy {
namespace {

std::vector<A3RpcMotionEntry> TestMotions() {
  return {
      {.id = 0, .name = "001_walk", .path = "motions/001_walk.csv",
       .total_ticks = 101},
      {.id = 1, .name = "002_wave", .path = "motions/002_wave.csv",
       .total_ticks = 51},
  };
}

TEST(A3MotionRpcControllerTest, RejectsUnknownMotionPath) {
  ManualControlState control;
  control.mode.store(static_cast<int>(DeployMode::kMotion));
  A3MotionRpcController controller(control, TestMotions());

  const auto result = controller.PlayMotion("motions/not-loaded.csv");

  EXPECT_FALSE(result.accepted);
  EXPECT_EQ(result.state, "rejected");
  EXPECT_EQ(control.motion_command_epoch.load(), 0u);
}

TEST(A3MotionRpcControllerTest, BlocksUntilSafeModeIsReady) {
  ManualControlState control;
  A3MotionRpcController controller(control, TestMotions());

  const auto idle = controller.PlayMotion("motions/001_walk.csv");
  EXPECT_FALSE(idle.accepted);
  EXPECT_EQ(idle.state, "blocked");

  control.mode.store(static_cast<int>(DeployMode::kPdStand));
  control.pd_stand_ready.store(false);
  const auto not_ready = controller.PlayMotion("motions/001_walk.csv");
  EXPECT_FALSE(not_ready.accepted);
  EXPECT_EQ(not_ready.state, "blocked");
}

TEST(A3MotionRpcControllerTest, SelectsAndStartsLoadedMotionByPath) {
  ManualControlState control;
  control.mode.store(static_cast<int>(DeployMode::kPdStand));
  control.pd_stand_ready.store(true);
  A3MotionRpcController controller(control, TestMotions());

  const auto result = controller.PlayMotion("motions/002_wave.csv");

  EXPECT_TRUE(result.accepted);
  EXPECT_EQ(result.play_id, 1u);
  EXPECT_EQ(result.motion_id, 1);
  EXPECT_EQ(result.motion_name, "002_wave");
  EXPECT_EQ(LoadDeployMode(control), DeployMode::kMotion);
  EXPECT_EQ(control.selected_motion_index.load(), 1);
  EXPECT_TRUE(control.motion_playing.load());
  EXPECT_GT(control.motion_command_epoch.load(), 0u);
}

TEST(A3MotionRpcControllerTest, ReportsCompletionAndSupportsStop) {
  ManualControlState control;
  control.mode.store(static_cast<int>(DeployMode::kMotion));
  A3MotionRpcController controller(control, TestMotions());
  ASSERT_TRUE(controller.PlayMotion("motions/001_walk.csv").accepted);

  control.motion_tick.store(100);
  control.motion_playing.store(false);
  control.motion_held.store(true);
  const auto complete = controller.GetPlaybackStatus();
  EXPECT_EQ(complete.state, "completed");
  EXPECT_EQ(complete.tick, 100u);
  EXPECT_EQ(complete.total_ticks, 101u);

  ASSERT_TRUE(controller.PlayMotion("motions/002_wave.csv").accepted);
  const auto stopped = controller.StopMotion();
  EXPECT_TRUE(stopped.accepted);
  EXPECT_EQ(stopped.state, "paused");
  EXPECT_FALSE(control.motion_playing.load());
}

}  // namespace
}  // namespace a3_deploy
