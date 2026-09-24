// Copyright (c) 2026, AgiBot Inc. All rights reserved.
#include <gtest/gtest.h>

#include "a3_deploy/a3_eval_episode_tracker.hpp"

namespace {

a3_deploy::EvalFrameV1 PlayingFrame(std::uint64_t frame_id,
                                    std::uint64_t reference_tick,
                                    std::int32_t motion_id) {
  a3_deploy::EvalFrameV1 frame;
  frame.frame_id = frame_id;
  frame.control_start_monotonic_ns =
      static_cast<std::int64_t>(frame_id * 20'000'000ULL);
  frame.deploy_mode = a3_deploy::DeployMode::kMotion;
  frame.motion_phase = a3_deploy::MotionPhase::kPlaying;
  frame.command_source = a3_deploy::CommandSource::kPolicy;
  frame.active_motion_id = motion_id;
  frame.selected_motion_id = motion_id;
  frame.reference_tick = reference_tick;
  frame.valid_mask = a3_deploy::kReferenceValid |
                     a3_deploy::kInferTimingValid |
                     a3_deploy::kSendTimingValid;
  frame.quality_flags = a3_deploy::kSyncComplete |
                        a3_deploy::kInferOk |
                        a3_deploy::kCommandSent;
  return frame;
}

}  // namespace

TEST(A3EvalEpisodeTracker, PauseEndsEpisodeAndReplayStartsNewEpisode) {
  a3_deploy::A3EvalEpisodeTracker tracker;
  a3_deploy::EvalTrackerOutput output;

  auto first = PlayingFrame(0, 0, 2);
  tracker.Update(first, 100, output);
  ASSERT_EQ(output.count, 1u);
  EXPECT_EQ(output.events[0].event_type,
            a3_deploy::EvalEventType::kEpisodeStarted);
  const auto first_episode = first.episode_id;
  ASSERT_NE(first_episode, 0u);

  auto second = PlayingFrame(1, 1, 2);
  tracker.Update(second, 100, output);
  EXPECT_EQ(output.count, 0u);
  EXPECT_EQ(second.episode_id, first_episode);

  auto paused = second;
  paused.frame_id = 2;
  paused.motion_phase = a3_deploy::MotionPhase::kPaused;
  paused.valid_mask &= ~a3_deploy::kReferenceValid;
  paused.reference_tick = a3_deploy::kA3EvalNoTick;
  tracker.Update(paused, 100, output);
  ASSERT_EQ(output.count, 1u);
  EXPECT_EQ(output.events[0].event_type,
            a3_deploy::EvalEventType::kEpisodeEnded);
  EXPECT_EQ(output.events[0].outcome,
            a3_deploy::EpisodeOutcome::kAborted);
  EXPECT_EQ(output.events[0].reason,
            a3_deploy::TerminationReason::kUserPause);
  EXPECT_EQ(output.events[0].reference_tick, 1u);

  // Demo 1 规定暂停后再次播放从 reference_tick=0 开始，并产生新 episode。
  auto replay = PlayingFrame(3, 0, 2);
  tracker.Update(replay, 100, output);
  ASSERT_EQ(output.count, 1u);
  EXPECT_EQ(output.events[0].event_type,
            a3_deploy::EvalEventType::kEpisodeStarted);
  EXPECT_NE(replay.episode_id, first_episode);
}

TEST(A3EvalEpisodeTracker, MotionSwitchAbortsOldAndStartsNewEpisode) {
  a3_deploy::A3EvalEpisodeTracker tracker;
  a3_deploy::EvalTrackerOutput output;

  auto first = PlayingFrame(0, 0, 1);
  tracker.Update(first, 50, output);
  ASSERT_EQ(output.count, 1u);
  const auto old_episode = first.episode_id;

  auto switched = PlayingFrame(1, 0, 4);
  tracker.Update(switched, 80, output);
  ASSERT_EQ(output.count, 3u);
  EXPECT_EQ(output.events[0].event_type,
            a3_deploy::EvalEventType::kMotionSelected);
  EXPECT_EQ(output.events[1].event_type,
            a3_deploy::EvalEventType::kEpisodeEnded);
  EXPECT_EQ(output.events[1].outcome,
            a3_deploy::EpisodeOutcome::kAborted);
  EXPECT_EQ(output.events[1].reason,
            a3_deploy::TerminationReason::kMotionSwitch);
  EXPECT_EQ(output.events[2].event_type,
            a3_deploy::EvalEventType::kEpisodeStarted);
  EXPECT_NE(switched.episode_id, old_episode);
}

TEST(A3EvalEpisodeTracker, TerminalHoldCompletesContinuousReference) {
  a3_deploy::A3EvalEpisodeTracker tracker;
  a3_deploy::EvalTrackerOutput output;

  for (std::uint64_t tick = 0; tick < 3; ++tick) {
    auto frame = PlayingFrame(tick, tick, 7);
    tracker.Update(frame, 3, output);
  }

  auto terminal = PlayingFrame(3, 2, 7);
  terminal.motion_phase = a3_deploy::MotionPhase::kTerminalHold;
  terminal.valid_mask &= ~a3_deploy::kReferenceValid;
  terminal.reference_tick = a3_deploy::kA3EvalNoTick;
  tracker.Update(terminal, 3, output);

  ASSERT_EQ(output.count, 1u);
  EXPECT_EQ(output.events[0].event_type,
            a3_deploy::EvalEventType::kEpisodeEnded);
  EXPECT_EQ(output.events[0].outcome,
            a3_deploy::EpisodeOutcome::kCompleted);
  EXPECT_EQ(output.events[0].reason,
            a3_deploy::TerminationReason::kEndOfReference);
  EXPECT_EQ(output.events[0].reference_tick, 2u);
}

TEST(A3EvalEpisodeTracker, ProcessExitClosesActiveEpisode) {
  a3_deploy::A3EvalEpisodeTracker tracker;
  a3_deploy::EvalTrackerOutput output;

  auto first = PlayingFrame(10, 0, 3);
  tracker.Update(first, 20, output);
  tracker.Finalize(11, 123456, output);

  ASSERT_EQ(output.count, 1u);
  EXPECT_EQ(output.events[0].event_type,
            a3_deploy::EvalEventType::kEpisodeEnded);
  EXPECT_EQ(output.events[0].outcome,
            a3_deploy::EpisodeOutcome::kAborted);
  EXPECT_EQ(output.events[0].reason,
            a3_deploy::TerminationReason::kProcessExit);
  EXPECT_EQ(output.events[0].event_time_monotonic_ns, 123456);
}
