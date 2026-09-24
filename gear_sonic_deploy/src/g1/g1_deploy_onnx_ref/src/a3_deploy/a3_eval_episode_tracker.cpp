// Copyright (c) 2026, AgiBot Inc. All rights reserved.
#include "a3_deploy/a3_eval_episode_tracker.hpp"

namespace a3_deploy {

namespace {

bool IsPlayingFrame(const EvalFrameV1& frame) noexcept {
  return frame.deploy_mode == DeployMode::kMotion &&
         frame.motion_phase == MotionPhase::kPlaying &&
         (frame.valid_mask & kReferenceValid) != 0 &&
         frame.active_motion_id != kA3EvalNoMotion &&
         frame.reference_tick != kA3EvalNoTick;
}

}  // namespace

EvalEventV1 A3EvalEpisodeTracker::MakeEvent_(
    EvalEventType type, const EvalFrameV1& frame) noexcept {
  EvalEventV1 event;
  event.event_id = next_event_id_++;
  event.frame_id = frame.frame_id;
  event.event_time_monotonic_ns = frame.control_start_monotonic_ns;
  event.event_type = type;
  event.episode_id = current_episode_id_;
  event.reference_tick = frame.reference_tick;
  event.from_motion_id = has_previous_ ? previous_.selected_motion_id
                                       : kA3EvalNoMotion;
  event.to_motion_id = frame.selected_motion_id;
  event.from_mode = has_previous_ ? previous_.deploy_mode : frame.deploy_mode;
  event.to_mode = frame.deploy_mode;
  event.from_phase = has_previous_ ? previous_.motion_phase : MotionPhase::kNone;
  event.to_phase = frame.motion_phase;
  return event;
}

void A3EvalEpisodeTracker::AccumulateQuality_(
    const EvalFrameV1& frame) noexcept {
  if ((frame.quality_flags & kWatchdogSafeHalt) != 0) {
    episode_quality_ok_ = false;
    episode_failure_reason_ = TerminationReason::kWatchdog;
  } else if ((frame.valid_mask & kInferTimingValid) != 0 &&
             (frame.quality_flags & kInferOk) == 0) {
    episode_quality_ok_ = false;
    episode_failure_reason_ = TerminationReason::kInferenceFailure;
  } else if ((frame.valid_mask & kSendTimingValid) != 0 &&
             (frame.quality_flags & kCommandSent) == 0) {
    episode_quality_ok_ = false;
    episode_failure_reason_ = TerminationReason::kCommandFailure;
  } else if ((frame.quality_flags & kSyncComplete) == 0) {
    episode_quality_ok_ = false;
    episode_failure_reason_ = TerminationReason::kSyncFailure;
  }
}

void A3EvalEpisodeTracker::StartEpisode_(
    EvalFrameV1& frame, std::uint64_t reference_num_ticks,
    EvalTrackerOutput& output) noexcept {
  episode_active_ = true;
  current_episode_id_ = next_episode_id_++;
  episode_motion_id_ = frame.active_motion_id;
  episode_reference_num_ticks_ = reference_num_ticks;
  episode_last_reference_tick_ = frame.reference_tick;
  episode_quality_ok_ = frame.reference_tick == 0;
  episode_failure_reason_ = episode_quality_ok_ ? TerminationReason::kNone
                                                : TerminationReason::kUnknown;
  frame.episode_id = current_episode_id_;
  AccumulateQuality_(frame);

  auto event = MakeEvent_(EvalEventType::kEpisodeStarted, frame);
  event.episode_id = current_episode_id_;
  event.from_motion_id = kA3EvalNoMotion;
  event.to_motion_id = episode_motion_id_;
  event.outcome = EpisodeOutcome::kInProgress;
  output.Push(event);
}

void A3EvalEpisodeTracker::EndEpisode_(
    const EvalFrameV1& frame, EpisodeOutcome outcome,
    TerminationReason reason, EvalTrackerOutput& output) noexcept {
  if (!episode_active_) return;
  if (outcome == EpisodeOutcome::kCompleted &&
      require_quality_for_completion_ && !episode_quality_ok_) {
    outcome = EpisodeOutcome::kFailed;
    reason = episode_failure_reason_ == TerminationReason::kNone
                 ? TerminationReason::kUnknown
                 : episode_failure_reason_;
  }

  auto event = MakeEvent_(EvalEventType::kEpisodeEnded, frame);
  event.episode_id = current_episode_id_;
  event.reference_tick = episode_last_reference_tick_;
  event.from_motion_id = episode_motion_id_;
  event.to_motion_id = frame.active_motion_id;
  event.outcome = outcome;
  event.reason = reason;
  output.Push(event);

  episode_active_ = false;
  current_episode_id_ = 0;
  episode_motion_id_ = kA3EvalNoMotion;
  episode_reference_num_ticks_ = 0;
  episode_last_reference_tick_ = kA3EvalNoTick;
  episode_quality_ok_ = true;
  episode_failure_reason_ = TerminationReason::kNone;
}

void A3EvalEpisodeTracker::Update(EvalFrameV1& frame,
                                  std::uint64_t reference_num_ticks,
                                  EvalTrackerOutput& output) noexcept {
  output.count = 0;
  frame.episode_id = 0;

  if (has_previous_ && frame.deploy_mode != previous_.deploy_mode) {
    output.Push(MakeEvent_(EvalEventType::kModeChanged, frame));
  }
  if (has_previous_ &&
      frame.selected_motion_id != previous_.selected_motion_id) {
    output.Push(MakeEvent_(EvalEventType::kMotionSelected, frame));
  }

  const bool playing = IsPlayingFrame(frame);
  if (episode_active_ && playing &&
      (frame.active_motion_id != episode_motion_id_ ||
       frame.reference_tick == 0)) {
    // 播放中的 motion 被替换，或者 tick 意外回到 0：旧 episode 必须截断。
    EndEpisode_(frame, EpisodeOutcome::kAborted,
                frame.active_motion_id != episode_motion_id_
                    ? TerminationReason::kMotionSwitch
                    : TerminationReason::kUserStop,
                output);
  }

  if (episode_active_ && !playing) {
    EpisodeOutcome outcome = EpisodeOutcome::kAborted;
    TerminationReason reason = TerminationReason::kUserStop;
    if ((frame.quality_flags & kWatchdogSafeHalt) != 0) {
      outcome = EpisodeOutcome::kFailed;
      reason = TerminationReason::kWatchdog;
    } else if (frame.deploy_mode != DeployMode::kMotion) {
      reason = TerminationReason::kModeSwitch;
    } else if (frame.selected_motion_id != episode_motion_id_) {
      reason = TerminationReason::kMotionSwitch;
    } else if (frame.motion_phase == MotionPhase::kTerminalHold &&
               episode_reference_num_ticks_ > 0 &&
               episode_last_reference_tick_ + 1 >=
                   episode_reference_num_ticks_) {
      outcome = EpisodeOutcome::kCompleted;
      reason = TerminationReason::kEndOfReference;
    } else if (frame.motion_phase == MotionPhase::kPaused) {
      reason = TerminationReason::kUserPause;
    }
    EndEpisode_(frame, outcome, reason, output);
  }

  if (!episode_active_ && playing) {
    StartEpisode_(frame, reference_num_ticks, output);
  } else if (episode_active_ && playing) {
    if (episode_last_reference_tick_ != kA3EvalNoTick &&
        frame.reference_tick != episode_last_reference_tick_ + 1) {
      episode_quality_ok_ = false;
      episode_failure_reason_ = TerminationReason::kUnknown;
    }
    episode_last_reference_tick_ = frame.reference_tick;
    frame.episode_id = current_episode_id_;
    AccumulateQuality_(frame);
  }

  previous_ = frame;
  has_previous_ = true;
}

void A3EvalEpisodeTracker::Finalize(std::uint64_t frame_id,
                                    std::int64_t monotonic_ns,
                                    EvalTrackerOutput& output) noexcept {
  output.count = 0;
  if (!episode_active_) return;
  EvalFrameV1 terminal = previous_;
  terminal.frame_id = frame_id;
  terminal.control_start_monotonic_ns = monotonic_ns;
  EndEpisode_(terminal, EpisodeOutcome::kAborted,
              TerminationReason::kProcessExit, output);
}

}  // namespace a3_deploy
