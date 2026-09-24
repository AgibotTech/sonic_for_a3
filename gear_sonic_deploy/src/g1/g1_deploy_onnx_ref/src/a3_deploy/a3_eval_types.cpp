// Copyright (c) 2026, AgiBot Inc. All rights reserved.
#include "a3_deploy/a3_eval_types.hpp"

namespace a3_deploy {

const char* MotionPhaseName(MotionPhase value) noexcept {
  switch (value) {
    case MotionPhase::kNone: return "NONE";
    case MotionPhase::kReady: return "READY";
    case MotionPhase::kPlaying: return "PLAYING";
    case MotionPhase::kPaused: return "PAUSED";
    case MotionPhase::kTerminalHold: return "TERMINAL_HOLD";
  }
  return "UNKNOWN";
}

const char* CommandSourceName(CommandSource value) noexcept {
  switch (value) {
    case CommandSource::kNone: return "NONE";
    case CommandSource::kPassive: return "PASSIVE";
    case CommandSource::kPdStand: return "PD_STAND";
    case CommandSource::kPolicy: return "POLICY";
    case CommandSource::kTeleop: return "TELEOP";
    case CommandSource::kSafeHalt: return "SAFE_HALT";
  }
  return "UNKNOWN";
}

const char* EvalEventTypeName(EvalEventType value) noexcept {
  switch (value) {
    case EvalEventType::kModeChanged: return "MODE_CHANGED";
    case EvalEventType::kMotionSelected: return "MOTION_SELECTED";
    case EvalEventType::kEpisodeStarted: return "EPISODE_STARTED";
    case EvalEventType::kEpisodeEnded: return "EPISODE_ENDED";
    case EvalEventType::kRecorderDropped: return "RECORDER_DROPPED";
  }
  return "UNKNOWN";
}

const char* EpisodeOutcomeName(EpisodeOutcome value) noexcept {
  switch (value) {
    case EpisodeOutcome::kNone: return "NONE";
    case EpisodeOutcome::kInProgress: return "IN_PROGRESS";
    case EpisodeOutcome::kCompleted: return "COMPLETED";
    case EpisodeOutcome::kAborted: return "ABORTED";
    case EpisodeOutcome::kFailed: return "FAILED";
  }
  return "UNKNOWN";
}

const char* TerminationReasonName(TerminationReason value) noexcept {
  switch (value) {
    case TerminationReason::kNone: return "NONE";
    case TerminationReason::kEndOfReference: return "END_OF_REFERENCE";
    case TerminationReason::kUserPause: return "USER_PAUSE";
    case TerminationReason::kUserStop: return "USER_STOP";
    case TerminationReason::kMotionSwitch: return "MOTION_SWITCH";
    case TerminationReason::kModeSwitch: return "MODE_SWITCH";
    case TerminationReason::kWatchdog: return "WATCHDOG";
    case TerminationReason::kFall: return "FALL";
    case TerminationReason::kInferenceFailure: return "INFERENCE_FAILURE";
    case TerminationReason::kCommandFailure: return "COMMAND_FAILURE";
    case TerminationReason::kSyncFailure: return "SYNC_FAILURE";
    case TerminationReason::kProcessExit: return "PROCESS_EXIT";
    case TerminationReason::kUnknown: return "UNKNOWN";
  }
  return "UNKNOWN";
}

}  // namespace a3_deploy
