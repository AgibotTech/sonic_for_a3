// Copyright (c) 2026, AgiBot Inc. All rights reserved.

#include "a3_deploy/a3_motion_rpc_controller.hpp"

#include <algorithm>
#include <sstream>
#include <system_error>

namespace a3_deploy {

namespace {

const A3RpcMotionEntry* FindMotionById(
    const std::vector<A3RpcMotionEntry>& motions, int id) {
  const auto it = std::find_if(
      motions.begin(), motions.end(),
      [id](const A3RpcMotionEntry& motion) { return motion.id == id; });
  return it == motions.end() ? nullptr : &*it;
}

A3RpcPlayResult ResultFor(const A3RpcMotionEntry* motion) {
  A3RpcPlayResult result;
  if (motion == nullptr) return result;
  result.motion_id = motion->id;
  result.motion_name = motion->name;
  result.motion_path = motion->path.string();
  result.total_ticks = motion->total_ticks;
  return result;
}

}  // namespace

A3MotionRpcController::A3MotionRpcController(
    ManualControlState& control, std::vector<A3RpcMotionEntry> motions)
    : control_(control), motions_(std::move(motions)) {
  for (auto& motion : motions_) {
    motion.path = NormalizePath_(motion.path);
  }
}

std::filesystem::path A3MotionRpcController::NormalizePath_(
    const std::filesystem::path& path) {
  std::error_code ec;
  auto absolute = std::filesystem::absolute(path, ec);
  if (ec) {
    ec.clear();
    absolute = path;
  }
  auto canonical = std::filesystem::weakly_canonical(absolute, ec);
  return (ec ? absolute : canonical).lexically_normal();
}

A3RpcPlayResult A3MotionRpcController::PlayMotion(
    const std::string& motion_path) {
  std::lock_guard lock(mutex_);
  const auto normalized = NormalizePath_(motion_path);
  const auto it = std::find_if(
      motions_.begin(), motions_.end(),
      [&normalized](const A3RpcMotionEntry& motion) {
        return motion.path == normalized;
      });
  if (it == motions_.end()) {
    A3RpcPlayResult result;
    result.state = "rejected";
    result.message = "motion_path is not in the loaded motion catalog";
    return result;
  }

  A3RpcPlayResult result = ResultFor(&*it);
  const DeployMode mode = LoadDeployMode(control_);
  if (mode == DeployMode::kIdle || mode == DeployMode::kPassive) {
    result.state = "blocked";
    result.message = "enter pd_stand and wait until ready first";
    return result;
  }
  if (mode == DeployMode::kPdStand &&
      !control_.pd_stand_ready.load(std::memory_order_acquire)) {
    result.state = "blocked";
    result.message = "pd_stand is not ready";
    return result;
  }

  current_play_id_ = next_play_id_++;
  current_motion_id_ = it->id;
  result.play_id = current_play_id_;
  result.accepted = true;
  result.state = "playing";
  result.message = "accepted";

  control_.remote_motion_active.store(false, std::memory_order_release);
  control_.selected_motion_index.store(it->id, std::memory_order_release);
  control_.motion_playing.store(true, std::memory_order_release);
  control_.motion_command_epoch.fetch_add(1, std::memory_order_acq_rel);
  if (mode != DeployMode::kMotion) {
    std::ostringstream ignored_log;
    RequestDeployMode(control_, DeployMode::kMotion, ignored_log);
    // RequestDeployMode only clears playback when leaving motion.
    control_.motion_playing.store(true, std::memory_order_release);
  }
  return result;
}

A3RpcPlaybackStatus A3MotionRpcController::GetPlaybackStatus() const {
  std::lock_guard lock(mutex_);
  A3RpcPlaybackStatus status;
  status.play_id = current_play_id_;
  status.motion_id = current_motion_id_;
  const auto* motion = FindMotionById(motions_, current_motion_id_);
  if (motion != nullptr) {
    status.motion_name = motion->name;
    status.motion_path = motion->path.string();
    status.total_ticks = motion->total_ticks;
  }
  status.tick = static_cast<std::size_t>(
      control_.motion_tick.load(std::memory_order_acquire));
  status.playing =
      control_.motion_playing.load(std::memory_order_acquire);
  status.held = control_.motion_held.load(std::memory_order_acquire);
  const DeployMode mode = LoadDeployMode(control_);
  status.deploy_mode = DeployModeName(mode);

  if (current_play_id_ == 0) {
    status.state = "idle";
  } else if (mode != DeployMode::kMotion) {
    status.state = "stopped";
  } else if (control_.selected_motion_index.load(std::memory_order_acquire) !=
             current_motion_id_) {
    status.state = "superseded";
  } else if (status.playing) {
    status.state = "playing";
  } else if (status.held && status.total_ticks > 0 &&
             status.tick + 1 >= status.total_ticks) {
    status.state = "completed";
  } else if (status.held) {
    status.state = "paused";
  } else {
    status.state = "ready";
  }
  return status;
}

A3RpcPlayResult A3MotionRpcController::StopMotion() {
  std::lock_guard lock(mutex_);
  const auto* motion = FindMotionById(motions_, current_motion_id_);
  A3RpcPlayResult result = ResultFor(motion);
  result.play_id = current_play_id_;
  if (current_play_id_ == 0 ||
      LoadDeployMode(control_) != DeployMode::kMotion) {
    result.state = "idle";
    result.message = "no RPC motion is active";
    return result;
  }
  control_.motion_playing.store(false, std::memory_order_release);
  control_.motion_command_epoch.fetch_add(1, std::memory_order_acq_rel);
  result.accepted = true;
  result.state = "paused";
  result.message = "stop accepted";
  return result;
}

}  // namespace a3_deploy
