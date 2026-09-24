// Copyright (c) 2026, AgiBot Inc. All rights reserved.
#pragma once

#include "a3_deploy/a3_manual_control.hpp"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <mutex>
#include <string>
#include <vector>

namespace a3_deploy {

struct A3RpcMotionEntry {
  int id = -1;
  std::string name;
  std::filesystem::path path;
  std::size_t total_ticks = 0;
};

struct A3RpcPlayResult {
  bool accepted = false;
  std::uint64_t play_id = 0;
  int motion_id = -1;
  std::string motion_name;
  std::string motion_path;
  std::size_t total_ticks = 0;
  std::string state;
  std::string message;
};

struct A3RpcPlaybackStatus {
  std::uint64_t play_id = 0;
  int motion_id = -1;
  std::string motion_name;
  std::string motion_path;
  std::size_t tick = 0;
  std::size_t total_ticks = 0;
  std::string deploy_mode;
  std::string state;
  bool playing = false;
  bool held = false;
};

class A3MotionRpcController {
 public:
  A3MotionRpcController(ManualControlState& control,
                        std::vector<A3RpcMotionEntry> motions);

  A3RpcPlayResult PlayMotion(const std::string& motion_path);
  A3RpcPlaybackStatus GetPlaybackStatus() const;
  A3RpcPlayResult StopMotion();

 private:
  static std::filesystem::path NormalizePath_(
      const std::filesystem::path& path);

  ManualControlState& control_;
  std::vector<A3RpcMotionEntry> motions_;
  mutable std::mutex mutex_;
  std::uint64_t next_play_id_ = 1;
  std::uint64_t current_play_id_ = 0;
  int current_motion_id_ = -1;
};

}  // namespace a3_deploy
