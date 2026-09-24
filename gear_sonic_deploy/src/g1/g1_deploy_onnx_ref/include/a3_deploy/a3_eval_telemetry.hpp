// Copyright (c) 2026, AgiBot Inc. All rights reserved.
//
// Foxglove/ROS2 实时可视化出口。调用方只执行 TryPush；ROS2 初始化、
// message 构造和 DDS publish 全部发生在独立 publisher thread。
#pragma once

#include "a3_deploy/a3_eval_spsc_queue.hpp"
#include "a3_deploy/a3_eval_types.hpp"

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <thread>

namespace a3_deploy {

struct A3EvalTelemetryOptions {
  bool enabled = false;
  double publish_hz = 10.0;
  std::size_t queue_capacity = 256;
  std::string node_name = "a3_eval_telemetry";
  std::string topic_prefix = "/a3/eval";
};

struct A3EvalTelemetryStatistics {
  std::uint64_t frames_accepted = 0;
  std::uint64_t frames_published = 0;
  std::uint64_t frames_dropped = 0;
};

class A3EvalTelemetryPublisher {
 public:
  explicit A3EvalTelemetryPublisher(A3EvalTelemetryOptions options);
  ~A3EvalTelemetryPublisher();

  A3EvalTelemetryPublisher(const A3EvalTelemetryPublisher&) = delete;
  A3EvalTelemetryPublisher& operator=(const A3EvalTelemetryPublisher&) = delete;

  bool Start(std::string* error = nullptr);
  void Stop() noexcept;
  bool TryPush(const EvalFrameV1& frame) noexcept;

  bool Enabled() const noexcept { return options_.enabled; }
  bool Running() const noexcept {
    return running_.load(std::memory_order_acquire);
  }
  A3EvalTelemetryStatistics Statistics() const noexcept;
  static bool Ros2AvailableAtBuildTime() noexcept;

 private:
  struct Impl;
  void PublisherMain_() noexcept;

  A3EvalTelemetryOptions options_;
  std::unique_ptr<A3EvalSpscQueue<EvalFrameV1>> queue_;
  std::unique_ptr<Impl> impl_;
  std::thread publisher_thread_;
  std::atomic<bool> running_{false};
  std::atomic<std::uint64_t> accepted_{0};
  std::atomic<std::uint64_t> published_{0};
  std::atomic<std::uint64_t> dropped_{0};
};

}  // namespace a3_deploy
