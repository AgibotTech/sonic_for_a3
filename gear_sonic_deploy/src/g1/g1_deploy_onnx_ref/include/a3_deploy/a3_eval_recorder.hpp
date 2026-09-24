// Copyright (c) 2026, AgiBot Inc. All rights reserved.
//
// 非阻塞 Evaluation Recorder：控制线程只负责 try_push，磁盘 I/O 由独立
// writer thread 完成。
#pragma once

#include "a3_deploy/a3_eval_episode_tracker.hpp"
#include "a3_deploy/a3_eval_mcap_writer.hpp"
#include "a3_deploy/a3_eval_spsc_queue.hpp"
#include "a3_deploy/a3_eval_telemetry.hpp"
#include "a3_deploy/a3_eval_types.hpp"

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <memory>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

namespace a3_deploy {

struct EvalMotionManifestEntry {
  std::int32_t motion_id = kA3EvalNoMotion;
  std::string name;
  std::string source_path;
  std::uint64_t num_ticks = 0;
};

struct A3EvalRecorderOptions {
  bool enabled = false;
  // false 保持连续录制；true 只提交从 tick 0 自然播放到末尾的完整 motion。
  bool complete_motion_only = false;
  std::filesystem::path output_root;
  std::string session_id;
  std::string backend_name;
  std::string state_clock_domain = "backend_defined";
  std::string runtime_config_path;
  std::string policy_model_path;
  double policy_hz = 50.0;
  double state_hz = 100.0;
  std::size_t frame_queue_capacity = 8192;
  std::size_t flush_frames = 512;
  std::vector<EvalMotionManifestEntry> motions;
  A3EvalTelemetryOptions telemetry;
  A3EvalMcapOptions episode_mcap;
};

struct A3EvalRecorderStatistics {
  std::uint64_t frames_accepted = 0;
  std::uint64_t frames_written = 0;
  std::uint64_t frames_dropped = 0;
  std::uint64_t frames_ignored = 0;
  std::uint64_t frames_discarded = 0;
  std::uint64_t episodes_started = 0;
  std::uint64_t episodes_committed = 0;
  std::uint64_t episodes_discarded = 0;
  std::uint64_t events_accepted = 0;
  std::uint64_t events_written = 0;
  std::uint64_t events_dropped = 0;
  std::uint64_t telemetry_accepted = 0;
  std::uint64_t telemetry_published = 0;
  std::uint64_t telemetry_dropped = 0;
  std::uint64_t mcap_files_written = 0;
  std::uint64_t mcap_write_failures = 0;
  std::uint64_t mcap_bytes_written = 0;
  std::uint64_t readable_mcap_files_written = 0;
  std::uint64_t readable_mcap_bytes_written = 0;
};

class A3EvalRecorder {
 public:
  explicit A3EvalRecorder(A3EvalRecorderOptions options);
  ~A3EvalRecorder();

  A3EvalRecorder(const A3EvalRecorder&) = delete;
  A3EvalRecorder& operator=(const A3EvalRecorder&) = delete;

  bool Start(std::string* error = nullptr);
  void Stop() noexcept;

  // 控制线程只调用这个非阻塞接口。队列满时返回 false，不等待 writer。
  bool TryPushFrame(const EvalFrameV1& frame) noexcept;

  bool Enabled() const noexcept { return options_.enabled; }
  bool Running() const noexcept { return running_.load(std::memory_order_acquire); }
  const std::filesystem::path& SessionDirectory() const noexcept {
    return session_dir_;
  }
  A3EvalRecorderStatistics Statistics() const noexcept;

 private:
  void WriterMain_() noexcept;
  std::uint64_t ReferenceTicksForMotion_(std::int32_t motion_id) const noexcept;
  void ProcessFrame_(EvalFrameV1& frame, std::uint64_t& since_flush) noexcept;
  void WriteFrame_(const EvalFrameV1& frame,
                   std::uint64_t& since_flush) noexcept;
  void WritePendingEpisodeMcap_() noexcept;
  void ProcessCompleteMotion_(EvalFrameV1* frame,
                              EvalTrackerOutput& output,
                              std::uint64_t& since_flush) noexcept;
  bool PendingEpisodeIsComplete_(const EvalEventV1& end_event) const noexcept;
  void ClearPendingEpisode_() noexcept;
  void ProcessTrackerEvents_(const EvalTrackerOutput& output) noexcept;
  void WriteSessionMetadata_();
  void WriteManifest_();
  void WriteEvent_(const EvalEventV1& event);
  void WriteSummary_() noexcept;

  A3EvalRecorderOptions options_;
  std::filesystem::path session_dir_;
  std::unique_ptr<A3EvalSpscQueue<EvalFrameV1>> frame_queue_;
  std::unique_ptr<A3EvalTelemetryPublisher> telemetry_;
  std::unique_ptr<A3EvalMcapWriter> mcap_writer_;
  std::ofstream frames_stream_;
  std::ofstream events_stream_;
  std::thread writer_thread_;
  std::atomic<bool> running_{false};
  // complete_motion_only + episode_mcap 时只产出 session/data 下的正式文件。
  bool mcap_only_output_{false};
  A3EvalEpisodeTracker episode_tracker_;
  std::uint64_t last_frame_id_{0};
  std::int64_t last_frame_time_ns_{0};
  bool have_last_frame_{false};
  std::int64_t last_telemetry_time_ns_{0};

  // complete_motion_only 模式只在 writer thread 使用：PLAYING 帧先留在
  // 内存，收到完整结束事件后才提交到 frames.bin。
  std::vector<EvalFrameV1> pending_episode_frames_;
  std::uint64_t pending_episode_id_{0};
  std::int32_t pending_motion_id_{kA3EvalNoMotion};
  std::uint64_t pending_reference_num_ticks_{0};
  A3EvalMcapSessionMetadata mcap_session_metadata_;
  std::unordered_map<std::int32_t, std::uint64_t> motion_repeat_counts_;
  std::unordered_map<std::uint64_t, std::string> episode_mcap_files_;
  std::unordered_map<std::uint64_t, std::string> episode_readable_mcap_files_;

  std::atomic<std::uint64_t> frames_accepted_{0};
  std::atomic<std::uint64_t> frames_written_{0};
  std::atomic<std::uint64_t> frames_dropped_{0};
  std::atomic<std::uint64_t> frames_ignored_{0};
  std::atomic<std::uint64_t> frames_discarded_{0};
  std::atomic<std::uint64_t> episodes_started_{0};
  std::atomic<std::uint64_t> episodes_committed_{0};
  std::atomic<std::uint64_t> episodes_discarded_{0};
  std::atomic<std::uint64_t> events_accepted_{0};
  std::atomic<std::uint64_t> events_written_{0};
  std::atomic<std::uint64_t> events_dropped_{0};
  std::atomic<std::uint64_t> mcap_files_written_{0};
  std::atomic<std::uint64_t> mcap_write_failures_{0};
  std::atomic<std::uint64_t> mcap_bytes_written_{0};
  std::atomic<std::uint64_t> readable_mcap_files_written_{0};
  std::atomic<std::uint64_t> readable_mcap_bytes_written_{0};

  // SPSC producer（driver thread）专用：某帧入队失败后，在下一帧留下缺口标记。
  bool pending_frame_gap_{false};
};

}  // namespace a3_deploy
