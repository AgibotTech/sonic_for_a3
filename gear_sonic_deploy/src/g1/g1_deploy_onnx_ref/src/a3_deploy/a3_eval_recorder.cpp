// Copyright (c) 2026, AgiBot Inc. All rights reserved.
#include "a3_deploy/a3_eval_recorder.hpp"

#include "robot_io/a3_layout_extra.hpp"

#include <algorithm>
#include <chrono>
#include <ctime>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <system_error>
#include <utility>

#include <yaml-cpp/yaml.h>

namespace a3_deploy {

namespace {

std::string MakeSessionId() {
  const auto now = std::chrono::system_clock::now();
  const auto tt = std::chrono::system_clock::to_time_t(now);
  std::tm tm{};
  localtime_r(&tt, &tm);
  std::ostringstream out;
  out << std::put_time(&tm, "%Y%m%d_%H%M%S");
  return out.str();
}

std::string JsonEscape(const std::string& value) {
  std::ostringstream out;
  for (const char c : value) {
    switch (c) {
      case '\\': out << "\\\\"; break;
      case '"': out << "\\\""; break;
      case '\n': out << "\\n"; break;
      case '\r': out << "\\r"; break;
      case '\t': out << "\\t"; break;
      default: out << c; break;
    }
  }
  return out.str();
}

bool IsSafeRelativeDirectory(const std::filesystem::path& path) {
  if (path.empty() || path.is_absolute()) return false;
  for (const auto& part : path) {
    if (part == "..") return false;
  }
  return true;
}

}  // namespace

A3EvalRecorder::A3EvalRecorder(A3EvalRecorderOptions options)
    : options_(std::move(options)),
      episode_tracker_(!options_.complete_motion_only) {}

A3EvalRecorder::~A3EvalRecorder() { Stop(); }

bool A3EvalRecorder::Start(std::string* error) {
  if (!options_.enabled) return true;
  if (running_.load(std::memory_order_acquire)) {
    if (error) *error = "evaluation recorder 已经运行";
    return false;
  }
  if (options_.output_root.empty()) {
    if (error) *error = "evaluation_recording.output_root 不能为空";
    return false;
  }
  if (options_.episode_mcap.enabled && !options_.complete_motion_only) {
    if (error) {
      *error = "evaluation_recording.episode_mcap 只支持 "
               "complete_motion_only=true";
    }
    return false;
  }
  if (options_.episode_mcap.enabled &&
      !A3EvalMcapWriter::AvailableAtBuildTime()) {
    if (error) *error = "当前二进制未编译 episode MCAP 支持";
    return false;
  }
  if (options_.episode_mcap.enabled &&
      !IsSafeRelativeDirectory(options_.episode_mcap.raw_directory)) {
    if (error) *error = "episode_mcap.raw_directory 必须是安全相对路径";
    return false;
  }
  if (options_.episode_mcap.enabled &&
      options_.episode_mcap.readable_copy_enabled &&
      !IsSafeRelativeDirectory(options_.episode_mcap.readable_directory)) {
    if (error) *error = "episode_mcap.readable_directory 必须是安全相对路径";
    return false;
  }
  if (options_.episode_mcap.enabled &&
      options_.episode_mcap.readable_copy_enabled &&
      options_.episode_mcap.raw_directory ==
          options_.episode_mcap.readable_directory) {
    if (error) *error = "原始和可读 MCAP 目录不能相同";
    return false;
  }
  mcap_only_output_ =
      options_.complete_motion_only && options_.episode_mcap.enabled;
  if (mcap_only_output_ && options_.episode_mcap.readable_copy_enabled) {
    if (error) {
      *error = "complete-motion raw-only 输出不再生成 data_read；请关闭 "
               "episode_mcap.readable_copy";
    }
    return false;
  }

  if (options_.session_id.empty()) options_.session_id = MakeSessionId();
  std::error_code ec;
  options_.output_root =
      std::filesystem::absolute(options_.output_root, ec).lexically_normal();
  if (ec) {
    if (error) {
      *error = "解析 recorder output_root 绝对路径失败: " + ec.message();
    }
    return false;
  }
  std::filesystem::create_directories(options_.output_root, ec);
  if (ec) {
    if (error) *error = "创建 recorder output_root 失败: " + ec.message();
    return false;
  }

  // 以 create_directory 的返回值占用唯一目录，避免同名进程并发启动或同一秒
  // 重启时覆盖既有 session。最多尝试原名加 999 个后缀。
  const std::string base_session_id = options_.session_id;
  bool session_created = false;
  for (int suffix = 0; suffix < 1000; ++suffix) {
    options_.session_id = base_session_id;
    if (suffix > 0) options_.session_id += "_" + std::to_string(suffix);
    const auto candidate =
        options_.output_root / ("session_" + options_.session_id);
    ec.clear();
    if (std::filesystem::create_directory(candidate, ec)) {
      session_dir_ = candidate;
      session_created = true;
      break;
    }
    if (ec) {
      if (error) *error = "创建 session 目录失败: " + ec.message();
      return false;
    }
  }
  if (!session_created) {
    if (error) *error = "无法分配唯一 session 目录（已尝试 1000 个名称）";
    return false;
  }

  if (options_.episode_mcap.enabled) {
    ec.clear();
    std::filesystem::create_directories(
        session_dir_ / options_.episode_mcap.raw_directory, ec);
    if (ec) {
      if (error) *error = "创建原始 MCAP 目录失败: " + ec.message();
      return false;
    }
    if (options_.episode_mcap.readable_copy_enabled) {
      ec.clear();
      std::filesystem::create_directories(
          session_dir_ / options_.episode_mcap.readable_directory, ec);
      if (ec) {
        if (error) *error = "创建可读 MCAP 目录失败: " + ec.message();
        return false;
      }
    }
  }

  if (!mcap_only_output_) {
    frames_stream_.open(session_dir_ / "frames.bin",
                        std::ios::binary | std::ios::trunc);
    events_stream_.open(session_dir_ / "events.jsonl", std::ios::trunc);
    if (!frames_stream_ || !events_stream_) {
      if (error) *error = "打开 frames.bin 或 events.jsonl 失败";
      return false;
    }
  }

  try {
    frame_queue_ = std::make_unique<A3EvalSpscQueue<EvalFrameV1>>(
        std::max<std::size_t>(2, options_.frame_queue_capacity));
    if (!mcap_only_output_) {
      const EvalFramesFileHeaderV1 header;
      frames_stream_.write(reinterpret_cast<const char*>(&header),
                           sizeof(header));
    }
    const auto system_now = std::chrono::system_clock::now();
    const auto monotonic_now = std::chrono::steady_clock::now();
    mcap_session_metadata_.session_id = options_.session_id;
    mcap_session_metadata_.backend_name = options_.backend_name;
    mcap_session_metadata_.state_clock_domain = options_.state_clock_domain;
    mcap_session_metadata_.runtime_config_path = options_.runtime_config_path;
    mcap_session_metadata_.policy_model_path = options_.policy_model_path;
    mcap_session_metadata_.policy_hz = options_.policy_hz;
    mcap_session_metadata_.state_hz = options_.state_hz;
    mcap_session_metadata_.system_time_anchor_ns =
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            system_now.time_since_epoch())
            .count();
    mcap_session_metadata_.monotonic_time_anchor_ns =
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            monotonic_now.time_since_epoch())
            .count();
    mcap_writer_ =
        std::make_unique<A3EvalMcapWriter>(options_.episode_mcap);
    if (mcap_only_output_) {
      WriteSessionMetadata_();
    } else {
      WriteManifest_();
    }
  } catch (const std::exception& e) {
    if (error) *error = std::string("初始化 evaluation recorder 失败: ") + e.what();
    frames_stream_.close();
    events_stream_.close();
    return false;
  }

  telemetry_ =
      std::make_unique<A3EvalTelemetryPublisher>(options_.telemetry);
  if (telemetry_->Enabled()) {
    std::string telemetry_error;
    if (!telemetry_->Start(&telemetry_error)) {
      // 可视化是旁路功能；启动失败不能阻止原始 evaluation 数据落盘。
      std::cerr << "[eval] realtime visualization disabled: "
                << telemetry_error << "\n";
    }
  }

  running_.store(true, std::memory_order_release);
  writer_thread_ = std::thread(&A3EvalRecorder::WriterMain_, this);
  return true;
}

void A3EvalRecorder::Stop() noexcept {
  if (!running_.exchange(false, std::memory_order_acq_rel)) return;
  if (writer_thread_.joinable()) writer_thread_.join();
  if (telemetry_) telemetry_->Stop();
  if (frames_stream_) {
    frames_stream_.flush();
    frames_stream_.close();
  }
  if (events_stream_) {
    events_stream_.flush();
    events_stream_.close();
  }
  if (!mcap_only_output_) WriteSummary_();
}

bool A3EvalRecorder::TryPushFrame(const EvalFrameV1& frame) noexcept {
  if (!Running() || !frame_queue_) {
    frames_dropped_.fetch_add(1, std::memory_order_relaxed);
    return false;
  }
  EvalFrameV1 queued_frame = frame;
  if (pending_frame_gap_) queued_frame.quality_flags |= kRecorderGap;
  if (!frame_queue_->TryPush(queued_frame)) {
    pending_frame_gap_ = true;
    frames_dropped_.fetch_add(1, std::memory_order_relaxed);
    return false;
  }
  pending_frame_gap_ = false;
  frames_accepted_.fetch_add(1, std::memory_order_relaxed);
  return true;
}

A3EvalRecorderStatistics A3EvalRecorder::Statistics() const noexcept {
  A3EvalRecorderStatistics stats;
  stats.frames_accepted = frames_accepted_.load(std::memory_order_relaxed);
  stats.frames_written = frames_written_.load(std::memory_order_relaxed);
  stats.frames_dropped = frames_dropped_.load(std::memory_order_relaxed);
  stats.frames_ignored = frames_ignored_.load(std::memory_order_relaxed);
  stats.frames_discarded = frames_discarded_.load(std::memory_order_relaxed);
  stats.episodes_started = episodes_started_.load(std::memory_order_relaxed);
  stats.episodes_committed =
      episodes_committed_.load(std::memory_order_relaxed);
  stats.episodes_discarded =
      episodes_discarded_.load(std::memory_order_relaxed);
  stats.events_accepted = events_accepted_.load(std::memory_order_relaxed);
  stats.events_written = events_written_.load(std::memory_order_relaxed);
  stats.events_dropped = events_dropped_.load(std::memory_order_relaxed);
  if (telemetry_) {
    const auto telemetry_stats = telemetry_->Statistics();
    stats.telemetry_accepted = telemetry_stats.frames_accepted;
    stats.telemetry_published = telemetry_stats.frames_published;
    stats.telemetry_dropped = telemetry_stats.frames_dropped;
  }
  stats.mcap_files_written =
      mcap_files_written_.load(std::memory_order_relaxed);
  stats.mcap_write_failures =
      mcap_write_failures_.load(std::memory_order_relaxed);
  stats.mcap_bytes_written =
      mcap_bytes_written_.load(std::memory_order_relaxed);
  stats.readable_mcap_files_written =
      readable_mcap_files_written_.load(std::memory_order_relaxed);
  stats.readable_mcap_bytes_written =
      readable_mcap_bytes_written_.load(std::memory_order_relaxed);
  return stats;
}

std::uint64_t A3EvalRecorder::ReferenceTicksForMotion_(
    std::int32_t motion_id) const noexcept {
  for (const auto& motion : options_.motions) {
    if (motion.motion_id == motion_id) return motion.num_ticks;
  }
  return 0;
}

void A3EvalRecorder::ProcessTrackerEvents_(
    const EvalTrackerOutput& output) noexcept {
  for (std::size_t i = 0; i < output.count; ++i) {
    events_accepted_.fetch_add(1, std::memory_order_relaxed);
    if (!mcap_only_output_) WriteEvent_(output.events[i]);
    events_written_.fetch_add(1, std::memory_order_relaxed);
  }
}

void A3EvalRecorder::WriteFrame_(const EvalFrameV1& frame,
                                 std::uint64_t& since_flush) noexcept {
  if (frames_stream_) {
    frames_stream_.write(reinterpret_cast<const char*>(&frame), sizeof(frame));
  }
  frames_written_.fetch_add(1, std::memory_order_relaxed);
  ++since_flush;
}

void A3EvalRecorder::WritePendingEpisodeMcap_() noexcept {
  if (!options_.episode_mcap.enabled || !mcap_writer_ ||
      pending_episode_frames_.empty()) {
    return;
  }

  const EvalMotionManifestEntry* motion = nullptr;
  for (const auto& candidate : options_.motions) {
    if (candidate.motion_id == pending_motion_id_) {
      motion = &candidate;
      break;
    }
  }
  if (motion == nullptr) {
    mcap_write_failures_.fetch_add(1, std::memory_order_relaxed);
    std::cerr << "[eval] MCAP skipped: motion manifest entry not found for id "
              << pending_motion_id_ << "\n";
    return;
  }

  // repeat 只统计同一 session 中成功生成正式 MCAP 的完整播放。
  const auto repeat_index = motion_repeat_counts_[pending_motion_id_] + 1;
  A3EvalMcapEpisodeMetadata episode;
  episode.episode_id = pending_episode_id_;
  episode.motion_id = pending_motion_id_;
  episode.motion_name = motion->name;
  episode.motion_source_path = motion->source_path;
  episode.motion_repeat_index = repeat_index;
  const auto result = mcap_writer_->WriteEpisode(
      session_dir_, mcap_session_metadata_, episode, pending_episode_frames_);
  if (!result.ok) {
    mcap_write_failures_.fetch_add(1, std::memory_order_relaxed);
    std::cerr << "[eval] MCAP write failed for episode "
              << pending_episode_id_ << ": " << result.error << "\n";
    return;
  }

  motion_repeat_counts_[pending_motion_id_] = repeat_index;
  episode_mcap_files_[pending_episode_id_] = result.relative_path.string();
  mcap_files_written_.fetch_add(1, std::memory_order_relaxed);
  mcap_bytes_written_.fetch_add(result.file_size, std::memory_order_relaxed);
  if (!result.readable_relative_path.empty()) {
    episode_readable_mcap_files_[pending_episode_id_] =
        result.readable_relative_path.string();
    readable_mcap_files_written_.fetch_add(1, std::memory_order_relaxed);
    readable_mcap_bytes_written_.fetch_add(
        result.readable_file_size, std::memory_order_relaxed);
  }
}

bool A3EvalRecorder::PendingEpisodeIsComplete_(
    const EvalEventV1& end_event) const noexcept {
  if (end_event.outcome != EpisodeOutcome::kCompleted ||
      end_event.reason != TerminationReason::kEndOfReference ||
      pending_episode_id_ == 0 ||
      end_event.episode_id != pending_episode_id_ ||
      pending_reference_num_ticks_ == 0 ||
      pending_episode_frames_.size() != pending_reference_num_ticks_) {
    return false;
  }

  for (std::size_t i = 0; i < pending_episode_frames_.size(); ++i) {
    const auto& frame = pending_episode_frames_[i];
    if (frame.episode_id != pending_episode_id_ ||
        frame.active_motion_id != pending_motion_id_ ||
        frame.deploy_mode != DeployMode::kMotion ||
        frame.motion_phase != MotionPhase::kPlaying ||
        frame.reference_tick != i ||
        (frame.valid_mask & kReferenceValid) == 0 ||
        (frame.valid_mask & kRobotStateValid) == 0 ||
        (frame.valid_mask & kRawActionValid) == 0 ||
        (frame.valid_mask & kRobotCommandValid) == 0 ||
        (frame.quality_flags & kRecorderGap) != 0) {
      return false;
    }
  }
  return true;
}

void A3EvalRecorder::ClearPendingEpisode_() noexcept {
  pending_episode_frames_.clear();
  pending_episode_id_ = 0;
  pending_motion_id_ = kA3EvalNoMotion;
  pending_reference_num_ticks_ = 0;
}

void A3EvalRecorder::ProcessCompleteMotion_(
    EvalFrameV1* frame, EvalTrackerOutput& output,
    std::uint64_t& since_flush) noexcept {
  // 一个 tick 可能同时结束旧 motion 并从 tick 0 启动新 motion；因此先
  // 结算 END，再处理 START，最后把当前 PLAYING 帧放入新 episode。
  for (std::size_t i = 0; i < output.count; ++i) {
    auto& event = output.events[i];
    if (event.event_type != EvalEventType::kEpisodeEnded ||
        event.episode_id != pending_episode_id_) {
      continue;
    }

    if (PendingEpisodeIsComplete_(event)) {
      if (!pending_episode_frames_.empty()) {
        if (!mcap_only_output_ && frames_stream_) {
          frames_stream_.write(
              reinterpret_cast<const char*>(pending_episode_frames_.data()),
              static_cast<std::streamsize>(pending_episode_frames_.size() *
                                           sizeof(EvalFrameV1)));
        }
        frames_written_.fetch_add(pending_episode_frames_.size(),
                                  std::memory_order_relaxed);
        since_flush += pending_episode_frames_.size();
      }
      // MCAP 复用当前 writer thread；完整性判断通过后才生成正式文件。
      WritePendingEpisodeMcap_();
      episodes_committed_.fetch_add(1, std::memory_order_relaxed);
    } else {
      frames_discarded_.fetch_add(pending_episode_frames_.size(),
                                  std::memory_order_relaxed);
      episodes_discarded_.fetch_add(1, std::memory_order_relaxed);
      // Reference 虽自然结束但数据字段不完整时，不能把事件保留为成功。
      if (event.outcome == EpisodeOutcome::kCompleted) {
        event.outcome = EpisodeOutcome::kFailed;
        event.reason = TerminationReason::kUnknown;
      }
    }
    ClearPendingEpisode_();
  }

  for (std::size_t i = 0; i < output.count; ++i) {
    const auto& event = output.events[i];
    if (event.event_type != EvalEventType::kEpisodeStarted) continue;

    // 防御异常事件序列：未结算的旧缓存绝不能混入新 episode。
    if (pending_episode_id_ != 0) {
      frames_discarded_.fetch_add(pending_episode_frames_.size(),
                                  std::memory_order_relaxed);
      episodes_discarded_.fetch_add(1, std::memory_order_relaxed);
      ClearPendingEpisode_();
    }
    pending_episode_id_ = event.episode_id;
    pending_motion_id_ = event.to_motion_id;
    pending_reference_num_ticks_ =
        ReferenceTicksForMotion_(pending_motion_id_);
    pending_episode_frames_.reserve(
        static_cast<std::size_t>(pending_reference_num_ticks_));
    episodes_started_.fetch_add(1, std::memory_order_relaxed);
  }

  bool staged = false;
  if (frame != nullptr && frame->episode_id != 0 &&
      frame->episode_id == pending_episode_id_ &&
      frame->deploy_mode == DeployMode::kMotion &&
      frame->motion_phase == MotionPhase::kPlaying) {
    pending_episode_frames_.push_back(*frame);
    staged = true;
  }
  if (frame != nullptr && !staged) {
    frames_ignored_.fetch_add(1, std::memory_order_relaxed);
  }
}

void A3EvalRecorder::ProcessFrame_(EvalFrameV1& frame,
                                   std::uint64_t& since_flush) noexcept {
  // episode 边界推导与 event 生成只在 writer thread 执行，不占用 policy。
  EvalTrackerOutput tracker_output;
  episode_tracker_.Update(frame,
                          ReferenceTicksForMotion_(frame.active_motion_id),
                          tracker_output);
  if (options_.complete_motion_only) {
    ProcessCompleteMotion_(&frame, tracker_output, since_flush);
  } else {
    WriteFrame_(frame, since_flush);
  }
  ProcessTrackerEvents_(tracker_output);

  last_frame_id_ = frame.frame_id;
  last_frame_time_ns_ = frame.control_start_monotonic_ns;
  have_last_frame_ = true;

  // 原始记录保持 50 Hz；这里只在 writer 侧按时间降采样后投递到第二条
  // SPSC 队列。telemetry 满只会丢实时画面，不会影响 frames.bin。
  if (telemetry_ && telemetry_->Running()) {
    const auto interval_ns = static_cast<std::int64_t>(
        1.0e9 / std::max(0.1, options_.telemetry.publish_hz));
    if (last_telemetry_time_ns_ == 0 ||
        frame.control_start_monotonic_ns <= 0 ||
        frame.control_start_monotonic_ns - last_telemetry_time_ns_ >=
            interval_ns) {
      telemetry_->TryPush(frame);
      last_telemetry_time_ns_ = frame.control_start_monotonic_ns;
    }
  }
}

void A3EvalRecorder::WriterMain_() noexcept {
  EvalFrameV1 frame;
  std::uint64_t since_flush = 0;
  for (;;) {
    bool progressed = false;
    while (frame_queue_ && frame_queue_->TryPop(frame)) {
      ProcessFrame_(frame, since_flush);
      progressed = true;
    }
    if (since_flush >= std::max<std::size_t>(1, options_.flush_frames)) {
      if (frames_stream_) frames_stream_.flush();
      if (events_stream_) events_stream_.flush();
      since_flush = 0;
    }
    if (!Running()) {
      bool more_frames = frame_queue_ && frame_queue_->TryPop(frame);
      if (more_frames) {
        ProcessFrame_(frame, since_flush);
        continue;
      }
      break;
    }
    if (!progressed) std::this_thread::sleep_for(std::chrono::milliseconds(2));
  }

  EvalTrackerOutput final_events;
  const auto final_frame_id = have_last_frame_ ? last_frame_id_ + 1 : 0;
  const auto final_time_ns = have_last_frame_ ? last_frame_time_ns_ : 0;
  episode_tracker_.Finalize(final_frame_id, final_time_ns, final_events);
  if (options_.complete_motion_only) {
    ProcessCompleteMotion_(nullptr, final_events, since_flush);
  }
  ProcessTrackerEvents_(final_events);
}

void A3EvalRecorder::WriteSessionMetadata_() {
  YAML::Node root;
  root["meta_schema_version"] = 1;
  root["data_schema"] = "a3_eval_raw_v8";
  root["session_id"] = options_.session_id;
  root["backend"] = options_.backend_name;
  root["state_clock_domain"] = options_.state_clock_domain;
  root["runtime_config_path"] = options_.runtime_config_path;
  root["policy_model_path"] = options_.policy_model_path;
  root["policy_hz"] = options_.policy_hz;
  root["state_hz"] = options_.state_hz;
  root["complete_motion_only"] = true;
  root["interrupted_episode_policy"] = "discard";
  root["episode_output"] = "one raw MCAP plus one meta YAML";

  const auto& names31 = robot_io::MakeA3Layout31().names;
  YAML::Node order31(YAML::NodeType::Sequence);
  YAML::Node order29(YAML::NodeType::Sequence);
  YAML::Node policy_to_sdk(YAML::NodeType::Sequence);
  for (const auto& name : names31) order31.push_back(name);
  for (const int sdk_index : robot_io::kA3PolicyToSdkIdx) {
    order29.push_back(names31.at(static_cast<std::size_t>(sdk_index)));
    policy_to_sdk.push_back(sdk_index);
  }
  order31.SetStyle(YAML::EmitterStyle::Flow);
  order29.SetStyle(YAML::EmitterStyle::Flow);
  policy_to_sdk.SetStyle(YAML::EmitterStyle::Flow);
  root["joint_layout"]["order_31"] = order31;
  root["joint_layout"]["order_29"] = order29;
  root["joint_layout"]["policy_to_sdk_index_31"] = policy_to_sdk;

  YAML::Node motions(YAML::NodeType::Sequence);
  for (const auto& motion : options_.motions) {
    YAML::Node item;
    item["motion_id"] = motion.motion_id;
    item["name"] = motion.name;
    item["source_path"] = motion.source_path;
    item["num_ticks"] = motion.num_ticks;
    motions.push_back(item);
  }
  root["motions"] = motions;
  root["notes"]["raw_only"] =
      "No frames.bin, events.jsonl, data_read, or realtime-derived torque";
  root["notes"]["gain_storage"] =
      "Each MCAP command frame stores kp/kd; episode meta YAML also stores "
      "profiles selected by gain_profile_id";

  YAML::Emitter emitter;
  emitter.SetIndent(2);
  emitter << root;
  if (!emitter.good()) {
    throw std::runtime_error("生成 session.meta.yaml 失败");
  }

  const auto data_dir = session_dir_ / options_.episode_mcap.raw_directory;
  const auto final_path = data_dir / "session.meta.yaml";
  const auto partial_path = data_dir / "session.meta.yaml.partial";
  std::ofstream out(partial_path, std::ios::trunc);
  if (!out) throw std::runtime_error("无法创建 session.meta.yaml.partial");
  out << emitter.c_str() << '\n';
  out.flush();
  if (!out) throw std::runtime_error("写入 session.meta.yaml 失败");
  out.close();
  std::error_code ec;
  std::filesystem::rename(partial_path, final_path, ec);
  if (ec) {
    const std::string rename_error = ec.message();
    std::filesystem::remove(partial_path, ec);
    throw std::runtime_error("提交 session.meta.yaml 失败: " + rename_error);
  }
}

void A3EvalRecorder::WriteManifest_() {
  std::ofstream out(session_dir_ / "manifest.json", std::ios::trunc);
  if (!out) throw std::runtime_error("无法创建 manifest.json");
  out << "{\n"
      << "  \"schema_version\": " << kA3EvalSchemaVersion << ",\n"
      << "  \"session_id\": \"" << JsonEscape(options_.session_id) << "\",\n"
      << "  \"backend\": \"" << JsonEscape(options_.backend_name) << "\",\n"
      << "  \"runtime_config_path\": \""
      << JsonEscape(options_.runtime_config_path) << "\",\n"
      << "  \"policy_model_path\": \""
      << JsonEscape(options_.policy_model_path) << "\",\n"
      << "  \"policy_hz\": " << options_.policy_hz << ",\n"
      << "  \"state_hz\": " << options_.state_hz << ",\n"
      << "  \"complete_motion_only\": "
      << (options_.complete_motion_only ? "true" : "false") << ",\n"
      << "  \"state_timestamp_clock_domain\": \""
      << JsonEscape(options_.state_clock_domain) << "\",\n"
      << "  \"state_data_ready_clock_domain\": \""
      << JsonEscape(options_.state_clock_domain) << "\",\n"
      << "  \"state_sync_ready_clock_domain\": \""
      << JsonEscape(options_.state_clock_domain) << "\",\n"
      << "  \"control_timestamp_clock_domain\": \"monotonic\",\n"
      << "  \"realtime_visualization\": {\"enabled\": "
      << (options_.telemetry.enabled ? "true" : "false")
      << ", \"publish_hz\": " << options_.telemetry.publish_hz
      << ", \"topic_prefix\": \""
      << JsonEscape(options_.telemetry.topic_prefix) << "\"},\n"
      << "  \"episode_mcap\": {\"enabled\": "
      << (options_.episode_mcap.enabled ? "true" : "false")
      << ", \"directory\": \""
      << JsonEscape(options_.episode_mcap.raw_directory)
      << "\", \"format\": \"raw_arrays\", \"environment\": \""
      << JsonEscape(options_.episode_mcap.environment)
      << "\", \"model_tag\": \""
      << JsonEscape(options_.episode_mcap.model_tag)
      << "\", \"topic_prefix\": \""
      << JsonEscape(options_.episode_mcap.topic_prefix)
      << "\", \"one_complete_episode_per_file\": true, "
      << "\"readable_copy\": {\"enabled\": "
      << (options_.episode_mcap.readable_copy_enabled ? "true" : "false")
      << ", \"directory\": \""
      << JsonEscape(options_.episode_mcap.readable_directory)
      << "\", \"format\": \"named_joint_compare\"}},\n"
      << "  \"frame_record_size\": " << sizeof(EvalFrameV1) << ",\n"
      << "  \"joint_order_31\": [";
  const auto& joint_names_31 = robot_io::MakeA3Layout31().names;
  for (std::size_t i = 0; i < joint_names_31.size(); ++i) {
    if (i != 0) out << ", ";
    out << "\"" << JsonEscape(joint_names_31[i]) << "\"";
  }
  out << "],\n"
      << "  \"joint_order_29\": [";
  for (std::size_t i = 0; i < robot_io::kA3PolicyToSdkIdx.size(); ++i) {
    if (i != 0) out << ", ";
    out << "\""
        << JsonEscape(joint_names_31.at(
               static_cast<std::size_t>(robot_io::kA3PolicyToSdkIdx[i])))
        << "\"";
  }
  out << "],\n"
      << "  \"policy_to_sdk_index_31\": [";
  for (std::size_t i = 0; i < robot_io::kA3PolicyToSdkIdx.size(); ++i) {
    if (i != 0) out << ", ";
    out << robot_io::kA3PolicyToSdkIdx[i];
  }
  out << "],\n"
      << "  \"motions\": [\n";
  for (std::size_t i = 0; i < options_.motions.size(); ++i) {
    const auto& motion = options_.motions[i];
    out << "    {\"motion_id\": " << motion.motion_id
        << ", \"name\": \"" << JsonEscape(motion.name)
        << "\", \"source_path\": \"" << JsonEscape(motion.source_path)
        << "\", \"num_ticks\": " << motion.num_ticks << "}";
    out << (i + 1 == options_.motions.size() ? "\n" : ",\n");
  }
  out << "  ],\n"
      << "  \"notes\": {\n"
      << "    \"pause_semantics\": \"pause ends the current episode; next play restarts at reference_tick 0\",\n"
      << "    \"joint_order_31\": \"A3 SDK/MuJoCo layout; see robot_io::MakeA3Layout31\",\n"
      << "    \"joint_order_29\": \"A3 MuJoCo policy view; neck excluded\",\n"
      << "    \"command_raw\": \"q_des/dq_des/tau_ff are recorded; kp/kd are carried for writer-side metadata\",\n"
      << "    \"tau_cmd_expected\": \"offline only: tau_ff + kp*(q_des-q) + kd*(dq_des-dq)\"\n"
      << "  }\n"
      << "}\n";
}

void A3EvalRecorder::WriteEvent_(const EvalEventV1& event) {
  events_stream_
      << "{\"event_id\":" << event.event_id
      << ",\"frame_id\":" << event.frame_id
      << ",\"time_monotonic_ns\":" << event.event_time_monotonic_ns
      << ",\"event_type\":\"" << EvalEventTypeName(event.event_type) << "\""
      << ",\"episode_id\":" << event.episode_id
      << ",\"reference_tick\":" << event.reference_tick
      << ",\"from_motion_id\":" << event.from_motion_id
      << ",\"to_motion_id\":" << event.to_motion_id
      << ",\"from_mode\":\"" << DeployModeName(event.from_mode) << "\""
      << ",\"to_mode\":\"" << DeployModeName(event.to_mode) << "\""
      << ",\"from_phase\":\"" << MotionPhaseName(event.from_phase) << "\""
      << ",\"to_phase\":\"" << MotionPhaseName(event.to_phase) << "\""
      << ",\"outcome\":\"" << EpisodeOutcomeName(event.outcome) << "\""
      << ",\"reason\":\"" << TerminationReasonName(event.reason) << "\""
      << ",\"dropped_frame_count\":" << event.dropped_frame_count;
  if (event.event_type == EvalEventType::kEpisodeEnded) {
    const auto mcap = episode_mcap_files_.find(event.episode_id);
    if (mcap != episode_mcap_files_.end()) {
      events_stream_ << ",\"mcap_file\":\"" << JsonEscape(mcap->second)
                     << "\"";
    } else {
      events_stream_ << ",\"mcap_file\":null";
    }
    const auto readable = episode_readable_mcap_files_.find(event.episode_id);
    if (readable != episode_readable_mcap_files_.end()) {
      events_stream_ << ",\"mcap_read_file\":\""
                     << JsonEscape(readable->second) << "\"";
    } else {
      events_stream_ << ",\"mcap_read_file\":null";
    }
  }
  events_stream_ << "}\n";
}

void A3EvalRecorder::WriteSummary_() noexcept {
  if (session_dir_.empty()) return;
  try {
    const auto stats = Statistics();
    std::ofstream out(session_dir_ / "summary.json", std::ios::trunc);
    if (!out) return;
    out << "{\n"
        << "  \"frames_accepted\": " << stats.frames_accepted << ",\n"
        << "  \"frames_written\": " << stats.frames_written << ",\n"
        << "  \"frames_dropped\": " << stats.frames_dropped << ",\n"
        << "  \"frames_ignored\": " << stats.frames_ignored << ",\n"
        << "  \"frames_discarded\": " << stats.frames_discarded << ",\n"
        << "  \"episodes_started\": " << stats.episodes_started << ",\n"
        << "  \"episodes_committed\": " << stats.episodes_committed << ",\n"
        << "  \"episodes_discarded\": " << stats.episodes_discarded << ",\n"
        << "  \"events_accepted\": " << stats.events_accepted << ",\n"
        << "  \"events_written\": " << stats.events_written << ",\n"
        << "  \"events_dropped\": " << stats.events_dropped << ",\n"
        << "  \"telemetry_accepted\": " << stats.telemetry_accepted << ",\n"
        << "  \"telemetry_published\": " << stats.telemetry_published << ",\n"
        << "  \"telemetry_dropped\": " << stats.telemetry_dropped << ",\n"
        << "  \"mcap_files_written\": " << stats.mcap_files_written << ",\n"
        << "  \"mcap_write_failures\": " << stats.mcap_write_failures
        << ",\n"
        << "  \"mcap_bytes_written\": " << stats.mcap_bytes_written << ",\n"
        << "  \"readable_mcap_files_written\": "
        << stats.readable_mcap_files_written << ",\n"
        << "  \"readable_mcap_bytes_written\": "
        << stats.readable_mcap_bytes_written << "\n"
        << "}\n";
  } catch (...) {
    // 析构/停止路径不能因 summary 写入失败而影响 deploy 正常退出。
  }
}

}  // namespace a3_deploy
