// Copyright (c) 2026, AgiBot Inc. All rights reserved.
//
// 完整 Motion Episode 的 MCAP 导出器。调用方必须位于 recorder writer
// thread；本类不创建线程，也不会进入 policy/control 实时路径。
#pragma once

#include "a3_deploy/a3_eval_types.hpp"

#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace a3_deploy {

struct A3EvalMcapOptions {
  bool enabled = false;
  // 文件名中的运行环境标识，推荐只使用 sim 或 real。
  std::string environment = "sim";
  // 为空时从 policy_model_path 自动提取，例如 model-step-010000。
  std::string model_tag;
  std::string topic_prefix = "/a3/eval";
  // data/ 保存固定顺序的原始数组，适合训练和程序读取。
  std::string raw_directory = "data";
  // data_read/ 保存完全相同的数值，但增加关节名层级，适合 Foxglove。
  bool readable_copy_enabled = false;
  std::string readable_directory = "data_read";
};

struct A3EvalMcapSessionMetadata {
  std::string session_id;
  std::string backend_name;
  std::string state_clock_domain;
  std::string runtime_config_path;
  std::string policy_model_path;
  double policy_hz = 50.0;
  double state_hz = 100.0;
  // 将 control monotonic 时间映射到 Unix epoch，供 Foxglove 正确显示时间轴。
  std::int64_t system_time_anchor_ns = 0;
  std::int64_t monotonic_time_anchor_ns = 0;
};

struct A3EvalMcapEpisodeMetadata {
  std::uint64_t episode_id = 0;
  std::int32_t motion_id = kA3EvalNoMotion;
  std::string motion_name;
  std::string motion_source_path;
  std::uint64_t motion_repeat_index = 0;
};

struct A3EvalMcapWriteResult {
  bool ok = false;
  // 原始数组 MCAP，相对 session 目录。
  std::filesystem::path relative_path;
  std::uint64_t file_size = 0;
  // 与原始 MCAP 同名的 sidecar，例如 clip.mcap -> clip.meta.yaml。
  std::filesystem::path metadata_relative_path;
  std::uint64_t metadata_file_size = 0;
  // 可读 MCAP 关闭时为空。
  std::filesystem::path readable_relative_path;
  std::uint64_t readable_file_size = 0;
  std::string error;
};

class A3EvalMcapWriter {
 public:
  explicit A3EvalMcapWriter(A3EvalMcapOptions options);

  static bool AvailableAtBuildTime() noexcept;

  // 将已验证完整的 episode 写为一个 MCAP。先生成 .mcap.partial，只有全部
  // 消息、metadata 和索引成功落盘后才原子重命名为正式 .mcap。
  A3EvalMcapWriteResult WriteEpisode(
      const std::filesystem::path& session_dir,
      const A3EvalMcapSessionMetadata& session,
      const A3EvalMcapEpisodeMetadata& episode,
      const std::vector<EvalFrameV1>& frames) noexcept;

 private:
  A3EvalMcapOptions options_;
};

}  // namespace a3_deploy
