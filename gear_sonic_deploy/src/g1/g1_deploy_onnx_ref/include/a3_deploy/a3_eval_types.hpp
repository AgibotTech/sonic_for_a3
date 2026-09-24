// Copyright (c) 2026, AgiBot Inc. All rights reserved.
//
// SONIC A3 Evaluation Recorder Demo 1 的稳定数据格式。
// 这里仅定义固定大小、可复制的数据对象，不执行文件 I/O。
#pragma once

#include "a3_deploy/a3_manual_control.hpp"
#include "robot_io/robot_io_backend.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <type_traits>

namespace a3_deploy {

// V7：reference 增加运行时重采样并应用 policy yaw 对齐后的 pelvis
// position/orientation，供 root tracking 和 MPJPE 分析直接使用。
// V6：采集层只保存原始/直接拷贝的数据。恢复最终 RobotCommand 的
// q_des/dq_des/tau_ff/kp/kd；writer thread 将五组 command 数组写入 raw
// MCAP，并额外按 gain_profile_id 去重生成便于阅读的 gain metadata。
// 不在控制线程计算或保存 tau_cmd_expected。
// 二进制布局改变时
// 必须提升版本，避免检查器把旧 session 按新结构误读。
inline constexpr std::uint32_t kA3EvalSchemaVersion = 7;
inline constexpr std::uint64_t kA3EvalNoTick =
    std::numeric_limits<std::uint64_t>::max();
inline constexpr std::int32_t kA3EvalNoMotion = -1;

// MOTION 内部阶段。PAUSED 在 Demo 1 中不支持续播；再次播放会从 tick 0
// 新建 episode。TERMINAL_HOLD 表示动作自然结束后保持末端状态。
enum class MotionPhase : std::uint8_t {
  kNone = 0,
  kReady = 1,
  kPlaying = 2,
  kPaused = 3,
  kTerminalHold = 4,
};

// 当前最终命令的真实来源，用于区分 deploy mode 与安全接管路径。
enum class CommandSource : std::uint8_t {
  kNone = 0,
  kPassive = 1,
  kPdStand = 2,
  kPolicy = 3,
  kTeleop = 4,
  kSafeHalt = 5,
};

enum class EvalEventType : std::uint8_t {
  kModeChanged = 0,
  kMotionSelected = 1,
  kEpisodeStarted = 2,
  kEpisodeEnded = 3,
  kRecorderDropped = 4,
};

enum class EpisodeOutcome : std::uint8_t {
  kNone = 0,
  kInProgress = 1,
  kCompleted = 2,
  kAborted = 3,
  kFailed = 4,
};

enum class TerminationReason : std::uint8_t {
  kNone = 0,
  kEndOfReference = 1,
  kUserPause = 2,
  kUserStop = 3,
  kMotionSwitch = 4,
  kModeSwitch = 5,
  kWatchdog = 6,
  kFall = 7,
  kInferenceFailure = 8,
  kCommandFailure = 9,
  kSyncFailure = 10,
  kProcessExit = 11,
  kUnknown = 12,
};

enum EvalValidBits : std::uint32_t {
  kRobotStateValid = 1u << 0,
  kPelvisImuValid = 1u << 1,
  kRawActionValid = 1u << 2,
  // bit 3 是旧 schema 的 q_des_policy 标记，V5 起保留不用。
  kRobotCommandValid = 1u << 4,
  kReferenceValid = 1u << 5,
  kInferTimingValid = 1u << 6,
  kSendTimingValid = 1u << 7,
  kTorsoImuValid = 1u << 8,
  // bit 9 是旧 schema 的 expected torque 标记，V6 起保留不用。
};

enum EvalQualityBits : std::uint32_t {
  kSyncComplete = 1u << 0,
  kSyncAligned = 1u << 1,
  kInferOk = 1u << 2,
  kCommandSent = 1u << 3,
  kWatchdogSafeHalt = 1u << 4,
  kRecorderGap = 1u << 5,
};

// 50 Hz 控制时间轴上的一帧。所有连续信号使用 float32；其精度显著高于
// 当前真机传感器噪声，同时可将长时间录制体积压低约一半。
struct EvalFrameV1 {
  std::uint64_t frame_id = 0;
  std::uint64_t state_tick = kA3EvalNoTick;
  std::uint64_t policy_tick = kA3EvalNoTick;
  std::uint64_t reference_tick = kA3EvalNoTick;
  std::uint64_t episode_id = 0;

  std::int32_t active_motion_id = kA3EvalNoMotion;
  std::int32_t selected_motion_id = kA3EvalNoMotion;

  std::int64_t control_start_monotonic_ns = 0;
  std::int64_t state_timestamp_ns = 0;
  std::int64_t state_data_ready_ns = 0;
  std::int64_t state_sync_ready_ns = 0;
  std::int64_t infer_start_monotonic_ns = 0;
  std::int64_t infer_end_monotonic_ns = 0;
  std::int64_t send_start_monotonic_ns = 0;
  std::int64_t send_end_monotonic_ns = 0;
  std::int64_t control_end_monotonic_ns = 0;

  DeployMode deploy_mode = DeployMode::kIdle;
  MotionPhase motion_phase = MotionPhase::kNone;
  CommandSource command_source = CommandSource::kNone;
  std::uint8_t gain_profile_id = 0;
  std::uint32_t valid_mask = 0;
  std::uint32_t quality_flags = 0;

  std::array<float, 31> q{};
  std::array<float, 31> dq{};
  std::array<float, 31> tau_est{};

  std::array<float, 4> pelvis_quat_wxyz{};
  std::array<float, 3> pelvis_gyro{};
  std::array<float, 4> torso_quat_wxyz{};
  std::array<float, 3> torso_gyro{};

  std::array<float, 29> raw_action{};

  // 最终 RobotCommand 的原始快照。五组数组均进入 raw MCAP；writer thread
  // 还会按 gain_profile_id 去重，将 kp/kd 副本写入 episode metadata。
  std::array<float, 31> command_q_des{};
  std::array<float, 31> command_dq_des{};
  std::array<float, 31> command_tau_ff{};
  std::array<float, 31> command_kp{};
  std::array<float, 31> command_kd{};

  std::array<float, 29> q_ref{};
  std::array<float, 29> dq_ref{};
  std::array<float, 3> reference_pelvis_position_m{};
  std::array<float, 4> reference_pelvis_quat_wxyz{
      1.0f, 0.0f, 0.0f, 0.0f};
};

// 稀疏事件记录。event_type 决定哪些字段有意义；未使用字段保持默认值。
struct EvalEventV1 {
  std::uint64_t event_id = 0;
  std::uint64_t frame_id = 0;
  std::int64_t event_time_monotonic_ns = 0;
  EvalEventType event_type = EvalEventType::kModeChanged;

  std::uint64_t episode_id = 0;
  std::uint64_t reference_tick = kA3EvalNoTick;
  std::int32_t from_motion_id = kA3EvalNoMotion;
  std::int32_t to_motion_id = kA3EvalNoMotion;
  DeployMode from_mode = DeployMode::kIdle;
  DeployMode to_mode = DeployMode::kIdle;
  MotionPhase from_phase = MotionPhase::kNone;
  MotionPhase to_phase = MotionPhase::kNone;
  EpisodeOutcome outcome = EpisodeOutcome::kNone;
  TerminationReason reason = TerminationReason::kNone;
  std::uint64_t dropped_frame_count = 0;
};

// frames.bin 文件头。record_size 使读取端可以拒绝不兼容的结构布局。
struct EvalFramesFileHeaderV1 {
  std::array<char, 8> magic{{'A', '3', 'E', 'V', 'A', 'L', '1', '\0'}};
  std::uint32_t schema_version = kA3EvalSchemaVersion;
  std::uint32_t record_size = sizeof(EvalFrameV1);
  std::uint32_t endian_marker = 0x01020304u;
  std::uint32_t reserved = 0;
};

static_assert(std::is_trivially_copyable_v<EvalFrameV1>);
static_assert(std::is_trivially_copyable_v<EvalEventV1>);

const char* MotionPhaseName(MotionPhase value) noexcept;
const char* CommandSourceName(CommandSource value) noexcept;
const char* EvalEventTypeName(EvalEventType value) noexcept;
const char* EpisodeOutcomeName(EpisodeOutcome value) noexcept;
const char* TerminationReasonName(TerminationReason value) noexcept;

}  // namespace a3_deploy
