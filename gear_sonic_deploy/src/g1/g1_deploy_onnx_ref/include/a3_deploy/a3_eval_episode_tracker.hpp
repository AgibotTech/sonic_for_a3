// Copyright (c) 2026, AgiBot Inc. All rights reserved.
//
// 根据连续 EvalFrame 推导 episode 边界和稀疏事件。
#pragma once

#include "a3_deploy/a3_eval_types.hpp"

#include <array>
#include <cstddef>
#include <cstdint>

namespace a3_deploy {

struct EvalTrackerOutput {
  std::array<EvalEventV1, 4> events{};
  std::size_t count = 0;

  void Push(const EvalEventV1& event) noexcept {
    if (count < events.size()) events[count++] = event;
  }
};

class A3EvalEpisodeTracker {
 public:
  explicit A3EvalEpisodeTracker(bool require_quality_for_completion = true)
      : require_quality_for_completion_(require_quality_for_completion) {}

  // 更新一帧并写入 frame.episode_id。reference_num_ticks 是当前 active
  // motion 的总帧数；非 MOTION 或未知时传 0。
  void Update(EvalFrameV1& frame,
              std::uint64_t reference_num_ticks,
              EvalTrackerOutput& output) noexcept;

  // 进程退出时补齐仍在进行的 episode，避免留下无结束事件的数据段。
  void Finalize(std::uint64_t frame_id,
                std::int64_t monotonic_ns,
                EvalTrackerOutput& output) noexcept;

  std::uint64_t CurrentEpisodeId() const noexcept {
    return episode_active_ ? current_episode_id_ : 0;
  }

 private:
  EvalEventV1 MakeEvent_(EvalEventType type,
                         const EvalFrameV1& frame) noexcept;
  void StartEpisode_(EvalFrameV1& frame,
                     std::uint64_t reference_num_ticks,
                     EvalTrackerOutput& output) noexcept;
  void EndEpisode_(const EvalFrameV1& frame,
                   EpisodeOutcome outcome,
                   TerminationReason reason,
                   EvalTrackerOutput& output) noexcept;
  void AccumulateQuality_(const EvalFrameV1& frame) noexcept;

  bool has_previous_{false};
  EvalFrameV1 previous_{};
  std::uint64_t next_event_id_{1};
  std::uint64_t next_episode_id_{1};
  bool episode_active_{false};
  std::uint64_t current_episode_id_{0};
  std::int32_t episode_motion_id_{kA3EvalNoMotion};
  std::uint64_t episode_reference_num_ticks_{0};
  std::uint64_t episode_last_reference_tick_{kA3EvalNoTick};
  bool episode_quality_ok_{true};
  TerminationReason episode_failure_reason_{TerminationReason::kNone};
  // 连续录制模式保留历史行为；complete-motion 模式由 recorder 只按
  // reference 播放完整性提交，质量标记不应筛掉低质量但完整的数据。
  bool require_quality_for_completion_{true};
};

}  // namespace a3_deploy
