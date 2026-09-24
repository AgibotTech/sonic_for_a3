// Copyright (c) 2026, AgiBot Inc. All rights reserved.
//
// Evaluation 数据链路共用的单生产者/单消费者无锁队列。
#pragma once

#include <atomic>
#include <cstddef>
#include <vector>

namespace a3_deploy {

template <typename T>
class A3EvalSpscQueue {
 public:
  explicit A3EvalSpscQueue(std::size_t capacity) : slots_(capacity + 1) {}

  // producer 侧只尝试一次；队列满时立即返回，不阻塞 policy thread。
  bool TryPush(const T& value) noexcept {
    const auto head = head_.load(std::memory_order_relaxed);
    const auto next = Next_(head);
    if (next == tail_.load(std::memory_order_acquire)) return false;
    slots_[head] = value;
    head_.store(next, std::memory_order_release);
    return true;
  }

  bool TryPop(T& value) noexcept {
    const auto tail = tail_.load(std::memory_order_relaxed);
    if (tail == head_.load(std::memory_order_acquire)) return false;
    value = slots_[tail];
    tail_.store(Next_(tail), std::memory_order_release);
    return true;
  }

 private:
  std::size_t Next_(std::size_t value) const noexcept {
    ++value;
    return value == slots_.size() ? 0 : value;
  }

  std::vector<T> slots_;
  alignas(64) std::atomic<std::size_t> head_{0};
  alignas(64) std::atomic<std::size_t> tail_{0};
};

}  // namespace a3_deploy
