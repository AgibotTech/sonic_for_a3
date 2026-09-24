// Copyright (c) 2026, AgiBot Inc. All rights reserved.
#include "a3_deploy/a3_eval_telemetry.hpp"

#include "robot_io/a3_layout_extra.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <thread>
#include <utility>
#include <vector>

#ifndef HAS_A3_EVAL_ROS2
#define HAS_A3_EVAL_ROS2 0
#endif

#if HAS_A3_EVAL_ROS2
#include <joint_msgs/msg/joint_state.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <std_msgs/msg/float64_multi_array.hpp>
#include <std_msgs/msg/int64_multi_array.hpp>
#endif

namespace a3_deploy {

#if HAS_A3_EVAL_ROS2
struct A3EvalTelemetryPublisher::Impl {
  std::shared_ptr<rclcpp::Context> context;
  std::shared_ptr<rclcpp::Node> node;
  rclcpp::Publisher<sensor_msgs::msg::JointState>::SharedPtr state_pub;
  rclcpp::Publisher<sensor_msgs::msg::JointState>::SharedPtr command_pub;
  rclcpp::Publisher<sensor_msgs::msg::JointState>::SharedPtr policy_pub;
  rclcpp::Publisher<sensor_msgs::msg::JointState>::SharedPtr reference_pub;
  // `sensor_msgs/JointState` 使用 name/position 平行数组，Foxglove 图表
  // 无法用关节名直接过滤。额外发布 joints[] 嵌套结构，保留旧 topic 兼容性。
  rclcpp::Publisher<joint_msgs::msg::JointState>::SharedPtr named_state_pub;
  rclcpp::Publisher<joint_msgs::msg::JointState>::SharedPtr named_command_pub;
  rclcpp::Publisher<joint_msgs::msg::JointState>::SharedPtr named_policy_pub;
  rclcpp::Publisher<joint_msgs::msg::JointState>::SharedPtr named_reference_pub;
  // aligned topics 统一使用完整31维 A3 layout。Policy/Reference 的29维值
  // 映射回对应 SDK slot，两个 neck slot 用 NaN 表示“不适用”。Policy 仅在
  // effort 中发布 raw_action，不再发布 q_des_policy。
  rclcpp::Publisher<sensor_msgs::msg::JointState>::SharedPtr aligned_state_pub;
  rclcpp::Publisher<sensor_msgs::msg::JointState>::SharedPtr aligned_command_pub;
  rclcpp::Publisher<sensor_msgs::msg::JointState>::SharedPtr aligned_policy_pub;
  rclcpp::Publisher<sensor_msgs::msg::JointState>::SharedPtr aligned_reference_pub;
  rclcpp::Publisher<sensor_msgs::msg::Imu>::SharedPtr imu_pub;
  rclcpp::Publisher<sensor_msgs::msg::Imu>::SharedPtr torso_imu_pub;
  rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr timing_pub;
  rclcpp::Publisher<std_msgs::msg::Int64MultiArray>::SharedPtr status_pub;
  std::vector<std::string> joint_names_31;
  std::vector<std::string> joint_names_29;
};
#else
struct A3EvalTelemetryPublisher::Impl {};
#endif

namespace {

std::string NormalizedPrefix(std::string prefix) {
  if (prefix.empty()) return "/a3/eval";
  if (prefix.front() != '/') prefix.insert(prefix.begin(), '/');
  while (prefix.size() > 1 && prefix.back() == '/') prefix.pop_back();
  return prefix;
}

#if HAS_A3_EVAL_ROS2
template <typename Array>
std::vector<double> ToDoubleVector(const Array& values) {
  return {values.begin(), values.end()};
}

template <typename Array>
std::vector<double> ToAlignedA3Vector(const Array& policy_values) {
  std::vector<double> aligned(
      robot_io::kA3Dof, std::numeric_limits<double>::quiet_NaN());
  const std::size_t count =
      std::min(policy_values.size(), robot_io::kA3PolicyToSdkIdx.size());
  for (std::size_t i = 0; i < count; ++i) {
    aligned[robot_io::kA3PolicyToSdkIdx[i]] = policy_values[i];
  }
  return aligned;
}

std::int64_t TickForRos(std::uint64_t tick) {
  return tick == kA3EvalNoTick ? -1 : static_cast<std::int64_t>(tick);
}

double DurationMs(std::int64_t start_ns, std::int64_t end_ns) {
  if (start_ns <= 0 || end_ns < start_ns) return 0.0;
  return static_cast<double>(end_ns - start_ns) / 1.0e6;
}

joint_msgs::msg::JointState MakeNamedJointState(
    const rclcpp::Time& stamp, const std::vector<std::string>& names,
    const std::vector<double>& positions,
    const std::vector<double>& velocities = {},
    const std::vector<double>& efforts = {}) {
  joint_msgs::msg::JointState message;
  message.header.stamp = stamp;
  message.joints.reserve(names.size());
  for (std::size_t i = 0; i < names.size(); ++i) {
    auto& joint = message.joints.emplace_back();
    joint.name = names[i];
    // sequence 在 telemetry 中固定表示稳定的 joint layout 索引。
    joint.sequence = static_cast<std::uint32_t>(i);
    if (i < positions.size()) joint.position = positions[i];
    if (i < velocities.size()) joint.velocity = velocities[i];
    if (i < efforts.size()) joint.effort = efforts[i];
  }
  return message;
}
#endif

}  // namespace

A3EvalTelemetryPublisher::A3EvalTelemetryPublisher(
    A3EvalTelemetryOptions options)
    : options_(std::move(options)) {
  options_.topic_prefix = NormalizedPrefix(std::move(options_.topic_prefix));
}

A3EvalTelemetryPublisher::~A3EvalTelemetryPublisher() { Stop(); }

bool A3EvalTelemetryPublisher::Ros2AvailableAtBuildTime() noexcept {
  return HAS_A3_EVAL_ROS2 != 0;
}

bool A3EvalTelemetryPublisher::Start(std::string* error) {
  if (!options_.enabled) return true;
  if (Running()) {
    if (error) *error = "evaluation telemetry 已经运行";
    return false;
  }
  if (!std::isfinite(options_.publish_hz) || options_.publish_hz <= 0.0) {
    if (error) *error = "realtime_visualization.publish_hz 必须大于 0";
    return false;
  }
#if !HAS_A3_EVAL_ROS2
  if (error) *error = "当前二进制未编译 ROS2 telemetry 支持";
  return false;
#else
  try {
    queue_ = std::make_unique<A3EvalSpscQueue<EvalFrameV1>>(
        std::max<std::size_t>(2, options_.queue_capacity));
    impl_ = std::make_unique<Impl>();

    // 使用私有 ROS2 context，避免与 deploy 中可能存在的 teleop ROS2
    // handler 共享全局 init/shutdown 生命周期。
    impl_->context = std::make_shared<rclcpp::Context>();
    impl_->context->init(0, nullptr);
    rclcpp::NodeOptions node_options;
    node_options.context(impl_->context);
    impl_->node = std::make_shared<rclcpp::Node>(options_.node_name,
                                                 node_options);
    const auto qos = rclcpp::SensorDataQoS().keep_last(5);
    const auto& prefix = options_.topic_prefix;
    impl_->state_pub = impl_->node->create_publisher<sensor_msgs::msg::JointState>(
        prefix + "/state", qos);
    impl_->command_pub =
        impl_->node->create_publisher<sensor_msgs::msg::JointState>(
            prefix + "/command", qos);
    impl_->policy_pub = impl_->node->create_publisher<sensor_msgs::msg::JointState>(
        prefix + "/policy", qos);
    impl_->reference_pub =
        impl_->node->create_publisher<sensor_msgs::msg::JointState>(
            prefix + "/reference", qos);
    impl_->named_state_pub =
        impl_->node->create_publisher<joint_msgs::msg::JointState>(
            prefix + "/state_named", qos);
    impl_->named_command_pub =
        impl_->node->create_publisher<joint_msgs::msg::JointState>(
            prefix + "/command_named", qos);
    impl_->named_policy_pub =
        impl_->node->create_publisher<joint_msgs::msg::JointState>(
            prefix + "/policy_named", qos);
    impl_->named_reference_pub =
        impl_->node->create_publisher<joint_msgs::msg::JointState>(
            prefix + "/reference_named", qos);
    impl_->aligned_state_pub =
        impl_->node->create_publisher<sensor_msgs::msg::JointState>(
            prefix + "/state_aligned", qos);
    impl_->aligned_command_pub =
        impl_->node->create_publisher<sensor_msgs::msg::JointState>(
            prefix + "/command_aligned", qos);
    impl_->aligned_policy_pub =
        impl_->node->create_publisher<sensor_msgs::msg::JointState>(
            prefix + "/policy_aligned", qos);
    impl_->aligned_reference_pub =
        impl_->node->create_publisher<sensor_msgs::msg::JointState>(
            prefix + "/reference_aligned", qos);
    impl_->imu_pub = impl_->node->create_publisher<sensor_msgs::msg::Imu>(
        prefix + "/pelvis_imu", qos);
    impl_->torso_imu_pub =
        impl_->node->create_publisher<sensor_msgs::msg::Imu>(
            prefix + "/torso_imu", qos);
    impl_->timing_pub =
        impl_->node->create_publisher<std_msgs::msg::Float64MultiArray>(
            prefix + "/timing_ms", qos);
    impl_->status_pub =
        impl_->node->create_publisher<std_msgs::msg::Int64MultiArray>(
            prefix + "/status", qos);

    impl_->joint_names_31 = robot_io::MakeA3Layout31().names;
    impl_->joint_names_29.reserve(robot_io::kA3PolicyDof);
    for (const int sdk_index : robot_io::kA3PolicyToSdkIdx) {
      impl_->joint_names_29.push_back(impl_->joint_names_31.at(sdk_index));
    }
  } catch (const std::exception& e) {
    if (error) *error = std::string("初始化 ROS2 telemetry 失败: ") + e.what();
    impl_.reset();
    queue_.reset();
    return false;
  }

  running_.store(true, std::memory_order_release);
  try {
    publisher_thread_ =
        std::thread(&A3EvalTelemetryPublisher::PublisherMain_, this);
  } catch (const std::exception& e) {
    running_.store(false, std::memory_order_release);
    if (error) *error = std::string("启动 telemetry thread 失败: ") + e.what();
    impl_.reset();
    queue_.reset();
    return false;
  }
  return true;
#endif
}

void A3EvalTelemetryPublisher::Stop() noexcept {
  if (!running_.exchange(false, std::memory_order_acq_rel)) return;
#if HAS_A3_EVAL_ROS2
  if (publisher_thread_.joinable()) publisher_thread_.join();
#endif
}

bool A3EvalTelemetryPublisher::TryPush(const EvalFrameV1& frame) noexcept {
  if (!Running() || !queue_ || !queue_->TryPush(frame)) {
    dropped_.fetch_add(1, std::memory_order_relaxed);
    return false;
  }
  accepted_.fetch_add(1, std::memory_order_relaxed);
  return true;
}

A3EvalTelemetryStatistics A3EvalTelemetryPublisher::Statistics() const noexcept {
  return {accepted_.load(std::memory_order_relaxed),
          published_.load(std::memory_order_relaxed),
          dropped_.load(std::memory_order_relaxed)};
}

void A3EvalTelemetryPublisher::PublisherMain_() noexcept {
#if HAS_A3_EVAL_ROS2
  auto publish_frame = [this](const EvalFrameV1& frame) {
    const auto stamp = impl_->node->now();

    sensor_msgs::msg::JointState state;
    state.header.stamp = stamp;
    state.name = impl_->joint_names_31;
    state.position = ToDoubleVector(frame.q);
    state.velocity = ToDoubleVector(frame.dq);
    state.effort = ToDoubleVector(frame.tau_est);
    impl_->state_pub->publish(state);
    impl_->named_state_pub->publish(MakeNamedJointState(
        stamp, impl_->joint_names_31, state.position, state.velocity,
        state.effort));
    impl_->aligned_state_pub->publish(state);

    sensor_msgs::msg::JointState command;
    command.header.stamp = stamp;
    command.name = impl_->joint_names_31;
    // 这里只发布最终 RobotCommand 的三组原始数组；kp/kd 属于 gain profile
    // metadata，理论 PD 力矩由离线分析工具重建。
    command.position = ToDoubleVector(frame.command_q_des);
    command.velocity = ToDoubleVector(frame.command_dq_des);
    command.effort = ToDoubleVector(frame.command_tau_ff);
    impl_->command_pub->publish(command);
    impl_->named_command_pub->publish(MakeNamedJointState(
        stamp, impl_->joint_names_31, command.position, {}, command.effort));
    impl_->aligned_command_pub->publish(command);

    sensor_msgs::msg::JointState policy;
    policy.header.stamp = stamp;
    policy.name = impl_->joint_names_29;
    // raw_action 不是力矩；这里只借用 effort 数组提供可按关节索引绘图的通道，
    // position 保持空数组，不再发布 policy q_des。
    policy.effort = ToDoubleVector(frame.raw_action);
    impl_->policy_pub->publish(policy);
    // 具名 topic 同样只在 effort 中提供 raw_action。
    impl_->named_policy_pub->publish(MakeNamedJointState(
        stamp, impl_->joint_names_29, policy.position, {}, policy.effort));

    sensor_msgs::msg::JointState aligned_policy;
    aligned_policy.header.stamp = stamp;
    aligned_policy.name = impl_->joint_names_31;
    aligned_policy.effort = ToAlignedA3Vector(frame.raw_action);
    impl_->aligned_policy_pub->publish(aligned_policy);

    sensor_msgs::msg::JointState reference;
    reference.header.stamp = stamp;
    reference.name = impl_->joint_names_29;
    reference.position = ToDoubleVector(frame.q_ref);
    reference.velocity = ToDoubleVector(frame.dq_ref);
    impl_->reference_pub->publish(reference);
    impl_->named_reference_pub->publish(MakeNamedJointState(
        stamp, impl_->joint_names_29, reference.position, reference.velocity));

    sensor_msgs::msg::JointState aligned_reference;
    aligned_reference.header.stamp = stamp;
    aligned_reference.name = impl_->joint_names_31;
    aligned_reference.position = ToAlignedA3Vector(frame.q_ref);
    aligned_reference.velocity = ToAlignedA3Vector(frame.dq_ref);
    impl_->aligned_reference_pub->publish(aligned_reference);

    sensor_msgs::msg::Imu imu;
    imu.header.stamp = stamp;
    imu.header.frame_id = "pelvis_imu";
    imu.orientation.w = frame.pelvis_quat_wxyz[0];
    imu.orientation.x = frame.pelvis_quat_wxyz[1];
    imu.orientation.y = frame.pelvis_quat_wxyz[2];
    imu.orientation.z = frame.pelvis_quat_wxyz[3];
    imu.angular_velocity.x = frame.pelvis_gyro[0];
    imu.angular_velocity.y = frame.pelvis_gyro[1];
    imu.angular_velocity.z = frame.pelvis_gyro[2];
    // 本 evaluation schema 不记录线加速度；-1 表示该字段不可用。
    imu.linear_acceleration_covariance[0] = -1.0;
    impl_->imu_pub->publish(imu);

    // Secondary IMU 尚未 ready 时不发布零值，消费者可据此区分“无数据”与
    // 真实的静止测量。
    if ((frame.valid_mask & kTorsoImuValid) != 0) {
      sensor_msgs::msg::Imu torso_imu;
      torso_imu.header.stamp = stamp;
      torso_imu.header.frame_id = "torso_imu";
      torso_imu.orientation.w = frame.torso_quat_wxyz[0];
      torso_imu.orientation.x = frame.torso_quat_wxyz[1];
      torso_imu.orientation.y = frame.torso_quat_wxyz[2];
      torso_imu.orientation.z = frame.torso_quat_wxyz[3];
      torso_imu.angular_velocity.x = frame.torso_gyro[0];
      torso_imu.angular_velocity.y = frame.torso_gyro[1];
      torso_imu.angular_velocity.z = frame.torso_gyro[2];
      torso_imu.linear_acceleration_covariance[0] = -1.0;
      impl_->torso_imu_pub->publish(torso_imu);
    }

    std_msgs::msg::Float64MultiArray timing;
    timing.layout.dim.resize(1);
    timing.layout.dim[0].label =
        "state_assemble,infer,send,control,telemetry_age_ms";
    timing.layout.dim[0].size = 5;
    timing.layout.dim[0].stride = 5;
    const auto now_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                            std::chrono::steady_clock::now().time_since_epoch())
                            .count();
    timing.data = {
        DurationMs(frame.state_data_ready_ns, frame.state_sync_ready_ns),
        DurationMs(frame.infer_start_monotonic_ns,
                   frame.infer_end_monotonic_ns),
        DurationMs(frame.send_start_monotonic_ns, frame.send_end_monotonic_ns),
        DurationMs(frame.control_start_monotonic_ns,
                   frame.control_end_monotonic_ns),
        DurationMs(frame.control_start_monotonic_ns, now_ns)};
    impl_->timing_pub->publish(timing);

    std_msgs::msg::Int64MultiArray status;
    status.layout.dim.resize(1);
    status.layout.dim[0].label =
        "frame,state,policy,reference,episode,active_motion,selected_motion,"
        "deploy_mode,motion_phase,command_source,valid_mask,quality_flags,gain";
    status.layout.dim[0].size = 13;
    status.layout.dim[0].stride = 13;
    status.data = {
        static_cast<std::int64_t>(frame.frame_id), TickForRos(frame.state_tick),
        TickForRos(frame.policy_tick), TickForRos(frame.reference_tick),
        static_cast<std::int64_t>(frame.episode_id), frame.active_motion_id,
        frame.selected_motion_id, static_cast<std::int64_t>(frame.deploy_mode),
        static_cast<std::int64_t>(frame.motion_phase),
        static_cast<std::int64_t>(frame.command_source), frame.valid_mask,
        frame.quality_flags, frame.gain_profile_id};
    impl_->status_pub->publish(status);
  };

  EvalFrameV1 frame;
  for (;;) {
    bool progressed = false;
    while (queue_ && queue_->TryPop(frame)) {
      try {
        publish_frame(frame);
        published_.fetch_add(1, std::memory_order_relaxed);
      } catch (...) {
        dropped_.fetch_add(1, std::memory_order_relaxed);
      }
      progressed = true;
    }
    if (!Running()) {
      if (!queue_ || !queue_->TryPop(frame)) break;
      try {
        publish_frame(frame);
        published_.fetch_add(1, std::memory_order_relaxed);
      } catch (...) {
        dropped_.fetch_add(1, std::memory_order_relaxed);
      }
      continue;
    }
    if (!progressed) std::this_thread::sleep_for(std::chrono::milliseconds(2));
  }

  try {
    if (impl_ && impl_->context) impl_->context->shutdown("telemetry stop");
  } catch (...) {
  }
  queue_.reset();
  impl_.reset();
#endif
}

}  // namespace a3_deploy
