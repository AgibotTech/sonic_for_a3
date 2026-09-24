// Copyright (c) 2026, AgiBot Inc. All rights reserved.
// 对应 notes/a3_backend_plan.md §PR 7 / PR 7a
#include "robot_io/a3_aimrt_backend.hpp"

#include "a3_deploy/a3_motion_rpc_controller.hpp"
#include "a3_io/o10_hand_bridge.hpp"

#include <algorithm>
#include <chrono>
#include <cctype>
#include <cstdint>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <thread>
#include <utility>

#ifdef ENABLE_A3_AIMRT_BACKEND
#include <filesystem>
#include <stdexcept>

#ifdef HAS_A3_MOTION_RPC
#include "a3_motion_control.aimrt_rpc.pb.h"
#endif
#include "aimrt_module_ros2_interface/channel/ros2_channel.h"
#include "joint_msgs/msg/detail/joint_command__rosidl_typesupport_fastrtps_cpp.hpp"
#include "joint_msgs/msg/detail/joint_state__rosidl_typesupport_fastrtps_cpp.hpp"
#include "joint_msgs/msg/joint_command.hpp"
#include "joint_msgs/msg/joint_state.hpp"
#include "sensor_msgs/msg/imu.hpp"
#include "sensor_msgs/msg/detail/joint_state__rosidl_typesupport_fastrtps_cpp.hpp"
#include "sensor_msgs/msg/joint_state.hpp"

#ifdef HAS_A3_HAND_PROTO
#include "aimdk/protocol/hal/hand/hand_channel.pb.h"
#endif

#if defined(HAS_A3_TA_PROTO) || defined(HAS_A3_HAND_PROTO)
#include "aimrt_module_protobuf_interface/channel/protobuf_channel.h"
#endif

#ifdef HAS_A3_TA_PROTO
#include "aimdk/protocol/ta/ta_channel.pb.h"
#endif
#endif

namespace robot_io {

namespace {

std::int64_t NowMonotonicNs() {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

std::int64_t NowSystemNs() {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
             std::chrono::system_clock::now().time_since_epoch())
      .count();
}

void Trim(std::string& s) {
  while (!s.empty() && std::isspace(static_cast<unsigned char>(s.front())))
    s.erase(s.begin());
  while (!s.empty() && std::isspace(static_cast<unsigned char>(s.back())))
    s.pop_back();
}

std::string NormalizeConfigToken(std::string value) {
  std::transform(value.begin(), value.end(), value.begin(),
                 [](unsigned char c) {
                   if (c == '-') return static_cast<char>('_');
                   return static_cast<char>(std::tolower(c));
                 });
  return value;
}

#if defined(ENABLE_A3_AIMRT_BACKEND) && defined(HAS_A3_ROS_MSGS)
void AnchorJointMsgsFastRtpsTypesupport() {
  // rosidl_typesupport_cpp resolves concrete FastRTPS support via dlopen() at
  // runtime. Taking these symbols keeps the vendored joint_msgs FastRTPS .so
  // loaded so direct launches do not need LD_LIBRARY_PATH.
  (void)ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(
      rosidl_typesupport_fastrtps_cpp, joint_msgs, msg, JointCommand)();
  (void)ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(
      rosidl_typesupport_fastrtps_cpp, joint_msgs, msg, JointState)();
  (void)ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(
      rosidl_typesupport_fastrtps_cpp, sensor_msgs, msg, JointState)();
}
#endif

void PrintBoolStatus(const char* name, bool ready, std::uint64_t count) {
  std::cerr << "  " << std::left << std::setw(12) << name
            << " ready=" << (ready ? "yes" : "no ")
            << " samples=" << count << "\n";
}

#if defined(ENABLE_A3_AIMRT_BACKEND) && defined(HAS_A3_MOTION_RPC)
void FillMotionReply(const a3_deploy::A3RpcPlayResult& result,
                     a3::deploy::rpc::MotionReply& reply) {
  reply.set_accepted(result.accepted);
  reply.set_play_id(result.play_id);
  reply.set_motion_id(result.motion_id);
  reply.set_motion_name(result.motion_name);
  reply.set_motion_path(result.motion_path);
  reply.set_total_ticks(result.total_ticks);
  reply.set_state(result.state);
  reply.set_message(result.message);
}

class A3MotionControlService final
    : public a3::deploy::rpc::MotionControlServiceSyncService {
 public:
  explicit A3MotionControlService(
      std::shared_ptr<a3_deploy::A3MotionRpcController> controller)
      : controller_(std::move(controller)) {}

  aimrt::rpc::Status PlayMotion(
      aimrt::rpc::ContextRef,
      const a3::deploy::rpc::PlayMotionRequest& request,
      a3::deploy::rpc::MotionReply& reply) override {
    FillMotionReply(controller_->PlayMotion(request.motion_path()), reply);
    return {};
  }

  aimrt::rpc::Status GetPlaybackStatus(
      aimrt::rpc::ContextRef,
      const a3::deploy::rpc::PlaybackStatusRequest&,
      a3::deploy::rpc::PlaybackStatusReply& reply) override {
    const auto status = controller_->GetPlaybackStatus();
    reply.set_play_id(status.play_id);
    reply.set_motion_id(status.motion_id);
    reply.set_motion_name(status.motion_name);
    reply.set_motion_path(status.motion_path);
    reply.set_tick(status.tick);
    reply.set_total_ticks(status.total_ticks);
    reply.set_deploy_mode(status.deploy_mode);
    reply.set_state(status.state);
    reply.set_playing(status.playing);
    reply.set_held(status.held);
    return {};
  }

  aimrt::rpc::Status StopMotion(
      aimrt::rpc::ContextRef,
      const a3::deploy::rpc::StopMotionRequest&,
      a3::deploy::rpc::MotionReply& reply) override {
    FillMotionReply(controller_->StopMotion(), reply);
    return {};
  }

 private:
  std::shared_ptr<a3_deploy::A3MotionRpcController> controller_;
};
#endif

}  // namespace

// ---------------------------------------------------------------------------
A3AimrtBackend::A3AimrtBackend() = default;

A3AimrtBackend::~A3AimrtBackend() {
  try {
    Stop();
  } catch (...) {
  }
}

// ---------------------------------------------------------------------------
bool A3AimrtBackend::ParseConfig_(const std::string& config) {
  // Reset to defaults.
  cfg_file_path_.clear();
  sync_cfg_ = a3_sync::SyncConfig{};
  publish_enabled_ = true;
  hand_bridge_enabled_ = true;
  auto_phase_ = true;
  sync_release_margin_ns_ = 500'000;
  actual_phase_ns_ = sync_cfg_.phase_ns;

  std::stringstream ss(config);
  std::string tok;
  while (std::getline(ss, tok, ',')) {
    auto eq = tok.find('=');
    if (eq == std::string::npos) continue;
    std::string key = tok.substr(0, eq);
    std::string val = tok.substr(eq + 1);
    Trim(key);
    Trim(val);
    if (key.empty()) continue;

    try {
      if (key == "cfg_file_path") {
        cfg_file_path_ = val;
      } else if (key == "sync_mode") {
        const auto mode = NormalizeConfigToken(val);
        if (mode == "header_interp") {
          sync_cfg_.sync_mode = a3_sync::SyncMode::HeaderInterp;
        } else if (mode == "min_skew_pair") {
          sync_cfg_.sync_mode = a3_sync::SyncMode::MinSkewPair;
        } else if (mode == "latest_cache") {
          sync_cfg_.sync_mode = a3_sync::SyncMode::LatestCache;
        } else if (mode == "latest_frame") {
          sync_cfg_.sync_mode = a3_sync::SyncMode::LatestFrame;
        } else {
          throw std::runtime_error(
              "sync_mode must be one of {header_interp,min_skew_pair,"
              "latest_cache,latest_frame}");
        }
      } else if (key == "sync_hz") {
        sync_cfg_.sync_hz = std::stod(val);
      } else if (key == "align_delay_ms") {
        sync_cfg_.align_delay_ns =
            static_cast<std::int64_t>(std::stod(val) * 1'000'000.0);
      } else if (key == "max_skew_ms") {
        sync_cfg_.max_skew_ns =
            static_cast<std::int64_t>(std::stod(val) * 1'000'000.0);
      } else if (key == "max_sample_age_ms") {
        sync_cfg_.max_sample_age_ns =
            static_cast<std::int64_t>(std::stod(val) * 1'000'000.0);
      } else if (key == "sync_ready_after_input_ms") {
        sync_cfg_.sync_ready_after_input_ns =
            static_cast<std::int64_t>(std::stod(val) * 1'000'000.0);
      } else if (key == "sync_release_margin_ms") {
        sync_release_margin_ns_ =
            static_cast<std::int64_t>(std::stod(val) * 1'000'000.0);
      } else if (key == "max_group_internal_skew_ms") {
        sync_cfg_.max_group_internal_skew_ns =
            static_cast<std::int64_t>(std::stod(val) * 1'000'000.0);
      } else if (key == "max_group_pair_skew_ms") {
        sync_cfg_.max_group_pair_skew_ns =
            static_cast<std::int64_t>(std::stod(val) * 1'000'000.0);
      } else if (key == "group_pair_search_depth") {
        sync_cfg_.group_pair_search_depth = std::stoi(val);
      } else if (key == "max_backtrack") {
        sync_cfg_.max_backtrack = std::stoi(val);
      } else if (key == "phase_ms") {
        sync_cfg_.phase_ns =
            static_cast<std::int64_t>(std::stod(val) * 1'000'000.0);
        actual_phase_ns_ = sync_cfg_.phase_ns;
        auto_phase_ = false;
      } else if (key == "auto_phase") {
        if (val == "true" || val == "1") {
          auto_phase_ = true;
        } else if (val == "false" || val == "0") {
          auto_phase_ = false;
        } else {
          throw std::runtime_error(
              "auto_phase must be one of {true,false,1,0}");
        }
      } else if (key == "publish_enabled") {
        if (val == "true" || val == "1") {
          publish_enabled_ = true;
        } else if (val == "false" || val == "0") {
          publish_enabled_ = false;
        } else {
          throw std::runtime_error(
              "publish_enabled must be one of {true,false,1,0}");
        }
      } else if (key == "hand_bridge_enabled") {
        if (val == "true" || val == "1") {
          hand_bridge_enabled_ = true;
        } else if (val == "false" || val == "0") {
          hand_bridge_enabled_ = false;
        } else {
          throw std::runtime_error(
              "hand_bridge_enabled must be one of {true,false,1,0}");
        }
      } else {
        std::cerr << "[a3_backend] WARN: unknown config key '" << key << "'\n";
      }
    } catch (const std::exception& e) {
      std::cerr << "[a3_backend] ERROR: parsing '" << key << "=" << val
                << "': " << e.what() << "\n";
      return false;
    }
  }
  if (!std::isfinite(sync_cfg_.sync_hz) || sync_cfg_.sync_hz <= 0.0) {
    std::cerr << "[a3_backend] ERROR: sync_hz must be > 0\n";
    return false;
  }
  if (sync_cfg_.align_delay_ns < 0 || sync_cfg_.max_skew_ns < 0 ||
      sync_cfg_.sync_ready_after_input_ns < 0 ||
      sync_release_margin_ns_ < 0 ||
      sync_cfg_.max_group_internal_skew_ns < 0 ||
      sync_cfg_.max_group_pair_skew_ns < 0) {
    std::cerr << "[a3_backend] ERROR: sync skew/delay thresholds must be "
                 "non-negative\n";
    return false;
  }
  if (sync_cfg_.group_pair_search_depth <= 0 ||
      sync_cfg_.group_pair_search_depth > 16) {
    std::cerr << "[a3_backend] ERROR: group_pair_search_depth must be in "
                 "[1,16]\n";
    return false;
  }
  if (sync_cfg_.max_backtrack <= 0) {
    std::cerr << "[a3_backend] ERROR: max_backtrack must be > 0\n";
    return false;
  }
  return true;
}

void A3AimrtBackend::CalibrateSyncPhase_() {
  if (!sync_loop_ || !auto_phase_ ||
      sync_cfg_.sync_mode != a3_sync::SyncMode::MinSkewPair) {
    actual_phase_ns_ = sync_cfg_.phase_ns;
    return;
  }

  std::this_thread::sleep_for(std::chrono::milliseconds(30));

  std::int64_t phase_ns = 0;
  std::int64_t latest_recv_ns = 0;
  std::int64_t pair_skew_ns = 0;
  if (!sync_loop_->EstimateAutoPhaseNs(sync_release_margin_ns_, phase_ns,
                                       latest_recv_ns, pair_skew_ns)) {
    actual_phase_ns_ = sync_cfg_.phase_ns;
    std::cerr << "[a3_backend] WARN: auto sync phase calibration failed; "
              << "using fallback phase_ms="
              << static_cast<double>(actual_phase_ns_) / 1e6 << "\n";
    return;
  }

  sync_cfg_.phase_ns = phase_ns;
  actual_phase_ns_ = phase_ns;
  (void)sync_loop_->SetPhaseNsForStartup(phase_ns);

  const auto period_ns =
      static_cast<std::int64_t>(1e9 / sync_cfg_.sync_hz);
  const double recv_phase_ms =
      period_ns > 0 ? static_cast<double>(latest_recv_ns % period_ns) / 1e6
                    : 0.0;
  std::cerr << "[a3_backend] auto sync phase calibrated: phase_ms="
            << static_cast<double>(phase_ns) / 1e6
            << " latest_recv_phase_ms=" << recv_phase_ms
            << " target_sync_wait_ms="
            << static_cast<double>(sync_release_margin_ns_) / 1e6
            << " group_pair_skew_ms="
            << static_cast<double>(pair_skew_ns) / 1e6 << "\n";
}

// ---------------------------------------------------------------------------
bool A3AimrtBackend::Init(const std::string& config) {
  if (!ParseConfig_(config)) return false;

#ifdef ENABLE_A3_AIMRT_BACKEND
  if (cfg_file_path_.empty()) {
    std::cerr << "[a3_backend] ERROR: cfg_file_path is required when "
                 "ENABLE_A3_AIMRT_BACKEND=ON\n";
    return false;
  }
  if (!std::filesystem::exists(cfg_file_path_)) {
    std::cerr << "[a3_backend] ERROR: cfg_file_path not found: "
              << cfg_file_path_ << "\n";
    return false;
  }
  aimrt_options_.cfg_file_path = cfg_file_path_;
#endif

  // Ring buffers.
  waist_ring_ =
      std::make_unique<a3_sync::RingBuffer<a3_sync::WaistSample>>(2048);
  leg_ring_ = std::make_unique<a3_sync::RingBuffer<a3_sync::LegSample>>(2048);
  arm_ring_ = std::make_unique<a3_sync::RingBuffer<a3_sync::ArmSample>>(2048);
  neck_ring_ =
      std::make_unique<a3_sync::RingBuffer<a3_sync::NeckSample>>(2048);
  pelvis_imu_ring_ =
      std::make_unique<a3_sync::RingBuffer<a3_sync::ImuSample>>(512);
  torso_imu_ring_ =
      std::make_unique<a3_sync::RingBuffer<a3_sync::ImuSample>>(512);

  // Sync loop.
  a3_sync::A3SyncLoop::Options loop_opts{};
  loop_opts.sync = sync_cfg_;
  sync_loop_ = std::make_unique<a3_sync::A3SyncLoop>(std::move(loop_opts));
  sync_loop_->RegisterStateCallback(
      [this](const RobotState& s) { OnSyncState_(s); });

  // Subscriber manager (owns 6 subscribers; rings are non-owning refs).
  a3_io::A3SubscriberManager::Options sub_opts{};
  sub_opts.waist_ring      = waist_ring_.get();
  sub_opts.leg_ring        = leg_ring_.get();
  sub_opts.arm_ring        = arm_ring_.get();
  sub_opts.neck_ring       = neck_ring_.get();
  sub_opts.pelvis_imu_ring = pelvis_imu_ring_.get();
  sub_opts.torso_imu_ring  = torso_imu_ring_.get();
  sub_opts.waist_sample_cb = [this](const a3_sync::WaistSample& s) {
    if (sync_loop_) sync_loop_->OnWaistState(s);
  };
  sub_opts.leg_sample_cb = [this](const a3_sync::LegSample& s) {
    if (sync_loop_) sync_loop_->OnLegState(s);
  };
  sub_opts.arm_sample_cb = [this](const a3_sync::ArmSample& s) {
    if (sync_loop_) sync_loop_->OnArmState(s);
  };
  sub_opts.neck_sample_cb = [this](const a3_sync::NeckSample& s) {
    if (sync_loop_) sync_loop_->OnNeckState(s);
  };
  sub_opts.pelvis_imu_sample_cb = [this](const a3_sync::ImuSample& s) {
    if (sync_loop_) sync_loop_->OnPelvisImu(s);
  };
  sub_opts.torso_imu_sample_cb = [this](const a3_sync::ImuSample& s) {
    if (sync_loop_) sync_loop_->OnTorsoImu(s);
  };
  sub_mgr_ = std::make_unique<a3_io::A3SubscriberManager>(sub_opts);

  inited_.store(true);
  return true;
}

// ---------------------------------------------------------------------------
#ifdef ENABLE_A3_AIMRT_BACKEND
bool A3AimrtBackend::InitAimrt_() {
  try {
    aimrt_core_.Initialize(aimrt_options_);
    module_handle_ = aimrt::CoreRef(
        aimrt_core_.GetModuleManager().CreateModule("A3BackendModule"));
  } catch (const std::exception& e) {
    std::cerr << "[a3_backend] AimRT Initialize failed: " << e.what() << "\n";
    return false;
  }
  return true;
}

bool A3AimrtBackend::RegisterPubSub_() {
#ifdef HAS_A3_ROS_MSGS
  try {
    AnchorJointMsgsFastRtpsTypesupport();

    auto ch = module_handle_.GetChannelHandle();

    // ---- Publishers ----
    if (publish_enabled_) {
      auto waist_pub = ch.GetPublisher("/body_drive/waist_joint_command");
      auto leg_pub   = ch.GetPublisher("/body_drive/leg_joint_command");
      auto arm_pub   = ch.GetPublisher("/body_drive/arm_joint_command");
      auto neck_pub  = ch.GetPublisher("/body_drive/neck_joint_command");

      if (!aimrt::channel::RegisterPublishType<joint_msgs::msg::JointCommand>(
              waist_pub))
        throw std::runtime_error("register waist publish failed");
      if (!aimrt::channel::RegisterPublishType<joint_msgs::msg::JointCommand>(
              leg_pub))
        throw std::runtime_error("register leg publish failed");
      if (!aimrt::channel::RegisterPublishType<joint_msgs::msg::JointCommand>(
              arm_pub))
        throw std::runtime_error("register arm publish failed");
      if (!aimrt::channel::RegisterPublishType<joint_msgs::msg::JointCommand>(
              neck_pub))
        throw std::runtime_error("register neck publish failed");
      auto waist_proxy = std::make_shared<
          aimrt::channel::PublisherProxy<joint_msgs::msg::JointCommand>>(
          waist_pub);
      auto leg_proxy =
          std::make_shared<aimrt::channel::PublisherProxy<
              joint_msgs::msg::JointCommand>>(leg_pub);
      auto arm_proxy =
          std::make_shared<aimrt::channel::PublisherProxy<
              joint_msgs::msg::JointCommand>>(arm_pub);
      auto neck_proxy = std::make_shared<
          aimrt::channel::PublisherProxy<joint_msgs::msg::JointCommand>>(
          neck_pub);
      a3_io::A3PublisherManager::Options pub_opts{};
      pub_opts.waist_publish_fn =
          [waist_proxy](const joint_msgs::msg::JointCommand& m) {
            waist_proxy->Publish(m);
          };
      pub_opts.leg_publish_fn =
          [leg_proxy](const joint_msgs::msg::JointCommand& m) {
            leg_proxy->Publish(m);
          };
      pub_opts.arm_publish_fn =
          [arm_proxy](const joint_msgs::msg::JointCommand& m) {
            arm_proxy->Publish(m);
          };
      pub_opts.neck_publish_fn =
          [neck_proxy](const joint_msgs::msg::JointCommand& m) {
            neck_proxy->Publish(m);
          };
      pub_mgr_ =
          std::make_unique<a3_io::A3PublisherManager>(std::move(pub_opts));
#ifdef HAS_A3_HAND_PROTO
      if (hand_bridge_enabled_) {
        auto hand_pub = ch.GetPublisher("/body_drive/hand_joint_command");
        if (!aimrt::channel::RegisterPublishType<
                aimdk::protocol::HandCommandChannel>(hand_pub))
          throw std::runtime_error("register hand command publish failed");
        auto hand_proxy = std::make_shared<aimrt::channel::PublisherProxy<
            aimdk::protocol::HandCommandChannel>>(hand_pub);
        hand_command_publish_fn_ =
            [hand_proxy](const aimdk::protocol::HandCommandChannel& m) {
              hand_proxy->Publish(m);
            };
      }
#endif
    } else {
      std::cerr << "[a3_backend] command publishers disabled "
                   "(publish_enabled=false)\n";
    }

#ifdef HAS_A3_HAND_PROTO
    if (hand_bridge_enabled_) {
      auto motion_hand_state_pub =
          ch.GetPublisher("/motion/control/hand_joint_state");
      if (!aimrt::channel::RegisterPublishType<sensor_msgs::msg::JointState>(
              motion_hand_state_pub))
        throw std::runtime_error("register motion hand state publish failed");
      auto motion_hand_state_proxy = std::make_shared<
          aimrt::channel::PublisherProxy<sensor_msgs::msg::JointState>>(
          motion_hand_state_pub);
      hand_state_publish_fn_ =
          [motion_hand_state_proxy](const sensor_msgs::msg::JointState& m) {
            motion_hand_state_proxy->Publish(m);
          };
    }
#endif

    // ---- Subscribers ----
    auto waist_sub = ch.GetSubscriber("/body_drive/waist_joint_state");
    auto leg_sub   = ch.GetSubscriber("/body_drive/leg_joint_state");
    auto arm_sub   = ch.GetSubscriber("/body_drive/arm_joint_state");
    auto neck_sub  = ch.GetSubscriber("/body_drive/neck_joint_state");
    auto pelvis_imu_sub = ch.GetSubscriber("/body_drive/pelvis_imu/data");
    auto torso_imu_sub  = ch.GetSubscriber("/body_drive/torso_imu/data");
    if (!aimrt::channel::Subscribe<joint_msgs::msg::JointState>(
            waist_sub,
            [this](const std::shared_ptr<const joint_msgs::msg::JointState>&
                       msg) { sub_mgr_->waist().OnMessage(msg); }))
      throw std::runtime_error("subscribe waist failed");
    if (!aimrt::channel::Subscribe<joint_msgs::msg::JointState>(
            leg_sub,
            [this](const std::shared_ptr<const joint_msgs::msg::JointState>&
                       msg) { sub_mgr_->leg().OnMessage(msg); }))
      throw std::runtime_error("subscribe leg failed");
    if (!aimrt::channel::Subscribe<joint_msgs::msg::JointState>(
            arm_sub,
            [this](const std::shared_ptr<const joint_msgs::msg::JointState>&
                       msg) { sub_mgr_->arm().OnMessage(msg); }))
      throw std::runtime_error("subscribe arm failed");
    if (!aimrt::channel::Subscribe<joint_msgs::msg::JointState>(
            neck_sub,
            [this](const std::shared_ptr<const joint_msgs::msg::JointState>&
                       msg) { sub_mgr_->neck().OnMessage(msg); }))
      throw std::runtime_error("subscribe neck failed");
    if (!aimrt::channel::Subscribe<sensor_msgs::msg::Imu>(
            pelvis_imu_sub,
            [this](const std::shared_ptr<const sensor_msgs::msg::Imu>& msg) {
              sub_mgr_->pelvis_imu().OnMessage(msg);
            }))
      throw std::runtime_error("subscribe pelvis_imu failed");
    if (!aimrt::channel::Subscribe<sensor_msgs::msg::Imu>(
            torso_imu_sub,
            [this](const std::shared_ptr<const sensor_msgs::msg::Imu>& msg) {
              sub_mgr_->torso_imu().OnMessage(msg);
            }))
      throw std::runtime_error("subscribe torso_imu failed");
#ifdef HAS_A3_HAND_PROTO
    if (hand_bridge_enabled_) {
      auto body_hand_state_sub =
          ch.GetSubscriber("/body_drive/hand_joint_state");
      auto motion_hand_cmd_sub =
          ch.GetSubscriber("/motion/control/hand_joint_command");
      if (!aimrt::channel::Subscribe<aimdk::protocol::HandStateChannel>(
              body_hand_state_sub,
              [this](const std::shared_ptr<
                     const aimdk::protocol::HandStateChannel>& msg) {
                if (msg) HandleBodyDriveHandState_(*msg);
              }))
        throw std::runtime_error("subscribe body hand state failed");
      if (!aimrt::channel::Subscribe<sensor_msgs::msg::JointState>(
              motion_hand_cmd_sub,
              [this](const std::shared_ptr<const sensor_msgs::msg::JointState>&
                         msg) {
                if (msg) HandleMotionHandCommand_(*msg);
              }))
        throw std::runtime_error("subscribe motion hand command failed");
    } else {
      std::cerr << "[a3_backend] hand bridge disabled "
                   "(hand_bridge_enabled=false)\n";
    }
#endif

#ifdef HAS_A3_TA_PROTO
    if (teleop_frame_cb_) {
      auto teleop_sub = ch.GetSubscriber(teleop_topic_);
      if (!aimrt::channel::Subscribe<aimdk::protocol::TaWholeBodyCommandChannel>(
              teleop_sub,
              [this](const std::shared_ptr<
                     const aimdk::protocol::TaWholeBodyCommandChannel>& msg) {
                if (!msg || !teleop_frame_cb_) return;
                const std::int64_t arrival_stamp_ns = NowSystemNs();
                a3_deploy::A3TeleopFrame frame{};
                std::string error;
                if (!a3_deploy::ConvertTaWholeBodyCommand(
                        *msg, arrival_stamp_ns, frame, &error)) {
                  static std::uint64_t reject_count = 0;
                  if ((reject_count++ % 100) == 0) {
                    std::cerr << "[a3_backend] WARN: reject "
                              << teleop_topic_ << ": " << error << "\n";
                  }
                  return;
                }
                // A3SyncLoop stamps RobotState with system_clock. Retimestamp
                // teleop commands onto that same local clock so the delayed
                // future window works for both live AimRT and ROS2 bag replay.
                frame.stamp_ns = arrival_stamp_ns;
                teleop_frame_cb_(frame);
              }))
        throw std::runtime_error("subscribe teleop whole_body_command failed");
      std::cerr << "[a3_backend] teleop subscriber enabled: "
                << teleop_topic_ << "\n";
    }
#else
    if (teleop_frame_cb_) {
      std::cerr << "[a3_backend] WARN: teleop callback configured but "
                   "HAS_A3_TA_PROTO is not defined; /ta subscriber disabled\n";
    }
#endif
  } catch (const std::exception& e) {
    std::cerr << "[a3_backend] RegisterPubSub_ failed: " << e.what() << "\n";
    return false;
  }
  return true;
#else
  std::cerr << "[a3_backend] ERROR: ENABLE_A3_AIMRT_BACKEND=ON but "
               "HAS_A3_ROS_MSGS is not defined (joint_msgs/sensor_msgs "
               "not available)\n";
  return false;
#endif
}

bool A3AimrtBackend::RegisterMotionRpc_() {
#ifdef HAS_A3_MOTION_RPC
  if (!motion_rpc_controller_) {
    std::cerr << "[a3_backend] motion RPC disabled: no controller configured\n";
    return true;
  }
  try {
    auto service =
        std::make_shared<A3MotionControlService>(motion_rpc_controller_);
    if (!module_handle_.GetRpcHandle().RegisterService(service.get())) {
      std::cerr << "[a3_backend] register MotionControlService failed\n";
      return false;
    }
    motion_rpc_service_ = std::move(service);
    std::cerr << "[a3_backend] motion HTTP RPC registered: "
                 "a3.deploy.rpc.MotionControlService\n";
    return true;
  } catch (const std::exception& e) {
    std::cerr << "[a3_backend] register motion RPC failed: " << e.what()
              << "\n";
    return false;
  }
#else
  if (motion_rpc_controller_) {
    std::cerr << "[a3_backend] WARN: motion RPC controller configured but "
                 "RPC bindings are unavailable\n";
  }
  return true;
#endif
}
#endif  // ENABLE_A3_AIMRT_BACKEND

// ---------------------------------------------------------------------------
bool A3AimrtBackend::Start() {
  if (!inited_.load()) {
    std::cerr << "[a3_backend] Start() called before successful Init()\n";
    return false;
  }
  if (started_.load()) return true;

#ifdef ENABLE_A3_AIMRT_BACKEND
  if (!InitAimrt_()) return false;
  if (!RegisterPubSub_()) return false;
  if (!RegisterMotionRpc_()) return false;
  try {
    shutdown_future_ = aimrt_core_.AsyncStart();
  } catch (const std::exception& e) {
    std::cerr << "[a3_backend] AsyncStart failed: " << e.what() << "\n";
    return false;
  }
#else
  std::cerr << "[a3_backend] WARN: built without ENABLE_A3_AIMRT_BACKEND; "
               "no AimRT subscribers are registered. Sync frames will stay "
               "missing unless samples are injected by tests.\n";
#endif

#ifdef ENABLE_A3_AIMRT_BACKEND
  bool all_ready = true;
  if (sub_mgr_) {
    all_ready = sub_mgr_->WaitAllReady(5.0);
    std::cerr << "[a3_backend] topic readiness after startup gate:\n";
    PrintBoolStatus("waist", sub_mgr_->waist().IsReady(),
                    sub_mgr_->waist().SampleCount());
    PrintBoolStatus("leg", sub_mgr_->leg().IsReady(),
                    sub_mgr_->leg().SampleCount());
    PrintBoolStatus("arm", sub_mgr_->arm().IsReady(),
                    sub_mgr_->arm().SampleCount());
    PrintBoolStatus("neck", sub_mgr_->neck().IsReady(),
                    sub_mgr_->neck().SampleCount());
    PrintBoolStatus("pelvis_imu", sub_mgr_->pelvis_imu().IsReady(),
                    sub_mgr_->pelvis_imu().SampleCount());
    PrintBoolStatus("torso_imu", sub_mgr_->torso_imu().IsReady(),
	                    sub_mgr_->torso_imu().SampleCount());
  }
  if (all_ready) {
    CalibrateSyncPhase_();
  } else {
    actual_phase_ns_ = sync_cfg_.phase_ns;
  }
#endif

  sync_loop_->Start();

#ifdef ENABLE_A3_AIMRT_BACKEND
  if (sync_loop_) {
    const auto s = sync_loop_->Statistics();
    std::cerr << "[a3_backend] sync startup stats: ticks=" << s.tick_total
              << " complete=" << s.frame_complete_total
              << " aligned=" << s.frame_aligned_total
              << " missing={waist:" << s.missing_waist
              << ", leg:" << s.missing_leg
              << ", arm:" << s.missing_arm
              << ", neck:" << s.missing_neck
              << ", pelvis_imu:" << s.missing_pelvis_imu
              << ", torso_imu:" << s.missing_torso_imu
              << "} stale={waist:" << s.stale_waist
              << ", leg:" << s.stale_leg
              << ", arm:" << s.stale_arm
              << ", neck:" << s.stale_neck
              << ", pelvis_imu:" << s.stale_pelvis_imu
              << ", torso_imu:" << s.stale_torso_imu
              << "} age_ms={waist:"
              << (static_cast<double>(s.last_age_waist_ns) / 1e6)
              << ", leg:" << (static_cast<double>(s.last_age_leg_ns) / 1e6)
              << ", arm:" << (static_cast<double>(s.last_age_arm_ns) / 1e6)
              << ", neck:" << (static_cast<double>(s.last_age_neck_ns) / 1e6)
              << ", pelvis_imu:"
              << (static_cast<double>(s.last_age_pelvis_imu_ns) / 1e6)
              << ", torso_imu:"
              << (static_cast<double>(s.last_age_torso_imu_ns) / 1e6)
              << "} last_skew_ms="
              << (static_cast<double>(s.last_skew_ns) / 1e6) << "\n";
  }
  if (!all_ready) {
    std::cerr << "[a3_backend] WARN: not all 6 topics ready within 5s; "
                 "proceeding — policy/watchdog handles stale samples.\n";
  }
#endif

  started_.store(true);
  return true;
}

// ---------------------------------------------------------------------------
void A3AimrtBackend::Stop() {
  if (!started_.load()) return;

  if (sync_loop_) sync_loop_->Stop();

#ifdef ENABLE_A3_AIMRT_BACKEND
  try {
    aimrt_core_.Shutdown();
  } catch (...) {
  }
  if (shutdown_future_.valid()) {
    try {
      shutdown_future_.get();
    } catch (...) {
    }
  }
#endif

  started_.store(false);
}

// ---------------------------------------------------------------------------
const JointLayout& A3AimrtBackend::GetLayout() const {
  return MakeA3Layout31();
}

double A3AimrtBackend::StateRateHz() const { return sync_cfg_.sync_hz; }

a3_sync::A3SyncStatistics::Snapshot A3AimrtBackend::SyncStatistics() const {
  if (!sync_loop_) return {};
  return sync_loop_->Statistics();
}

a3_sync::A3SyncStatistics::LatencySnapshot
A3AimrtBackend::ConsumeLatencyStatistics() {
  if (!sync_loop_) return {};
  return sync_loop_->ConsumeLatencyStatistics();
}

void A3AimrtBackend::RegisterStateCallback(StateCallback cb) {
  user_cb_ = std::move(cb);
}

bool A3AimrtBackend::UsesLatestStatePull() const noexcept {
  return sync_cfg_.sync_mode == a3_sync::SyncMode::LatestCache;
}

bool A3AimrtBackend::TryReadLatestState(RobotState& state) {
  return sync_loop_ && sync_loop_->TryReadLatestState(state);
}

void A3AimrtBackend::SetTeleopFrameCallback(TeleopFrameCallback cb) {
  teleop_frame_cb_ = std::move(cb);
}

void A3AimrtBackend::SetTeleopTopic(std::string topic) {
  if (!topic.empty()) teleop_topic_ = std::move(topic);
}

void A3AimrtBackend::SetMotionRpcController(
    std::shared_ptr<a3_deploy::A3MotionRpcController> controller) {
  motion_rpc_controller_ = std::move(controller);
}

void A3AimrtBackend::OnSyncState_(const RobotState& state) {
  if (user_cb_) user_cb_(state);
}

// ---------------------------------------------------------------------------
bool A3AimrtBackend::SendCommand(const RobotCommand& cmd) {
  if (cmd.q_des.size() != kA3Dof) return false;
  if (cmd.dq_des.size() != kA3Dof) return false;
  if (cmd.tau_ff.size() != kA3Dof) return false;
  if (cmd.kp.size() != kA3Dof) return false;
  if (cmd.kd.size() != kA3Dof) return false;

  const std::uint32_t seq =
      seq_counter_.fetch_add(1, std::memory_order_relaxed);
  const std::int64_t stamp_ns = NowMonotonicNs();

  if (test_capture_fn_) {
    test_capture_fn_("waist", stamp_ns, seq, cmd);
    test_capture_fn_("leg", stamp_ns, seq, cmd);
    test_capture_fn_("arm", stamp_ns, seq, cmd);
    test_capture_fn_("neck", stamp_ns, seq, cmd);
  }

#if defined(ENABLE_A3_AIMRT_BACKEND) && defined(HAS_A3_ROS_MSGS)
  if (pub_mgr_) pub_mgr_->PublishAll(stamp_ns, seq, cmd);
#endif

  return true;
}

// ---------------------------------------------------------------------------
// Test injection helpers — safe before Start(), unaffected by the AimRT gate.
void A3AimrtBackend::InjectWaistSample_ForTest(const a3_sync::WaistSample& s) {
  if (!waist_ring_) return;
  waist_ring_->write([&](a3_sync::WaistSample& slot) { slot = s; });
  if (sync_loop_) sync_loop_->OnWaistState(s);
}
void A3AimrtBackend::InjectLegSample_ForTest(const a3_sync::LegSample& s) {
  if (!leg_ring_) return;
  leg_ring_->write([&](a3_sync::LegSample& slot) { slot = s; });
  if (sync_loop_) sync_loop_->OnLegState(s);
}
void A3AimrtBackend::InjectArmSample_ForTest(const a3_sync::ArmSample& s) {
  if (!arm_ring_) return;
  arm_ring_->write([&](a3_sync::ArmSample& slot) { slot = s; });
  if (sync_loop_) sync_loop_->OnArmState(s);
}
void A3AimrtBackend::InjectNeckSample_ForTest(const a3_sync::NeckSample& s) {
  if (!neck_ring_) return;
  neck_ring_->write([&](a3_sync::NeckSample& slot) { slot = s; });
  if (sync_loop_) sync_loop_->OnNeckState(s);
}
void A3AimrtBackend::InjectPelvisImu_ForTest(const a3_sync::ImuSample& s) {
  if (!pelvis_imu_ring_) return;
  pelvis_imu_ring_->write([&](a3_sync::ImuSample& slot) { slot = s; });
  if (sync_loop_) sync_loop_->OnPelvisImu(s);
}
void A3AimrtBackend::InjectTorsoImu_ForTest(const a3_sync::ImuSample& s) {
  if (!torso_imu_ring_) return;
  torso_imu_ring_->write([&](a3_sync::ImuSample& slot) { slot = s; });
  if (sync_loop_) sync_loop_->OnTorsoImu(s);
}

void A3AimrtBackend::SetTestCaptureFn_ForTest(TestCaptureFn fn) {
  test_capture_fn_ = std::move(fn);
}

#if defined(HAS_A3_ROS_MSGS) && defined(HAS_A3_HAND_PROTO)
bool A3AimrtBackend::HandleMotionHandCommand_(
    const sensor_msgs::msg::JointState& msg) {
  if (!hand_bridge_enabled_) return false;
  const std::uint32_t seq =
      hand_seq_counter_.fetch_add(1, std::memory_order_relaxed);
  const std::int64_t stamp_ns = NowSystemNs();
  aimdk::protocol::HandCommandChannel out;
  std::string error;
  if (!a3_io::ConvertO10JointStateToHandCommandChannel(msg, seq, stamp_ns,
                                                        out, &error)) {
    static std::uint64_t reject_count = 0;
    if ((reject_count++ % 100) == 0) {
      std::cerr << "[a3_backend] WARN: reject "
                   "/motion/control/hand_joint_command: "
                << error << "\n";
    }
    return false;
  }

  if (hand_command_capture_fn_) {
    hand_command_capture_fn_(stamp_ns, seq, out);
  }
  if (hand_command_publish_fn_) {
    hand_command_publish_fn_(out);
  }
  return true;
}

bool A3AimrtBackend::HandleBodyDriveHandState_(
    const aimdk::protocol::HandStateChannel& msg) {
  if (!hand_bridge_enabled_) return false;
  const std::int64_t stamp_ns = NowSystemNs();
  sensor_msgs::msg::JointState out;
  std::string error;
  if (!a3_io::ConvertO10HandStateChannelToJointState(msg, stamp_ns, out,
                                                     &error)) {
    static std::uint64_t reject_count = 0;
    if ((reject_count++ % 100) == 0) {
      std::cerr << "[a3_backend] WARN: reject "
                   "/body_drive/hand_joint_state: "
                << error << "\n";
    }
    return false;
  }

  if (hand_state_capture_fn_) {
    hand_state_capture_fn_(stamp_ns, out);
  }
  if (hand_state_publish_fn_) {
    hand_state_publish_fn_(out);
  }
  return true;
}

void A3AimrtBackend::SetHandCommandCaptureFn_ForTest(
    HandCommandCaptureFn fn) {
  hand_command_capture_fn_ = std::move(fn);
}

void A3AimrtBackend::SetHandStateCaptureFn_ForTest(HandStateCaptureFn fn) {
  hand_state_capture_fn_ = std::move(fn);
}

bool A3AimrtBackend::InjectMotionHandCommand_ForTest(
    const sensor_msgs::msg::JointState& msg) {
  return HandleMotionHandCommand_(msg);
}

bool A3AimrtBackend::InjectBodyDriveHandState_ForTest(
    const aimdk::protocol::HandStateChannel& msg) {
  return HandleBodyDriveHandState_(msg);
}
#endif

// ---------------------------------------------------------------------------
// Strong factory definition — overrides the weak fallback in
// backend_factory.cpp whenever this TU is linked in.
std::unique_ptr<RobotIOBackend> CreateA3AimrtBackend() {
  return std::make_unique<A3AimrtBackend>();
}

}  // namespace robot_io
