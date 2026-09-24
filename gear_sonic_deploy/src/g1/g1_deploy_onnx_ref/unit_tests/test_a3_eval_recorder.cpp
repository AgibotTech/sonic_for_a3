// Copyright (c) 2026, AgiBot Inc. All rights reserved.
#include <gtest/gtest.h>
#include <yaml-cpp/yaml.h>

#include "a3_deploy/a3_eval_recorder.hpp"

#include <filesystem>
#include <fstream>
#include <iterator>
#include <set>
#include <string>
#include <unistd.h>

#if HAS_A3_EVAL_MCAP
#include <mcap/reader.hpp>
#endif

namespace {

class ScopedCurrentPath {
 public:
  explicit ScopedCurrentPath(const std::filesystem::path& path)
      : original_(std::filesystem::current_path()) {
    std::filesystem::current_path(path);
  }

  ~ScopedCurrentPath() {
    std::error_code ec;
    std::filesystem::current_path(original_, ec);
  }

  ScopedCurrentPath(const ScopedCurrentPath&) = delete;
  ScopedCurrentPath& operator=(const ScopedCurrentPath&) = delete;

 private:
  std::filesystem::path original_;
};

std::string ReadText(const std::filesystem::path& path) {
  std::ifstream in(path);
  return {std::istreambuf_iterator<char>(in),
          std::istreambuf_iterator<char>()};
}

#if HAS_A3_EVAL_MCAP
std::string McapBytesToString(const mcap::ByteArray& bytes) {
  return {reinterpret_cast<const char*>(bytes.data()), bytes.size()};
}
#endif

a3_deploy::EvalFrameV1 CompletePlayingFrame(std::uint64_t frame_id,
                                            std::uint64_t reference_tick) {
  a3_deploy::EvalFrameV1 frame;
  frame.frame_id = frame_id;
  frame.state_tick = frame_id;
  frame.policy_tick = frame_id;
  frame.reference_tick = reference_tick;
  frame.deploy_mode = a3_deploy::DeployMode::kMotion;
  frame.motion_phase = a3_deploy::MotionPhase::kPlaying;
  frame.command_source = a3_deploy::CommandSource::kPolicy;
  frame.gain_profile_id = 2;
  frame.active_motion_id = 0;
  frame.selected_motion_id = 0;
  frame.valid_mask =
      a3_deploy::kRobotStateValid | a3_deploy::kPelvisImuValid |
      a3_deploy::kTorsoImuValid | a3_deploy::kRawActionValid |
      a3_deploy::kRobotCommandValid |
      a3_deploy::kReferenceValid | a3_deploy::kInferTimingValid |
      a3_deploy::kSendTimingValid;
  frame.quality_flags = a3_deploy::kSyncComplete |
                        a3_deploy::kInferOk |
                        a3_deploy::kCommandSent;
  return frame;
}

}  // namespace

TEST(A3EvalSpscQueue, FullQueueReturnsImmediatelyAndPreservesOrder) {
  a3_deploy::A3EvalSpscQueue<int> queue(2);
  EXPECT_TRUE(queue.TryPush(11));
  EXPECT_TRUE(queue.TryPush(22));
  EXPECT_FALSE(queue.TryPush(33));

  int value = 0;
  ASSERT_TRUE(queue.TryPop(value));
  EXPECT_EQ(value, 11);
  ASSERT_TRUE(queue.TryPop(value));
  EXPECT_EQ(value, 22);
  EXPECT_FALSE(queue.TryPop(value));
}

TEST(A3EvalTypes, UsesReferencePelvisSchemaV7) {
  EXPECT_EQ(a3_deploy::kA3EvalSchemaVersion, 7u);
  EXPECT_TRUE(std::is_trivially_copyable_v<a3_deploy::EvalFrameV1>);
  // 与 Python inspector 的 EvalFrameV7 ctypes 布局保持一致。
  EXPECT_EQ(sizeof(a3_deploy::EvalFrameV1), 1560u);
}

TEST(A3EvalTelemetry, DedicatedPublisherThreadDrainsQueueOnStop) {
  if (!a3_deploy::A3EvalTelemetryPublisher::Ros2AvailableAtBuildTime()) {
    GTEST_SKIP() << "binary built without ROS2 telemetry";
  }
  // 受限 CI/sandbox 中 HOME 可能只读；ROS 日志写入临时目录即可。
  ::setenv("ROS_LOG_DIR", "/tmp/a3_eval_ros_test_logs", 1);
  a3_deploy::A3EvalTelemetryOptions options;
  options.enabled = true;
  options.queue_capacity = 8;
  options.node_name = "a3_eval_telemetry_unit_test";
  options.topic_prefix = "/a3/eval_unit_test";
  a3_deploy::A3EvalTelemetryPublisher telemetry(std::move(options));
  std::string error;
  ASSERT_TRUE(telemetry.Start(&error)) << error;

  a3_deploy::EvalFrameV1 frame;
  for (std::uint64_t i = 0; i < 3; ++i) {
    frame.frame_id = i;
    ASSERT_TRUE(telemetry.TryPush(frame));
  }
  telemetry.Stop();

  const auto stats = telemetry.Statistics();
  EXPECT_EQ(stats.frames_accepted, 3u);
  EXPECT_EQ(stats.frames_published, 3u);
  EXPECT_EQ(stats.frames_dropped, 0u);
}

TEST(A3EvalRecorder, WritesFramesEventsManifestAndSummary) {
  const auto root = std::filesystem::temp_directory_path() /
                    ("a3_eval_recorder_test_" +
                     std::to_string(static_cast<long long>(::getpid())));
  std::error_code ec;
  std::filesystem::remove_all(root, ec);

  a3_deploy::A3EvalRecorderOptions options;
  options.enabled = true;
  options.output_root = root;
  options.session_id = "unit_test";
  options.backend_name = "mujoco_test";
  options.runtime_config_path = "/tmp/demo.yaml";
  options.policy_model_path = "/tmp/model.onnx";
  options.frame_queue_capacity = 16;
  options.flush_frames = 1;
  options.motions.push_back({0, "demo_motion", "/tmp/demo.csv", 1});

  a3_deploy::A3EvalRecorder recorder(std::move(options));
  std::string error;
  ASSERT_TRUE(recorder.Start(&error)) << error;

  a3_deploy::EvalFrameV1 frame;
  frame.frame_id = 0;
  frame.policy_tick = 9;
  frame.deploy_mode = a3_deploy::DeployMode::kMotion;
  frame.motion_phase = a3_deploy::MotionPhase::kPlaying;
  frame.active_motion_id = 0;
  frame.selected_motion_id = 0;
  frame.reference_tick = 0;
  frame.valid_mask = a3_deploy::kReferenceValid;
  frame.quality_flags = a3_deploy::kSyncComplete;
  frame.q[0] = 1.25f;
  ASSERT_TRUE(recorder.TryPushFrame(frame));

  auto terminal = frame;
  terminal.frame_id = 1;
  terminal.policy_tick = 10;
  terminal.motion_phase = a3_deploy::MotionPhase::kTerminalHold;
  terminal.valid_mask = 0;
  ASSERT_TRUE(recorder.TryPushFrame(terminal));
  recorder.Stop();

  const auto session = recorder.SessionDirectory();
  ASSERT_TRUE(std::filesystem::is_directory(session));
  EXPECT_EQ(std::filesystem::file_size(session / "frames.bin"),
            sizeof(a3_deploy::EvalFramesFileHeaderV1) +
                2 * sizeof(a3_deploy::EvalFrameV1));

  const auto manifest = ReadText(session / "manifest.json");
  EXPECT_NE(manifest.find("\"backend\": \"mujoco_test\""),
            std::string::npos);
  EXPECT_NE(manifest.find("\"name\": \"demo_motion\""),
            std::string::npos);

  const auto events = ReadText(session / "events.jsonl");
  EXPECT_NE(events.find("\"event_type\":\"EPISODE_STARTED\""),
            std::string::npos);
  EXPECT_NE(events.find("\"event_type\":\"EPISODE_ENDED\""),
            std::string::npos);
  EXPECT_NE(events.find("\"outcome\":\"COMPLETED\""), std::string::npos);

  const auto summary = ReadText(session / "summary.json");
  EXPECT_NE(summary.find("\"frames_written\": 2"), std::string::npos);
  EXPECT_NE(summary.find("\"events_written\": 2"), std::string::npos);

  std::filesystem::remove_all(root, ec);
}

TEST(A3EvalRecorder, ExistingSessionIsNeverOverwritten) {
  const auto root = std::filesystem::temp_directory_path() /
                    ("a3_eval_collision_test_" +
                     std::to_string(static_cast<long long>(::getpid())));
  std::error_code ec;
  std::filesystem::remove_all(root, ec);
  const auto existing = root / "session_collision";
  ASSERT_TRUE(std::filesystem::create_directories(existing));
  {
    std::ofstream marker(existing / "keep.txt");
    marker << "must survive";
  }

  a3_deploy::A3EvalRecorderOptions options;
  options.enabled = true;
  options.output_root = root;
  options.session_id = "collision";
  a3_deploy::A3EvalRecorder recorder(std::move(options));
  std::string error;
  ASSERT_TRUE(recorder.Start(&error)) << error;
  recorder.Stop();

  EXPECT_EQ(recorder.SessionDirectory().filename(), "session_collision_1");
  EXPECT_EQ(ReadText(existing / "keep.txt"), "must survive");
  EXPECT_NE(ReadText(recorder.SessionDirectory() / "manifest.json")
                .find("\"session_id\": \"collision_1\""),
            std::string::npos);
  std::filesystem::remove_all(root, ec);
}

TEST(A3EvalRecorder, CompleteMotionOnlyCommitsNaturalFullEpisode) {
  const auto root = std::filesystem::temp_directory_path() /
                    ("a3_eval_complete_test_" +
                     std::to_string(static_cast<long long>(::getpid())));
  std::error_code ec;
  std::filesystem::remove_all(root, ec);

  a3_deploy::A3EvalRecorderOptions options;
  options.enabled = true;
  options.complete_motion_only = true;
  options.output_root = root;
  options.session_id = "complete";
  options.frame_queue_capacity = 16;
  options.flush_frames = 1;
  options.motions.push_back({0, "two_tick_motion", "/tmp/two.csv", 2});

  a3_deploy::A3EvalRecorder recorder(std::move(options));
  std::string error;
  ASSERT_TRUE(recorder.Start(&error)) << error;
  // V6 正式完整数据必须同时具备 state/action/command/reference。
  auto first = CompletePlayingFrame(0, 0);
  auto second = CompletePlayingFrame(1, 1);
  ASSERT_TRUE(recorder.TryPushFrame(first));
  ASSERT_TRUE(recorder.TryPushFrame(second));
  auto terminal = CompletePlayingFrame(2, 1);
  terminal.motion_phase = a3_deploy::MotionPhase::kTerminalHold;
  terminal.valid_mask = 0;
  ASSERT_TRUE(recorder.TryPushFrame(terminal));
  recorder.Stop();

  const auto session = recorder.SessionDirectory();
  EXPECT_EQ(std::filesystem::file_size(session / "frames.bin"),
            sizeof(a3_deploy::EvalFramesFileHeaderV1) +
                2 * sizeof(a3_deploy::EvalFrameV1));
  const auto summary = ReadText(session / "summary.json");
  EXPECT_NE(summary.find("\"episodes_committed\": 1"), std::string::npos);
  EXPECT_NE(summary.find("\"episodes_discarded\": 0"), std::string::npos);
  std::filesystem::remove_all(root, ec);
}

TEST(A3EvalRecorder, WritesOneRawMcapAndMetadataForCompleteEpisode) {
  if (!a3_deploy::A3EvalMcapWriter::AvailableAtBuildTime()) {
    GTEST_SKIP() << "binary built without MCAP support";
  }
#if HAS_A3_EVAL_MCAP
  const auto root = std::filesystem::temp_directory_path() /
                    ("a3_eval_mcap_test_" +
                     std::to_string(static_cast<long long>(::getpid())));
  std::error_code ec;
  std::filesystem::remove_all(root, ec);

  a3_deploy::A3EvalRecorderOptions options;
  options.enabled = true;
  options.complete_motion_only = true;
  options.output_root = root;
  options.session_id = "20260729_161206";
  options.backend_name = "a3";
  options.state_clock_domain = "system";
  options.runtime_config_path = "/tmp/demo.yaml";
  options.policy_model_path = "/tmp/model_step_010000_g1.onnx";
  options.frame_queue_capacity = 16;
  options.flush_frames = 1;
  options.episode_mcap.enabled = true;
  options.episode_mcap.environment = "sim";
  options.episode_mcap.readable_copy_enabled = false;
  options.motions.push_back({0, "two_tick_motion", "/tmp/two.csv", 2});

  a3_deploy::A3EvalRecorder recorder(std::move(options));
  std::string error;
  ASSERT_TRUE(recorder.Start(&error)) << error;
  auto first = CompletePlayingFrame(0, 0);
  first.state_timestamp_ns = 1785300000000000000LL;
  first.control_start_monotonic_ns = 1000000000LL;
  first.control_end_monotonic_ns = 1003000000LL;
  first.infer_start_monotonic_ns = 1000100000LL;
  first.infer_end_monotonic_ns = 1002800000LL;
  first.send_start_monotonic_ns = 1002800000LL;
  first.send_end_monotonic_ns = 1002900000LL;
  first.q[22] = 1.25f;
  first.q_ref[20] = 1.5f;
  first.reference_pelvis_position_m = {1.25f, -0.5f, 0.88f};
  first.reference_pelvis_quat_wxyz = {
      0.9238795f, 0.0f, 0.0f, 0.3826834f};
  first.command_q_des[22] = 1.5f;
  first.command_dq_des[22] = 0.1f;
  first.command_tau_ff[22] = 0.3f;
  first.command_kp[22] = 250.0f;
  first.command_kd[22] = 8.0f;
  first.raw_action[20] = 1.25f;
  auto second = first;
  second.frame_id = 1;
  second.state_tick = 2;
  second.policy_tick = 1;
  second.reference_tick = 1;
  second.state_timestamp_ns += 20000000LL;
  second.control_start_monotonic_ns += 20000000LL;
  second.control_end_monotonic_ns += 20000000LL;
  second.infer_start_monotonic_ns += 20000000LL;
  second.infer_end_monotonic_ns += 20000000LL;
  second.send_start_monotonic_ns += 20000000LL;
  second.send_end_monotonic_ns += 20000000LL;
  ASSERT_TRUE(recorder.TryPushFrame(first));
  ASSERT_TRUE(recorder.TryPushFrame(second));
  auto terminal = second;
  terminal.frame_id = 2;
  terminal.motion_phase = a3_deploy::MotionPhase::kTerminalHold;
  terminal.valid_mask = 0;
  ASSERT_TRUE(recorder.TryPushFrame(terminal));
  recorder.Stop();

  const auto data_dir = recorder.SessionDirectory() / "data";
  ASSERT_TRUE(std::filesystem::is_directory(data_dir));
  std::vector<std::filesystem::path> mcap_files;
  for (const auto& entry : std::filesystem::directory_iterator(data_dir)) {
    if (entry.path().extension() == ".mcap") mcap_files.push_back(entry.path());
  }
  ASSERT_EQ(mcap_files.size(), 1u);
  EXPECT_NE(mcap_files.front().filename().string().find(
                "20260729_161206_sim_model-step-010000_ep000001_m000_"
                "two_tick_motion_rep001.mcap"),
            std::string::npos);

  mcap::McapReader reader;
  const auto open_status = reader.open(mcap_files.front().string());
  ASSERT_TRUE(open_status.ok()) << open_status.message;
  std::set<std::string> topics;
  std::size_t messages = 0;
  bool saw_raw_state_payload = false;
  bool saw_raw_reference_payload = false;
  bool saw_raw_command_payload = false;
  for (const auto& view : reader.readMessages()) {
    ++messages;
    topics.insert(view.channel->topic);
    EXPECT_EQ(view.channel->messageEncoding, "json");
    ASSERT_NE(view.schema, nullptr);
    EXPECT_EQ(view.schema->encoding, "jsonschema");
    YAML::Node schema;
    ASSERT_NO_THROW(schema = YAML::Load(McapBytesToString(view.schema->data)));
    EXPECT_TRUE(schema.IsMap());
    const std::string payload(
        reinterpret_cast<const char*>(view.message.data),
        static_cast<std::size_t>(view.message.dataSize));
    YAML::Node payload_node;
    ASSERT_NO_THROW(payload_node = YAML::Load(payload));
    ASSERT_TRUE(payload_node.IsMap());
    if (view.channel->topic == "/a3/eval/state" &&
        payload.find("\"q\":[") != std::string::npos &&
        payload.find("left_knee_joint") == std::string::npos) {
      saw_raw_state_payload = true;
      EXPECT_EQ(payload_node["q"].size(), 31u);
      EXPECT_EQ(payload_node["dq"].size(), 31u);
      EXPECT_EQ(payload_node["tau_est"].size(), 31u);
    }
    if (view.channel->topic == "/a3/eval/reference") {
      saw_raw_reference_payload = true;
      EXPECT_TRUE(schema["properties"]["pelvis_position_m"]);
      EXPECT_TRUE(schema["properties"]["pelvis_quat_wxyz"]);
      EXPECT_EQ(payload_node["q_ref"].size(), 29u);
      EXPECT_EQ(payload_node["dq_ref"].size(), 29u);
      EXPECT_EQ(payload_node["pelvis_position_m"].size(), 3u);
      EXPECT_EQ(payload_node["pelvis_quat_wxyz"].size(), 4u);
      EXPECT_FLOAT_EQ(
          payload_node["pelvis_position_m"][0].as<float>(), 1.25f);
      EXPECT_NEAR(
          payload_node["pelvis_quat_wxyz"][0].as<float>(), 0.9238795f, 1e-6f);
    }
    if (view.channel->topic == "/a3/eval/command") {
      saw_raw_command_payload = true;
      EXPECT_EQ(payload_node["q_des"].size(), 31u);
      EXPECT_EQ(payload_node["dq_des"].size(), 31u);
      EXPECT_EQ(payload_node["tau_ff"].size(), 31u);
      EXPECT_EQ(payload_node["kp"].size(), 31u);
      EXPECT_EQ(payload_node["kd"].size(), 31u);
      EXPECT_FALSE(payload_node["tau_cmd_expected"]);
      EXPECT_FLOAT_EQ(payload_node["q_des"][22].as<float>(), 1.5f);
      EXPECT_FLOAT_EQ(payload_node["dq_des"][22].as<float>(), 0.1f);
      EXPECT_FLOAT_EQ(payload_node["tau_ff"][22].as<float>(), 0.3f);
      EXPECT_FLOAT_EQ(payload_node["kp"][22].as<float>(), 250.0f);
      EXPECT_FLOAT_EQ(payload_node["kd"][22].as<float>(), 8.0f);
    }
  }
  reader.close();
  EXPECT_EQ(messages, 16u);
  EXPECT_EQ(topics.size(), 8u);
  EXPECT_TRUE(topics.contains("/a3/eval/state"));
  EXPECT_TRUE(topics.contains("/a3/eval/reference"));
  EXPECT_TRUE(topics.contains("/a3/eval/pelvis_imu"));
  EXPECT_TRUE(topics.contains("/a3/eval/status"));
  EXPECT_TRUE(saw_raw_state_payload);
  EXPECT_TRUE(saw_raw_reference_payload);
  EXPECT_TRUE(saw_raw_command_payload);

  const auto session_dir = recorder.SessionDirectory();
  EXPECT_FALSE(std::filesystem::exists(session_dir / "frames.bin"));
  EXPECT_FALSE(std::filesystem::exists(session_dir / "events.jsonl"));
  EXPECT_FALSE(std::filesystem::exists(session_dir / "manifest.json"));
  EXPECT_FALSE(std::filesystem::exists(session_dir / "summary.json"));
  EXPECT_FALSE(std::filesystem::exists(session_dir / "data_read"));

  const auto session_meta = data_dir / "session.meta.yaml";
  ASSERT_TRUE(std::filesystem::is_regular_file(session_meta));
  const YAML::Node session_node = YAML::LoadFile(session_meta.string());
  EXPECT_EQ(session_node["data_schema"].as<std::string>(), "a3_eval_raw_v8");
  EXPECT_EQ(session_node["interrupted_episode_policy"].as<std::string>(),
            "discard");

  std::vector<std::filesystem::path> episode_meta_files;
  for (const auto& entry : std::filesystem::directory_iterator(data_dir)) {
    const auto name = entry.path().filename().string();
    if (name.ends_with(".meta.yaml") && name != "session.meta.yaml") {
      episode_meta_files.push_back(entry.path());
    }
  }
  ASSERT_EQ(episode_meta_files.size(), 1u);
  const YAML::Node episode_meta =
      YAML::LoadFile(episode_meta_files.front().string());
  EXPECT_EQ(episode_meta["data_schema"].as<std::string>(), "a3_eval_raw_v8");
  EXPECT_EQ(episode_meta["episode"]["outcome"].as<std::string>(),
            "COMPLETED");
  EXPECT_EQ(episode_meta["episode"]["frame_count"].as<std::size_t>(), 2u);
  EXPECT_EQ(episode_meta["gains"]["per_frame_selector"].as<std::string>(),
            "/a3/eval/status.gain_profile_id");
  EXPECT_EQ(episode_meta["gains"]["per_frame_values"].as<std::string>(),
            "/a3/eval/command.kp[31], /a3/eval/command.kd[31]");
  const auto profile = episode_meta["gains"]["profiles"][2];
  ASSERT_TRUE(profile.IsMap());
  EXPECT_EQ(profile["kp"].size(), 31u);
  EXPECT_EQ(profile["kd"].size(), 31u);
  EXPECT_FLOAT_EQ(profile["kp"][22].as<float>(), 250.0f);
  EXPECT_FLOAT_EQ(profile["kd"][22].as<float>(), 8.0f);
  EXPECT_EQ(
      episode_meta["signals"]["reference"].as<std::string>(),
      "q_ref[29], dq_ref[29], pelvis_position_m[3], pelvis_quat_wxyz[4]");
  EXPECT_FALSE(episode_meta["offline_derivation"]
                           ["calculated_during_recording"]
                               .as<bool>());
  std::filesystem::remove_all(root, ec);
#endif
}

TEST(A3EvalRecorder, RelativeOutputRootSurvivesWorkingDirectoryChange) {
  if (!a3_deploy::A3EvalMcapWriter::AvailableAtBuildTime()) {
    GTEST_SKIP() << "binary built without MCAP support";
  }
#if HAS_A3_EVAL_MCAP
  const auto root = std::filesystem::temp_directory_path() /
                    ("a3_eval_relative_root_test_" +
                     std::to_string(static_cast<long long>(::getpid())));
  std::error_code ec;
  std::filesystem::remove_all(root, ec);
  ASSERT_TRUE(std::filesystem::create_directories(root / "launch"));
  ASSERT_TRUE(std::filesystem::create_directories(root / "runtime"));

  {
    ScopedCurrentPath launch_cwd(root / "launch");
    a3_deploy::A3EvalRecorderOptions options;
    options.enabled = true;
    options.complete_motion_only = true;
    options.output_root = "eval_records";
    options.session_id = "cwd_change";
    options.frame_queue_capacity = 16;
    options.flush_frames = 1;
    options.episode_mcap.enabled = true;
    options.episode_mcap.environment = "real";
    options.episode_mcap.readable_copy_enabled = false;
    options.motions.push_back({0, "two_tick_motion", "/tmp/two.csv", 2});

    a3_deploy::A3EvalRecorder recorder(std::move(options));
    std::string error;
    ASSERT_TRUE(recorder.Start(&error)) << error;
    EXPECT_TRUE(recorder.SessionDirectory().is_absolute());

    std::filesystem::current_path(root / "runtime");
    ASSERT_TRUE(recorder.TryPushFrame(CompletePlayingFrame(0, 0)));
    ASSERT_TRUE(recorder.TryPushFrame(CompletePlayingFrame(1, 1)));
    auto terminal = CompletePlayingFrame(2, 1);
    terminal.motion_phase = a3_deploy::MotionPhase::kTerminalHold;
    terminal.valid_mask = 0;
    ASSERT_TRUE(recorder.TryPushFrame(terminal));
    recorder.Stop();

    const auto data_dir = recorder.SessionDirectory() / "data";
    EXPECT_TRUE(std::ranges::any_of(
        std::filesystem::directory_iterator(data_dir),
        [](const auto& entry) { return entry.path().extension() == ".mcap"; }));
  }
  std::filesystem::remove_all(root, ec);
#endif
}

TEST(A3EvalRecorder, CompleteMotionOnlyDiscardsInterruptedEpisode) {
  const auto root = std::filesystem::temp_directory_path() /
                    ("a3_eval_abort_test_" +
                     std::to_string(static_cast<long long>(::getpid())));
  std::error_code ec;
  std::filesystem::remove_all(root, ec);

  a3_deploy::A3EvalRecorderOptions options;
  options.enabled = true;
  options.complete_motion_only = true;
  options.output_root = root;
  options.session_id = "abort";
  options.frame_queue_capacity = 16;
  options.flush_frames = 1;
  options.episode_mcap.enabled =
      a3_deploy::A3EvalMcapWriter::AvailableAtBuildTime();
  options.episode_mcap.environment = "sim";
  options.episode_mcap.readable_copy_enabled = false;
  options.motions.push_back({0, "three_tick_motion", "/tmp/three.csv", 3});

  a3_deploy::A3EvalRecorder recorder(std::move(options));
  std::string error;
  ASSERT_TRUE(recorder.Start(&error)) << error;
  ASSERT_TRUE(recorder.TryPushFrame(CompletePlayingFrame(0, 0)));
  auto paused = CompletePlayingFrame(1, 0);
  paused.motion_phase = a3_deploy::MotionPhase::kPaused;
  paused.valid_mask = 0;
  ASSERT_TRUE(recorder.TryPushFrame(paused));
  recorder.Stop();

  const auto session = recorder.SessionDirectory();
  if (a3_deploy::A3EvalMcapWriter::AvailableAtBuildTime()) {
    EXPECT_FALSE(std::filesystem::exists(session / "frames.bin"));
    EXPECT_FALSE(std::filesystem::exists(session / "events.jsonl"));
    EXPECT_FALSE(std::filesystem::exists(session / "data_read"));
    EXPECT_TRUE(std::filesystem::is_regular_file(
        session / "data" / "session.meta.yaml"));
    EXPECT_TRUE(std::filesystem::directory_iterator(session / "data") !=
                std::filesystem::directory_iterator{});
    EXPECT_TRUE(std::ranges::none_of(
        std::filesystem::directory_iterator(session / "data"),
        [](const auto& entry) { return entry.path().extension() == ".mcap"; }));
  } else {
    EXPECT_EQ(std::filesystem::file_size(session / "frames.bin"),
              sizeof(a3_deploy::EvalFramesFileHeaderV1));
  }
  std::filesystem::remove_all(root, ec);
}

TEST(A3EvalRecorder, CompleteMotionOnlyDiscardsEpisodeOnProcessStop) {
  const auto root = std::filesystem::temp_directory_path() /
                    ("a3_eval_stop_test_" +
                     std::to_string(static_cast<long long>(::getpid())));
  std::error_code ec;
  std::filesystem::remove_all(root, ec);

  a3_deploy::A3EvalRecorderOptions options;
  options.enabled = true;
  options.complete_motion_only = true;
  options.output_root = root;
  options.session_id = "stop";
  options.frame_queue_capacity = 16;
  options.episode_mcap.enabled =
      a3_deploy::A3EvalMcapWriter::AvailableAtBuildTime();
  options.episode_mcap.environment = "sim";
  options.episode_mcap.readable_copy_enabled = false;
  options.motions.push_back({0, "three_tick_motion", "/tmp/three.csv", 3});

  a3_deploy::A3EvalRecorder recorder(std::move(options));
  std::string error;
  ASSERT_TRUE(recorder.Start(&error)) << error;
  ASSERT_TRUE(recorder.TryPushFrame(CompletePlayingFrame(0, 0)));
  recorder.Stop();

  const auto session = recorder.SessionDirectory();
  if (a3_deploy::A3EvalMcapWriter::AvailableAtBuildTime()) {
    EXPECT_FALSE(std::filesystem::exists(session / "frames.bin"));
    EXPECT_FALSE(std::filesystem::exists(session / "events.jsonl"));
    EXPECT_FALSE(std::filesystem::exists(session / "data_read"));
    EXPECT_TRUE(std::ranges::none_of(
        std::filesystem::directory_iterator(session / "data"),
        [](const auto& entry) { return entry.path().extension() == ".mcap"; }));
  } else {
    EXPECT_EQ(std::filesystem::file_size(session / "frames.bin"),
              sizeof(a3_deploy::EvalFramesFileHeaderV1));
  }
  std::filesystem::remove_all(root, ec);
}
