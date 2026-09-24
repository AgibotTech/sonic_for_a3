// Copyright (c) 2026, AgiBot Inc. All rights reserved.
#include "a3_deploy/a3_eval_mcap_writer.hpp"

#include "robot_io/a3_layout_extra.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <locale>
#include <map>
#include <sstream>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>

#include <yaml-cpp/yaml.h>

#ifndef HAS_A3_EVAL_MCAP
#define HAS_A3_EVAL_MCAP 0
#endif

#if HAS_A3_EVAL_MCAP
#define MCAP_IMPLEMENTATION
#include <mcap/mcap.hpp>
#endif

namespace a3_deploy {

namespace {

constexpr std::string_view kNamedJointStateSchema = R"json({
  "title":"A3NamedJointState","type":"object",
  "properties":{
    "frame_id":{"type":"integer"},"timestamp_ns":{"type":"integer"},
    "joints":{"type":"array","items":{"type":"object","properties":{
      "name":{"type":"string"},"sequence":{"type":"integer"},
      "position":{"type":["number","null"]},
      "velocity":{"type":["number","null"]},
      "effort":{"type":["number","null"]}
    },"required":["name","sequence"]}}
  },"required":["frame_id","timestamp_ns","joints"]
})json";

constexpr std::string_view kRawStateSchema = R"json({
  "title":"A3RawRobotState","type":"object",
  "properties":{
    "frame_id":{"type":"integer"},"timestamp_ns":{"type":"integer"},
    "q":{"type":"array","minItems":31,"maxItems":31,"items":{"type":["number","null"]}},
    "dq":{"type":"array","minItems":31,"maxItems":31,"items":{"type":["number","null"]}},
    "tau_est":{"type":"array","minItems":31,"maxItems":31,"items":{"type":["number","null"]}}
  },"required":["frame_id","timestamp_ns","q","dq","tau_est"]
})json";

constexpr std::string_view kRawCommandSchema = R"json({
  "title":"A3RawRobotCommand","type":"object",
  "properties":{
    "frame_id":{"type":"integer"},"timestamp_ns":{"type":"integer"},
    "q_des":{"type":"array","minItems":31,"maxItems":31,"items":{"type":["number","null"]}},
    "dq_des":{"type":"array","minItems":31,"maxItems":31,"items":{"type":["number","null"]}},
    "tau_ff":{"type":"array","minItems":31,"maxItems":31,"items":{"type":["number","null"]}},
    "kp":{"type":"array","minItems":31,"maxItems":31,"items":{"type":["number","null"]}},
    "kd":{"type":"array","minItems":31,"maxItems":31,"items":{"type":["number","null"]}}
  },"required":["frame_id","timestamp_ns","q_des","dq_des","tau_ff","kp","kd"]
})json";

constexpr std::string_view kRawPolicySchema = R"json({
  "title":"A3RawPolicyOutput","type":"object",
  "properties":{
    "frame_id":{"type":"integer"},"timestamp_ns":{"type":"integer"},
    "raw_action":{"type":"array","minItems":29,"maxItems":29,"items":{"type":["number","null"]}}
  },"required":["frame_id","timestamp_ns","raw_action"]
})json";

constexpr std::string_view kRawReferenceSchema = R"json({
  "title":"A3RawMotionReference","type":"object",
  "properties":{
    "frame_id":{"type":"integer"},"timestamp_ns":{"type":"integer"},
    "q_ref":{"type":"array","minItems":29,"maxItems":29,"items":{"type":["number","null"]}},
    "dq_ref":{"type":"array","minItems":29,"maxItems":29,"items":{"type":["number","null"]}},
    "pelvis_position_m":{"type":"array","minItems":3,"maxItems":3,"items":{"type":["number","null"]}},
    "pelvis_quat_wxyz":{"type":"array","minItems":4,"maxItems":4,"items":{"type":["number","null"]}}
  },"required":["frame_id","timestamp_ns","q_ref","dq_ref","pelvis_position_m","pelvis_quat_wxyz"]
})json";

constexpr std::string_view kRawImuSchema = R"json({
  "title":"A3RawImu","type":"object",
  "properties":{
    "frame_id":{"type":"integer"},"timestamp_ns":{"type":"integer"},
    "quat_wxyz":{"type":"array","minItems":4,"maxItems":4,"items":{"type":["number","null"]}},
    "gyro":{"type":"array","minItems":3,"maxItems":3,"items":{"type":["number","null"]}}
  },"required":["frame_id","timestamp_ns","quat_wxyz","gyro"]
})json";

constexpr std::string_view kImuSchema = R"json({
  "title":"A3Imu","type":"object",
  "properties":{
    "frame_id":{"type":"integer"},"timestamp_ns":{"type":"integer"},
    "quaternion_wxyz":{"type":"object","properties":{
      "w":{"type":["number","null"]},"x":{"type":["number","null"]},
      "y":{"type":["number","null"]},"z":{"type":["number","null"]}
    }},
    "gyro_rad_s":{"type":"object","properties":{
      "x":{"type":["number","null"]},"y":{"type":["number","null"]},
      "z":{"type":["number","null"]}
    }}
  },"required":["frame_id","timestamp_ns","quaternion_wxyz","gyro_rad_s"]
})json";

constexpr std::string_view kTimingSchema = R"json({
  "title":"A3EvalTiming","type":"object",
  "properties":{
    "frame_id":{"type":"integer"},"timestamp_ns":{"type":"integer"},
    "state_timestamp_ns":{"type":"integer"},
    "state_data_ready_ns":{"type":"integer"},
    "state_sync_ready_ns":{"type":"integer"},
    "control_start_monotonic_ns":{"type":"integer"},
    "infer_start_monotonic_ns":{"type":"integer"},
    "infer_end_monotonic_ns":{"type":"integer"},
    "send_start_monotonic_ns":{"type":"integer"},
    "send_end_monotonic_ns":{"type":"integer"},
    "control_end_monotonic_ns":{"type":"integer"},
    "inference_ms":{"type":"number"},"send_ms":{"type":"number"},
    "control_ms":{"type":"number"}
  },"required":["frame_id","timestamp_ns"]
})json";

constexpr std::string_view kStatusSchema = R"json({
  "title":"A3EvalStatus","type":"object",
  "properties":{
    "frame_id":{"type":"integer"},"timestamp_ns":{"type":"integer"},
    "state_tick":{"type":"integer"},"policy_tick":{"type":"integer"},
    "reference_tick":{"type":"integer"},"episode_id":{"type":"integer"},
    "active_motion_id":{"type":"integer"},"selected_motion_id":{"type":"integer"},
    "deploy_mode":{"type":"string"},"motion_phase":{"type":"string"},
    "command_source":{"type":"string"},"gain_profile_id":{"type":"integer"},
    "valid_mask":{"type":"integer"},"quality_flags":{"type":"integer"}
  },"required":["frame_id","timestamp_ns","episode_id"]
})json";

std::string JsonEscape(std::string_view value) {
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

std::ostringstream MakeJsonStream() {
  std::ostringstream out;
  out.imbue(std::locale::classic());
  out << std::setprecision(9);
  return out;
}

void AppendNumber(std::ostringstream& out, float value) {
  if (std::isfinite(value)) {
    out << value;
  } else {
    out << "null";
  }
}

std::string NormalizePrefix(std::string prefix) {
  if (prefix.empty()) return "/a3/eval";
  if (prefix.front() != '/') prefix.insert(prefix.begin(), '/');
  while (prefix.size() > 1 && prefix.back() == '/') prefix.pop_back();
  return prefix;
}

std::string SanitizeToken(std::string_view value, std::size_t max_size = 96) {
  std::string result;
  result.reserve(std::min(value.size(), max_size));
  bool previous_separator = false;
  for (const unsigned char c : value) {
    const bool ascii_alnum =
        (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
        (c >= '0' && c <= '9');
    if (ascii_alnum || c == '-') {
      result.push_back(static_cast<char>(c));
      previous_separator = false;
    } else if (!previous_separator && !result.empty()) {
      result.push_back('_');
      previous_separator = true;
    }
    if (result.size() >= max_size) break;
  }
  while (!result.empty() && result.back() == '_') result.pop_back();
  return result.empty() ? "unknown" : result;
}

std::string DeriveModelTag(std::string_view configured,
                           std::string_view model_path) {
  if (!configured.empty()) return SanitizeToken(configured, 64);
  const auto marker = model_path.find("model_step_");
  if (marker != std::string_view::npos) {
    const auto digits_begin = marker + std::string_view("model_step_").size();
    auto digits_end = digits_begin;
    while (digits_end < model_path.size() && model_path[digits_end] >= '0' &&
           model_path[digits_end] <= '9') {
      ++digits_end;
    }
    if (digits_end > digits_begin) {
      return "model-step-" +
             std::string(model_path.substr(digits_begin, digits_end - digits_begin));
    }
  }
  const auto first_model = model_path.substr(0, model_path.find(" + "));
  return SanitizeToken(std::filesystem::path(first_model).stem().string(), 64);
}

std::string MakeEpisodeFilename(const A3EvalMcapOptions& options,
                                const A3EvalMcapSessionMetadata& session,
                                const A3EvalMcapEpisodeMetadata& episode) {
  std::ostringstream out;
  out << SanitizeToken(session.session_id, 40) << '_'
      << SanitizeToken(options.environment, 16) << '_'
      << DeriveModelTag(options.model_tag, session.policy_model_path)
      << "_ep" << std::setw(6) << std::setfill('0') << episode.episode_id
      << "_m" << std::setw(3) << std::setfill('0') << episode.motion_id
      << '_' << SanitizeToken(episode.motion_name)
      << "_rep" << std::setw(3) << std::setfill('0')
      << episode.motion_repeat_index << ".mcap";
  return out.str();
}

std::uint64_t FrameEpochNs(const A3EvalMcapSessionMetadata& session,
                           const EvalFrameV1& frame,
                           std::size_t frame_index) {
  // A3 backend 的 state timestamp 属于 system_clock 时优先使用源时间；
  // 其它 backend 则用 session anchor 将 control monotonic 映射到 Unix epoch。
  if (session.state_clock_domain == "system" && frame.state_timestamp_ns > 0) {
    return static_cast<std::uint64_t>(frame.state_timestamp_ns);
  }
  if (session.system_time_anchor_ns > 0 &&
      session.monotonic_time_anchor_ns > 0 &&
      frame.control_start_monotonic_ns > 0) {
    const auto delta =
        frame.control_start_monotonic_ns - session.monotonic_time_anchor_ns;
    if (delta >= -session.system_time_anchor_ns) {
      return static_cast<std::uint64_t>(session.system_time_anchor_ns + delta);
    }
  }
  const double hz = session.policy_hz > 0.0 ? session.policy_hz : 50.0;
  const auto step_ns = static_cast<std::uint64_t>(1.0e9 / hz);
  return static_cast<std::uint64_t>(std::max<std::int64_t>(
             0, session.system_time_anchor_ns)) +
         static_cast<std::uint64_t>(frame_index) * step_ns;
}

double DurationMs(std::int64_t start_ns, std::int64_t end_ns) {
  if (start_ns <= 0 || end_ns < start_ns) return 0.0;
  return static_cast<double>(end_ns - start_ns) / 1.0e6;
}

template <std::size_t N>
std::string NamedJointJson(
    const EvalFrameV1& frame, std::uint64_t timestamp_ns,
    const std::vector<std::string>& names,
    const std::array<float, N>* positions,
    const std::array<float, N>* velocities,
    const std::array<float, N>* efforts) {
  auto out = MakeJsonStream();
  out << "{\"frame_id\":" << frame.frame_id
      << ",\"timestamp_ns\":" << timestamp_ns << ",\"joints\":[";
  for (std::size_t i = 0; i < N; ++i) {
    if (i != 0) out << ',';
    out << "{\"name\":\"" << JsonEscape(names.at(i))
        << "\",\"sequence\":" << i;
    if (positions != nullptr) {
      out << ",\"position\":";
      AppendNumber(out, (*positions)[i]);
    }
    if (velocities != nullptr) {
      out << ",\"velocity\":";
      AppendNumber(out, (*velocities)[i]);
    }
    if (efforts != nullptr) {
      out << ",\"effort\":";
      AppendNumber(out, (*efforts)[i]);
    }
    out << '}';
  }
  out << "]}";
  return out.str();
}

template <std::size_t N>
void AppendArray(std::ostringstream& out, const std::array<float, N>& values) {
  out << '[';
  for (std::size_t i = 0; i < N; ++i) {
    if (i != 0) out << ',';
    AppendNumber(out, values[i]);
  }
  out << ']';
}

std::string RawStateJson(const EvalFrameV1& frame,
                         std::uint64_t timestamp_ns) {
  auto out = MakeJsonStream();
  out << "{\"frame_id\":" << frame.frame_id
      << ",\"timestamp_ns\":" << timestamp_ns << ",\"q\":";
  AppendArray(out, frame.q);
  out << ",\"dq\":";
  AppendArray(out, frame.dq);
  out << ",\"tau_est\":";
  AppendArray(out, frame.tau_est);
  out << '}';
  return out.str();
}

std::string RawCommandJson(const EvalFrameV1& frame,
                           std::uint64_t timestamp_ns) {
  auto out = MakeJsonStream();
  out << "{\"frame_id\":" << frame.frame_id
      << ",\"timestamp_ns\":" << timestamp_ns << ",\"q_des\":";
  AppendArray(out, frame.command_q_des);
  out << ",\"dq_des\":";
  AppendArray(out, frame.command_dq_des);
  out << ",\"tau_ff\":";
  AppendArray(out, frame.command_tau_ff);
  out << ",\"kp\":";
  AppendArray(out, frame.command_kp);
  out << ",\"kd\":";
  AppendArray(out, frame.command_kd);
  out << '}';
  return out.str();
}

std::string RawPolicyJson(const EvalFrameV1& frame,
                          std::uint64_t timestamp_ns) {
  auto out = MakeJsonStream();
  out << "{\"frame_id\":" << frame.frame_id
      << ",\"timestamp_ns\":" << timestamp_ns << ",\"raw_action\":";
  AppendArray(out, frame.raw_action);
  out << '}';
  return out.str();
}

std::string RawReferenceJson(const EvalFrameV1& frame,
                             std::uint64_t timestamp_ns) {
  auto out = MakeJsonStream();
  out << "{\"frame_id\":" << frame.frame_id
      << ",\"timestamp_ns\":" << timestamp_ns << ",\"q_ref\":";
  AppendArray(out, frame.q_ref);
  out << ",\"dq_ref\":";
  AppendArray(out, frame.dq_ref);
  out << ",\"pelvis_position_m\":";
  AppendArray(out, frame.reference_pelvis_position_m);
  out << ",\"pelvis_quat_wxyz\":";
  AppendArray(out, frame.reference_pelvis_quat_wxyz);
  out << '}';
  return out.str();
}

std::string RawImuJson(const EvalFrameV1& frame,
                       std::uint64_t timestamp_ns,
                       const std::array<float, 4>& quat,
                       const std::array<float, 3>& gyro) {
  auto out = MakeJsonStream();
  out << "{\"frame_id\":" << frame.frame_id
      << ",\"timestamp_ns\":" << timestamp_ns << ",\"quat_wxyz\":";
  AppendArray(out, quat);
  out << ",\"gyro\":";
  AppendArray(out, gyro);
  out << '}';
  return out.str();
}

std::string JsonStringArray(const std::vector<std::string>& values) {
  std::ostringstream out;
  out << '[';
  for (std::size_t i = 0; i < values.size(); ++i) {
    if (i != 0) out << ',';
    out << '"' << JsonEscape(values[i]) << '"';
  }
  out << ']';
  return out.str();
}

std::string JsonIntegerArray(
    const std::array<int, robot_io::kA3PolicyDof>& values) {
  std::ostringstream out;
  out << '[';
  for (std::size_t i = 0; i < values.size(); ++i) {
    if (i != 0) out << ',';
    out << values[i];
  }
  out << ']';
  return out.str();
}

struct GainProfileSnapshot {
  std::uint8_t id = 0;
  CommandSource command_source = CommandSource::kNone;
  std::array<float, 31> kp{};
  std::array<float, 31> kd{};
};

using GainProfileMap = std::map<std::uint8_t, GainProfileSnapshot>;

bool SameGainValues(const GainProfileSnapshot& profile,
                    const EvalFrameV1& frame) noexcept {
  return profile.kp == frame.command_kp && profile.kd == frame.command_kd;
}

bool CollectGainProfiles(const std::vector<EvalFrameV1>& frames,
                         GainProfileMap& profiles,
                         std::string& error) {
  profiles.clear();
  for (const auto& frame : frames) {
    if ((frame.valid_mask & kRobotCommandValid) == 0) {
      error = "完整 episode 中存在无效 RobotCommand 帧";
      return false;
    }
    auto [it, inserted] = profiles.try_emplace(frame.gain_profile_id);
    if (inserted) {
      it->second.id = frame.gain_profile_id;
      it->second.command_source = frame.command_source;
      it->second.kp = frame.command_kp;
      it->second.kd = frame.command_kd;
    } else if (!SameGainValues(it->second, frame)) {
      error = "同一 gain_profile_id 在 episode 内对应了不同 kp/kd";
      return false;
    }
  }
  if (profiles.empty()) {
    error = "完整 episode 没有 gain profile";
    return false;
  }
  return true;
}

std::string GainProfilesJson(const GainProfileMap& profiles) {
  auto out = MakeJsonStream();
  out << '{';
  bool first_profile = true;
  for (const auto& [id, profile] : profiles) {
    if (!first_profile) out << ',';
    first_profile = false;
    out << '"' << static_cast<unsigned>(id) << "\":{\"command_source\":\""
        << CommandSourceName(profile.command_source) << "\",\"kp\":";
    AppendArray(out, profile.kp);
    out << ",\"kd\":";
    AppendArray(out, profile.kd);
    out << '}';
  }
  out << '}';
  return out.str();
}

template <std::size_t N>
YAML::Node FloatArrayYaml(const std::array<float, N>& values) {
  YAML::Node node(YAML::NodeType::Sequence);
  for (const float value : values) node.push_back(value);
  node.SetStyle(YAML::EmitterStyle::Flow);
  return node;
}

YAML::Node StringArrayYaml(const std::vector<std::string>& values) {
  YAML::Node node(YAML::NodeType::Sequence);
  for (const auto& value : values) node.push_back(value);
  node.SetStyle(YAML::EmitterStyle::Flow);
  return node;
}

bool WriteEpisodeMetadataYaml(
    const A3EvalMcapOptions& options,
    const std::filesystem::path& session_dir,
    const std::filesystem::path& mcap_relative_path,
    std::uint64_t mcap_file_size,
    const A3EvalMcapSessionMetadata& session,
    const A3EvalMcapEpisodeMetadata& episode,
    const std::vector<EvalFrameV1>& frames,
    const GainProfileMap& profiles,
    std::filesystem::path& metadata_relative_path,
    std::uint64_t& metadata_file_size,
    std::string& error) {
  const auto& names31 = robot_io::MakeA3Layout31().names;
  std::vector<std::string> names29;
  names29.reserve(robot_io::kA3PolicyDof);
  for (const int sdk_index : robot_io::kA3PolicyToSdkIdx) {
    names29.push_back(names31.at(static_cast<std::size_t>(sdk_index)));
  }

  YAML::Node root;
  root["meta_schema_version"] = 1;
  root["data_schema"] = "a3_eval_raw_v8";
  root["mcap"]["file"] = mcap_relative_path.filename().string();
  root["mcap"]["size_bytes"] = mcap_file_size;

  root["session"]["session_id"] = session.session_id;
  root["session"]["environment"] = options.environment;
  root["session"]["backend"] = session.backend_name;
  root["session"]["state_clock_domain"] = session.state_clock_domain;
  root["session"]["runtime_config_path"] = session.runtime_config_path;
  root["session"]["policy_model_path"] = session.policy_model_path;
  root["session"]["model_tag"] =
      DeriveModelTag(options.model_tag, session.policy_model_path);
  root["session"]["policy_hz"] = session.policy_hz;
  root["session"]["state_hz"] = session.state_hz;

  root["episode"]["episode_id"] = episode.episode_id;
  root["episode"]["motion_id"] = episode.motion_id;
  root["episode"]["motion_name"] = episode.motion_name;
  root["episode"]["motion_source_path"] = episode.motion_source_path;
  root["episode"]["motion_repeat_index"] = episode.motion_repeat_index;
  root["episode"]["outcome"] = "COMPLETED";
  root["episode"]["termination_reason"] = "END_OF_REFERENCE";
  root["episode"]["frame_count"] = frames.size();
  root["episode"]["reference_tick_first"] = frames.front().reference_tick;
  root["episode"]["reference_tick_last"] = frames.back().reference_tick;
  root["episode"]["timestamp_first_ns"] =
      FrameEpochNs(session, frames.front(), 0);
  root["episode"]["timestamp_last_ns"] =
      FrameEpochNs(session, frames.back(), frames.size() - 1);

  root["joint_layout"]["order_31"] = StringArrayYaml(names31);
  root["joint_layout"]["order_29"] = StringArrayYaml(names29);
  YAML::Node policy_to_sdk(YAML::NodeType::Sequence);
  for (const int index : robot_io::kA3PolicyToSdkIdx) {
    policy_to_sdk.push_back(index);
  }
  policy_to_sdk.SetStyle(YAML::EmitterStyle::Flow);
  root["joint_layout"]["policy_to_sdk_index_31"] = policy_to_sdk;

  root["signals"]["state"] = "q[31], dq[31], tau_est[31]";
  root["signals"]["command"] =
      "q_des[31], dq_des[31], tau_ff[31], kp[31], kd[31]";
  root["signals"]["policy"] = "raw_action[29]";
  root["signals"]["reference"] =
      "q_ref[29], dq_ref[29], pelvis_position_m[3], pelvis_quat_wxyz[4]";
  root["signals"]["imu"] =
      "pelvis/torso quat_wxyz[4], gyro[3]";

  root["gains"]["per_frame_selector"] =
      "/a3/eval/status.gain_profile_id";
  root["gains"]["per_frame_values"] =
      "/a3/eval/command.kp[31], /a3/eval/command.kd[31]";
  YAML::Node profile_nodes(YAML::NodeType::Map);
  for (const auto& [id, profile] : profiles) {
    YAML::Node profile_node;
    profile_node["command_source"] =
        CommandSourceName(profile.command_source);
    profile_node["kp"] = FloatArrayYaml(profile.kp);
    profile_node["kd"] = FloatArrayYaml(profile.kd);
    profile_nodes[static_cast<unsigned>(id)] = profile_node;
  }
  root["gains"]["profiles"] = profile_nodes;

  root["offline_derivation"]["tau_cmd_expected_formula"] =
      "tau_ff + kp*(q_des-q) + kd*(dq_des-dq)";
  root["offline_derivation"]["calculated_during_recording"] = false;
  root["offline_derivation"]["driver_internal_limit_included"] = false;

  YAML::Emitter emitter;
  emitter.SetIndent(2);
  emitter << root;
  if (!emitter.good()) {
    error = "生成 episode metadata YAML 失败";
    return false;
  }

  const auto stem = mcap_relative_path.stem().string();
  metadata_relative_path =
      mcap_relative_path.parent_path() / (stem + ".meta.yaml");
  const auto final_path = session_dir / metadata_relative_path;
  const auto partial_path = final_path.string() + ".partial";
  std::error_code ec;
  if (std::filesystem::exists(final_path, ec) ||
      std::filesystem::exists(partial_path, ec)) {
    error = "episode metadata 目标已存在，拒绝覆盖: " + final_path.string();
    return false;
  }
  {
    std::ofstream out(partial_path, std::ios::trunc);
    if (!out) {
      error = "无法创建 episode metadata 临时文件";
      return false;
    }
    out << emitter.c_str() << '\n';
    out.flush();
    if (!out) {
      error = "写入 episode metadata 失败";
      out.close();
      std::filesystem::remove(partial_path, ec);
      return false;
    }
  }
  std::filesystem::rename(partial_path, final_path, ec);
  if (ec) {
    error = "提交 episode metadata 失败: " + ec.message();
    std::filesystem::remove(partial_path, ec);
    return false;
  }
  metadata_file_size = std::filesystem::file_size(final_path, ec);
  if (ec) metadata_file_size = 0;
  return true;
}

std::string JointCompareSchema(const std::vector<std::string>& names31) {
  std::ostringstream out;
  out << R"json({"title":"A3JointCompare","type":"object","properties":{"frame_id":{"type":"integer"},"timestamp_ns":{"type":"integer"},"joints":{"type":"object","properties":{)json";
  for (std::size_t i = 0; i < names31.size(); ++i) {
    if (i != 0) out << ',';
    out << '"' << JsonEscape(names31[i]) << R"json(":{"type":"object","properties":{"q":{"type":["number","null"]},"dq":{"type":["number","null"]},"tau_est":{"type":["number","null"]},"command_q_des":{"type":["number","null"]},"command_dq_des":{"type":["number","null"]},"command_tau_ff":{"type":["number","null"]},"raw_action":{"type":["number","null"]},"q_ref":{"type":["number","null"]},"dq_ref":{"type":["number","null"]}}})json";
  }
  out << R"json(}}},"required":["frame_id","timestamp_ns","joints"]})json";
  return out.str();
}

std::string JointCompareJson(const EvalFrameV1& frame,
                             std::uint64_t timestamp_ns,
                             const std::vector<std::string>& names31) {
  std::array<int, robot_io::kA3Dof> sdk_to_policy{};
  sdk_to_policy.fill(-1);
  for (std::size_t policy = 0;
       policy < robot_io::kA3PolicyToSdkIdx.size(); ++policy) {
    sdk_to_policy[static_cast<std::size_t>(
        robot_io::kA3PolicyToSdkIdx[policy])] = static_cast<int>(policy);
  }

  auto out = MakeJsonStream();
  out << "{\"frame_id\":" << frame.frame_id
      << ",\"timestamp_ns\":" << timestamp_ns << ",\"joints\":{";
  for (std::size_t sdk = 0; sdk < names31.size(); ++sdk) {
    if (sdk != 0) out << ',';
    out << '"' << JsonEscape(names31[sdk]) << "\":{\"q\":";
    AppendNumber(out, frame.q[sdk]);
    out << ",\"dq\":";
    AppendNumber(out, frame.dq[sdk]);
    out << ",\"tau_est\":";
    AppendNumber(out, frame.tau_est[sdk]);
    out << ",\"command_q_des\":";
    AppendNumber(out, frame.command_q_des[sdk]);
    out << ",\"command_dq_des\":";
    AppendNumber(out, frame.command_dq_des[sdk]);
    out << ",\"command_tau_ff\":";
    AppendNumber(out, frame.command_tau_ff[sdk]);
    const int policy = sdk_to_policy[sdk];
    if (policy >= 0) {
      const auto index = static_cast<std::size_t>(policy);
      out << ",\"raw_action\":";
      AppendNumber(out, frame.raw_action[index]);
      out << ",\"q_ref\":";
      AppendNumber(out, frame.q_ref[index]);
      out << ",\"dq_ref\":";
      AppendNumber(out, frame.dq_ref[index]);
    } else {
      out << ",\"raw_action\":null,\"q_ref\":null,\"dq_ref\":null";
    }
    out << '}';
  }
  out << "}}";
  return out.str();
}

std::string ImuJson(const EvalFrameV1& frame, std::uint64_t timestamp_ns,
                    const std::array<float, 4>& quat,
                    const std::array<float, 3>& gyro) {
  auto out = MakeJsonStream();
  out << "{\"frame_id\":" << frame.frame_id
      << ",\"timestamp_ns\":" << timestamp_ns
      << ",\"quaternion_wxyz\":{\"w\":";
  AppendNumber(out, quat[0]);
  out << ",\"x\":";
  AppendNumber(out, quat[1]);
  out << ",\"y\":";
  AppendNumber(out, quat[2]);
  out << ",\"z\":";
  AppendNumber(out, quat[3]);
  out << "},\"gyro_rad_s\":{\"x\":";
  AppendNumber(out, gyro[0]);
  out << ",\"y\":";
  AppendNumber(out, gyro[1]);
  out << ",\"z\":";
  AppendNumber(out, gyro[2]);
  out << "}}";
  return out.str();
}

std::string TimingJson(const EvalFrameV1& frame,
                       std::uint64_t timestamp_ns) {
  auto out = MakeJsonStream();
  out << "{\"frame_id\":" << frame.frame_id
      << ",\"timestamp_ns\":" << timestamp_ns
      << ",\"state_timestamp_ns\":" << frame.state_timestamp_ns
      << ",\"state_data_ready_ns\":" << frame.state_data_ready_ns
      << ",\"state_sync_ready_ns\":" << frame.state_sync_ready_ns
      << ",\"control_start_monotonic_ns\":"
      << frame.control_start_monotonic_ns
      << ",\"infer_start_monotonic_ns\":"
      << frame.infer_start_monotonic_ns
      << ",\"infer_end_monotonic_ns\":" << frame.infer_end_monotonic_ns
      << ",\"send_start_monotonic_ns\":" << frame.send_start_monotonic_ns
      << ",\"send_end_monotonic_ns\":" << frame.send_end_monotonic_ns
      << ",\"control_end_monotonic_ns\":" << frame.control_end_monotonic_ns
      << ",\"inference_ms\":"
      << DurationMs(frame.infer_start_monotonic_ns,
                    frame.infer_end_monotonic_ns)
      << ",\"send_ms\":"
      << DurationMs(frame.send_start_monotonic_ns,
                    frame.send_end_monotonic_ns)
      << ",\"control_ms\":"
      << DurationMs(frame.control_start_monotonic_ns,
                    frame.control_end_monotonic_ns)
      << '}';
  return out.str();
}

std::int64_t TickForJson(std::uint64_t tick) {
  return tick == kA3EvalNoTick ? -1 : static_cast<std::int64_t>(tick);
}

std::string StatusJson(const EvalFrameV1& frame,
                       std::uint64_t timestamp_ns) {
  auto out = MakeJsonStream();
  out << "{\"frame_id\":" << frame.frame_id
      << ",\"timestamp_ns\":" << timestamp_ns
      << ",\"state_tick\":" << TickForJson(frame.state_tick)
      << ",\"policy_tick\":" << TickForJson(frame.policy_tick)
      << ",\"reference_tick\":" << TickForJson(frame.reference_tick)
      << ",\"episode_id\":" << frame.episode_id
      << ",\"active_motion_id\":" << frame.active_motion_id
      << ",\"selected_motion_id\":" << frame.selected_motion_id
      << ",\"deploy_mode\":\"" << DeployModeName(frame.deploy_mode)
      << "\",\"motion_phase\":\"" << MotionPhaseName(frame.motion_phase)
      << "\",\"command_source\":\""
      << CommandSourceName(frame.command_source)
      << "\",\"gain_profile_id\":"
      << static_cast<unsigned>(frame.gain_profile_id)
      << ",\"valid_mask\":" << frame.valid_mask
      << ",\"quality_flags\":" << frame.quality_flags << '}';
  return out.str();
}

#if HAS_A3_EVAL_MCAP
bool WriteJsonMessage(mcap::McapWriter& writer, mcap::ChannelId channel_id,
                      std::uint32_t sequence, std::uint64_t timestamp_ns,
                      const std::string& json, std::string* error) {
  mcap::Message message;
  message.channelId = channel_id;
  message.sequence = sequence;
  message.logTime = timestamp_ns;
  message.publishTime = timestamp_ns;
  message.data = reinterpret_cast<const std::byte*>(json.data());
  message.dataSize = json.size();
  const auto status = writer.write(message);
  if (status.ok()) return true;
  if (error) *error = status.message;
  return false;
}

enum class McapRepresentation { kRawArrays, kReadableNamed };

struct SingleMcapWriteResult {
  bool ok = false;
  std::filesystem::path relative_path;
  std::uint64_t file_size = 0;
  std::string error;
};

SingleMcapWriteResult WriteEpisodeMcapFile(
    const A3EvalMcapOptions& options,
    const std::filesystem::path& session_dir,
    const std::filesystem::path& relative_directory,
    const std::string& filename,
    const A3EvalMcapSessionMetadata& session,
    const A3EvalMcapEpisodeMetadata& episode,
    const std::vector<EvalFrameV1>& frames,
    const GainProfileMap& gain_profiles,
    McapRepresentation representation) noexcept {
  SingleMcapWriteResult result;
  const auto data_dir = session_dir / relative_directory;
  std::error_code ec;
  std::filesystem::create_directories(data_dir, ec);
  if (ec) {
    result.error =
        "创建 MCAP 目录失败 (" + data_dir.string() + "): " + ec.message();
    return result;
  }

  const auto final_path = data_dir / filename;
  const auto partial_path = data_dir / (filename + ".partial");
  if (std::filesystem::exists(final_path, ec) ||
      std::filesystem::exists(partial_path, ec)) {
    result.error = "MCAP 目标文件已存在，拒绝覆盖: " + final_path.string();
    return result;
  }

  mcap::McapWriter writer;
  bool writer_open = false;
  try {
    mcap::McapWriterOptions writer_options("");
    writer_options.compression = mcap::Compression::Zstd;
    writer_options.compressionLevel = mcap::CompressionLevel::Fastest;
    writer_options.chunkSize = 1024u * 1024u;
    writer_options.library = "sonic-a3-eval-recorder";
    const auto open_status = writer.open(partial_path.string(), writer_options);
    if (!open_status.ok()) {
      result.error = "打开 MCAP 临时文件失败: " + open_status.message;
      return result;
    }
    writer_open = true;

    const auto& names31 = robot_io::MakeA3Layout31().names;
    std::vector<std::string> names29;
    names29.reserve(robot_io::kA3PolicyDof);
    for (const int sdk_index : robot_io::kA3PolicyToSdkIdx) {
      names29.push_back(names31.at(static_cast<std::size_t>(sdk_index)));
    }

    const auto first_timestamp = FrameEpochNs(session, frames.front(), 0);
    const auto last_timestamp =
        FrameEpochNs(session, frames.back(), frames.size() - 1);
    mcap::Metadata metadata;
    metadata.name = "a3_eval_episode";
    metadata.metadata = {
        {"session_id", session.session_id},
        {"environment", options.environment},
        {"backend", session.backend_name},
        {"model_tag", DeriveModelTag(options.model_tag,
                                      session.policy_model_path)},
        {"policy_model_path", session.policy_model_path},
        {"runtime_config_path", session.runtime_config_path},
        {"episode_id", std::to_string(episode.episode_id)},
        {"motion_id", std::to_string(episode.motion_id)},
        {"motion_name", episode.motion_name},
        {"motion_source_path", episode.motion_source_path},
        {"motion_repeat_index", std::to_string(episode.motion_repeat_index)},
        {"frame_count", std::to_string(frames.size())},
        {"reference_tick_first", std::to_string(frames.front().reference_tick)},
        {"reference_tick_last", std::to_string(frames.back().reference_tick)},
        {"outcome", "COMPLETED"},
        {"termination_reason", "END_OF_REFERENCE"},
        {"policy_hz", std::to_string(session.policy_hz)},
        {"state_hz", std::to_string(session.state_hz)},
        {"eval_schema_version", std::to_string(kA3EvalSchemaVersion)},
        {"timestamp_first_ns", std::to_string(first_timestamp)},
        {"timestamp_last_ns", std::to_string(last_timestamp)},
        {"representation",
         representation == McapRepresentation::kRawArrays
             ? "raw_arrays"
             : "readable_named"},
        {"joint_order_31_json", JsonStringArray(names31)},
        {"joint_order_29_json", JsonStringArray(names29)},
        {"policy_to_sdk_index_31_json",
         JsonIntegerArray(robot_io::kA3PolicyToSdkIdx)},
        {"reference_fields",
         "q_ref[29],dq_ref[29],pelvis_position_m[3],pelvis_quat_wxyz[4]"},
        {"reference_scope",
         "runtime-resampled joint and yaw-aligned pelvis reference"},
        {"command_fields",
         "q_des[31],dq_des[31],tau_ff[31],kp[31],kd[31]"},
        {"gain_profile_selector", "status.gain_profile_id"},
        {"gain_profiles_json", GainProfilesJson(gain_profiles)},
        {"offline_tau_cmd_expected_formula",
         "tau_ff + kp*(q_des-q) + kd*(dq_des-dq)"},
        {"derived_during_recording", "false"},
        {"driver_internal_torque_limits_included", "false"},
    };
    const auto metadata_status = writer.write(metadata);
    if (!metadata_status.ok()) {
      result.error = "写入 MCAP metadata 失败: " + metadata_status.message;
      writer.terminate();
      writer_open = false;
      std::filesystem::remove(partial_path, ec);
      return result;
    }

    const auto& prefix = options.topic_prefix;
    if (representation == McapRepresentation::kRawArrays) {
      mcap::Schema state_schema("a3.eval.RawRobotState", "jsonschema",
                                kRawStateSchema);
      mcap::Schema command_schema("a3.eval.RawRobotCommand", "jsonschema",
                                  kRawCommandSchema);
      mcap::Schema policy_schema("a3.eval.RawPolicyOutput", "jsonschema",
                                 kRawPolicySchema);
      mcap::Schema reference_schema("a3.eval.RawMotionReference", "jsonschema",
                                    kRawReferenceSchema);
      mcap::Schema imu_schema("a3.eval.RawImu", "jsonschema", kRawImuSchema);
      mcap::Schema timing_schema("a3.eval.Timing", "jsonschema",
                                 kTimingSchema);
      mcap::Schema status_schema("a3.eval.Status", "jsonschema",
                                 kStatusSchema);
      writer.addSchema(state_schema);
      writer.addSchema(command_schema);
      writer.addSchema(policy_schema);
      writer.addSchema(reference_schema);
      writer.addSchema(imu_schema);
      writer.addSchema(timing_schema);
      writer.addSchema(status_schema);

      mcap::Channel state_channel(prefix + "/state", "json", state_schema.id);
      mcap::Channel command_channel(prefix + "/command", "json",
                                    command_schema.id);
      mcap::Channel policy_channel(prefix + "/policy", "json",
                                   policy_schema.id);
      mcap::Channel reference_channel(prefix + "/reference", "json",
                                      reference_schema.id);
      mcap::Channel pelvis_channel(prefix + "/pelvis_imu", "json",
                                   imu_schema.id);
      mcap::Channel torso_channel(prefix + "/torso_imu", "json",
                                  imu_schema.id);
      mcap::Channel timing_channel(prefix + "/timing", "json",
                                   timing_schema.id);
      mcap::Channel status_channel(prefix + "/status", "json",
                                   status_schema.id);
      writer.addChannel(state_channel);
      writer.addChannel(command_channel);
      writer.addChannel(policy_channel);
      writer.addChannel(reference_channel);
      writer.addChannel(pelvis_channel);
      writer.addChannel(torso_channel);
      writer.addChannel(timing_channel);
      writer.addChannel(status_channel);

      for (std::size_t index = 0; index < frames.size(); ++index) {
        const auto& frame = frames[index];
        const auto timestamp_ns = FrameEpochNs(session, frame, index);
        const auto sequence = static_cast<std::uint32_t>(
            std::min<std::size_t>(index + 1,
                                  std::numeric_limits<std::uint32_t>::max()));
        const std::array<std::pair<mcap::ChannelId, std::string>, 8> messages{{
            {state_channel.id, RawStateJson(frame, timestamp_ns)},
            {command_channel.id, RawCommandJson(frame, timestamp_ns)},
            {policy_channel.id, RawPolicyJson(frame, timestamp_ns)},
            {reference_channel.id, RawReferenceJson(frame, timestamp_ns)},
            {pelvis_channel.id,
             RawImuJson(frame, timestamp_ns, frame.pelvis_quat_wxyz,
                        frame.pelvis_gyro)},
            {torso_channel.id,
             RawImuJson(frame, timestamp_ns, frame.torso_quat_wxyz,
                        frame.torso_gyro)},
            {timing_channel.id, TimingJson(frame, timestamp_ns)},
            {status_channel.id, StatusJson(frame, timestamp_ns)},
        }};
        std::string write_error;
        for (const auto& [channel_id, json] : messages) {
          if (!WriteJsonMessage(writer, channel_id, sequence, timestamp_ns,
                                json, &write_error)) {
            result.error = "写入原始 MCAP message 失败: " + write_error;
            writer.terminate();
            writer_open = false;
            std::filesystem::remove(partial_path, ec);
            return result;
          }
        }
      }
    } else {
      const std::string compare_schema_json = JointCompareSchema(names31);
      mcap::Schema joint_schema("a3.eval.NamedJointState", "jsonschema",
                                kNamedJointStateSchema);
      mcap::Schema compare_schema("a3.eval.JointCompare", "jsonschema",
                                  compare_schema_json);
      mcap::Schema imu_schema("a3.eval.Imu", "jsonschema", kImuSchema);
      mcap::Schema timing_schema("a3.eval.Timing", "jsonschema",
                                 kTimingSchema);
      mcap::Schema status_schema("a3.eval.Status", "jsonschema",
                                 kStatusSchema);
      writer.addSchema(joint_schema);
      writer.addSchema(compare_schema);
      writer.addSchema(imu_schema);
      writer.addSchema(timing_schema);
      writer.addSchema(status_schema);

      mcap::Channel state_channel(prefix + "/state_named", "json",
                                  joint_schema.id);
      mcap::Channel command_channel(prefix + "/command_named", "json",
                                    joint_schema.id);
      mcap::Channel policy_channel(prefix + "/policy_named", "json",
                                   joint_schema.id);
      mcap::Channel reference_channel(prefix + "/reference_named", "json",
                                      joint_schema.id);
      mcap::Channel compare_channel(prefix + "/joint_compare", "json",
                                    compare_schema.id);
      mcap::Channel pelvis_channel(prefix + "/pelvis_imu", "json",
                                   imu_schema.id);
      mcap::Channel torso_channel(prefix + "/torso_imu", "json",
                                  imu_schema.id);
      mcap::Channel timing_channel(prefix + "/timing", "json",
                                   timing_schema.id);
      mcap::Channel status_channel(prefix + "/status", "json",
                                   status_schema.id);
      writer.addChannel(state_channel);
      writer.addChannel(command_channel);
      writer.addChannel(policy_channel);
      writer.addChannel(reference_channel);
      writer.addChannel(compare_channel);
      writer.addChannel(pelvis_channel);
      writer.addChannel(torso_channel);
      writer.addChannel(timing_channel);
      writer.addChannel(status_channel);

      for (std::size_t index = 0; index < frames.size(); ++index) {
        const auto& frame = frames[index];
        const auto timestamp_ns = FrameEpochNs(session, frame, index);
        const auto sequence = static_cast<std::uint32_t>(
            std::min<std::size_t>(index + 1,
                                  std::numeric_limits<std::uint32_t>::max()));
        const std::array<std::pair<mcap::ChannelId, std::string>, 9> messages{{
            {state_channel.id,
             NamedJointJson<31>(frame, timestamp_ns, names31, &frame.q,
                                &frame.dq, &frame.tau_est)},
            {command_channel.id,
             NamedJointJson<31>(frame, timestamp_ns, names31,
                                &frame.command_q_des,
                                &frame.command_dq_des,
                                &frame.command_tau_ff)},
            {policy_channel.id,
             NamedJointJson<29>(frame, timestamp_ns, names29, nullptr, nullptr,
                                &frame.raw_action)},
            {reference_channel.id,
             NamedJointJson<29>(frame, timestamp_ns, names29, &frame.q_ref,
                                &frame.dq_ref, nullptr)},
            {compare_channel.id,
             JointCompareJson(frame, timestamp_ns, names31)},
            {pelvis_channel.id,
             ImuJson(frame, timestamp_ns, frame.pelvis_quat_wxyz,
                     frame.pelvis_gyro)},
            {torso_channel.id,
             ImuJson(frame, timestamp_ns, frame.torso_quat_wxyz,
                     frame.torso_gyro)},
            {timing_channel.id, TimingJson(frame, timestamp_ns)},
            {status_channel.id, StatusJson(frame, timestamp_ns)},
        }};
        std::string write_error;
        for (const auto& [channel_id, json] : messages) {
          if (!WriteJsonMessage(writer, channel_id, sequence, timestamp_ns,
                                json, &write_error)) {
            result.error = "写入可读 MCAP message 失败: " + write_error;
            writer.terminate();
            writer_open = false;
            std::filesystem::remove(partial_path, ec);
            return result;
          }
        }
      }
    }

    writer.close();
    writer_open = false;
    std::filesystem::rename(partial_path, final_path, ec);
    if (ec) {
      result.error = "提交 MCAP 文件失败: " + ec.message();
      std::filesystem::remove(partial_path, ec);
      return result;
    }
    result.ok = true;
    result.relative_path = relative_directory / filename;
    result.file_size = std::filesystem::file_size(final_path, ec);
    if (ec) result.file_size = 0;
    return result;
  } catch (const std::exception& exception) {
    if (writer_open) writer.terminate();
    std::filesystem::remove(partial_path, ec);
    result.error = std::string("生成 MCAP 时发生异常: ") + exception.what();
    return result;
  } catch (...) {
    if (writer_open) writer.terminate();
    std::filesystem::remove(partial_path, ec);
    result.error = "生成 MCAP 时发生未知异常";
    return result;
  }
}
#endif

}  // namespace

A3EvalMcapWriter::A3EvalMcapWriter(A3EvalMcapOptions options)
    : options_(std::move(options)) {
  options_.topic_prefix = NormalizePrefix(std::move(options_.topic_prefix));
}

bool A3EvalMcapWriter::AvailableAtBuildTime() noexcept {
  return HAS_A3_EVAL_MCAP != 0;
}

A3EvalMcapWriteResult A3EvalMcapWriter::WriteEpisode(
    const std::filesystem::path& session_dir,
    const A3EvalMcapSessionMetadata& session,
    const A3EvalMcapEpisodeMetadata& episode,
    const std::vector<EvalFrameV1>& frames) noexcept {
  A3EvalMcapWriteResult result;
  if (!options_.enabled) {
    result.error = "episode MCAP 未启用";
    return result;
  }
  if (frames.empty()) {
    result.error = "完整 episode 没有可写入帧";
    return result;
  }
#if !HAS_A3_EVAL_MCAP
  (void)session_dir;
  (void)session;
  (void)episode;
  result.error = "当前二进制未编译 MCAP 支持";
  return result;
#else
  GainProfileMap gain_profiles;
  if (!CollectGainProfiles(frames, gain_profiles, result.error)) {
    return result;
  }
  const auto filename = MakeEpisodeFilename(options_, session, episode);
  const auto raw = WriteEpisodeMcapFile(
      options_, session_dir, options_.raw_directory, filename, session,
      episode, frames, gain_profiles, McapRepresentation::kRawArrays);
  if (!raw.ok) {
    result.error = "原始数组 MCAP: " + raw.error;
    return result;
  }

  result.relative_path = raw.relative_path;
  result.file_size = raw.file_size;
  if (!WriteEpisodeMetadataYaml(
          options_, session_dir, result.relative_path, result.file_size,
          session, episode, frames, gain_profiles,
          result.metadata_relative_path, result.metadata_file_size,
          result.error)) {
    std::error_code remove_error;
    std::filesystem::remove(session_dir / result.relative_path, remove_error);
    result.relative_path.clear();
    result.file_size = 0;
    return result;
  }
  if (options_.readable_copy_enabled) {
    const auto readable = WriteEpisodeMcapFile(
        options_, session_dir, options_.readable_directory, filename, session,
        episode, frames, gain_profiles, McapRepresentation::kReadableNamed);
    if (!readable.ok) {
      std::error_code remove_error;
      std::filesystem::remove(session_dir / raw.relative_path, remove_error);
      std::filesystem::remove(session_dir / result.metadata_relative_path,
                              remove_error);
      result.relative_path.clear();
      result.metadata_relative_path.clear();
      result.file_size = 0;
      result.metadata_file_size = 0;
      result.error = "可读 MCAP: " + readable.error;
      return result;
    }
    result.readable_relative_path = readable.relative_path;
    result.readable_file_size = readable.file_size;
  }
  result.ok = true;
  return result;
#endif
}

}  // namespace a3_deploy
