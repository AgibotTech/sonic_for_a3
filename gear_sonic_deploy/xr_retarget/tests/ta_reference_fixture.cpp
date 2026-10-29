#include "a3_deploy/a3_teleop_reference.hpp"
#include "a3_policy_parameters.hpp"
#include "aimdk/protocol/ta/ta_channel.pb.h"
#include <fstream>
#include <iostream>
#include <iterator>
#include <nlohmann/json.hpp>
int main(int argc, char** argv) {
  if (argc != 2) return 64;
  std::ifstream in(argv[1], std::ios::binary);
  std::string bytes((std::istreambuf_iterator<char>(in)), {});
  aimdk::protocol::TaWholeBodyCommandChannel msg;
  if (!msg.ParseFromString(bytes)) return 2;
  a3_deploy::A3TeleopFrame frame{};
  std::string error;
  if (!a3_deploy::ConvertTaWholeBodyCommand(msg, 1, frame, &error)) {
    std::cerr << error; return 3;
  }
  nlohmann::json result = {{"stamp_ns", frame.stamp_ns}, {"q_mujoco", frame.q_mujoco},
                           {"dq_mujoco", frame.dq_mujoco}, {"quat", frame.pelvis_quat_wxyz}};
  std::cout << result.dump();
}
