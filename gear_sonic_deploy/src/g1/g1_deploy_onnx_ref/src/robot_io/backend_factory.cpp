// Copyright (c) 2026, AgiBot Inc. All rights reserved.
#include "robot_io/robot_io_backend.hpp"

#include <iostream>
#include <memory>
#include <string>

namespace robot_io {

// The A3 helper is defined in its own translation unit. It is a no-op
// (returns nullptr) when the A3 backend is not compiled in for this build.
std::unique_ptr<RobotIOBackend> CreateA3AimrtBackend();

std::unique_ptr<RobotIOBackend> CreateBackend(const std::string& name) {
  if (name == "g1") {
    std::cerr << "[robot_io] legacy backend 'g1' is not part of this A3 "
                 "release."
              << std::endl;
    return nullptr;
  }
  if (name == "a3") {
    auto backend = CreateA3AimrtBackend();
    if (!backend) {
      std::cerr << "[robot_io] A3 backend requested but not compiled in "
                   "(build with -DENABLE_A3_BACKEND=ON)."
                << std::endl;
    }
    return backend;
  }
  std::cerr << "[robot_io] CreateBackend: unknown backend name '" << name
            << "'. Known name: 'a3'." << std::endl;
  return nullptr;
}

// Weak fallback keeps non-AimRT unit-test builds linkable.
__attribute__((weak)) std::unique_ptr<RobotIOBackend> CreateA3AimrtBackend() {
  return nullptr;
}

}  // namespace robot_io
