#!/bin/bash

set -e

source install/share/aimrt_msgs/local_setup.bash

./aimrt_main --cfg_file_path=./cfg/examples_hardware_joint_ros2_cfg.yaml
