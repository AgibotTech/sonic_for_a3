#!/bin/bash

# default values
DEFAULT_SHOULDER_POS=0.723
DEFAULT_ELBOW_POS=-1.88

# read command line arguments
SHOULDER_POS=${1:-$DEFAULT_SHOULDER_POS}
ELBOW_POS=${2:-$DEFAULT_ELBOW_POS}

# build json data
JSON_DATA=$(cat <<EOF
{
  "header": {
    "stamp": {
      "sec": 1234567890,
      "nanosec": 123456789
    },
    "frame_id": "base_link"
  },
  "joints": [
    {
      "name": "shoulder_actuator",
      "position": $SHOULDER_POS,
      "velocity": 0.0,
      "effort": 0.0,
      "stiffness": 0.0,
      "damping": 0.0
    },
    {
      "name": "elbow_actuator",
      "position": $ELBOW_POS,
      "velocity": 0.1,
      "effort": 0.0,
      "stiffness": 0.0,
      "damping": 0.0
    }
  ]
}
EOF
)

# send json data to server
curl -i \
  -H 'content-type:application/json' \
  -X POST 'http://127.0.0.1:50080/channel/%2Fexamples_hardware%2Fjoint%2Fjoint_command/ros2%3Aaimrt_msgs%2Fmsg%2FJointCommandArray' \
  -d "$JSON_DATA"