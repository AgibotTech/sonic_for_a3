"""Publish XR retarget frames on AimRT's ROS 2 TA protobuf channel."""
from __future__ import annotations

from pathlib import Path
import sys
import time

import numpy as np

from sonic_a3.contract.joint_order import IL_JOINT_NAMES
from sonic_a3.stream.a3_ref_stream import A3RefFrame

# Generated directly from the deploy's checked-in AimRT protocol definitions.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'ta_proto'))
from aimdk.protocol.ta import ta_channel_pb2, ta_whole_body_command_pb2  # noqa: E402

LEG_NAMES = tuple(f'{side}_{joint}_joint' for side in ('left', 'right') for joint in (
    'hip_pitch', 'hip_roll', 'hip_yaw', 'knee', 'ankle_pitch', 'ankle_roll'))
WAIST_NAMES = tuple(f'waist_{joint}_joint' for joint in ('yaw', 'roll', 'pitch'))
ARM_NAMES = tuple(f'{side}_{joint}_joint' for side in ('left', 'right') for joint in (
    'shoulder_pitch', 'shoulder_roll', 'shoulder_yaw', 'elbow', 'wrist_roll', 'wrist_pitch', 'wrist_yaw'))
_INDEX = {name: i for i, name in enumerate(IL_JOINT_NAMES)}
ROS_TYPE_SUFFIX = 'pb_3Aaimdk_2Eprotocol_2ETaWholeBodyCommandChannel'


def frame_to_ta(frame: A3RefFrame, *, wall_ns: int | None = None, monotonic_s: float | None = None):
    """Map IsaacLab order into the TA body layout, preserving sample time."""
    q = np.asarray(frame.dof_il, dtype=np.float64)
    dq = np.asarray(frame.dof_vel_il, dtype=np.float64)
    pos = np.asarray(frame.root_pos, dtype=np.float64)
    quat = np.asarray(frame.root_quat_wxyz, dtype=np.float64)
    if q.shape != (29,) or dq.shape != (29,) or pos.shape != (3,) or quat.shape != (4,):
        raise ValueError('Expected 29 joint positions/velocities, 3 root coordinates and 4 quaternion values')
    if not all(np.isfinite(v).all() for v in (q, dq, pos, quat)) or not np.isfinite(frame.stamp_s):
        raise ValueError('Non-finite XR reference frame')
    norm = np.linalg.norm(quat)
    if norm < 1e-8:
        raise ValueError('Invalid root quaternion')
    wall_ns = time.time_ns() if wall_ns is None else wall_ns
    monotonic_s = time.monotonic() if monotonic_s is None else monotonic_s
    stamp = wall_ns + round((float(frame.stamp_s) - monotonic_s) * 1e9)
    if stamp <= 0:
        raise ValueError('Invalid sample time')
    msg = ta_channel_pb2.TaWholeBodyCommandChannel()
    msg.header.seq = int(frame.frame_index) % (1 << 32)
    msg.header.timestamp.seconds, msg.header.timestamp.nanos = divmod(stamp, 1_000_000_000)
    data = msg.data
    data.joint_layout = ta_whole_body_command_pb2.TaJointLayout_BODY_31
    data.pelvis_pose.quat_wxyz.extend((quat / norm).tolist())
    data.pelvis_pose.position_xyz.extend(pos.tolist())
    data.leg_command.angles_rad.extend(q[_INDEX[n]] for n in LEG_NAMES)
    data.waist_command.angles_rad.extend(q[_INDEX[n]] for n in WAIST_NAMES)
    data.arm_command.angles_rad.extend(q[_INDEX[n]] for n in ARM_NAMES)
    data.joint_velocities.velocities_rad_s.extend(dq[_INDEX[n]] for n in LEG_NAMES + WAIST_NAMES)
    data.joint_velocities.velocities_rad_s.extend((0.0, 0.0))  # no head command
    data.joint_velocities.velocities_rad_s.extend(dq[_INDEX[n]] for n in ARM_NAMES)
    return msg


class TaRosPublisher:
    def __init__(self, topic='/ta/whole_body_command'):
        import rclpy
        from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
        from ros2_plugin_proto.msg import RosMsgWrapper

        self._rclpy = rclpy
        self._wrapper_type = RosMsgWrapper
        rclpy.init(args=None)
        self._node = rclpy.create_node('xr_a3_retarget_publisher')
        # Reliable publisher is compatible with both reliable and best-effort subscribers.
        qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.VOLATILE)
        self.topic = topic.rstrip('/') + '/' + ROS_TYPE_SUFFIX
        self._publisher = self._node.create_publisher(RosMsgWrapper, self.topic, qos)

    def send(self, frame: A3RefFrame):
        wrapper = self._wrapper_type()
        wrapper.serialization_type = 'pb'
        wrapper.data = [bytes((byte,)) for byte in frame_to_ta(frame).SerializeToString()]
        self._publisher.publish(wrapper)

    def send_control(self, state):
        # TA has no pause message: stopping frames activates deploy's stale handling.
        pass

    def close(self):
        self._node.destroy_node()
        self._rclpy.shutdown()
