"""Validate TA protobuf mapping and ROS 2 wire transport without a robot."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest

import numpy as np

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
from sonic_a3.stream.a3_ref_stream import A3RefFrame
from sonic_a3.stream.ta_ros_publisher import TaRosPublisher, frame_to_ta


class TaPublisherTest(unittest.TestCase):
    def frame(self):
        return A3RefFrame(42, 10.0, np.arange(29, dtype=float), np.arange(29, dtype=float) / 10,
                          np.array([1, 2, 3]), np.array([1, 0, 0, 0]))

    def test_cpp_consumer_mapping(self):
        deploy = ROOT / 'gear_sonic_deploy/src/g1/g1_deploy_onnx_ref'
        proto = deploy / 'proto'
        with tempfile.TemporaryDirectory() as d:
            binary, payload = Path(d)/'fixture', Path(d)/'message.pb'
            names = ['aimdk/protocol/common/timestamp.proto', 'aimdk/protocol/common/control_source.proto',
                     'aimdk/protocol/common/header.proto', 'aimdk/protocol/ta/ta_whole_body_command.proto',
                     'aimdk/protocol/ta/ta_channel.proto']
            subprocess.run(['/usr/bin/protoc', '-I'+str(proto), '--cpp_out='+d,
                            *(str(proto/n) for n in names)], check=True, capture_output=True)
            sources = [str(Path(d)/n.replace('.proto', '.pb.cc')) for n in names]
            subprocess.run(['g++', '-std=c++20', '-O1', '-DHAS_A3_TA_PROTO', '-I'+str(deploy/'include'),
                            '-I'+d, str(HERE/'tests/ta_reference_fixture.cpp'),
                            str(deploy/'src/a3_deploy/a3_teleop_reference.cpp'), *sources,
                            '-lprotobuf', '-pthread', '-o', str(binary)], check=True, capture_output=True)
            frame = self.frame()
            payload.write_bytes(frame_to_ta(frame, wall_ns=20_000_000_000, monotonic_s=10.1).SerializeToString())
            data = json.loads(subprocess.check_output([str(binary), str(payload)]))
            self.assertEqual(data['stamp_ns'], 19_900_000_000)
            expected = np.array([2,5,8,11,15,19,21,23,25,27,12,16,20,22,24,26,28,0,3,6,9,13,17,1,4,7,10,14,18])
            np.testing.assert_array_equal(data['q_mujoco'], expected)
            np.testing.assert_allclose(data['dq_mujoco'], expected / 10)
            np.testing.assert_array_equal(data['quat'], [1,0,0,0])

    def test_ros2_roundtrip(self):
        import rclpy
        from rclpy.qos import QoSProfile, ReliabilityPolicy
        from ros2_plugin_proto.msg import RosMsgWrapper
        publisher = TaRosPublisher('/test/xr_ta')
        receiver = rclpy.create_node('xr_ta_test_subscriber')
        received = []
        sub = receiver.create_subscription(RosMsgWrapper, publisher.topic, received.append,
                                           QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT))
        deadline = time.monotonic() + 8
        try:
            while not received and time.monotonic() < deadline:
                publisher.send(self.frame())
                rclpy.spin_once(receiver, timeout_sec=.05)
            self.assertTrue(received, 'ROS 2 wrapper not received')
            from aimdk.protocol.ta import ta_channel_pb2
            wrapper = received[-1]
            self.assertEqual(wrapper.serialization_type, 'pb')
            msg = ta_channel_pb2.TaWholeBodyCommandChannel()
            msg.ParseFromString(b''.join(wrapper.data))
            self.assertEqual(msg.header.seq, 42)
            self.assertEqual(len(msg.data.leg_command.angles_rad), 12)
            self.assertEqual(len(msg.data.arm_command.angles_rad), 14)
        finally:
            receiver.destroy_subscription(sub)
            receiver.destroy_node()
            publisher.close()


    def test_synthetic_ik_to_ros2(self):
        import rclpy
        from rclpy.qos import QoSProfile, ReliabilityPolicy
        from ros2_plugin_proto.msg import RosMsgWrapper
        from aimdk.protocol.ta import ta_channel_pb2
        rclpy.init()
        receiver = rclpy.create_node('xr_ta_synthetic_test')
        topic = '/test/xr_synthetic'
        received = []
        from sonic_a3.stream.ta_ros_publisher import ROS_TYPE_SUFFIX
        sub = receiver.create_subscription(RosMsgWrapper, topic + '/' + ROS_TYPE_SUFFIX, received.append,
                                           QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT))
        csv = ROOT / 'a3_data/agibot_a3/001_walk_front_slow.csv'
        proc = subprocess.Popen([sys.executable, str(HERE/'scripts/pico_a3_retarget_publisher.py'),
                                 '--source', 'synthetic', '--synthetic-from', str(csv),
                                 '--topic', topic, '--duration', '4', '--start-delay', '.5'],
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        try:
            deadline = time.monotonic() + 12
            while len(received) < 4 and time.monotonic() < deadline and proc.poll() is None:
                rclpy.spin_once(receiver, timeout_sec=.1)
            out, err = proc.communicate(timeout=8)
            self.assertEqual(proc.returncode, 0, out + err)
            self.assertGreaterEqual(len(received), 4, out + err)
            msgs = []
            for wrapper in received:
                m = ta_channel_pb2.TaWholeBodyCommandChannel()
                m.ParseFromString(b''.join(wrapper.data))
                msgs.append(m)
            self.assertTrue(all(len(m.data.leg_command.angles_rad) == 12 for m in msgs))
            stamps = [m.header.timestamp.seconds * 1_000_000_000 + m.header.timestamp.nanos for m in msgs]
            self.assertTrue(all(b > a for a, b in zip(stamps, stamps[1:])))
        finally:
            if proc.poll() is None:
                proc.kill(); proc.communicate()
            receiver.destroy_subscription(sub)
            receiver.destroy_node()
            rclpy.shutdown()


if __name__ == '__main__':
    unittest.main()
