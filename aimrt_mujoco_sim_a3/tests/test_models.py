"""Check both T3D0 MJCF models against the body-drive configuration."""
from pathlib import Path
import unittest

import mujoco
import yaml

BIN = Path(__file__).resolve().parents[1] / 'src/models/bin'


class ModelContractTest(unittest.TestCase):
    def test_body_drive_bindings(self):
        for name in ('a3_t3d0', 'a3u_t3d0'):
            with self.subTest(model=name):
                cfg = yaml.safe_load((BIN / 'cfg' / f'{name}.yaml').read_text())['MujocoSimModule']
                model = mujoco.MjModel.from_xml_path(str(BIN / cfg['simulation_model_path']))
                joints = [j['bind_joint'] for group in cfg['joint_groups'].values() for j in group]
                self.assertEqual(len(joints), 31)
                for joint in joints:
                    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint)
                    self.assertGreaterEqual(jid, 0, joint)
                    self.assertTrue(any(model.actuator_trnid[i, 0] == jid for i in range(model.nu)), joint)
                    for kind in (mujoco.mjtSensor.mjSENS_JOINTPOS, mujoco.mjtSensor.mjSENS_JOINTVEL,
                                 mujoco.mjtSensor.mjSENS_JOINTACTFRC):
                        self.assertTrue(any(model.sensor_type[i] == kind and model.sensor_objid[i] == jid
                                            for i in range(model.nsensor)), (joint, kind))
                for pub in cfg['publisher_options']:
                    if pub['type'] == 'imu_sensor_ros2':
                        for key, name in pub['options'].items():
                            obj = mujoco.mjtObj.mjOBJ_SITE if key == 'bind_site' else mujoco.mjtObj.mjOBJ_SENSOR
                            self.assertGreaterEqual(mujoco.mj_name2id(model, obj, name), 0, name)


if __name__ == '__main__':
    unittest.main()
