"""Release checks for the 089 robot and self-collision contract."""
import importlib.util
from pathlib import Path
import unittest
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation

import yaml

ROOT = Path(__file__).resolve().parents[1]


class TrainingContractTest(unittest.TestCase):
    def test_late_collision_asset_and_pairs(self):
        config = yaml.safe_load((ROOT / 'gear_sonic/config/exp/manager/universal_token/all_modes/sonic_a3_089.yaml').read_text())
        robot = config['manager_env']['config']['robot']
        self.assertTrue(robot['disable_passive_foot'])
        self.assertTrue(robot['enable_self_collisions'])
        self.assertTrue(robot['enable_self_contact_reward_sensors'])
        self.assertTrue(robot['dynamic_ankle']['enabled'])
        spec = importlib.util.spec_from_file_location('a3_collision_check', ROOT / 'gear_sonic/envs/manager_env/robots/a3_self_collision.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        pairs = module.load_reviewed_pairs(ROOT / robot['self_collision_review_manifest'], ROOT / robot['asset_path'])
        self.assertEqual(set(pairs), {tuple(sorted(pair)) for pair in (
            ('pelvis_link', 'torso_Link'),
            ('left_wrist_roll_Link', 'left_wrist_yaw_Link'),
            ('right_wrist_roll_Link', 'right_wrist_yaw_Link'),
        )})
        rewards = config['manager_env']['rewards']
        events = config['manager_env']['events']
        self.assertEqual(rewards['self_collision_contacts']['weight'], -0.1)
        self.assertEqual(rewards['tracking_wrist_joint_angle']['weight'], 0.5)
        self.assertIsNotNone(events['ankle_motor_inertia'])
        self.assertIsNotNone(events['robot_joint_friction_ankle'])


    def test_mujoco_collision_geometry_and_exclusions(self):
        robot = ROOT / 'gear_sonic/data/assets/robot_description/urdf/a3/a3_089_rigid_foot.urdf'
        mjcf = ROOT / 'gear_sonic/data/assets/robot_description/mjcf/a3_089_rigid_foot_loop.xml'
        u = ET.parse(robot).getroot()
        m = ET.parse(mjcf).getroot()
        bodies = {b.get('name'): b for b in m.findall('.//worldbody//body')}
        geoms = {g.get('name'): g for g in m.findall('.//worldbody//geom') if g.get('class') == 'collision'}
        collisions = [c for link in u.findall('link') for c in link.findall('collision')]
        for link in u.findall('link'):
            body_name = link.get('name')
            if body_name in ('left_hand_Link', 'right_hand_Link'):
                body_name = body_name.replace('_hand_Link', '_wrist_yaw_Link')
            for collision in link.findall('collision'):
                self.assertIsNotNone(bodies[body_name].find("geom[@name='089_" + collision.get('name') + "']"))
        self.assertEqual(len(geoms), len(collisions))
        for collision in collisions:
            geom = geoms['089_' + collision.get('name')]
            shape, origin = collision.find('geometry')[0], collision.find('origin')
            self.assertEqual(geom.get('type'), shape.tag)
            np.testing.assert_allclose(np.fromstring(geom.get('pos'), sep=' '), np.fromstring(origin.get('xyz', '0 0 0'), sep=' '), atol=1e-10)
            expected = Rotation.from_euler('xyz', np.fromstring(origin.get('rpy', '0 0 0'), sep=' '))
            q = np.fromstring(geom.get('quat'), sep=' ')
            actual = Rotation.from_quat([q[1], q[2], q[3], q[0]])
            self.assertLess((expected.inv() * actual).magnitude(), 1e-10)
            if shape.tag == 'box':
                size = np.fromstring(shape.get('size'), sep=' ') / 2
            elif shape.tag == 'capsule':
                size = [float(shape.get('radius')), float(shape.get('length')) / 2]
            elif shape.tag == 'sphere':
                size = [float(shape.get('radius'))]
            else:
                asset = m.find(f"asset/mesh[@name='{geom.get('mesh')}']")
                self.assertEqual(asset.get('file'), '../' + str(Path(shape.get('filename')).relative_to('meshes')))
                continue
            np.testing.assert_allclose(np.fromstring(geom.get('size'), sep=' '), size, atol=1e-10)
        exclusions = {frozenset((x.get('body1'), x.get('body2'))) for x in m.findall('contact/exclude')}
        self.assertEqual(exclusions, {frozenset(p) for p in (('pelvis_link', 'torso_Link'), ('left_wrist_roll_Link', 'left_wrist_yaw_Link'), ('right_wrist_roll_Link', 'right_wrist_yaw_Link'))})


if __name__ == '__main__':
    unittest.main()
