"""Self-contact sensing contract independent of support-contact exemptions."""
import xml.etree.ElementTree as ET

import torch


def imported_body_names(urdf_path):
    """Bodies retained by this contract's merge_fixed_joints=True importer."""
    root = ET.parse(urdf_path).getroot()
    merged = {joint.find('child').get('link') for joint in root.findall('joint')
              if joint.get('type') == 'fixed'}
    return sorted(link.get('name') for link in root.findall('link')
                  if link.get('name') not in merged)


def sensor_specs(body_names):
    """Upper-triangular one-to-many sensors: each unordered pair exactly once."""
    names = sorted(body_names)
    if len(names) != len(set(names)) or len(names) < 2:
        raise ValueError('Self-contact sensing requires unique imported body names')
    return [(f'self_contact_{i:02d}', body, names[i+1:]) for i, body in enumerate(names[:-1])]


def count_pair_contacts(force_history, threshold):
    """Count pairs with a normal force above threshold in the history window.

    Input [env, history, 1 sensor body, filtered bodies, xyz]. Norm is taken
    before aggregation so opposing contacts on different pairs do not cancel.
    """
    if threshold < 0:
        raise ValueError('Contact force threshold must be nonnegative')
    if force_history.ndim != 5 or force_history.shape[2] != 1 or force_history.shape[-1] != 3:
        raise ValueError('Expected one-to-many contact history [N,H,1,F,3]')
    return (torch.linalg.vector_norm(force_history[:, :, 0], dim=-1).amax(dim=1) > threshold).sum(dim=-1)


def self_collision_contacts(env, threshold=1.0):
    """Additive self-contact penalty; ground is absent from every filter list.

    Existing undesired_contacts stays unchanged: some non-exempt bodies may
    also incur that legacy term. Weight requires dynamic contribution review.
    """
    names = sorted(name for name in env.scene.sensors if name.startswith('self_contact_'))
    if not names:
        raise ValueError('Self-collision reward has no configured pair sensors')
    result = torch.zeros(env.num_envs, device=env.device)
    for name in names:
        history = env.scene.sensors[name].data.force_matrix_w_history
        if history is None:
            raise RuntimeError(f'Self-contact sensor {name} has no filtered force history')
        result += count_pair_contacts(history, threshold)
    return result


def configure_self_contact_sensors(scene_cfg, robot_config):
    """Opt-in only; all other experiments retain their existing sensor cost."""
    if not robot_config.get('enable_self_contact_reward_sensors', False):
        return
    if not robot_config.get('enable_self_collisions', False):
        raise ValueError('Self-contact sensing requires physical self-collisions')
    if not robot_config.get('merge_fixed_joints', True):
        raise ValueError('Self-contact sensor mapping requires merge_fixed_joints=True')
    from isaaclab.sensors import ContactSensorCfg
    fast_update = robot_config.get('self_contact_every_step_fast_path', False)
    extra = {}
    if fast_update:
        from .a3_every_step_contact_sensor import EveryStepContactSensor
        extra['class_type'] = EveryStepContactSensor
    names = imported_body_names(robot_config['asset_path'])
    for name, body, filters in sensor_specs(names):
        setattr(scene_cfg, name, ContactSensorCfg(
            prim_path='{ENV_REGEX_NS}/Robot/'+body,
            filter_prim_paths_expr=['{ENV_REGEX_NS}/Robot/'+other for other in filters],
            history_length=3, update_period=0.0,
            track_air_time=False, track_pose=False, debug_vis=False, **extra))
