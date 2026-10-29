"""Explicit, review-gated A3 self-collision filtering before environment cloning."""
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET


def load_reviewed_pairs(path, asset_path):
    manifest = json.loads(Path(path).read_text())
    if manifest.get('status') != 'approved':
        raise ValueError('Self-collision pair review has not been approved')
    digest = hashlib.sha256(Path(asset_path).read_bytes()).hexdigest()
    if manifest.get('asset_sha256') != digest:
        raise ValueError('Collision review does not match this URDF')
    for mesh in ET.parse(asset_path).getroot().findall('.//collision/geometry/mesh'):
        path = Path(asset_path).parent / mesh.get('filename')
        expected = manifest.get('collision_mesh_sha256', {}).get(mesh.get('filename'), path.stem)
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f'Collision mesh changed since asset assembly: {path}')
    pairs = []
    for row in manifest['pairs']:
        if not row.get('reason') or not row.get('evidence'):
            raise ValueError('Every excluded pair requires a reason and evidence')
        pair = tuple(sorted((row['body1'], row['body2'])))
        if pair[0] == pair[1] or pair in pairs:
            raise ValueError('Invalid or duplicate self-collision pair')
        pairs.append(pair)
    return pairs


def author_pair_filters(prim, pairs):
    from pxr import Usd, UsdPhysics
    by_name = {}
    for child in Usd.PrimRange(prim):
        if child.HasAPI(UsdPhysics.RigidBodyAPI):
            by_name.setdefault(child.GetName(), []).append(child)
    for first, second in pairs:
        if len(by_name.get(first, [])) != 1 or len(by_name.get(second, [])) != 1:
            raise ValueError(f'Excluded pair does not resolve to unique rigid bodies: {first}, {second}')
        def endpoints(body):
            # A filter on an articulation-root prim may address the entire
            # articulation, not just its root link. Restrict that endpoint to
            # its collision shapes, otherwise pelvis<->torso can hide hands<->torso.
            if not body.HasAPI(UsdPhysics.ArticulationRootAPI):
                return [body]
            for child in list(Usd.PrimRange(body)):
                if child.IsInstance() and child.GetName() == 'collisions':
                    child.SetInstanceable(False)
            shapes = [child for child in Usd.PrimRange(body)
                      if child.HasAPI(UsdPhysics.CollisionAPI)]
            if not shapes:
                raise ValueError(f'Articulation-root body has no editable collision shapes: {body.GetPath()}')
            return shapes

        for left in endpoints(by_name[first][0]):
            for right in endpoints(by_name[second][0]):
                for source, target in ((left, right), (right, left)):
                    rel = UsdPhysics.FilteredPairsAPI.Apply(source).CreateFilteredPairsRel()
                    rel.AddTarget(target.GetPath())
                    if target.GetPath() not in rel.GetTargets():
                        raise RuntimeError('Failed to author self-collision filter')


def configure_self_collision(robot_cfg, config):
    if 'enable_self_collisions' not in config:
        return
    enabled = bool(config['enable_self_collisions'])
    robot_cfg.spawn.self_collision = enabled
    robot_cfg.spawn.articulation_props.enabled_self_collisions = enabled
    if not enabled:
        return
    pairs = load_reviewed_pairs(config['self_collision_review_manifest'], robot_cfg.spawn.asset_path)
    import isaaclab.sim as sim_utils
    from isaaclab.sim.utils import clone

    if robot_cfg.spawn.func is not sim_utils.spawn_from_urdf:
        raise ValueError('Self-collision wrapper requires the standard URDF spawner; compose explicitly')

    @clone
    def spawn_reviewed(prim_path, cfg, translation=None, orientation=None, **kwargs):
        spawn = getattr(sim_utils.spawn_from_urdf, '__wrapped__', sim_utils.spawn_from_urdf)
        prim = spawn(prim_path, cfg, translation=translation, orientation=orientation, **kwargs)
        author_pair_filters(prim, pairs)
        return prim

    robot_cfg.spawn.func = spawn_reviewed
