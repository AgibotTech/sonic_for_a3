#!/usr/bin/env python3
"""Build the A3 089 closed-loop MJCF collision geometry from its training URDF."""
from pathlib import Path
import math
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[2]
ASSETS = ROOT / 'gear_sonic/data/assets/robot_description'
URDF = ASSETS / 'urdf/a3/a3_089_rigid_foot.urdf'
BASE = ASSETS / 'mjcf/a3_loop_template.xml'
OUTPUT = ASSETS / 'mjcf/a3_089_rigid_foot_loop.xml'


def quat(rpy):
    r, p, y = (float(v) * 0.5 for v in rpy.split())
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    return ' '.join(f'{v:.15g}' for v in (cr*cp*cy+sr*sp*sy, sr*cp*cy-cr*sp*sy, cr*sp*cy+sr*cp*sy, cr*cp*sy-sr*sp*cy))


def build():
    urdf = ET.parse(URDF).getroot()
    tree = ET.parse(BASE)
    root = tree.getroot()
    root.set('model', 'A3_089_rigid_foot_loop')
    bodies = {body.get('name'): body for body in root.findall('.//worldbody//body')}
    for body in bodies.values():
        for geom in list(body.findall('geom')):
            if geom.get('class') == 'collision':
                body.remove(geom)
    asset = root.find('asset')
    existing_mesh = {}
    count = 0
    for link in urdf.findall('link'):
        name = link.get('name')
        collisions = link.findall('collision')
        if not collisions:
            continue
        if name not in bodies:
            # Both hands are fixed at identity to their wrist-yaw body.
            joint = next(j for j in urdf.findall('joint') if j.find('child').get('link') == name)
            origin = joint.find('origin')
            if joint.get('type') != 'fixed' or origin.get('xyz') != '0 0 0' or origin.get('rpy') != '0 0 0':
                raise ValueError(f'Cannot merge non-identity collision link {name}')
            name = joint.find('parent').get('link')
        body = bodies[name]
        for collision in collisions:
            shape = collision.find('geometry')[0]
            origin = collision.find('origin')
            attrs = {'name': '089_' + collision.get('name'), 'class': 'collision', 'type': shape.tag,
                     'pos': origin.get('xyz', '0 0 0'), 'quat': quat(origin.get('rpy', '0 0 0'))}
            if shape.tag == 'mesh':
                filename = shape.get('filename')
                if filename not in existing_mesh:
                    mesh_name = f'collision_089_mesh_{len(existing_mesh):02d}'
                    ET.SubElement(asset, 'mesh', {'name': mesh_name, 'file': '../' + str(Path(filename).relative_to('meshes'))})
                    existing_mesh[filename] = mesh_name
                attrs['mesh'] = existing_mesh[filename]
            elif shape.tag == 'capsule':
                attrs['size'] = f"{shape.get('radius')} {float(shape.get('length')) / 2:.15g}"
            elif shape.tag == 'sphere':
                attrs['size'] = shape.get('radius')
            elif shape.tag == 'box':
                attrs['size'] = ' '.join(f'{float(v)/2:.15g}' for v in shape.get('size').split())
            else:
                raise ValueError(f'Unsupported collision geometry {shape.tag}')
            ET.SubElement(body, 'geom', attrs)
            count += 1
    # The URDF has fixed, massless-foot-geometry merges. The source loop model
    # already contains the same rigid-foot mass and inertia, and retains its
    # ankle and waist linkage constraints.
    contact = root.find('contact')
    for exclude in list(contact.findall('exclude')):
        contact.remove(exclude)
    for first, second in [('pelvis_link', 'torso_Link'), ('left_wrist_roll_Link', 'left_wrist_yaw_Link'), ('right_wrist_roll_Link', 'right_wrist_yaw_Link')]:
        ET.SubElement(contact, 'exclude', body1=first, body2=second)
    ET.indent(tree, space='  ')
    tree.write(OUTPUT, encoding='unicode', xml_declaration=False)
    print(f'{OUTPUT}: {count} collision shapes, {len(existing_mesh)} mesh assets, 3 excluded pairs')


if __name__ == '__main__':
    build()
