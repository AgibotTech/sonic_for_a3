# A3 robot descriptions

- `a3_035_passive_foot.urdf`: 035 / 200k training model with two passive foot joints per side.
- `a3_089_rigid_foot.urdf`: 089 / 136k training model with rigidly merged feet and reviewed self-collision pairs.
- `meshes/visual/`: body and foot visual geometry shared by both models and the A3 MuJoCo assets.
- `meshes/collision_035/`: 035 collision geometry.
- `meshes/collision_089/`: 089 collision meshes named by body; primitive collision shapes are in the URDF.

The 089 collision asset hash and per-mesh hashes are recorded in `gear_sonic/config/collision/a3_089_rigid_foot_pairs.json`.
