# A3 MuJoCo models

- `a3_motion_skeleton.xml`: open-chain reference skeleton for motion conversion and training.
- `a3_loop_template.xml`: closed-chain template used to regenerate the 089 simulation model.
- `a3_035_passive_foot_loop.xml`: 035 simulation model with two passive foot joints per side.
- `a3_089_rigid_foot_loop.xml`: 089 simulation model with rigid feet and its reviewed collision geometry.

Regenerate the 089 model with `python gear_sonic/data_process/build_a3_089_mjcf.py` after changing its URDF.
