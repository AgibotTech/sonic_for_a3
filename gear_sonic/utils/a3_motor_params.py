"""Canonical A3 motor parameters shared by training and MuJoCo sim2sim.

Keep these values in the training convention (joint-side reflected rotor
inertia).  IsaacLab actuator configs and the standalone MuJoCo runner both
import this module so that the two physics paths cannot silently diverge.
"""

ARMATURE_PFP_78_58 = 30.118e-6 * 20.03 * 20.03
ARMATURE_PFP_93_65 = 138.5069e-6 * 21.906 * 21.906
ARMATURE_PFP_41_48 = 1.359e-6 * 24.415 * 24.415
ARMATURE_PFP_59_60 = 10.1374e-6 * 22.136 * 22.136
ARMATURE_PFP_110_75 = 300.851e-6 * 20.0 * 20.0
