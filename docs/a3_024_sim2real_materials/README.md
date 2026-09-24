# A3 Sim2Real Materials

This directory is the reviewable evidence package for the A3 sim-to-real modeling workflow. Each numbered folder is one technical question; its README is the index for the raw inputs, derived tables, code snapshot, figures, validation, and generated robot assets. Paths in this package are relative to this directory.

## Contents

| Topic | Question answered | Start here | Main raw or reference files | Derived / released outputs |
| --- | --- | --- | --- | --- |
| [`01_serial_parallel_torque_coupling/`](01_serial_parallel_torque_coupling/) | How are serial pitch/roll commands mapped to two physical motors? | [`01_serial_parallel_torque_coupling/README.md`](01_serial_parallel_torque_coupling/README.md) | Jacobian CSV grids, armature reference | Runtime NPZ tables, solver library, torque-space figure |
| [`02_tn_curves/`](02_tn_curves/) | Which motor torque-speed limits and reward are used? | [`02_tn_curves/README.md`](02_tn_curves/README.md) | `reward_024_tn_curves.csv`, reference config | T-N figure, code/config snapshots |
| [`03_passive_foot/`](03_passive_foot/) | How were the two passive-foot joints fitted and packaged? | [`03_passive_foot/README.md`](03_passive_foot/README.md) | Five GB18030 bench CSVs, manifest | Fit parameters, RMSE, URDF/MJCF and meshes |
| [`licenses/`](licenses/) | Third-party and source-license notices | — | — | — |
| [`tools/`](tools/) | Rebuild helper for Jacobian runtime tables | — | CSV grids from topic 01 | NPZ and optional diagnostic PNGs in a separate output directory |

## Authority and provenance

The files here are a frozen evidence and release package. Files under `code_snapshot/` are for review and provenance; the authoritative runtime implementations remain under `gear_sonic/`. Generated files are labelled as such in the topic READMEs. The package does not turn bench measurements into a claim of dynamic sim2real accuracy.

Use the SHA-256 values in `03_passive_foot/raw/raw_manifest.csv` to verify that the five raw CSV exports were not changed. The original GB18030 encoding, units, and source labels are preserved.
