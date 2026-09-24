# A3 motor-model Jacobian tables

These tables contain
`J = d(pitch, roll) / d(motor_0, motor_1)` for the left ankle, right
ankle, and waist parallel mechanisms. They were scanned from the x86 solver
mirror used by this repository:

`gear_sonic/data/assets/robot_description/solver/a3_loop/lib/liba3_ankle_waist_solver.a`

X86 solver SHA-256:

`ceced0794654c6534a1fd444dce123ad6dfdacb132725873edc63039e8c5be15`

The real A3 HAL uses the same `AnkleAnalyticalSolver` and
`WaistAnalyticalSolver` interfaces and calls `DIK`/`IDyn` before sending
parallel motor commands. The inspected ARM HAL artifacts had these SHA-256
values:

- `liba3_ankle_waist_solver.a`:
  `ef7c0ca5c10b3745584698b1ad77729fba564d835530a73be39c3f6ce0f75ce6`
- `liba3_ankle_waist_solver.so`:
  `414f5c466661ed7c360ec7da0067ebdc7d9b3385bd239935e39801b5b5e05e71`

The ARM and x86 public headers differ only in their include path. The ARM
binary cannot be executed in the x86 training environment, so the tests below
check the packaged tables against the x86 mirror used for training and
deployment validation. The human-readable CSV source grids and a portable
CSV-to-NPZ builder are in
[`docs/a3_024_sim2real_materials/01_serial_parallel_torque_coupling/jacobian_tables/`](../../../docs/a3_024_sim2real_materials/01_serial_parallel_torque_coupling/jacobian_tables/).
The runtime keeps the checked NPZ tables in this directory.

Table SHA-256 values:

- `left_ankle_jacobi.npz`:
  `0d2fe1c7c91ff6246db686d425ec0ee65ea07a3fc15e7b0c225ca3723a50107a`
- `right_ankle_jacobi.npz`:
  `512ee781069a9255e341b52393558e1a7249124a58fcfd9adee0bfc2da997adf`
- `waist_jacobi.npz`:
  `2b6189cc35ab55cade6d12a6f11085e10b865859eb357e15f438c96b15e1185c`

`tests/test_a3_motor_model.py` and `tests/test_a3_motor_reward.py` check the
packaged tables against the x86 C++ solver's inverse dynamics and differential
inverse kinematics at grid points.
