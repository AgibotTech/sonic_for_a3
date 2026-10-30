# A3 T3D0 AimRT MuJoCo simulator

This standalone simulator publishes A3 body-drive state and receives joint
commands over iceoryx. It supports standard A3 T3D0 (`a3_t3d0`) and A3U T3D0
(`a3u_t3d0`), without the ping-pong robot or its racket-specific topics.
The A3U neck pitch is mapped to the `ultra_head_pitch_joint` model joint.

## Build and start

On a Linux x86-64 machine with ROS 2 Humble and the build dependencies listed
in the upstream AimRT/MuJoCo project:

```bash
cd aimrt_mujoco_sim_a3
source /opt/ros/humble/setup.bash
JOBS=10 ./build.sh
cd build/install/bin
./run_a3_sim.sh a3_t3d0    # or a3u_t3d0
```

The first build fetches AimRT and MuJoCo dependencies. The launcher starts
iceoryx RouDi if none is running. It opens the MuJoCo viewer and requires a
working graphical display. In a separate terminal start the 089 x86 deploy
package with `A3_TRANSPORT=iceoryx ./run_a3.sh`.

The model and transport configurations are under `src/models/bin/cfg/`. These
are standard serial-joint simulation models; the A3 model is not the 089
training closed-loop collision/dynamic ankle model. Simulation is not a
substitute for validation of the physical robot.

## Provenance

The simulator source is derived from `aimrt_mujoco_sim_a3` revision
`61ad378ad88937214aee738a23224f7944784948`; its Mulan PSL v2 license is in
[LICENSE](LICENSE). A3/A3U robot model assets originate from AgiBot's A3 model
release; see [third-party notices](../docs/THIRD_PARTY_NOTICES.md) and
[Mulan PSL v2](../licenses/MulanPSL-2.0.txt).
