# Third-Party and Asset Notices

## A3 robot description

The A3 URDF/MJCF and associated mesh assets in this release are modified from
the public [AgibotTech/A3-A3U-robot-model](https://github.com/AgibotTech/A3-A3U-robot-model).
The upstream project publishes them under the [Mulan Permissive Software
License v2](https://github.com/AgibotTech/A3-A3U-robot-model/blob/main/LICENSE.txt).
For self-contained redistribution, an unmodified copy of that upstream license
is included at [licenses/MulanPSL-2.0.txt](licenses/MulanPSL-2.0.txt).

When redistributing A3-derived assets, retain the upstream copyright and
license notices and comply with Mulan PSL v2.  This notice does not relicense
the upstream asset, does not grant trademark rights, and does not resolve the
terms for source code and 035 model artifacts in the repository [LICENSE](../LICENSE).

## Rockchip cross-build sysroot

`gear_sonic_deploy/thirdparty/rockchip_sysroot/rockchip-1.0-aarch64-sysroot.tar.gz`
is downloaded from [Hugging Face](https://huggingface.co/sonic-for-a3/sonic/tree/main/rockchip_sysroot) with
`python download_from_hf.py --component sysroot`. It is a metadata-sanitized, build-only AArch64 sysroot snapshot for the
documented Rockchip package build. It contains target ROS Jazzy headers and
libraries plus required system headers/libraries; it is not a Thor OS image or
a runtime root filesystem. The project owner has authorized distribution of
this snapshot as a build input. Preserve the third-party notices carried in
the extracted sysroot; this repository does not grant additional rights in its
ROS, operating-system, or other third-party components.

## Rockchip RKNN runtime

The bundled RKNN runtime is third-party vendor software, not Apache-2.0
project code. Original Rockchip header notices and the upstream LICENSE are
retained; see the [runtime redistribution note](../gear_sonic_deploy/thirdparty/rknn_runtime/README.md).

## 035 policy artifacts

The metadata-sanitized 035 step-200,000 PT checkpoint and matching ONNX/RKNN
artifacts are hosted on [Hugging Face](https://huggingface.co/sonic-for-a3/sonic/tree/main/035_step200000).
Use `python download_from_hf.py --component pt onnx rknn` from the repository root.
The source tree retains portable configurations and a pinned checksum manifest.
The 035 PT checkpoint, matching ONNX/RKNN exports and accompanying release
configurations are licensed under the [Apache License 2.0](../LICENSE).
Upstream copyright and third-party notices are retained; this grant does not
relicense third-party software, robot assets or datasets.

## Selected20 motion CSVs — distribution authorized for this release

The 20 flat A3 motion CSVs under `a3_data/agibot_a3/` originate from the `human_mocap_data_selected_20_inplace_sonic`
dataset snapshot.  Only the raw CSVs are retained;
derived PKLs, videos, and the workbook are excluded.  The project owner has
authorized distribution of these listed release files with this repository.

The supplied source snapshot has no separate data-license file.  The release
authorization applies only to the 20 listed CSVs;
it does not assert an upstream license, relicense the omitted source snapshot,
or authorize redistribution of its omitted derived PKLs, videos, or workbook.
Retain this provenance boundary and any applicable attribution when making a
derivative release.

## Passive-foot raw compression CSVs — distribution authorized for this release

The five original quasi-static force/displacement CSVs under
`docs/a3_024_sim2real_materials/03_passive_foot/raw/` are a compact, byte-identical subset
of the A3 metal-shoe toe-cantilever source archive. The archive did not contain
a separate data license. The repository [LICENSE](../LICENSE) is a source-code
license/notice and must not be treated as a grant for these measurements.

The project owner has authorized distribution of these five listed CSVs with
this repository. The release documents the source archive hash and each CSV
hash so that the included data are auditable. This authorization is limited to
the five release files: it does not assert a separate upstream data license or
authorize redistribution of the full archive, its photographs, Word reports,
or any other measurement material. Retain the provenance and coordinate-system
caveats recorded in the raw-data README when making a derivative release.

## Deploy-only motion CSVs — distribution authorized for this release

The project owner has authorized distribution of the following five deploy
reference clips with this repository:

- `gear_sonic_deploy/assets/a3_runtime/remote_motions/remote_forward_short.csv`
- `gear_sonic_deploy/assets/a3_runtime/remote_motions/remote_backward_short.csv`
- `gear_sonic_deploy/assets/a3_runtime/remote_motions/remote_turn_left_45.csv`
- `gear_sonic_deploy/assets/a3_runtime/remote_motions/remote_turn_right_45.csv`
- `gear_sonic_deploy/assets/a3_runtime/teleop_motions/BMD_0319_a3_filtered_20260319_164305__stand_Skeleton0.csv`

The four `remote_motions/` files are direction-key reference clips, and the
`teleop_motions/` file is the optional teleop standing/idle reference. They are
separate from the selected20 training/sim2sim set and do not enter its normal
playback playlist. Source clip labels in `remote_motions/manifest.txt` are
provenance labels only. No separate upstream license is asserted for any
larger source motion dataset or recording; this authorization applies only to
the five listed release files.

## Excluded material

The edited real-robot demonstration video and its cover under `media/a3/`
are included with the project owner's authorization. Other raw real-robot
recordings, raw robot telemetry, incident logs, private source motion datasets, and the full
passive-foot bench archive (including its photos and Word reports) are excluded. The compact
T–N tables, selected20 CSVs, the five raw passive-foot CSVs, the five
deploy-only reference clips, the passive-foot parameter summary, and the
edited demonstration video and cover are the
only listed data assets authorized for distribution here; the boundaries above
continue to apply.
