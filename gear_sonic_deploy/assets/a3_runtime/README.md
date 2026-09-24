# A3 Runtime Assets — 035 step 200,000

This directory contains the small, runtime-required asset set for the public
A3 035 release.  The source configuration
[`../../src/g1/g1_deploy_onnx_ref/config/a3_runtime_config.yaml`](../../src/g1/g1_deploy_onnx_ref/config/a3_runtime_config.yaml)
selects this exact model pair by default.

| Artifact | Source-release state | Contract |
| --- | --- | --- |
| `models/035_step200000/model_step_200000_g1.onnx` | Hugging Face download | `obs_dict: float32[1,1570]` → `action: float32[1,29]` |
| `models/035_step200000/model_step_200000_a3_fast.onnx` | Hugging Face download | `obs_dict: float32[1,1570]` → `action: float32[1,29]` |
| `rknn_models/035_step200000/model_step_200000_g1.rknn` | Hugging Face download | RK3588, FP16/non-quantized |
| `rknn_models/035_step200000/model_step_200000_a3_fast.rknn` | Hugging Face download | RK3588, FP16/non-quantized |

The ONNX and RKNN binaries are hosted on [Hugging Face](https://huggingface.co/sonic-for-a3/sonic/tree/main/035_step200000).
From the repository root, download them before inference, conversion, or deployment:

```bash
python -m pip install huggingface_hub
python download_from_hf.py --component onnx rknn
# Download only one format with --component onnx or --component rknn.
```

The `.rknn.json` sidecars document the expected source-model contract. The PT
checkpoint is downloaded with `python download_from_hf.py --component pt` under [`../../../checkpoints/035_step200000/`](../../../checkpoints/035_step200000/README.md)
and is also published separately. The downloader verifies all files against the pinned release manifest before installing them.

Other assets:

- The normal playback playlist is the 20 selected 120 Hz A3 flat CSVs under
  [`../../../a3_data/agibot_a3/`](../../../a3_data/agibot_a3/).  The source
  runtime configuration points there directly so the repository carries one
  canonical copy; the package builder copies all 20 into package-local
  `motions/`. Runtime keeps every fourth row, treats the resulting stream as
  30 Hz, then resamples to the 50 Hz policy timeline.
- `remote_motions/*.csv`: direction-key clips; they are independent of the
  normal playlist.
- `teleop_motions/*.csv`: the idle/startup reference for the optional teleop
  path.

The 20 normal-playback CSVs and these five deploy-only reference clips are
separately scoped data assets. The project owner has authorized their
distribution with this release; the deploy-only set does not imply a license
for a larger source motion dataset or recording. See
[`docs/THIRD_PARTY_NOTICES.md`](../../../docs/THIRD_PARTY_NOTICES.md) for the
exact file list and provenance boundary.

Rerun the downloader to verify existing files before a release or hardware bring-up.
