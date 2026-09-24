# A3 035 step-200,000 checkpoint

Download the PT and matching portable configuration from [Hugging Face](https://huggingface.co/sonic-for-a3/sonic/tree/main/035_step200000), from the source repository root:

```bash
python -m pip install huggingface_hub
python download_from_hf.py --component pt
(cd checkpoints/035_step200000 && sha256sum -c SHA256SUMS)
```

035 uses the VR3 / no-ankle-orientation training recipe. `tracking_vr_5point_local` uses reward weight 3.0 for torso and wrists. Use the adjacent `config.yaml`, `model_config.yaml` and `meta.yaml` for export / evaluation.

The public PT preserves policy, value, optimizer and scheduler states exactly. Internal training arguments, run metadata and motion-library state are omitted; step 200,000 and the learning rate are retained. It supports inference and fine-tuning, but is not an exact original-environment resume snapshot. The portable configuration is a path-scrubbed derivative of the original resolved training configuration.

The 035 PT checkpoint, matching ONNX/RKNN exports and accompanying release configurations are licensed under the [Apache License 2.0](LICENSE). The binary is ignored by Git; the source repository retains this download entry and checksum only.
