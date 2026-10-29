# A3 089 step-136,000 checkpoint

This directory prepares the 089 / 136k release. The PT is ignored by Git and is currently available locally; no public download has been published yet. Verify it with `sha256sum -c SHA256SUMS`.

The checkpoint preserves the original policy, value, optimizer and scheduler tensors, and the training step. Internal training arguments and motion-library state are omitted; it is not an exact training-resume snapshot. The adjacent path-scrubbed configuration includes the later rigid-foot collision model, self-contact terms and dynamic ankle model. The public training example uses a different motion dataset and is not an exact original-environment resume.

The checkpoint provides G1 and A3-fast encoders, not an SMPL encoder. The matching ONNX and RKNN artifacts are not yet publicly hosted.

The release checkpoint and configuration are provided under the [Apache License 2.0](LICENSE); the binary is distributed separately from Git.
