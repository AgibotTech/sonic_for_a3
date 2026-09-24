#!/usr/bin/env python3
"""Unit tests for the Kimodo training-dataset sim2sim runner."""

from __future__ import annotations

import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

from run_kimodo_training_dataset_eval import main, select_clip_symlinks


class SelectClipSymlinksTest(unittest.TestCase):
    def test_selects_only_direct_csv_symlink_clips(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source"
            clips = root / "clips"
            source.mkdir()
            clips.mkdir()

            for stem in ("clip_b", "clip_a"):
                target = source / f"{stem}.csv"
                target.write_text("Frame\n0\n", encoding="utf-8")
                (clips / f"{stem}.csv").symlink_to(target)

            (clips / "manifest.csv").write_text("motion_key\nclip_a\n", encoding="utf-8")
            (clips / "ordinary.csv").write_text("Frame\n0\n", encoding="utf-8")
            (clips / "nested").mkdir()
            (clips / "nested" / "nested.csv").symlink_to(source / "clip_a.csv")

            self.assertEqual(
                [path.name for path in select_clip_symlinks(clips)],
                ["clip_a.csv", "clip_b.csv"],
            )

    def test_dry_run_keeps_native30_a3_fast_10ms_defaults(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source"
            clips = root / "clips"
            source.mkdir()
            clips.mkdir()
            target = source / "clip.csv"
            target.write_text("Frame\n0\n", encoding="utf-8")
            (clips / "clip.csv").symlink_to(target)
            stdout = StringIO()
            with redirect_stdout(stdout):
                result = main(
                    [
                        "--checkpoint",
                        str(root / "future.pt"),
                        "--clips-dir",
                        str(clips),
                        "--metrics-dir",
                        str(root / "metrics"),
                        "--dry-run",
                    ]
                )
            self.assertEqual(result, 0)
            output = stdout.getvalue()
            self.assertIn("source_fps=30", output)
            self.assertIn("frame_stride=1", output)
            self.assertIn("encoder_mode=a3_fast", output)
            self.assertIn("action_delay_ms=10", output)
            self.assertIn("--csv-source-fps 30.0", output)
            self.assertIn("--csv-frame-stride 1", output)
            self.assertIn("--encoder-mode a3_fast", output)
            self.assertIn("--action-delay-ms 10.0", output)
            self.assertNotIn("--output-video", output)
            self.assertFalse((root / "metrics").exists())

    def test_dry_run_emits_per_clip_preview_video_args_without_creating_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source"
            clips = root / "clips"
            source.mkdir()
            clips.mkdir()
            target = source / "clip.csv"
            target.write_text("Frame\n0\n", encoding="utf-8")
            (clips / "clip.csv").symlink_to(target)
            metrics_dir = root / "metrics"
            videos_dir = root / "videos"
            stdout = StringIO()
            with redirect_stdout(stdout):
                result = main(
                    [
                        "--checkpoint",
                        str(root / "future.pt"),
                        "--clips-dir",
                        str(clips),
                        "--metrics-dir",
                        str(metrics_dir),
                        "--videos-dir",
                        str(videos_dir),
                        "--preview-video",
                        "--dry-run",
                    ]
                )
            self.assertEqual(result, 0)
            output = stdout.getvalue()
            self.assertIn(f"--output-video {videos_dir / 'clip.mp4'}", output)
            self.assertIn("--preview-video", output)
            self.assertFalse(metrics_dir.exists())
            self.assertFalse(videos_dir.exists())


if __name__ == "__main__":
    unittest.main()
