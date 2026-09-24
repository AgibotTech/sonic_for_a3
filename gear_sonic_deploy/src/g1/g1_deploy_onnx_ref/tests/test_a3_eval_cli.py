#!/usr/bin/env python3

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "tools" / "a3_eval_cli.py"
SPEC = importlib.util.spec_from_file_location("a3_eval_cli", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class EvalCliTest(unittest.TestCase):
    def test_default_rpc_port_uses_dedicated_motion_endpoint(self):
        self.assertEqual(MODULE.RPC_BASE, "http://127.0.0.1:50901")

    def test_counts_only_completed_nonempty_mcap(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "eval_records"
            data = root / "session_test" / "data"
            data.mkdir(parents=True)
            (data / "episode.mcap").write_bytes(b"mcap")
            (data / "episode.meta.yaml").write_text(
                """
meta_schema_version: 1
mcap:
  file: episode.mcap
episode:
  motion_name: 001_walk
  motion_source_path: /tmp/motions/001_walk.csv
  outcome: COMPLETED
""",
                encoding="utf-8",
            )
            (data / "missing.meta.yaml").write_text(
                """
mcap:
  file: missing.mcap
episode:
  motion_source_path: /tmp/motions/001_walk.csv
  outcome: COMPLETED
""",
                encoding="utf-8",
            )
            self.assertEqual(
                MODULE.scan_completed_recordings(root), {"001_walk.csv": 1}
            )

    def test_attempt_state_round_trip(self):
        with tempfile.TemporaryDirectory() as temp:
            state = Path(temp) / "eval_records" / "state.json"
            MODULE.save_attempts({"001_walk.csv": 2}, state)
            self.assertEqual(
                json.loads(state.read_text(encoding="utf-8"))["attempts"],
                {"001_walk.csv": 2},
            )
            self.assertEqual(
                MODULE.load_attempts(state), {"001_walk.csv": 2}
            )

    def test_finds_video_with_matching_basename(self):
        with tempfile.TemporaryDirectory() as temp:
            old_work_dir = MODULE.WORK_DIR
            try:
                MODULE.WORK_DIR = Path(temp)
                video = Path(temp) / "videos" / "001_walk.mp4"
                video.parent.mkdir()
                video.write_bytes(b"video")
                motion = MODULE.Motion(
                    path=Path(temp) / "csv" / "001_walk.csv",
                    name="001_walk",
                )
                self.assertEqual(MODULE.find_video(motion), video.resolve())
            finally:
                MODULE.WORK_DIR = old_work_dir


if __name__ == "__main__":
    unittest.main()
