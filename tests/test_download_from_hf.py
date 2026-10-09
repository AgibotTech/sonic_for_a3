"""Downloader integrity and component selection regression tests (no network)."""
import contextlib
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('a3_download', ROOT / 'download_from_hf.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class DownloadTests(unittest.TestCase):
    def test_corrupt_download_preserves_existing_destination(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            destination = root / 'model.pt'
            destination.write_bytes(b'old')
            cached = root / 'cache'
            cached.write_bytes(b'bad')
            entry = dict(destination='model.pt', remote='remote.pt', size=3,
                         sha256=hashlib.sha256(b'new').hexdigest())
            def download(**kwargs):
                self.assertEqual(kwargs['revision'], 'pinned')
                return cached
            with self.assertRaisesRegex(RuntimeError, 'mismatch'):
                module.install_file(entry, root, download, 'repo', 'pinned')
            self.assertEqual(destination.read_bytes(), b'old')
            cached.write_bytes(b'new')
            module.install_file(entry, root, download, 'repo', 'pinned')
            self.assertEqual(destination.read_bytes(), b'new')
            def unexpected_download(**kwargs):
                self.fail('verified existing file must not download')
            module.install_file(entry, root, unexpected_download, 'repo', 'pinned')
            self.assertFalse(list(root.glob('.hf-*')))

    def test_each_group_selects_only_requested_artifacts(self):
        manifest = json.loads((ROOT / 'a3_hf_manifest.json').read_text())
        for group in ['pt', 'onnx', 'rknn', 'sysroot']:
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                module.main(['--component', group, '--list'])
            for entry in manifest['files']:
                expected = entry['group'] == group or (entry['group'] == 'model_metadata' and group != 'sysroot')
                self.assertEqual(entry['destination'] in out.getvalue(), expected)

    def test_shared_metadata_selected_once_and_not_for_sysroot(self):
        manifest = json.loads((ROOT / 'a3_hf_manifest.json').read_text())
        for groups in [['pt'], ['onnx'], ['rknn'], ['onnx', 'rknn'], ['runtime'], ['all']]:
            entries = module.selected_entries(manifest, groups)
            metadata = [e for e in entries if e['group'] == 'model_metadata']
            self.assertEqual(len(metadata), 1)
            self.assertEqual(metadata[0]['remote'], 'config.json')
        self.assertTrue(all(e['group'] == 'sysroot' for e in module.selected_entries(manifest, ['sysroot'])))

    def test_manifest_has_safe_unique_destinations(self):
        entries = json.loads((ROOT / 'a3_hf_manifest.json').read_text())['files']
        self.assertEqual(len(entries), len({e['destination'] for e in entries}))
        for entry in entries:
            self.assertFalse(Path(entry['destination']).is_absolute())
            self.assertNotIn('..', Path(entry['destination']).parts)
            self.assertEqual(len(entry['sha256']), 64)


if __name__ == '__main__':
    unittest.main()
