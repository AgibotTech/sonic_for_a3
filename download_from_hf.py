#!/usr/bin/env python3
"""Download A3 035 / 200k artifacts into their configured source-tree paths.

Examples (run from the repository root):
  python download_from_hf.py --component pt
  python download_from_hf.py --component onnx rknn
  python download_from_hf.py --component sysroot
  python download_from_hf.py --component all

Defaults to runtime models (ONNX + RKNN). Every file is size/SHA-256 checked.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile

ROOT = Path(__file__).resolve().parent


def matches(path, entry):
    if not path.is_file() or path.stat().st_size != entry['size']:
        return False
    sha = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            sha.update(chunk)
    return sha.hexdigest() == entry['sha256']


def install_file(entry, output_dir, download, repo_id, revision, token=None, force=False):
    destination = output_dir / entry['destination']
    if not force and matches(destination, entry):
        print(f"[verified] {entry['destination']}")
        return
    cached = Path(download(repo_id=repo_id, filename=entry['remote'], revision=revision,
                           token=token, force_download=force))
    if not matches(cached, entry):
        raise RuntimeError(f"Size/SHA-256 mismatch: {entry['remote']}; retry with --force")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=destination.parent, prefix='.hf-', delete=False) as f:
            temporary = Path(f.name)
            with cached.open('rb') as source:
                shutil.copyfileobj(source, f)
        if not matches(temporary, entry):
            raise RuntimeError(f"Copy verification failed: {destination}")
        temporary.chmod(0o644)
        os.replace(temporary, destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    print(f"[downloaded] {entry['destination']}")


def main(argv=None):
    manifest = json.loads((ROOT / 'a3_hf_manifest.json').read_text())
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--component', nargs='+', choices=['pt', 'onnx', 'rknn', 'sysroot', 'runtime', 'all'],
                        default=['runtime'], help='one or more groups; default: runtime (ONNX + RKNN)')
    parser.add_argument('--output-dir', type=Path, default=ROOT, help='source-tree root to populate')
    parser.add_argument('--repo-id', default=os.environ.get('A3_HF_REPO_ID', manifest['repo_id']))
    parser.add_argument('--revision', default=os.environ.get('A3_HF_REVISION', manifest['revision']),
                        help='defaults to the pinned release commit; checksums always remain enforced')
    parser.add_argument('--token', default=os.environ.get('HF_TOKEN'))
    parser.add_argument('--force', action='store_true', help='redownload and replace even verified files')
    parser.add_argument('--list', action='store_true', help='show selected files without downloading')
    args = parser.parse_args(argv)
    groups = set(args.component)
    if 'all' in groups:
        groups.update(['pt', 'onnx', 'rknn', 'sysroot'])
    if 'runtime' in groups:
        groups.update(['onnx', 'rknn'])
    entries = [e for e in manifest['files'] if e['group'] in groups]
    print(f"Repository: {args.repo_id} @ {args.revision}")
    print(f"Selected: {len(entries)} files, {sum(e['size'] for e in entries) / 1e6:.1f} MB")
    if args.list:
        for e in entries:
            print(f"{e['group']}: {e['remote']} -> {e['destination']}")
        return 0
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        parser.error('Install the dependency: python -m pip install huggingface_hub')
    for entry in entries:
        install_file(entry, args.output_dir, hf_hub_download, args.repo_id, args.revision, args.token, args.force)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
