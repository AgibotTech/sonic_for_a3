"""Preflight must distinguish Git LFS assets from HF model downloads."""
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

spec = importlib.util.spec_from_file_location('preflight', Path(__file__).resolve().parents[1] / 'check_environment.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_lfs_missing_pointer_and_real_asset(tmp_path, monkeypatch):
    monkeypatch.setattr(module.shutil, 'which', lambda _: '/usr/bin/git-lfs')
    monkeypatch.setattr(module.subprocess, 'run', lambda *a, **k: SimpleNamespace(returncode=0, stdout='mesh.STL\n'))
    assert not module.check_git_lfs(tmp_path)
    mesh = tmp_path / 'mesh.STL'
    mesh.write_text('version https://git-lfs.github.com/spec/v1\noid sha256:example\n')
    assert not module.check_git_lfs(tmp_path)
    mesh.write_bytes(b'actual mesh content')
    assert module.check_git_lfs(tmp_path)


def test_hf_missing_corrupt_and_correct_artifact(tmp_path):
    data = b'model'
    entry = dict(group='onnx', destination='model.onnx', size=len(data), sha256=hashlib.sha256(data).hexdigest())
    (tmp_path / 'a3_hf_manifest.json').write_text(json.dumps({'files': [entry]}))
    assert not module.check_hf_artifacts(['onnx'], tmp_path)
    (tmp_path / 'model.onnx').write_bytes(b'wrong')
    assert not module.check_hf_artifacts(['onnx'], tmp_path)
    (tmp_path / 'model.onnx').write_bytes(data)
    assert module.check_hf_artifacts(['onnx'], tmp_path)
    assert module.check_hf_artifacts([], tmp_path)
