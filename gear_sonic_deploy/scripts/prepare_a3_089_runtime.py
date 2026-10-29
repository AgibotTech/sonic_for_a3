#!/usr/bin/env python3
"""Stage verified 089 ONNX artifacts from the 089 checkpoint bundle."""
import argparse
import hashlib
import shutil
from pathlib import Path

EXPECTED = {
    'g1': '9006131bab66eb63f4d3502ad38ddae32e794b5b76d743061a2fea5cfe62530a',
    'a3_fast': '7225c52ecc8ca8866f2b0ec58d938806fe6479fc07f7df9e88d2e310bcefa5bc',
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True, type=Path, help='089 artifact bundle directory')
    parser.add_argument('--include-rknn', action='store_true', help='also stage verified RKNN models and sidecars')
    args = parser.parse_args()
    sources = {}
    for encoder, expected in EXPECTED.items():
        path = args.source / 'onnx' / f'exp089_model_step_136000_{encoder}.onnx'
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != expected:
            raise SystemExit(f'SHA256 mismatch: {path}')
        sources[encoder] = path
    dest = Path(__file__).resolve().parents[1] / 'assets/a3_runtime/models/exp089_step136000'
    dest.mkdir(parents=True, exist_ok=True)
    for encoder, path in sources.items():
        target = dest / f'model_step_136000_{encoder}.onnx'
        shutil.copyfile(path, target)
        print(f'{encoder}: {target}')
    if args.include_rknn:
        rknn_expected = {
            'g1': 'ef220b56aabe85c73976a509ad6bb9f397bfc3bea93862ba97152830d73b3552',
            'a3_fast': 'bbc990cf417437c4ded7c066f55500bea091028af5f476bf7ca9860d5fd379bb',
        }
        rknn_dest = dest.parents[1] / 'rknn_models/exp089_step136000'
        for encoder, expected in rknn_expected.items():
            source = args.source / 'rknn' / f'exp089_model_step_136000_{encoder}.rknn'
            if hashlib.sha256(source.read_bytes()).hexdigest() != expected:
                raise SystemExit(f'SHA256 mismatch: {source}')
            sidecar = source.with_suffix('.rknn.json')
            if not sidecar.is_file():
                raise SystemExit(f'Missing RKNN sidecar: {sidecar}')
            rknn_dest.mkdir(parents=True, exist_ok=True)
            target = rknn_dest / f'model_step_136000_{encoder}.rknn'
            shutil.copyfile(source, target)
            shutil.copyfile(sidecar, target.with_suffix('.rknn.json'))
            print(f'{encoder} RKNN: {target}')


if __name__ == '__main__':
    main()
