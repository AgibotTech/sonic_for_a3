#!/usr/bin/env python3
"""Export only 024/035 scalar display data from a local TensorBoard to Pages.

Run explicitly when refreshing the public snapshot; no event files, text logs,
wall-clock timestamps, machine paths, or other runs are exported.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from urllib.parse import urlencode, urlparse
from urllib.request import ProxyHandler, build_opener


def export(base, output):
    parsed = urlparse(base)
    if parsed.scheme != 'http' or parsed.hostname not in ('127.0.0.1', 'localhost', '::1'):
        raise ValueError('Use a loopback HTTP TensorBoard endpoint')
    def get(route, **query):
        url = base.rstrip('/') + route + ('?' + urlencode(query) if query else '')
        with build_opener(ProxyHandler({})).open(url, timeout=60) as response:
            return json.load(response)
    tags = get('/data/plugin/scalars/tags')
    runs = {}
    for label in ('024', '035'):
        matches = [name for name in tags if name.startswith(label + '_')]
        if len(matches) != 1:
            raise ValueError(f'Expected exactly one {label} run; got {len(matches)}')
        runs[label] = matches[0]
    tasks = [(label, tag) for label, run in runs.items() for tag in sorted(tags[run])]
    def fetch(task):
        label, tag = task
        rows = get('/data/plugin/scalars/scalars', run=runs[label], tag=tag)
        points = [[int(step), float(value) if math.isfinite(value) else None]
                  for _, step, value in rows]
        return label, tag, points
    data = {'schema_version': 1, 'exported_at': datetime.now(timezone.utc).isoformat(),
            'source': 'TensorBoard scalar display API',
            'sampling': 'Display snapshot; TensorBoard may sample up to 5,000 points per series. Not the full event log.',
            'runs': {'024': {}, '035': {}}}
    with ThreadPoolExecutor(max_workers=8) as pool:
        for label, tag, points in pool.map(fetch, tasks):
            if not points:
                raise ValueError(f'Empty scalar series: {label}/{tag}')
            data['runs'][label][tag] = points
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(data, separators=(',', ':'), allow_nan=False) + '\n')
    for label, series in data['runs'].items():
        steps = [p[0] for points in series.values() for p in points]
        print(label, 'metrics:', len(series), 'points:', len(steps), 'steps:', min(steps), max(steps))
    print('Export bytes:', output.stat().st_size)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tensorboard-url', default='http://127.0.0.1:16006')
    parser.add_argument('--output', type=Path, default=Path('site/training/scalars.json'))
    args = parser.parse_args()
    export(args.tensorboard_url, args.output)
