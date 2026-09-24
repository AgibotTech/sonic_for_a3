#!/usr/bin/env python3
"""Compatibility entry point; prefer `python download_from_hf.py --component runtime`."""
from pathlib import Path
import runpy

if __name__ == '__main__':
    runpy.run_path(str(Path(__file__).resolve().parents[2] / 'download_from_hf.py'), run_name='__main__')
