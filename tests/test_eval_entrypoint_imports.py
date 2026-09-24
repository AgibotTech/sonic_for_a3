"""Direct eval invocation must use this checkout, not an editable sibling."""
import ast
from pathlib import Path
import subprocess
import sys


def test_eval_bootstrap_precedes_installed_checkout(tmp_path):
    root = Path(__file__).resolve().parents[1]
    # A different editable installation earlier on sys.path reproduces the bug.
    other = tmp_path / 'other'
    (other / 'gear_sonic').mkdir(parents=True)
    (other / 'gear_sonic/__init__.py').write_text('')
    script = root / 'gear_sonic/eval_agent_trl.py'
    tree = ast.parse(script.read_text())
    bootstrap = []
    for node in tree.body:
        if isinstance(node, ast.Try):
            break  # Do not import/start Isaac Lab in this import-resolution test.
        bootstrap.append(node)
    source = ast.unparse(ast.Module(body=bootstrap, type_ignores=[]))
    check = f'''
import importlib.util, sys
from pathlib import Path
sys.path[:0] = [{str(script.parent)!r}, {str(other)!r}, {str(root)!r}]
exec({source!r}, {{'__file__': {str(script)!r}}})
assert Path(importlib.util.find_spec('gear_sonic').origin).parent == Path({str(root / 'gear_sonic')!r})
assert {str(script.parent)!r} not in sys.path
'''
    subprocess.run([sys.executable, '-c', check], check=True)
