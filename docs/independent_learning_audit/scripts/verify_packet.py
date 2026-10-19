"""Verify portable payload hashes, without importing robotics or training code."""
import hashlib
import json
from pathlib import Path
import sys


def verify(root):
    root = Path(root).resolve()
    manifest = json.loads((root / 'audit_manifest.json').read_text())
    errors = []
    for row in manifest['files']:
        p = (root / row['path']).resolve()
        if not p.is_relative_to(root):
            errors.append('Unsafe path: ' + row['path'])
        elif not p.is_file():
            errors.append('Missing: ' + row['path'])
        elif p.stat().st_size != row['bytes'] or hashlib.sha256(p.read_bytes()).hexdigest() != row['sha256']:
            errors.append('Hash/size mismatch: ' + row['path'])
    return errors


if __name__ == '__main__':
    problems = verify(sys.argv[1] if len(sys.argv) > 1 else Path(__file__).resolve().parents[1])
    print(json.dumps(dict(ok=not problems, errors=problems), ensure_ascii=False))
    raise SystemExit(bool(problems))
