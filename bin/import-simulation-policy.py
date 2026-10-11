#!/usr/bin/env python3
"""Import a checked simulation policy as an inactive candidate, not motion."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from simulator.transfer import import_archive

parser = argparse.ArgumentParser()
parser.add_argument("archive", type=Path)
parser.add_argument("--robot-root", type=Path, default=Path("/home/vlad/Explorer"))
parser.add_argument("--register", action="store_true", help="Register an inactive simulator candidate")
args = parser.parse_args()
print(json.dumps(import_archive(args.archive, args.robot_root, register=args.register), ensure_ascii=False, indent=2))
