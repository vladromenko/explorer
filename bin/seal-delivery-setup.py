#!/usr/bin/env python3
"""Bind existing passed physical records to delivery settings; no robot commands."""
import argparse
import json
from pathlib import Path
import sys


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1])
    parser.add_argument('--record',action='append',required=True,
                        help='Existing physical acceptance JSON; repeat for controller, arm and geometry')
    parser.add_argument('--check',action='store_true',help='Check records/settings without writing acceptance file')
    args=parser.parse_args();root=args.root.resolve()
    sys.path.insert(0,str(root/'src'))
    from delivery_robot import seal_setup
    profile=json.loads((root/'config/controller-profile.json').read_text())
    result=seal_setup(root,profile['firmware_source_sha256'],profile['calibration_sha256'],args.record,write=not args.check)
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':
    try:main()
    except (OSError,ValueError,KeyError,TypeError) as exc:
        raise SystemExit('Приёмка не оформлена; робот не запускался: '+str(exc))
