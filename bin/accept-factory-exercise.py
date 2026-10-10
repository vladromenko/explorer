#!/usr/bin/env python3
"""Read saved physical exercise data, annotate observation, seal validated setup."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from factory_delivery import SCOPES, load_factory_settings, next_exercises, record_exercise, seal_factory_exercises


def main(argv=None):
    parser=argparse.ArgumentParser(description="Оформление уже выполненных упражнений factory delivery; без движения")
    parser.add_argument("--root",type=Path,default=ROOT)
    commands=parser.add_subparsers(dest="command",required=True)
    commands.add_parser("status")
    record=commands.add_parser("record")
    record.add_argument("--scope",choices=sorted(SCOPES),required=True)
    record.add_argument("--source",required=True)
    record.add_argument("--observation",action="append",required=True)
    record.add_argument("--observed",action="store_true")
    record.add_argument("--operator-result",choices=("passed","failed"),required=True)
    seal=commands.add_parser("seal")
    seal.add_argument("--record",action="append",required=True)
    seal.add_argument("--write-manifest",action="store_true")
    args=parser.parse_args(argv)
    try:
        if args.command=="record":
            result=record_exercise(args.root,args.scope,args.source,args.observation,args.observed,args.operator_result)
        elif args.command=="seal":
            result=seal_factory_exercises(args.root,args.record,args.write_manifest)
        else:
            result=next_exercises(args.root)
            try:
                settings=load_factory_settings(args.root)
                result.update(prerequisites_valid=True,destination=settings["destination"])
            except (OSError,ValueError,KeyError,TypeError,ImportError) as exc:
                result.update(prerequisites_valid=False,blocked_by=str(exc))
        print(json.dumps(result,ensure_ascii=False,allow_nan=False,indent=2))
        return 0
    except (OSError,ValueError,KeyError,TypeError,ImportError) as exc:
        print("Не сохранено: "+str(exc),file=sys.stderr)
        return 1


if __name__=="__main__":
    raise SystemExit(main())
