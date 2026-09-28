#!/usr/bin/env python3
"""CLI client of the same authenticated laboratory API used by web and Telegram."""
import argparse
import json
from pathlib import Path
import time
import urllib.request
import urllib.error
import uuid

p=argparse.ArgumentParser(description='Explorer: анализ без движения')
p.add_argument('experiment',nargs='?',help='E01…E18; без аргумента — каталог')
p.add_argument('--mode',choices=('observe','shadow','replay'),default='observe')
p.add_argument('--params',default='{}',help='JSON параметров; replay: {"run_id":"…"}')
p.add_argument('--root',type=Path,default=Path('/home/vlad/Explorer'))
args=p.parse_args()
token=(args.root/'config/access_token').read_text().strip()
payload=None if not args.experiment else dict(experiment=args.experiment,mode=args.mode,
    params=json.loads(args.params),request_id=uuid.uuid4().hex,issued_at=time.time())
req=urllib.request.Request('http://127.0.0.1:8080/api/experiments'+('/run' if payload else ''),
    data=json.dumps(payload).encode() if payload else None,
    headers={'Authorization':'Bearer '+token,'Content-Type':'application/json'})
try:
    with urllib.request.urlopen(req,timeout=20) as response:result=json.load(response)
    print(json.dumps(result,ensure_ascii=False,indent=2))
except urllib.error.HTTPError as exc:
    raise SystemExit(json.load(exc).get('detail','Запрос отклонён')) from None
