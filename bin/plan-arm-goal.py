#!/usr/bin/env python3
"""Request a fresh MoveIt plan through the single web owner."""
import argparse
import json
from pathlib import Path
import urllib.request

ROOT=Path('/home/vlad/Explorer')
parser=argparse.ArgumentParser();parser.add_argument('degrees',nargs=6,type=int);args=parser.parse_args()
token=(ROOT/'config/access_token').read_text().strip()
request=urllib.request.Request('http://127.0.0.1:8080/api/arm/trajectory/plan',
    data=json.dumps({'goal_deg':args.degrees}).encode(),
    headers={'Authorization':'Bearer '+token,'Content-Type':'application/json'})
with urllib.request.urlopen(request,timeout=180) as response:print(json.dumps(json.load(response),ensure_ascii=False))
