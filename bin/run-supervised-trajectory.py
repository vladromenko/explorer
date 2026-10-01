#!/usr/bin/env python3
"""Keep the existing web trajectory lease while a nearby observer watches."""
import argparse
import json
from pathlib import Path
import time
import urllib.request
import urllib.error

ROOT=Path('/home/vlad/Explorer')
parser=argparse.ArgumentParser();parser.add_argument('plan_id');args=parser.parse_args()
token=(ROOT/'config/access_token').read_text().strip()
def api(path,body=None):
    request=urllib.request.Request('http://127.0.0.1:8080/api/'+path,
        data=None if body is None else json.dumps(body).encode(),
        headers={'Authorization':'Bearer '+token,'Content-Type':'application/json'})
    with urllib.request.urlopen(request,timeout=15) as response:return json.load(response)
state=api('arm/trajectory/start',dict(plan_id=args.plan_id,observing=True));session=state['session']
deadline=time.monotonic()+300
while state.get('busy') and time.monotonic()<deadline:
    time.sleep(.2)
    try:state=api('arm/trajectory/lease',dict(session=session,held=True))
    except urllib.error.HTTPError as exc:
        if exc.code!=409:raise
        state=api('arm/trajectory')
if state.get('busy'):
    api('arm/trajectory/stop',{});raise SystemExit('Trajectory exceeded observer deadline')
print(json.dumps(state,ensure_ascii=False))
