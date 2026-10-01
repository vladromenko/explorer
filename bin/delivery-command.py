#!/usr/bin/env python3
"""Local CLI for the same authenticated delivery endpoint used by the web UI."""
import json
from pathlib import Path
import sys
import urllib.error
import urllib.request
operation=sys.argv[1] if len(sys.argv)>1 else 'status'
if operation not in ('start','status','cancel'):
    raise SystemExit('Usage: explorer delivery [start|status|cancel]')
root=Path('/home/vlad/Explorer');token=(root/'config/access_token').read_text().strip()
path='/api/delivery'+('' if operation=='status' else '/'+operation)
request=urllib.request.Request('http://127.0.0.1:8080'+path,
    data=None if operation=='status' else b'{}',
    headers={'Authorization':'Bearer '+token,'Content-Type':'application/json'})
try:
    with urllib.request.urlopen(request,timeout=10) as response:result=json.load(response)
except urllib.error.HTTPError as exc:
    print(exc.read().decode());raise SystemExit(1)
print(json.dumps(result,ensure_ascii=False,indent=2))
