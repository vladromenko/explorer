#!/usr/bin/env python3
"""Query the same read-only readiness summary used by the web interface."""
import json
from pathlib import Path
import urllib.request
token=(Path('/home/vlad/Explorer/config/access_token')).read_text().strip()
request=urllib.request.Request('http://127.0.0.1:8080/api/readiness',headers={'Authorization':'Bearer '+token})
with urllib.request.urlopen(request,timeout=10) as response:
    print(json.dumps(json.load(response),ensure_ascii=False,indent=2))
