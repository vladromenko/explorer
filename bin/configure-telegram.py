#!/usr/bin/env python3
"""Hidden SSH token entry and read-only Telegram identity/webhook validation."""
import getpass
import json
import os
from pathlib import Path
import re
import urllib.request

root=Path('/home/vlad/Explorer')
secret=Path.home()/'.config/explorer/secrets/telegram-token'
secret.parent.mkdir(parents=True,exist_ok=True,mode=0o700)
os.chmod(secret.parent,0o700)
token=getpass.getpass('Telegram token (hidden): ').strip()
if not re.fullmatch(r'8850343219:[A-Za-z0-9_-]{30,60}',token):
    raise SystemExit('Token format or expected bot ID mismatch')
network=urllib.request.build_opener(urllib.request.ProxyHandler({'https':'http://127.0.0.1:18780'}))
def call(method):
    request=urllib.request.Request('https://api.telegram.org/bot'+token+'/'+method,data=b'{}',headers={'Content-Type':'application/json'})
    try:
        with network.open(request,timeout=15) as response:result=json.load(response)
    except Exception as exc:
        raise SystemExit('Telegram check failed: '+type(exc).__name__) from None
    if not result.get('ok'):raise SystemExit('Telegram rejected request')
    return result['result']
me=call('getMe')
if me['id']!=8850343219:raise SystemExit('Unexpected bot identity')
hook=call('getWebhookInfo')
if hook.get('url'):raise SystemExit('Existing webhook found; configuration was not changed')
fd=os.open(secret,os.O_WRONLY|os.O_CREAT|os.O_TRUNC,0o600)
with os.fdopen(fd,'w') as f:f.write(token)
config_path=root/'config/telegram.json'
config=json.loads(config_path.read_text()) if config_path.exists() else {}
config.update(enabled=True,https_proxy='http://127.0.0.1:18780',token_file=str(secret),bot_id=me['id'],bot_username=me['username'])
config.setdefault('allowed_user_ids',[]);config.setdefault('allowed_chat_ids',[])
fd=os.open(config_path,os.O_WRONLY|os.O_CREAT|os.O_TRUNC,0o600)
with os.fdopen(fd,'w') as f:json.dump(config,f)
print(json.dumps(dict(validated=True,bot_id=me['id'],username=me['username'],webhook_present=False,paired=bool(config['allowed_user_ids']))))
