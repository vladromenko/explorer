"""Read-only status for the private phone connection."""
import json
from pathlib import Path
import subprocess

def status(root):
    root=Path(root)
    command=[str(root/'vendor/tailscale/tailscale'),'--socket='+str(root/'data/tailscale/tailscaled.sock'),'status','--json']
    try:
        result=subprocess.run(command,capture_output=True,text=True,timeout=3,check=True)
        data=json.loads(result.stdout);self_state=data.get('Self') or {}
        name=str(self_state.get('DNSName') or '').rstrip('.')
        connected=data.get('BackendState')=='Running' and self_state.get('Online') is True
        return {'connected':connected,'backend_state':data.get('BackendState','Unknown'),
                'url':'http://'+name+':8080/mobile' if connected and name else None,
                'private_tailnet_only':True,'changes_default_routes':False}
    except (OSError,ValueError,subprocess.SubprocessError):
        return {'connected':False,'backend_state':'Unavailable','url':None,
                'private_tailnet_only':True,'changes_default_routes':False}
