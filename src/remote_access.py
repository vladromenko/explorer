"""Read-only status for the private phone connection."""
import json
import ipaddress
from pathlib import Path
import subprocess

def local_address():
    """Return a private LAN address without changing routes or DNS."""
    try:
        result=subprocess.run(['ip','-j','-4','addr','show','up'],capture_output=True,text=True,timeout=2,check=True)
        candidates=[]
        for interface in json.loads(result.stdout):
            name=interface.get('ifname','')
            if name in ('lo','docker0','l4tbr0') or name.startswith(('tailscale','br-','veth')):continue
            for value in interface.get('addr_info',[]):
                address=value.get('local','')
                try:parsed=ipaddress.ip_address(address)
                except ValueError:continue
                if parsed.version==4 and parsed.is_private:candidates.append((name,address))
        return candidates[0] if candidates else (None,None)
    except (OSError,ValueError,subprocess.SubprocessError):
        return (None,None)

def status(root):
    root=Path(root)
    interface,local_ip=local_address()
    local_url='http://'+local_ip+':8080/mobile' if local_ip else None
    command=[str(root/'vendor/tailscale/tailscale'),'--socket='+str(root/'data/tailscale/tailscaled.sock'),'status','--json']
    try:
        result=subprocess.run(command,capture_output=True,text=True,timeout=3,check=True)
        data=json.loads(result.stdout);self_state=data.get('Self') or {}
        name=str(self_state.get('DNSName') or '').rstrip('.')
        connected=data.get('BackendState')=='Running' and self_state.get('Online') is True
        remote_url='http://'+name+':8080/mobile' if connected and name else None
        return {'connected':connected,'backend_state':data.get('BackendState','Unknown'),
                'url':remote_url,'tailscale_url':remote_url,
                'local_url':local_url,'local_mdns_url':'http://explorer.local:8080/mobile',
                'local_interface':interface,'local_access_enabled':bool(local_url),
                'private_tailnet_only':True,'changes_default_routes':False}
    except (OSError,ValueError,subprocess.SubprocessError):
        return {'connected':False,'backend_state':'Unavailable','url':None,'tailscale_url':None,
                'local_url':local_url,'local_mdns_url':'http://explorer.local:8080/mobile',
                'local_interface':interface,'local_access_enabled':bool(local_url),
                'private_tailnet_only':True,'changes_default_routes':False}
