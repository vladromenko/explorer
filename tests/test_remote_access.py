import json
from unittest.mock import patch
from remote_access import status,local_address

def test_running_tailnet_returns_private_mobile_url(tmp_path):
    payload={'BackendState':'Running','Self':{'Online':True,'DNSName':'explorer.example.ts.net.'}}
    with patch('remote_access.local_address',return_value=('wlan0','192.168.0.177')), patch('remote_access.subprocess.run') as run:
        run.return_value.stdout=json.dumps(payload)
        result=status(tmp_path)
    assert result['url']=='http://explorer.example.ts.net:8080/mobile'
    assert result['local_url']=='http://192.168.0.177:8080/mobile'
    assert result['private_tailnet_only'] and not result['changes_default_routes']

def test_unavailable_is_not_reported_as_connected(tmp_path):
    with patch('remote_access.local_address',return_value=(None,None)), patch('remote_access.subprocess.run',side_effect=OSError):
        assert status(tmp_path)['connected'] is False

def test_local_private_address():
    payload=[{'ifname':'lo','addr_info':[{'local':'127.0.0.1'}]},
             {'ifname':'wlan0','addr_info':[{'local':'192.168.0.177'}]}]
    with patch('remote_access.subprocess.run') as run:
        run.return_value.stdout=json.dumps(payload)
        assert local_address()==('wlan0','192.168.0.177')
