import json
from unittest.mock import patch
from remote_access import status

def test_running_tailnet_returns_private_mobile_url(tmp_path):
    payload={'BackendState':'Running','Self':{'Online':True,'DNSName':'explorer.example.ts.net.'}}
    with patch('remote_access.subprocess.run') as run:
        run.return_value.stdout=json.dumps(payload)
        result=status(tmp_path)
    assert result['url']=='http://explorer.example.ts.net:8080/mobile'
    assert result['private_tailnet_only'] and not result['changes_default_routes']

def test_unavailable_is_not_reported_as_connected(tmp_path):
    with patch('remote_access.subprocess.run',side_effect=OSError):
        assert status(tmp_path)['connected'] is False
