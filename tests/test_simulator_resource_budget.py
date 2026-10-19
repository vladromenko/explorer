from types import SimpleNamespace

from simulator import resource_budget as budget


def test_macos_budget_records_kernel_estimate_without_lowering_reserve(monkeypatch):
    monkeypatch.setattr(budget.sys,'platform','darwin')
    monkeypatch.setattr(budget.psutil,'virtual_memory',lambda:SimpleNamespace(total=1000,available=170))
    calls=[]
    def run(command,**kwargs):
        calls.append(command);assert kwargs['timeout']==2
        return SimpleNamespace(stdout='Stats: 12\nSystem-wide memory free percentage: 34%\n')
    monkeypatch.setattr(budget.subprocess,'run',run)
    result=budget.memory_budget()
    assert calls==[['/usr/bin/memory_pressure','-Q']]
    assert result['available_bytes']==340 and result['psutil_available_bytes']==170
    assert result['source']=='macos_memory_pressure' and result['minimum_available_fraction']==.15


def test_failed_or_malformed_native_probe_falls_back_to_psutil(monkeypatch):
    monkeypatch.setattr(budget.sys,'platform','darwin')
    monkeypatch.setattr(budget.psutil,'virtual_memory',lambda:SimpleNamespace(total=1000,available=170))
    monkeypatch.setattr(budget.subprocess,'run',lambda *a,**kw:SimpleNamespace(stdout='unexpected output'))
    result=budget.memory_budget()
    assert result['available_bytes']==170 and result['source']=='psutil'
    def fail(*a,**kw):raise OSError('unavailable')
    monkeypatch.setattr(budget.subprocess,'run',fail)
    assert budget.memory_budget()['available_bytes']==170
