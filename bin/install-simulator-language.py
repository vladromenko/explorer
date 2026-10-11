#!/usr/bin/env python3
"""Download an official local planner model without altering the main Ollama service."""
import json
import os
from pathlib import Path
import sys
import urllib.request

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from simulator.language_learning import LocalLanguageModel, ENDPOINT, MODEL

volume=Path(os.environ.get('EXPLORER_SIM_VOLUME','/Volumes/ExplorerSim'))
if not volume.is_mount():
    raise SystemExit('Подключите внешний диск '+str(volume))
root=volume/'ExplorerSim'
(root/'logs').mkdir(parents=True,exist_ok=True)
model=LocalLanguageModel(root)
try:
    model.ensure()
    request=urllib.request.Request(ENDPOINT+'/api/pull',data=json.dumps(dict(model=MODEL,stream=True)).encode(),headers={'Content-Type':'application/json'})
    last=None
    with urllib.request.urlopen(request,timeout=120) as response:
        for line in response:
            value=json.loads(line)
            if value.get('error'):
                raise ValueError(value['error'])
            fraction=int(100*value.get('completed',0)/max(1,value.get('total',1)))
            marker=(value.get('status'),fraction//10)
            if marker!=last:
                print(value.get('status'),str(fraction)+'%',flush=True)
                last=marker
    status=model.status()
    if not status['available']:
        raise ValueError('Файл модели не появился в локальном реестре')
    (root/'logs/language-model-install.json').write_text(json.dumps(status,ensure_ascii=False,indent=2))
    print('Локальный помощник установлен:',status['model'],status['digest'])
finally:
    model.close()
