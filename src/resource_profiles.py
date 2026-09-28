"""Serialize heavy job admission; never stop control or sensor services."""
import json
from pathlib import Path
import subprocess
import threading
import time

MODES={'work':('Работа',{'llm','voice','grounding','preview','policy'}),
       'experiment':('Эксперимент',{'grounding','preview','policy'}),
       'training':('Обучение',{'train'})}

class ResourceProfiles:
    def __init__(self,root,busy):
        self.root=Path(root);self.busy=busy;self.lock=threading.RLock()
        # Restart does not resume training or an experimental workload.
        self.mode='work'

    def status(self):
        return dict(mode=self.mode,label=MODES[self.mode][0],heavy_jobs=self.busy(),
                    modes=[dict(id=k,label=v[0]) for k,v in MODES.items()],
                    limits='Одна тяжёлая задача; системные лимиты памяти остаются включены')

    def select(self,mode):
        if mode not in MODES:raise ValueError('Неизвестный профиль')
        with self.lock:
            if self.busy():raise ValueError('Сначала завершите текущую тяжёлую задачу')
            if mode=='training':
                state=json.loads((self.root/'data/status.json').read_text())
                if not 0<=time.time()-state['at']<2 or not state.get('stop_latched') or any(state.get('velocity',[])):
                    raise ValueError('Для обучения нужны свежая телеметрия и STOP')
            if mode!='work':self.release_idle_models()
            self.mode=mode
            return self.status()

    def release_idle_models(self):
        subprocess.run(['systemctl','--user','stop','explorer-llm.service','explorer-speech.service'],
                       check=True,timeout=10)

    def admit(self,kind,operation,*args):
        with self.lock:
            if kind not in MODES[self.mode][1]:
                raise ValueError('Выберите подходящий профиль во вкладке «Эксперименты»')
            busy=[job for job in self.busy() if job!=kind]
            if busy:raise ValueError('Тяжёлая задача уже выполняется: '+', '.join(busy))
            state=json.loads((self.root/'data/status.json').read_text())
            # MemAvailable includes reclaimable page cache. No promise of zero OOM.
            memory={line.split(':')[0]:int(line.split()[1]) for line in Path('/proc/meminfo').read_text().splitlines()}
            if kind in ('train','grounding','preview','policy'):
                self.release_idle_models()
                memory={line.split(':')[0]:int(line.split()[1]) for line in Path('/proc/meminfo').read_text().splitlines()}
            if memory.get('MemAvailable',0)<1500*1024:
                raise ValueError('Недостаточно свободной памяти; тяжёлая задача отложена')
            if not 0<=time.time()-state['at']<2:raise ValueError('Нет свежего состояния робота')
            return operation(*args)
