"""Hold-to-run replay of a validated ACT policy through the finite arm gate.

Base motion stays latched off. This is supervised skill testing, not autonomous
success verification. Inference runs without actuator access in a bounded unit.
"""
import json
import subprocess
import threading
import time
import uuid
from pathlib import Path
import cv2
import numpy as np
from lerobot_bridge import write_json,training_budget


def bounded_goal(start,action):
    a=np.asarray(action,dtype=float);s=np.asarray(start,dtype=float)
    if a.shape!=(6,) or s.shape!=(6,) or not np.isfinite(a).all() or not np.isfinite(s).all():
        raise ValueError('Некорректное действие модели')
    if np.max(np.abs(a-s))>2.00001:raise ValueError('Модель запросила шаг больше 2°; выполнение остановлено')
    if np.any(a<[0,0,0,0,0,30]) or np.any(a>[180,180,180,180,270,180]):
        raise ValueError('Действие модели вне диапазона приводов')
    return np.rint(a).astype(int).tolist()


class PolicyExecution:
    def __init__(self,root,jobs,teaching,manual,preview):
        self.root=Path(root);self.jobs=jobs;self.teaching=teaching;self.manual=manual;self.preview=preview
        self.lock=threading.Lock();self.cancelled=threading.Event();self.lease=0.;self.session=None
        self.state=dict(phase='idle',success_verified=False);self.unit=None

    def status(self):return dict(self.state,busy=self.lock.locked(),requires_held_button=True,base_motion=False)

    def heartbeat(self,session,held):
        if not self.lock.locked() or session!=self.session:raise ValueError('Нет этого сеанса выполнения')
        if held:self.lease=time.monotonic()+.6
        else:self.stop()
        return self.status()

    def stop(self):
        self.cancelled.set();self.lease=0
        self.manual.stop()
        return dict(stopping=True,success_verified=False)

    def permit(self):
        if self.cancelled.is_set() or time.monotonic()>self.lease:raise ValueError('Кнопка отпущена или связь с панелью потеряна')
        if self.manual.stop_revision!=self.revision:raise ValueError('Нажат STOP')

    def start(self,task,observing):
        if observing is not True:raise ValueError('Требуется наблюдатель у робота')
        if not self.lock.acquire(blocking=False):raise ValueError('Навык уже выполняется')
        reserved=False
        try:
            if not self.teaching.lock.acquire(blocking=False):raise ValueError('Рука занята')
            reserved=True
            if self.teaching.active or self.preview.lock.locked():raise ValueError('Завершите показ или предварительный расчёт')
            training_budget(self.root)
            jobs=self.jobs.status()['jobs']
            if any(j['state'] in ('queued','exporting','training','validating') for j in jobs):raise ValueError('Дождитесь окончания обучения')
            candidates=[j for j in jobs if j['task']==task and j['state']=='validated_offline' and j.get('validation',{}).get('improves_hold_baseline')]
            if not candidates:raise ValueError('Сначала запишите показы и обучите проверенную модель этого навыка')
            self.teaching.observation()
            job=candidates[-1];self.session=uuid.uuid4().hex
            folder=self.root/'data/policy-runs'/self.session;folder.mkdir(parents=True)
            checkpoint=self.root/'data/learning-jobs'/job['id']/'model/checkpoints/last/pretrained_model'
            write_json(folder/'config.json',dict(checkpoint=str(checkpoint),task=task))
            self.revision=self.manual.stop_revision;self.cancelled.clear();self.lease=time.monotonic()+.6
            self.state=dict(phase='loading',session=self.session,task=task,model_job=job['id'],steps=0,success_verified=False)
            self.unit='explorer-policy-'+self.session
            threading.Thread(target=self.run,args=(folder,),daemon=True).start()
            return self.status()
        except Exception:
            if reserved:self.teaching.lock.release()
            self.lock.release();raise

    def wait_file(self,path,seconds,process):
        end=time.monotonic()+seconds
        while not path.exists():
            self.permit()
            if process.poll() is not None:raise ValueError('Процесс модели завершился; смотрите журнал')
            if time.monotonic()>end:raise ValueError('Превышено время расчёта модели')
            time.sleep(.025)
        self.permit()
        return json.loads(path.read_text())

    def observation(self):
        end=time.monotonic()+3
        while time.monotonic()<end:
            self.permit()
            pose,_=self.teaching.observation()
            arm=json.loads((self.root/'data/arm-state.json').read_text())
            with np.load(self.root/'data/rgbd-snapshot.npz',allow_pickle=False) as sample:
                stamp=float(sample['stamp'])
                if 0<=time.time()-stamp<1.5 and stamp>arm.get('updated_at',arm['at'])+.05:
                    return pose,sample['rgb'].copy(),stamp
            time.sleep(.025)
        raise ValueError('Нет свежего кадра после завершения движения')

    def run(self,folder):
        process=None;log=None
        try:
            log=(folder/'worker.log').open('w')
            process=subprocess.Popen(['systemd-run','--user','--quiet','--wait','--pipe','--collect',
                '--unit='+self.unit,'--property=MemoryMax=2500M','--property=PartOf=explorer.target',
                '--property=Nice=15','--property=CPUWeight=10','--property=RuntimeMaxSec=120',
                str(self.root/'.venv-learning/bin/python'),str(self.root/'bin/policy-worker.py'),str(folder)],stdout=log,stderr=log)
            self.wait_file(folder/'ready.json',60,process)
            for step in range(30):
                self.permit();training_budget(self.root)
                pose,image,observed_at=self.observation()
                if not cv2.imwrite(str(folder/f'{step:03d}.jpg'),image):raise ValueError('Не удалось сохранить кадр')
                write_json(folder/f'{step:03d}-request.json',dict(pose=pose,observed_at=observed_at))
                self.state.update(phase='thinking')
                prediction=self.wait_file(folder/f'{step:03d}-result.json',2,process)
                if prediction.get('observed_at')!=observed_at or prediction.get('start_deg')!=pose:
                    raise ValueError('Ответ модели не соответствует текущему кадру и положению')
                if time.time()-observed_at>2:raise ValueError('Результат модели устарел')
                goal=bounded_goal(pose,prediction['proposed_deg'])
                self.permit()
                if goal==pose:
                    self.state.update(phase='hold',reason='Модель предложила удержать положение');return
                self.state.update(phase='moving')
                # Rechecks live state, geometry, limits, telemetry and STOP revision.
                result=self.manual.move(pose,goal,expected_stop_revision=self.revision,execution_permit=self.permit,source="supervised_policy")
                self.state['steps']=step+1
                write_json(folder/f'{step:03d}-command.json',result)
                # Wait for an image acquired after the command completed.
                time.sleep(.35);self.permit()
            self.state.update(phase='limit_reached',reason='30 шагов: проверьте результат перед следующим запуском')
        except (OSError,ValueError,KeyError,subprocess.SubprocessError) as exc:
            self.state.update(phase='stopped',reason=str(exc))
        finally:
            try:
                if process is not None:
                    subprocess.run(['systemctl','--user','stop',self.unit+'.service'],capture_output=True,timeout=3,check=False)
                    try:process.wait(timeout=2)
                    except subprocess.TimeoutExpired:process.terminate()
                write_json(folder/'result.json',dict(self.state,ended=time.time(),success_verified=False))
            except (OSError,subprocess.SubprocessError) as exc:self.state['cleanup_error']=str(exc)
            finally:
                if log:log.close()
                self.teaching.lock.release();self.lock.release()
