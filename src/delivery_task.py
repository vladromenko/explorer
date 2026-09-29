"""One local, cancellable pick-and-deliver transaction with evidence at boundaries.

The robot port supplies real navigation, measured trajectories and observations.
This coordinator never turns an ACK, elapsed command or missing image into success.
"""
import json
import copy
import threading
import time
import uuid
from pathlib import Path


class DeliveryTask:
    def __init__(self, root, robot, world=None):
        self.root, self.robot = Path(root), robot
        self.lock = threading.RLock()
        self.active = None
        self.last = None
        self.generation = 0
        self.worker_id = None
        self.stopping = set()
        self.recovery_errors = []
        self.folder = self.root/'data/delivery-runs'
        self.world = world
        self.folder.mkdir(parents=True, exist_ok=True)
        # In-flight work never resumes after a process restart.
        for path in self.folder.glob('*.json'):
            try:
                value = json.loads(path.read_text())
                if value.get('state') == 'running':
                    value.update(state='interrupted', reason='Jetson task process restarted', ended=time.time())
                    self._save(value)
                if self.last is None or value.get('started',0)>self.last.get('started',0):
                    self.last=value
            except (OSError,ValueError,TypeError,AttributeError,KeyError) as exc:
                self.recovery_errors.append(path.name+': '+str(exc))

    def _save(self, value):
        mid=value['id']
        if not isinstance(mid,str) or not mid or Path(mid).name!=mid:
            raise ValueError('Invalid delivery record ID')
        path = self.folder/(mid+'.json')
        temporary = path.with_suffix('.tmp')
        temporary.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2))
        temporary.replace(path)

    def status(self):
        with self.lock:
            value=dict(active=copy.deepcopy(self.active), last=copy.deepcopy(self.last),
                        busy=self.worker_id is not None or bool(self.stopping),
                        cancelling=(self.worker_id is not None or bool(self.stopping)) and self.active is None,
                        local_execution=True, browser_required=False)
            recovery_errors=list(self.recovery_errors)
        value['blocked_by']=self.robot.blockers()+recovery_errors
        return value

    def start(self):
        with self.lock:
            if self.active or self.worker_id is not None or self.stopping:
                raise ValueError('Доставка уже выполняется')
            recovery_errors=list(self.recovery_errors)
        reasons = self.robot.blockers()+recovery_errors
        with self.lock:
            if self.active or self.worker_id is not None or self.stopping:
                raise ValueError('Доставка уже выполняется')
            if reasons:
                raise ValueError('; '.join(reasons))
            self.generation += 1
            generation = self.generation
            self.active = dict(id=uuid.uuid4().hex, generation=generation, state='running',
                               phase='preparing', started=time.time(), started_monotonic=time.monotonic(),
                               events=[], delivered=False)
            try:self._save(self.active)
            except (OSError,ValueError):
                self.active=None
                raise
            mid = self.active['id']
            self.worker_id=mid
            try:
                threading.Thread(target=self.run, args=(mid,generation), daemon=True).start()
            except Exception as exc:
                self.active.update(state='failed', ended=time.time(), reason='Worker did not start: '+str(exc))
                self._save(self.active); self.last=self.active; self.active=None; self.worker_id=None
                raise
            return dict(id=mid, accepted=True, completed=False)

    def check_current(self, mid, generation):
        with self.lock:
            if (not self.active or self.active['id'] != mid or self.generation != generation or
                self.active['state'] != 'running'):
                raise ValueError('Доставка отменена; старый результат недействителен')
            if time.monotonic()-self.active['started_monotonic'] > 900:
                raise ValueError('Истёк срок доставки')

    def permit(self, mid, generation):
        self.check_current(mid, generation)
        self.robot.permit(mid)
        self.check_current(mid, generation)

    def stage(self, mid, generation, name, operation):
        self.permit(mid, generation)
        with self.lock:
            self.check_current(mid, generation)
            self.active['phase'] = name
            self._save(self.active)
        if self.world:self.world.record_action('delivery',name,'started')
        try:result = operation()
        except Exception as exc:
            if self.world:self.world.record_action('delivery',name,'failed',{'reason':str(exc)})
            raise
        self.permit(mid, generation)  # including late accepted goals and results
        with self.lock:
            self.check_current(mid, generation)
            self.active['events'].append(dict(stage=name, at=time.time(), result=result))
            self._save(self.active)
        if self.world:self.world.record_action('delivery',name,'succeeded',result if isinstance(result,dict) else {'result':str(result)})
        return result

    def cancel(self):
        with self.lock:
            self.generation += 1
            value = self.active
            if value:
                value.update(state='cancelled', ended=time.time(), reason='Остановлено пользователем')
                self.stopping.add(value['id'])
                self.last=value;self.active=None
                try:self._save(value)
                except (OSError,ValueError) as exc:
                    value['record_error']=str(exc)
                    self.recovery_errors.append('Не удалось сохранить отмену: '+str(exc))
        if value:
            try:
                self.robot.stop(value['id'])
            except Exception as exc:
                with self.lock:
                    value['stop_error']=str(exc)
                    self._save(value)
            finally:
                with self.lock:self.stopping.discard(value['id'])
        return dict(cancel_requested=True, physical_stop_confirmed=False)

    def run(self, mid, generation):
        result, reason = 'failed', None
        try:
            self.check_current(mid,generation)
            self.robot.begin(mid,check=lambda:self.check_current(mid,generation))
            stage = lambda name, fn:self.stage(mid,generation,name,fn)
            target = stage('find', self.robot.find)
            approach=stage('approach', lambda:self.robot.approach(target))
            if approach.get('reobserve_required'):
                target=stage('refind_after_base_alignment',self.robot.find_aligned)
                approach=stage('approach_after_base_alignment',lambda:self.robot.approach(target))
                if approach.get('reobserve_required'):raise ValueError('Коррекция базы не сошлась после повторного измерения')
            stage('hold_base', self.robot.hold)
            stage('inspect', self.robot.inspect)
            target = stage('reobserve', lambda:self.robot.reobserve(target))
            stage('align', lambda:self.robot.align(target))
            stage('pregrasp', lambda:self.robot.pregrasp(target))
            stage('observe_before', self.robot.observe)
            stage('grasp', lambda:self.robot.grasp(target))
            before = stage('observe_grasp', self.robot.observe)
            stage('lift', self.robot.lift)
            held = stage('verify_hold', lambda:self.robot.verify_hold(before))
            if held.get('outcome') == 'failure':
                stage('regrasp', lambda:self.robot.regrasp(target))
                before = stage('observe_regrasp', self.robot.observe)
                stage('relift', self.robot.lift)
                held = stage('verify_regrasp', lambda:self.robot.verify_hold(before))
            if held.get('outcome') != 'success':
                raise ValueError('Захват не подтверждён: '+held.get('outcome','unknown'))
            stage('transport_pose', self.robot.transport)
            stage('carry', self.robot.carry)
            stage('hold_destination', self.robot.hold)
            stage('support', self.robot.support)
            before_release = stage('observe_release', self.robot.observe)
            stage('release', self.robot.release)
            stage('withdraw', self.robot.withdraw)
            placed = stage('verify_place', lambda:self.robot.verify_place(before_release))
            if placed.get('outcome') != 'success':
                raise ValueError('Размещение не подтверждено: '+placed.get('outcome','unknown'))
            result = 'succeeded'
        except Exception as exc:
            reason = str(exc)
        finally:
            try:
                self.robot.finish(mid, result == 'succeeded')
            except Exception as exc:
                result, reason = 'failed', 'Не удалось завершить остановку: '+str(exc)
            with self.lock:
                try:
                    if self.active and self.active['id'] == mid and self.generation == generation:
                        self.active.update(state=result, ended=time.time(), reason=reason,
                                           delivered=result == 'succeeded')
                        self._save(self.active); self.last=self.active; self.active=None
                finally:
                    if self.worker_id==mid:self.worker_id=None
