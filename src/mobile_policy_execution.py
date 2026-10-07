"""Observed 9D policy trial through Explorer's local mission and arm gates."""
import json
from pathlib import Path
import subprocess
import threading
import time
import uuid

import cv2
import numpy as np

from arm_commissioning import coordinated_policy_status
from lerobot_bridge import write_json
from mobile_policy_contract import bound_action,read_bundle


class MobilePolicyExecution:
    def __init__(self,root,jobs,missions,arm,emit):
        self.root=Path(root)
        self.jobs=jobs
        self.missions=missions
        self.arm=arm
        self.emit=emit
        self.lock=threading.Lock()
        self.cancelled=threading.Event()
        self.session=None
        self.mission=None
        self.lease=0.0
        self.target=[0.0,0.0,0.0]
        self.target_at=0.0
        self.latest_proposal=None
        self.state={"phase":"idle","physical_success_verified":False}

    def _candidate(self,skill_id):
        jobs=[job for job in self.jobs.status()["jobs"] if job.get("skill_id")==skill_id and
              job.get("state")=="validated_offline" and job.get("validation",{}).get("improves_hold_baseline") is True]
        if not jobs:raise ValueError("Нет кандидата, проверенного на отдельном показе")
        job=jobs[-1]
        bundle,checkpoint=read_bundle(self.root/"data/learning-jobs"/job["id"])
        if bundle.get("skill_id")!=skill_id:raise ValueError("Модель относится к другому навыку")
        return job,bundle,checkpoint

    def preflight(self,skill_id):
        blockers=[]
        try:job,bundle,checkpoint=self._candidate(skill_id)
        except (OSError,ValueError,KeyError) as exc:
            job=None;bundle=None;checkpoint=None;blockers.append(str(exc))
        try:
            state=json.loads((self.root/"data/status.json").read_text())
            if not 0<=time.time()-state.get("at",0)<.8:blockers.append("Состояние робота устарело")
            flags=state.get("commissioning",{})
            required=("base_commissioned","arm_commissioned","lidar_tf_validated","camera_tf_validated",
                      "mcu_watchdog_verified","localization_verified","gripper_calibrated")
            blockers.extend("Не принята калибровка: "+key for key in required if flags.get(key) is not True)
            if state.get("power",{}).get("state") in ("LOW_POWER","CRITICAL","CHARGING","UNKNOWN"):
                blockers.append("Питание не допускает попытку")
        except (OSError,ValueError,KeyError) as exc: blockers.append("Нет состояния робота: "+str(exc))
        try:
            arm=self.arm.status()
            if arm.get("blocked_by"):blockers.append("Рука: "+arm["blocked_by"])
        except (OSError,ValueError) as exc:blockers.append("Рука: "+str(exc))
        return {"ready":not blockers,"blocked_by":blockers,"job_id":job["id"] if job else None,
                "model_candidate":bool(job),"physical_success_verified":False,
                "observed_trial_required":True}

    def status(self):
        return dict(self.state,busy=self.lock.locked(),session=self.session,
                    observed_lease_s=max(0.0,self.lease-time.monotonic()),
                    physical_success_verified=False)

    def start(self,skill_id,observing):
        if observing is not True:raise ValueError("Нужен наблюдающий оператор")
        check=self.preflight(skill_id)
        if not check["ready"]:raise ValueError("; ".join(check["blocked_by"]))
        if not self.lock.acquire(blocking=False):raise ValueError("Попытка уже идёт")
        try:
            job,bundle,checkpoint=self._candidate(skill_id)
            self.session=uuid.uuid4().hex
            self.mission=None
            self.cancelled.clear()
            self.lease=time.monotonic()+1.0
            self.target=[0.0,0.0,0.0]
            self.target_at=0.0
            folder=self.root/"data/mobile-policy-runs"/self.session
            folder.mkdir(parents=True)
            write_json(folder/"config.json",{"job_folder":str(self.root/"data/learning-jobs"/job["id"]),
                                             "skill_id":skill_id,"model_job":job["id"]})
            self.state={"phase":"loading","run_id":self.session,"skill_id":skill_id,
                        "model_job":job["id"],"steps":0,"physical_success_verified":False}
            threading.Thread(target=self._run,args=(folder,),daemon=True,name="explorer-mobile-policy").start()
            return self.status()
        except Exception:
            self.lock.release()
            raise

    def heartbeat(self,session,observing):
        if session!=self.session or not self.lock.locked():raise ValueError("Попытка не найдена")
        if observing is not True:self.stop()
        else:self.lease=time.monotonic()+1.0
        return self.status()

    def _lease_permit(self):
        if self.cancelled.is_set() or time.monotonic()>self.lease:
            raise ValueError("Наблюдение прекращено или попытка отменена")

    def _permit(self):
        self._lease_permit()
        state=json.loads((self.root/"data/status.json").read_text())
        coordinated_policy_status(state,time.time())
        if state.get("mission")!=self.mission:raise ValueError("Владение миссией изменилось")
        if state.get("reason") in ("OBSTACLE","SENSOR OR BATTERY FAULT","STOP LATCHED"):
            raise ValueError("Движение остановлено контролем шасси: "+state["reason"])

    def stop(self):
        self.cancelled.set()
        self.target=[0.0,0.0,0.0]
        self.arm.stop()
        mission=self.mission
        if mission:
            try:self.missions.finish(mission,"cancelled",{"reason":"Operator stop or manual takeover"})
            except (OSError,ValueError):pass
        return {"stopping":True,"run_id":self.session}

    def takeover(self,manual_action=None):
        if not self.lock.locked():return None
        event={"at":time.time(),"kind":"human_takeover","proposed_action":self.latest_proposal,
               "first_manual_action":manual_action,"old_policy_sequence_invalidated":True}
        folder=self.root/"data/mobile-policy-runs"/self.session
        with (folder/"events.jsonl").open("a") as stream:stream.write(json.dumps(event,ensure_ascii=False)+"\n")
        self.stop()
        return event

    def _fresh_observation(self):
        self._permit()
        arm=self.arm.reference(coordinated_policy_status)
        status=json.loads((self.root/"data/status.json").read_text())
        with np.load(self.root/"data/rgbd-snapshot.npz",allow_pickle=False) as frame:
            stamp=float(frame["stamp"])
            if not 0<=time.time()-stamp<.9:raise ValueError("Камера устарела")
            image=frame["rgb"].copy()
        command=[float(value) for value in arm["servo_deg"]]+[float(value) for value in status["velocity"]]
        if len(command)!=9 or not np.isfinite(command).all():raise ValueError("Нет девяти достоверных оценок команд")
        return command,image,stamp

    def _wait_file(self,path,timeout,process):
        deadline=time.monotonic()+timeout
        while not path.exists():
            self._lease_permit()
            if process.poll() is not None:raise ValueError("Процесс модели остановился")
            if time.monotonic()>deadline:raise ValueError("Ответ модели опоздал")
            time.sleep(.02)
        return json.loads(path.read_text())

    def _base_heartbeat(self,mission):
        while not self.cancelled.is_set() and self.mission==mission:
            try:
                self._permit()
                velocity=self.target if time.monotonic()-self.target_at<.5 else [0.0,0.0,0.0]
                self.emit("autonomy_lease",mission=mission)
                self.emit("policy_drive",mission=mission,velocity=velocity)
            except (OSError,ValueError,KeyError):
                self.cancelled.set()
            time.sleep(.08)

    def _run(self,folder):
        process=None
        mission=uuid.uuid4().hex
        try:
            with (folder/"worker.log").open("w") as log:
                process=subprocess.Popen(["systemd-run","--user","--quiet","--wait","--pipe","--collect",
                    "--unit=explorer-mobile-policy-"+self.session,"--property=MemoryMax=2500M",
                    "--property=CPUWeight=10","--property=RuntimeMaxSec=180",
                    str(self.root/".venv-learning/bin/python"),str(self.root/"bin/mobile-policy-worker.py"),str(folder)],
                    stdout=log,stderr=log)
                self._wait_file(folder/"ready.json",60,process)
                self._lease_permit()
                self.emit("mode",mode="AUTONOMOUS")
                self.emit("clear_stop")
                deadline=time.monotonic()+2
                while time.monotonic()<deadline:
                    self._lease_permit()
                    state=json.loads((self.root/"data/status.json").read_text())
                    if state.get("mode")=="AUTONOMOUS" and state.get("stop_latched") is False:break
                    time.sleep(.03)
                else:raise ValueError("Автономный режим не подтвердился")
                self.missions.begin_compound(mission,"mobile_policy",120,self._lease_permit)
                self.mission=mission
                self.missions.emit("resume_base",mission=mission)
                deadline=time.monotonic()+2
                while time.monotonic()<deadline:
                    self._lease_permit()
                    state=json.loads((self.root/"data/status.json").read_text())
                    if state.get("mission")==mission and state.get("mode")=="AUTONOMOUS":break
                    time.sleep(.03)
                else:raise ValueError("Контроллер не подтвердил начало миссии")
                threading.Thread(target=self._base_heartbeat,args=(mission,),daemon=True).start()
                self.state["phase"]="observing"
                for step in range(300):
                    self._permit()
                    command,image,stamp=self._fresh_observation()
                    if not cv2.imwrite(str(folder/f"{step:04d}.jpg"),image):raise ValueError("Кадр не сохранён")
                    write_json(folder/f"{step:04d}-request.json",{"state":command,"observed_at":stamp,
                        "reset_history":step==0})
                    prediction=self._wait_file(folder/f"{step:04d}-result.json",1.0,process)
                    if prediction.get("observed_at")!=stamp or prediction.get("state")!=command or time.time()-stamp>1.2:
                        raise ValueError("Предсказание относится к устаревшему наблюдению")
                    action=bound_action(command,prediction["action"])
                    self.latest_proposal=action["proposed"]
                    self._permit()
                    self.target=action["issued_body_velocity"]
                    self.target_at=time.monotonic()
                    goal=action["issued_joint_goal_deg"]
                    if goal!=[int(value) for value in command[:6]]:
                        self.arm.move([int(value) for value in command[:6]],goal,
                            expected_stop_revision=self.arm.stop_revision,execution_permit=self._permit,
                            source="supervised_mobile_policy",speed="teleop")
                    event={"at":time.time(),"step":step,"observed_at":stamp,
                           "state_source":"command_estimate","proposed":prediction["action"],
                           "issued":action,"physical_attainment_measured":False}
                    with (folder/"events.jsonl").open("a") as stream:
                        stream.write(json.dumps(event,ensure_ascii=False)+"\n")
                    self.state.update(phase="running",steps=step+1,last_action=action)
                    if time.monotonic()-self.target_at<.1:time.sleep(.1)
                self.state.update(phase="limit_reached",reason="Достигнут предел наблюдаемой попытки")
        except (OSError,ValueError,KeyError,subprocess.SubprocessError) as exc:
            self.state.update(phase="stopped",reason=str(exc))
        finally:
            self.target=[0.0,0.0,0.0]
            self.cancelled.set()
            if self.mission==mission:
                try:self.missions.finish(mission,"cancelled",{"reason":self.state.get("reason","Trial ended without task verification")})
                except (OSError,ValueError):pass
            if process and process.poll() is None:
                subprocess.run(["systemctl","--user","stop","explorer-mobile-policy-"+self.session+".service"],
                               capture_output=True,timeout=3,check=False)
            self.state["ended"]=time.time()
            write_json(folder/"result.json",self.state)
            self.mission=None
            self.lock.release()
