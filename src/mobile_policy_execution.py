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
from mobile_demonstrations import episode_quality,sample


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
        self.intervention_id=None
        self.skills=None
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
        intervention_state=None
        if self.intervention_id:
            try:
                record=json.loads((self.root/"data/mobile-demonstrations"/self.intervention_id/"episode.json").read_text())
                intervention_state={key:record.get(key) for key in ("id","state","outcome","samples","quality")}
            except (OSError,ValueError):pass
        return dict(self.state,busy=self.lock.locked(),session=self.session,
                    observed_lease_s=max(0.0,self.lease-time.monotonic()),
                    intervention_id=self.intervention_id,intervention=intervention_state,
                    physical_success_verified=False)

    def start(self,skill_id,observing):
        if observing is not True:raise ValueError("Нужен наблюдающий оператор")
        check=self.preflight(skill_id)
        if not check["ready"]:raise ValueError("; ".join(check["blocked_by"]))
        if not self.lock.acquire(blocking=False):raise ValueError("Попытка уже идёт")
        try:
            job,bundle,checkpoint=self._candidate(skill_id)
            self.policy_sample_version=bundle["format"]
            self.session=uuid.uuid4().hex
            self.mission=None
            self.cancelled.clear()
            self.lease=time.monotonic()+1.0
            self.target=[0.0,0.0,0.0]
            self.target_at=0.0
            self.intervention_id=None
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
        flags=state.get("commissioning",{})
        if flags.get("camera_tf_validated") is not True or flags.get("gripper_calibrated") is not True:
            raise ValueError("Привязка камеры или захвата перестала быть принятой")
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
        session=self.session
        skill_id=self.state.get("skill_id")
        job_id=self.state.get("model_job")
        event={"at":time.time(),"kind":"human_takeover","proposed_action":self.latest_proposal,
               "first_manual_action":manual_action,"old_policy_sequence_invalidated":True}
        folder=self.root/"data/mobile-policy-runs"/session
        with (folder/"events.jsonl").open("a") as stream:stream.write(json.dumps(event,ensure_ascii=False)+"\n")
        self.stop()
        identifier=uuid.uuid4().hex
        self.intervention_id=identifier
        threading.Thread(target=self._capture_intervention,args=(identifier,session,skill_id,job_id),
                         daemon=True,name="explorer-human-correction").start()
        return event

    def _capture_intervention(self,identifier,session,skill_id,job_id):
        ready=False
        deadline=time.monotonic()+10
        while time.monotonic()<deadline:
            try:control=json.loads((self.root/"data/manual-teleop.json").read_text())
            except (OSError,ValueError):control={}
            actions=control.get("normalized_actions") or {}
            if time.time()-control.get("at",0)<.8 and any(abs(float(value))>.08 for value in actions.values()):
                ready=True;break
            time.sleep(.1)
        if not ready:
            if self.intervention_id==identifier:self.intervention_id=None
            return
        folder=self.root/"data/mobile-demonstrations"/identifier
        folder.mkdir(parents=True,exist_ok=True)
        record={"id":identifier,"name":"Исправление навыка","skill_id":skill_id,
                "source":"operator_mobile_demonstration","kind":"human_intervention",
                "intervention_of":session,"model_job":job_id,"state":"recording",
                "outcome":"unknown","started":time.time(),"samples":0,
                "joint_state_source":"commanded_not_measured","dataset_kind":"mobile_manipulation_9dof",
                "label_source":"operator","stage":"human_correction"}
        write_json(folder/"episode.json",record)
        begun=time.monotonic();last_motion=begun;last_frame=-1.0
        try:
            while time.monotonic()-begun<60:
                now=time.time()
                try:
                    control=json.loads((self.root/"data/manual-teleop.json").read_text())
                except (OSError,ValueError):control={}
                actions=control.get("normalized_actions") or {}
                moving=now-control.get("at",0)<.8 and any(abs(float(value))>.08 for value in actions.values())
                if moving:last_motion=time.monotonic()
                if time.monotonic()-last_motion>10:break
                try:item,image=sample(self.root,now)
                except (OSError,ValueError,KeyError):item=image=None
                if item and item["image_stamp"]>last_frame:
                    index=record["samples"]
                    filename=f"{index:06d}.jpg"
                    if not cv2.imwrite(str(folder/filename),image):raise ValueError("Кадр исправления не сохранён")
                    item.update(image=filename,stage="human_correction",intervention_of=session,
                                proposed_policy_action=self.latest_proposal,
                                executed_action_source="operator_manual_control")
                    with (folder/"samples.jsonl").open("a") as stream:
                        stream.write(json.dumps(item,ensure_ascii=False)+"\n")
                    record["samples"]+=1;last_frame=item["image_stamp"]
                    write_json(folder/"episode.json",record)
                time.sleep(.2)
        except (OSError,ValueError) as exc:record["capture_error"]=str(exc)
        record["ended"]=time.time()
        record["quality"]=episode_quality(folder)
        record["state"]="complete" if record["samples"] and not record.get("capture_error") else "interrupted"
        try:record["outcome"]=json.loads((folder/"episode.json").read_text()).get("outcome","unknown")
        except (OSError,ValueError):pass
        write_json(folder/"episode.json",record)
        if self.skills:
            try:self.skills.ingest(record,folder/"episode.json")
            except (OSError,ValueError,KeyError):pass

    def label_intervention(self,identifier,outcome):
        if outcome not in ("success","failure","unknown"):raise ValueError("Неизвестный результат исправления")
        if not isinstance(identifier,str) or len(identifier)!=32 or any(ch not in "0123456789abcdef" for ch in identifier):
            raise ValueError("Некорректный идентификатор исправления")
        path=self.root/"data/mobile-demonstrations"/identifier/"episode.json"
        record=json.loads(path.read_text())
        if record.get("kind")!="human_intervention":raise ValueError("Это не исправление оператора")
        if record.get("state")=="recording":raise ValueError("Сначала закончите исправляющее движение")
        if record.get("outcome") not in ("unknown",outcome):raise ValueError("Результат уже оценён иначе")
        record["outcome"]=outcome
        write_json(path,record)
        if record.get("state")=="complete" and self.skills:self.skills.ingest(record,path)
        return {"id":identifier,"outcome":outcome,"state":record.get("state"),
                "quality":record.get("quality"),"physical_success_verified":False}

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
        estimate=((status.get("arm_command_state") or {}).get("q_estimated_deg") or command[:6]) if (
            getattr(self,"policy_sample_version",None)=="explorer_mobile_act_bundle_v2") else command[:6]
        observation=[float(value) for value in estimate]+command[6:]
        if len(observation)!=9 or not np.isfinite(observation).all():raise ValueError("Нет оценки для политики")
        return command,image,stamp,observation

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
                    command,image,stamp,observation=self._fresh_observation()
                    if not cv2.imwrite(str(folder/f"{step:04d}.jpg"),image):raise ValueError("Кадр не сохранён")
                    write_json(folder/f"{step:04d}-request.json",{"state":observation,"observed_at":stamp,
                        "reset_history":step==0})
                    prediction=self._wait_file(folder/f"{step:04d}-result.json",1.0,process)
                    if prediction.get("observed_at")!=stamp or prediction.get("state")!=observation or time.time()-stamp>1.2:
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
                           "observation_state":observation,"applied_command_reference":command,
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
