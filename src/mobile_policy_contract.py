"""Portable mobile policy contract; no ROS, UART or actuator imports."""
import hashlib
import json
import math
from pathlib import Path

ACTION_ORDER=["base","shoulder","elbow","wrist_pitch","wrist_roll","gripper","vx","vy","wz"]
UNITS=["deg"]*6+["m/s","m/s","rad/s"]
JOINT_LIMITS=[(0,180)]*4+[(0,270),(30,180)]
BASE_LIMITS=[.8,.72,1.67]


def read_bundle(job_folder):
    folder=Path(job_folder).resolve()
    bundle=json.loads((folder/"bundle.json").read_text())
    if bundle.get("format")!="explorer_mobile_act_bundle_v1":raise ValueError("Неверный формат комплекта")
    if bundle.get("dataset_kind")!="mobile_manipulation_9dof":raise ValueError("Комплект не для шасси и руки Explorer")
    if bundle.get("action_order")!=ACTION_ORDER or bundle.get("observation_order")!=ACTION_ORDER or bundle.get("units")!=UNITS:
        raise ValueError("Порядок или единицы команд не совпадают с Explorer")
    if bundle.get("joint_state_source")!="command_estimate":raise ValueError("Недостоверный источник положения руки")
    relative=Path(bundle["checkpoint_relative"])
    if relative.is_absolute() or ".." in relative.parts:raise ValueError("Неверный путь checkpoint")
    checkpoint=(folder/relative).resolve()
    if not checkpoint.is_relative_to(folder):raise ValueError("Checkpoint вне задания")
    digests=bundle.get("checkpoint_sha256",{})
    if not {"model.safetensors","config.json"}.issubset(digests):raise ValueError("Неполный комплект модели")
    for name,digest in digests.items():
        path=(checkpoint/name).resolve()
        if not path.is_relative_to(checkpoint) or not path.is_file():raise ValueError("Файл модели отсутствует")
        if hashlib.sha256(path.read_bytes()).hexdigest()!=digest:raise ValueError("Контрольная сумма модели не совпала")
    model=json.loads((checkpoint/"config.json").read_text())
    if (model.get("type")!="act" or model.get("input_features",{}).get("observation.state",{}).get("shape")!=[9] or
        model.get("input_features",{}).get("observation.images.wrist",{}).get("shape")!=[3,240,320] or
        model.get("output_features",{}).get("action",{}).get("shape")!=[9] or
        model.get("n_action_steps")!=1):
        raise ValueError("Конфигурация модели не соответствует девяти командам Explorer")
    return bundle,checkpoint


def bound_action(current,proposed,base_trial_limits=(.12,.12,.35)):
    if len(current)!=9 or len(proposed)!=9:raise ValueError("Ожидались девять координат")
    values=[float(value) for value in proposed]
    reference=[float(value) for value in current]
    if not all(math.isfinite(value) for value in values+reference):raise ValueError("Нечисловое действие политики")
    goal=[]
    arm_limited=False
    for index,(minimum,maximum) in enumerate(JOINT_LIMITS):
        if not minimum<=values[index]<=maximum or abs(values[index]-reference[index])>20:
            raise ValueError("Недопустимый шаг сустава "+str(index+1))
        step=max(-2.0,min(2.0,values[index]-reference[index]))
        value=int(round(reference[index]+step))
        arm_limited=arm_limited or abs(value-values[index])>1e-9
        goal.append(value)
    speed=[]
    for index,limit in enumerate(BASE_LIMITS):
        value=values[6+index]
        if abs(value)>limit*1.2:raise ValueError("Модель запросила скорость вне шкалы обучения")
        trial=min(limit,base_trial_limits[index])
        speed.append(max(-trial,min(trial,value)))
    return {"proposed":values,"issued_joint_goal_deg":goal,"issued_body_velocity":speed,
            "arm_limited":arm_limited,
            "base_limited":any(abs(speed[index]-values[6+index])>1e-9 for index in range(3)),
            "attainment_measured":False}
