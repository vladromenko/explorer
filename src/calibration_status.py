"""Read-only, evidence-scoped calibration and capability status."""
import json
import hashlib
from pathlib import Path
import time


def _read(path):
    try:
        value=json.loads(Path(path).read_text())
        return value if isinstance(value,dict) else {}
    except (OSError,ValueError,TypeError):
        return {}


def status(root, now=None):
    root=Path(root);now=time.time() if now is None else now
    acceptance=_read(root/'config/calibration-acceptance.json')
    records=json.loads(json.dumps(acceptance.get('records',{})))
    for name,item in records.items():
        if item.get('state')!='accepted':continue
        paths=item.get('evidence',[]);hashes=item.get('evidence_sha256',{})
        errors=[]
        for relative in paths:
            path=(root/relative).resolve()
            if not path.is_relative_to(root.resolve()) or hashes.get(relative) is None:
                errors.append(relative+': unbound evidence');continue
            try:digest=hashlib.sha256(path.read_bytes()).hexdigest()
            except OSError:
                errors.append(relative+': missing');continue
            if digest!=hashes[relative]:errors.append(relative+': digest changed')
        if not paths:errors.append('no evidence')
        item['evidence_valid']=not errors;item['evidence_errors']=errors
    live=_read(root/'data/status.json')
    lidar=_read(root/'data/lidar_geometry.json')
    fresh=type(live.get('at')) in (int,float) and 0 <= now-live['at'] < 3
    accepted=sorted(name for name,item in records.items() if item.get('state')=='accepted' and item.get('evidence_valid'))
    remaining=sorted(name for name,item in records.items() if name not in accepted)
    unlocked=[
        dict(id='manual_holonomic_drive',name='Ручное всенаправленное движение шасси',available='chassis' in accepted),
        dict(id='gamepad_base_arm',name='Совместное ручное управление шасси и рукой',available=all(x in accepted for x in ('chassis','arm_motion'))),
        dict(id='dual_lidar_mapping',name='Карта по двум лидарам',available='lidar_geometry' in accepted),
        dict(id='stationary_camera_arm_geometry',name='3D-привязка камеры к руке после завершённой команды',available='hand_eye' in accepted),
        dict(id='demonstration_capture',name='Запись демонстраций движения и захвата',available=all(x in accepted for x in ('chassis','arm_motion','hand_eye'))),
    ]
    blocked=[
        dict(id='global_object_memory',name='Достоверная 3D-память объектов на карте',blocked_by=['localization']),
        dict(id='autonomous_exploration',name='Автономное исследование и поиск по квартире',blocked_by=['localization']),
        dict(id='closed_loop_grasp',name='Захват с подтверждением положения и удержания',blocked_by=['arm_feedback','gripper']),
        dict(id='pick_and_deliver',name='Полный автономный pick-and-deliver',blocked_by=['localization','arm_feedback','gripper']),
    ]
    return dict(at=now,updated_at=acceptance.get('updated_at'),robot_live=fresh,
                lidar_live=bool(lidar and type(lidar.get('at')) in (int,float) and 0<=now-lidar['at']<2),
                records=records,accepted=accepted,remaining=remaining,
                unlocked=unlocked,blocked=blocked)
