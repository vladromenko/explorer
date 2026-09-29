"""Non-actuating relocalization exercise using map pose and both lidar scans."""
import json,math,time,uuid
from pathlib import Path
import numpy as np

class LocalizationExercise:
    def __init__(self,root,pose):
        self.root=Path(root);self.pose=pose;self.scans={};self.reference=None
        (self.root/'data/autonomy-evidence').mkdir(parents=True,exist_ok=True)

    def scan(self,name,msg):
        values=np.asarray(msg.ranges,dtype=float)
        values[(values<msg.range_min)|(values>msg.range_max)|~np.isfinite(values)]=np.nan
        if len(values)>180:values=values[np.linspace(0,len(values)-1,180).astype(int)]
        self.scans[name]=(values,time.monotonic())

    def snapshot(self):
        state=json.loads((self.root/'data/status.json').read_text())
        if time.time()-state.get('at',0)>2 or any(abs(v)>1e-3 for v in state.get('velocity',[])):
            raise ValueError('Робот должен неподвижно стоять со свежими датчиками')
        if set(self.scans)!= {'scan0','scan1'} or any(time.monotonic()-v[1]>.7 for v in self.scans.values()):
            raise ValueError('Нет двух свежих лидаров')
        return dict(pose=self.pose(),scans={k:v[0].tolist() for k,v in self.scans.items()},
                    map_epoch=json.loads((self.root/'data/map_session.json').read_text())['id'])

    def begin(self):
        self.reference=self.snapshot()
        return dict(active=True,instruction='Отъедьте и вернитесь к физической отметке с выбранного направления.')

    def capture(self,heading):
        if heading not in ('front','left','right','back'):raise ValueError('Выберите направление, с которого начался маршрут')
        if not self.reference:raise ValueError('Сначала зафиксируйте контрольную точку')
        current=self.snapshot()
        if current['map_epoch']!=self.reference['map_epoch']:raise ValueError('Карта изменилась во время упражнения')
        a,b=self.reference['pose'],current['pose'];translation=math.hypot(a['x']-b['x'],a['y']-b['y'])
        yaw=abs(math.degrees(math.atan2(math.sin(a['yaw']-b['yaw']),math.cos(a['yaw']-b['yaw']))))
        lidar=[]
        for name in ('scan0','scan1'):
            x=np.asarray(self.reference['scans'][name]);y=np.asarray(current['scans'][name]);valid=np.isfinite(x)&np.isfinite(y)
            if valid.sum()<60:raise ValueError('Недостаточно общих измерений лидара')
            lidar.append(float(np.median(np.abs(x[valid]-y[valid]))))
        outcome='success' if max(lidar)<=.08 and translation<=.15 and yaw<=10 else 'failure'
        row=dict(schema='explorer_autonomy_evidence_v1',kind='localization_return',at=time.time(),hardware_executed=True,simulation=False,
                 verifier='dual_lidar_return_v1',verifier_outcome=outcome,start_heading_bucket=heading,translation_error_m=translation,
                 yaw_error_deg=yaw,lidar_median_absolute_error_m=lidar,map_epoch=current['map_epoch'])
        path=self.root/'data/autonomy-evidence'/(time.strftime('%Y%m%d-%H%M%S')+'-localization_return-'+uuid.uuid4().hex[:8]+'.json')
        path.write_text(json.dumps(row,ensure_ascii=False,allow_nan=False,indent=2)+'\n')
        return row
