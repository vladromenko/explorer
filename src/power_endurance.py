"""Measured power history and software endurance profiles without invented SOC."""
import json
import math
from pathlib import Path
import statistics
import time


PROFILES={
 'PERFORMANCE':dict(perception_active_fps=4.,perception_idle_fps=2.,llm_idle_timeout_s=180,
                    description='Высокая частота зрения для активной манипуляции.'),
 'BALANCED':dict(perception_active_fps=3.,perception_idle_fps=1.,llm_idle_timeout_s=90,
                 description='Обычная работа и выгрузка тяжёлых моделей после простоя.'),
 'ENDURANCE':dict(perception_active_fps=2.,perception_idle_fps=.5,llm_idle_timeout_s=45,
                  description='Редкое фоновое зрение; высокая частота возвращается только задаче.'),
 'AUTO':dict(perception_active_fps=3.,perception_idle_fps=.5,llm_idle_timeout_s=60,
             description='Профиль выбирается по текущей физической задаче и резерву напряжения.'),
}


def read(path):
    try:return json.loads(Path(path).read_text())
    except (OSError,ValueError,TypeError):return {}


class PowerEndurance:
    def __init__(self,root,heavy_jobs=lambda:[]):
        self.root=Path(root);self.heavy_jobs=heavy_jobs;self.path=self.root/'data/power-policy.json'
        if not self.path.exists():self.select('AUTO')

    def select(self,profile):
        if profile not in PROFILES:raise ValueError('Unknown endurance profile')
        value=dict(PROFILES[profile],profile=profile,selected_at=time.time(),
            nvpmodel_switching=False,nvpmodel_reason='privileged helper is not installed and no transition was physically validated',
            battery_soc_available=False,whole_robot_current_available=False)
        temporary=self.path.with_suffix('.tmp');temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2));temporary.replace(self.path)
        return self.status()

    def activity(self):
        status=read(self.root/'data/status.json');arm=status.get('arm_command_state',{});jobs=self.heavy_jobs()
        if 'train' in jobs:return 'TRAIN'
        if 'llm' in jobs:return 'LLM'
        if 'grounding' in jobs:return 'GROUNDING'
        if arm.get('phase')=='command_in_progress':return 'ARM'
        velocity=status.get('velocity',[0,0,0])
        if any(abs(float(value))>.001 for value in velocity):return 'NAVIGATION' if status.get('mode')=='AUTONOMOUS' else 'BASE'
        perception=read(self.root/'data/perception.json')
        if 0<=time.time()-perception.get('image_stamp',0)<3:return 'OBSERVE'
        return 'IDLE'

    def samples(self,hours=6):
        path=self.root/'data/power-telemetry.jsonl'
        if not path.exists():return []
        cutoff=time.time()-hours*3600;rows=[]
        for line in path.read_text().splitlines()[-12000:]:
            try:item=json.loads(line)
            except (ValueError,TypeError):item={}
            if item.get('at',0)>=cutoff and type(item.get('battery_voltage_v')) in (int,float):rows.append(item)
        return rows

    @staticmethod
    def trend(rows):
        usable=[row for row in rows if row.get('charging') is not True and row.get('state') not in ('CHARGING','UNKNOWN')]
        if len(usable)<120:return dict(available=False,reason='need at least 10 minutes of real discharge samples')
        x=[float(row['at']) for row in usable];y=[float(row['battery_voltage_v']) for row in usable]
        duration=max(x)-min(x)
        if duration<600:return dict(available=False,reason='observed discharge interval is shorter than 10 minutes')
        mean_x=statistics.fmean(x);mean_y=statistics.fmean(y);den=sum((value-mean_x)**2 for value in x)
        slope=sum((a-mean_x)*(b-mean_y) for a,b in zip(x,y))/den if den else 0.
        if not math.isfinite(slope) or slope>=-1e-7:
            return dict(available=False,reason='no stable negative voltage trend in observed interval',duration_s=duration)
        slope_v_h=slope*3600;remaining=(y[-1]-10.8)/-slope_v_h*60
        return dict(available=True,duration_s=duration,samples=len(usable),slope_v_per_hour=slope_v_h,
            estimated_minutes_to_10_8v=max(0.,remaining),method='least_squares_voltage_trend_not_SOC',
            uncertainty='load and battery relaxation can change this estimate')

    def status(self):
        policy=read(self.path);power=read(self.root/'data/power.json');rows=self.samples()
        rails=power.get('resources',{}).get('jetson_rails',{})
        rail_w=sum(float(item.get('power_w',0)) for item in rails.values()) if rails else None
        activity=self.activity();effective=policy.get('profile','AUTO')
        if effective=='AUTO':
            effective='PERFORMANCE' if activity in ('ARM','NAVIGATION','GROUNDING') else 'BALANCED' if activity in ('TRAIN','LLM') else 'ENDURANCE'
        groups={}
        for row in rows:
            name=row.get('activity','UNCLASSIFIED');watts=sum(float(item.get('power_w',0)) for item in row.get('resources',{}).get('jetson_rails',{}).values())
            groups.setdefault(name,[]).append(watts)
        consumers=[dict(activity=name,samples=len(values),mean_jetson_rail_w=statistics.fmean(values))
                   for name,values in groups.items() if values]
        consumers.sort(key=lambda item:item['mean_jetson_rail_w'],reverse=True)
        return dict(at=time.time(),selected_profile=policy.get('profile','AUTO'),effective_profile=effective,
            current_activity=activity,policy=policy,battery_voltage_v=power.get('battery_voltage_v'),
            battery_soc=None,whole_robot_power_w=None,jetson_rail_power_w=rail_w,jetson_rails=rails,
            nvpmodel=power.get('resources',{}).get('nvpmodel'),heavy_jobs=self.heavy_jobs(),
            discharge_trend=self.trend(rows),activity_power=consumers,
            session_runtime_s=read(self.root/'data/runtime.json').get('uptime_s'),
            current_optimisation='adaptive perception and on-demand heavy models; motor/sensor duty cycling unchanged',
            claims={'soc':'unavailable','remaining_runtime':'voltage-trend estimate only when available',
                    'whole_robot_current':'unavailable','nvpmodel_changed':False})
