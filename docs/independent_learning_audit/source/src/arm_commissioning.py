"""Supervised near-home probes, not general manipulation or measured feedback."""
import math

HOME = [90, 125, 3, 0, 90, 30]
HARD_LIMITS = [(0, 180)] * 4 + [(0, 270), (30, 180)]


def validate_incremental(start, pose, runtime_ms, coordinated=False):
    if len(start)!=6 or len(pose)!=6 or any(type(v) is not int for v in start+pose):
        raise ValueError('Two complete integer poses required')
    if type(runtime_ms) is not int or not 3000<=runtime_ms<=5000:
        raise ValueError('Incremental observation requires 3000..5000 ms')
    for a,b,(lo,hi) in zip(start,pose,HARD_LIMITS):
        if not lo<=a<=hi or not lo<=b<=hi or abs(a-b)>10:
            raise ValueError('Step exceeds hard limits or 10 degrees')
    if not coordinated and sum(a!=b for a,b in zip(start,pose))!=1:
        raise ValueError('Change exactly one joint per observed step')


def validate_pose(pose, runtime_ms):
    if len(pose) != 6 or any(type(v) is not int for v in pose):
        raise ValueError('Six integer servo angles required')
    if type(runtime_ms) is not int or not 1000 <= runtime_ms <= 5000:
        raise ValueError('Commissioning runtime must be 1000..5000 ms')
    for v, h, (lo, hi) in zip(pose, HOME, HARD_LIMITS):
        if not lo <= v <= hi:
            raise ValueError('Hardware angle range exceeded')
        if abs(v-h) > 10:
            raise ValueError('Outside supervised near-home commissioning region')


def stationary_status(s, now):
    age = now - s['at']
    velocity = s['velocity']
    battery = s['battery']
    if not math.isfinite(age) or not 0 <= age < 1:
        raise ValueError('Controller status is stale')
    if not (s['stop_latched'] is True or s.get('base_hold_confirmed') is True) or len(velocity) != 3:
        raise ValueError('Base requires latched STOP or confirmed stationary hold')
    if s.get('power',{}).get('state') in ('CRITICAL','CHARGING','UNKNOWN','LOW_POWER'):
        raise ValueError('Power policy blocks arm commissioning')
    if any(not math.isfinite(v) or abs(v) > .001 for v in velocity):
        raise ValueError('Base command is not zero')
    observed=s.get('odom_velocity')
    if not isinstance(observed,list) or len(observed)!=3 or any(
            not isinstance(v,(int,float)) or not math.isfinite(v) or abs(v)>=limit
            for v,limit in zip(observed,[.005,.005,.02])):
        raise ValueError('Measured base velocity is missing or not stationary')
    if not isinstance(battery, (int, float)) or not math.isfinite(battery) or battery < 11:
        raise ValueError('Battery below commissioning threshold')
    if any(not math.isfinite(s['sensor_age'].get(k, math.inf)) or
           not 0 <= s['sensor_age'].get(k, math.inf) < limit
           for k, limit in [('odom', .5), ('battery', 2)]):
        raise ValueError('MCU telemetry is stale')


def coordinated_status(s, now):
    """Validate slow, supervised mobile manipulation without requiring base HOLD."""
    age=now-s['at'];velocity=s.get('velocity');observed=s.get('odom_velocity')
    if not math.isfinite(age) or not 0<=age<1:raise ValueError('Controller status is stale')
    if s.get('mode')!='MANUAL' or s.get('stop_latched') is True or s.get('mission'):
        raise ValueError('Совместное управление требует ручного режима без STOP')
    if not isinstance(velocity,list) or len(velocity)!=3 or any(
            not isinstance(v,(int,float)) or not math.isfinite(v) or abs(v)>limit
            for v,limit in zip(velocity,[.81,.73,1.68])):
        raise ValueError('Команда шасси вне диапазона совместного управления')
    if not isinstance(observed,list) or len(observed)!=3 or any(
            not isinstance(v,(int,float)) or not math.isfinite(v) or abs(v)>limit
            for v,limit in zip(observed,[.95,.87,1.95])):
        raise ValueError('Фактическая скорость шасси вне диапазона совместного управления')
    if s.get('power',{}).get('state') in ('CRITICAL','CHARGING','UNKNOWN','LOW_POWER'):
        raise ValueError('Power policy blocks mobile manipulation')
    battery=s.get('battery')
    if not isinstance(battery,(int,float)) or not math.isfinite(battery) or battery<11:
        raise ValueError('Battery below mobile manipulation threshold')
    if any(not math.isfinite(s.get('sensor_age',{}).get(k,math.inf)) or
           not 0<=s['sensor_age'].get(k,math.inf)<limit for k,limit in [('odom',.5),('battery',2)]):
        raise ValueError('MCU telemetry is stale')


def coordinated_policy_status(s,now):
    """Policy arm steps require the same live local mission as policy base steps."""
    age=now-s.get("at",0)
    if not math.isfinite(age) or not 0<=age<.7:raise ValueError("Состояние контроллера устарело")
    if s.get("mode")!="AUTONOMOUS" or s.get("stop_latched") is True or not s.get("mission"):
        raise ValueError("Нет действующей автономной миссии")
    flags=s.get("commissioning",{})
    if not all(flags.get(key) is True for key in ("base_commissioned","arm_commissioned",
            "lidar_tf_validated","mcu_watchdog_verified","localization_verified")):
        raise ValueError("Калибровка мобильной манипуляции не принята")
    for key,ttl in (("odom",.5),("battery",2.),("scan0",.6),("scan1",.6)):
        value=s.get("sensor_age",{}).get(key,math.inf)
        if not isinstance(value,(int,float)) or not 0<=value<ttl:
            raise ValueError("Датчик устарел: "+key)
    if s.get("power",{}).get("state") in ("CRITICAL","CHARGING","UNKNOWN","LOW_POWER"):
        raise ValueError("Питание не допускает движение")
    for key,limits in (("velocity",(.13,.13,.36)),("odom_velocity",(.30,.30,.65))):
        values=s.get(key)
        if not isinstance(values,list) or len(values)!=3 or any(
                not isinstance(value,(int,float)) or not math.isfinite(value) or abs(value)>limit
                for value,limit in zip(values,limits)):
            raise ValueError("Скорость вне диапазона наблюдаемой попытки")
