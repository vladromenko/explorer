"""Read-only explanation of prepared software and live physical readiness."""
import json
from pathlib import Path
import time


def read(root, name):
    try:
        value=json.loads((Path(root)/name).read_text())
        return value if isinstance(value,dict) else {}
    except (OSError,ValueError):return {}


def status(root, delivery, now=None):
    now=time.time() if now is None else now
    profile=read(root,'config/controller-profile.json')
    registry=read(root,'config/controller-releases.json')
    state=read(root,'data/controller-state.json')
    identity=state.get('identity') or {}
    fresh=(type(state.get('at')) in (int,float) and 0<=now-state['at']<=3
           and state.get('telemetry_fresh') is True)
    current=identity.get('source_sha256') if fresh else None
    staged=registry.get('staged_source_sha256')
    entry=registry.get('entries',{}).get(staged,{})
    selected=bool(current and current==profile.get('firmware_source_sha256'))
    pending=bool(staged and current!=staged)
    controller_ready=selected and profile.get('hardware_accepted') is True and not profile.get('telemetry_only',True)
    if not fresh:message='Ожидаю подключение контроллера и свежие измерения.'
    elif pending:message='Jetson подготовлен к новому образу. Сейчас подключена прежняя прошивка.'
    elif not selected:message='Прошивка подключённого контроллера не соответствует выбранному профилю.'
    elif not controller_ready:message='Контроллер на связи. Требуется физическая приёмка шасси и руки.'
    else:message='Контроллер на связи, его физическая приёмка сохранена.'
    navigation={}
    for component in ('planning','navigation'):
        value=read(root,'data/'+component+'-health.json')
        recent=type(value.get('at')) in (int,float) and 0<=now-value['at']<=4
        navigation[component]=dict(ready=bool(recent and value.get('stage')=='active' and value.get('inputs_ready')),
            stage=value.get('stage') if recent else 'unavailable',waiting_on=value.get('waiting_on',[]) if recent else ['health_stale'])
    reasons=list(delivery.get('blocked_by',[]))
    if not controller_ready:reasons.insert(0,message)
    if not all(v['ready'] for v in navigation.values()):reasons.append('Навигация ожидает свежие датчики, карту или запуск серверов.')
    ready=not reasons
    if ready:message='Готов к запуску принятого сценария доставки.'
    last=delivery.get('last') or {}
    return dict(message=message,controller_connected=fresh,controller_accepted=controller_ready,
        staged_release=entry.get('version'),firmware_installation_pending=pending,
        navigation=navigation,delivery_ready=ready,delivery_busy=bool(delivery.get('busy')),
        blocked_by=list(dict.fromkeys(reasons)),
        last_delivery_verified=last.get('delivered') is True,
        physical_acceptance_required=not controller_ready)
