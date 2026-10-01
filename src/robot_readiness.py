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
    factory=read(root,'config/factory-runtime.json')
    if not profile and factory.get('transport')=='micro_ros':
        live=read(root,'data/status.json');flags=live.get('commissioning',{})
        calibration=read(root,'config/calibration-acceptance.json').get('records',{})
        connected=(0<=now-live.get('at',0)<3 and live.get('sensor_age',{}).get('odom',99)<.5)
        pose=live.get('arm_command_state',{});fault=read(root,'data/arm-telemetry-fault.json')
        arm_ready=(connected and pose.get('boot_id')==live.get('boot_id') and
                   pose.get('phase')=='command_elapsed_observation_required' and pose.get('at',0)>fault.get('at',0))
        accepted=all(flags.get(k) is True for k in ('base_commissioned','mcu_watchdog_verified','arm_commissioned'))
        navigation={}
        for component in ('planning','navigation'):
            health=read(root,'data/'+component+'-health.json')
            ready=0<=now-health.get('at',0)<4 and health.get('stage')=='active' and health.get('inputs_ready') is True
            navigation[component]=dict(ready=ready,stage=health.get('stage') if ready else 'unavailable')
        if navigation['navigation']['ready']:navigation['planning']=dict(ready=True,stage='active',source='navigation_planner_server')
        reasons=list(delivery.get('blocked_by',[]))
        if 'watchdog_1_0_starves_battery_sampling' in factory.get('known_issues',[]):
            reasons.append('В патче контроллера 1.0 команды мешают опросу батареи при непрерывной езде; обновление на стоянке восстановлено на Jetson')
        if not connected:reasons.insert(0,'Нет свежей телеметрии robotio')
        if not flags.get('mcu_watchdog_verified'):reasons.append('Не принята остановка после истечения действительной команды')
        if flags.get('localization_verified') is not True:reasons.append('Нужна физическая проверка локализации и повторного определения позы')
        if flags.get('gripper_calibrated') is not True:reasons.append('Нужна калибровка раскрытия и физическая проверка удержания предмета')
        return dict(message=('Шасси, два лидара, конечные команды руки и hand-eye приняты в указанном объёме. '
                             'Полная доставка ожидает локализацию и захват.'),
            controller_connected=connected,controller_accepted=accepted,firmware_compatible=connected,
            staged_release=None,firmware_installation_pending=False,navigation=navigation,
            arm_reference_initialized=arm_ready,arm_command_enabled=arm_ready,
            firmware_image_sha256=factory.get('image_sha256'),firmware_identity_source='operator_flash_record',
            delivery_ready=False,delivery_busy=bool(delivery.get('busy')),blocked_by=list(dict.fromkeys(reasons)),
            calibration=calibration,physical_validation_status='accepted_scoped',
            physical_acceptance_required=True,last_delivery_verified=False)
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
    arm=state.get('arm') or {}
    command_mode=profile.get('manual_reference_version')==1
    arm_ready=bool(selected and fresh and arm.get('command_enabled'))
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
    if command_mode:
        message='Рука готова по ручной привязке; положение расчётное.' if arm_ready else arm.get('blocked_by') or 'Ожидаю ручную исходную позу'
        if not arm_ready:reasons.insert(0,message)
    elif not controller_ready:reasons.insert(0,message)
    if not all(v['ready'] for v in navigation.values()):reasons.append('Навигация ожидает свежие датчики, карту или запуск серверов.')
    ready=not reasons
    if ready:message='Готов к запуску принятого сценария доставки.'
    last=delivery.get('last') or {}
    return dict(message=message,controller_connected=fresh,controller_accepted=controller_ready,
        staged_release=entry.get('version'),firmware_installation_pending=pending,
        navigation=navigation,delivery_ready=ready,delivery_busy=bool(delivery.get('busy')),
        firmware_compatible=selected,arm_reference_initialized=bool(arm.get('reference_valid')),
        arm_command_enabled=arm_ready,physical_validation_status='accepted' if controller_ready else 'not_accepted',
        blocked_by=list(dict.fromkeys(reasons)),
        last_delivery_verified=last.get('delivered') is True,
        physical_acceptance_required=not controller_ready)
