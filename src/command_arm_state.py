"""One CommandOnly host contract. No UART access or inferred measurements."""
import math
import json
from pathlib import Path

SOURCE_SHA = 'ed0213dea86b3f77d39113d851d2bf36364ad9903b0540aba619d7c354078469'


def execution_state(arm, root):
    """Attach the active host curve without advancing the MCU command estimate."""
    if not arm.get('reference_valid'):return arm
    try:record=json.loads((Path(root)/'data/arm-state.json').read_text())
    except (OSError,ValueError):return arm
    if (arm.get('phase')=='EXECUTING' and record.get('phase')=='command_in_progress' and
        record.get('controller_boot_id')==arm.get('boot_id') and record.get('session')==arm.get('session') and
        record.get('command_generation')==arm.get('command_generation')):
        goal=record.get('q_goal')
        if isinstance(goal,list) and len(goal)==6 and all(type(q) in (int,float) and math.isfinite(q) for q in goal):
            arm.update(q_goal=goal,trajectory=record.get('trajectory'),trajectory_time=record.get('trajectory_time'))
    elif arm.get('phase') in ('READY','COMMAND_HOLD'):arm['q_goal']=arm['q_command']
    return arm


def describe(state, calibration, now_ns, profile=None):
    ref=state.get('manual_reference') or {}; controller=state.get('controller') or {}
    identity=state.get('identity') or {}; stamp=ref.get('received_monotonic_ns')
    compatible=identity.get('source_sha256')==SOURCE_SHA
    if profile is not None:
        compatible=compatible and profile.get('firmware_source_sha256')==SOURCE_SHA and profile.get('manual_reference_version')==1
    fresh=bool(state.get('telemetry_fresh') and type(stamp) is int and 0<=now_ns-stamp<300_000_000)
    session=bool(state.get('session_state')=='active' and ref.get('session')==controller.get('session') and ref.get('boot')==identity.get('boot'))
    initialized=ref.get('goal_verified_mask')==63
    torque='ON' if ref.get('torque_on_mask')==63 else 'OFF' if ref.get('torque_off_mask')==63 else 'UNKNOWN'
    fault=ref.get('error',0) or controller.get('fault',0) or state.get('fault')
    valid=bool(compatible and fresh and session and ref.get('reference_valid') and initialized and torque=='ON' and not fault)
    wire=ref.get('phase','UNKNOWN')
    phase=wire
    if wire in ('RELEASING','VERIFY_OFF'):phase='MANUAL_SETUP'
    if wire in ('CAPTURING','STAGING','VERIFY_TARGET','RECHECK_POSITION','ENABLING','VERIFY_ON'):phase='CAPTURING_REFERENCE'
    if wire=='CANCEL_PENDING':phase='EXECUTING'
    if wire=='FAULT' or fault and wire!='WAIT_REFERENCE':phase='ERROR'
    if not compatible or not fresh:phase='UNKNOWN'
    raw=ref.get('raw_sent',[]); origin=ref.get('raw_reference',[])
    if len(raw)!=6 or len(origin)!=6 or len(calibration)!=6:valid=False
    q=None; qref=None; degrees=None
    if valid:
        try:
            for c,r in zip(calibration,raw):c.target_raw(r*c.radians_per_tick+c.radians_at_raw_zero)
            q=[r*c.radians_per_tick+c.radians_at_raw_zero for c,r in zip(calibration,raw)]
            qref=[r*c.radians_per_tick+c.radians_at_raw_zero for c,r in zip(calibration,origin)]
            degrees=[r*c.physical_degrees_per_tick+c.physical_degrees_at_raw_zero for c,r in zip(calibration,raw)]
            if not all(math.isfinite(x) for x in q+qref+degrees):raise ValueError('nonfinite command')
        except (ValueError,TypeError):valid=False;q=None;qref=None;degrees=None;phase='ERROR'
    source='unknown'
    if valid:
        source='operator_reference' if raw==origin and wire=='READY' and ref.get('generation')==ref.get('reference_generation',ref.get('generation')) else 'command_estimate'
        if phase=='READY' and source=='command_estimate':phase='COMMAND_HOLD'
    reason=None
    if not compatible:reason='Неверная firmware identity CommandOnly'
    elif not fresh:reason='Нет свежего состояния контроллера'
    elif fault:reason='STM32: '+str(ref.get('error_name') if ref.get('error') else fault)
    elif not session:reason='Сессия контроллера не активна'
    elif not valid:reason='Нужна ручная исходная поза: '+phase
    return dict(phase=phase,wire_phase=wire,state_source=source,source=source,measured=False,
        estimated=valid,estimated_only=True,reference_valid=valid,reference_source='operator_reference' if valid else 'unknown',
        firmware_compatible=compatible,transport_ready=fresh,session_active=session,
        targets_initialized=initialized,torque_state=torque,command_enabled=valid and wire in ('READY','EXECUTING','CANCEL_PENDING'),
        blocked_by=reason,error=ref.get('error_name'),reference_pose=qref,q_reference=qref,
        q_command=q,q_estimated=q,q_goal=qref if source=='operator_reference' else None,trajectory=None,trajectory_time=0. if valid else None,
        command_generation=ref.get('generation'),reference_generation=ref.get('reference_generation'),
        last_accepted_command=q,last_command_time=ref.get('last_sent_us'),servo_deg=degrees,position_rad=q,
        raw_command=raw if valid else None,boot_id=identity.get('boot'),session=ref.get('session'),
        stop_type='commanded_hold',physical_stop_verified=False,physical_validation_status='not_accepted',
        feedback_available=False,all_fresh=False,joints=[],force_measured=False)
