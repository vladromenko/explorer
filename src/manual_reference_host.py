"""Manual-reference wire state and gates; pure, testable, no serial side effects."""
from dataclasses import replace
import json
import math
from pathlib import Path
import struct
import time
from controller_protocol import RESULT_NAMES

PHASES = ('WAIT_REFERENCE', 'RELEASING', 'VERIFY_OFF', 'MANUAL_SETUP', 'CAPTURING',
          'STAGING', 'VERIFY_TARGET', 'RECHECK_POSITION', 'ENABLING', 'VERIFY_ON',
          'READY', 'EXECUTING', 'FAULT', 'CANCEL_PENDING')
OPERATIONS = frozenset(('CLEAR', 'OPEN', 'CALIBRATION', 'MR_BEGIN', 'MR_CAPTURE',
    'MR_ABORT', 'MR_KEEPALIVE', 'MR_MOVE', 'MR_STOP', 'MR_DIAGNOSTIC', 'ARM_ENABLE',
    'ARM', 'ARM_CANCEL', 'CANCEL', 'HOLD'))
STOP_OPERATIONS = frozenset(('MR_ABORT', 'MR_STOP', 'ARM_CANCEL', 'CANCEL', 'HOLD'))


def decode_reference(payload, now_ns):
    if len(payload) != 244:
        raise ValueError('manual-reference status length mismatch')
    acquired, boot, session, generation = struct.unpack_from('<QQQI', payload)
    phase, error, failed, flags = payload[28:32]
    if phase >= len(PHASES) or failed > 6 or flags & ~7:
        raise ValueError('bad manual-reference state')
    arrays = {name: list(struct.unpack_from('<6f', payload, offset)) for name, offset in
              (('position_rad', 60), ('slopes', 84), ('offsets', 108), ('lower', 132), ('upper', 156))}
    if not all(math.isfinite(x) for values in arrays.values() for x in values):
        raise ValueError('non-finite manual-reference state')
    valid = bool(flags & 1)
    if valid and (any(x == 0 for x in arrays['slopes']) or
                  any(a >= b for a, b in zip(arrays['lower'], arrays['upper']))):
        raise ValueError('bad manual-reference calibration')
    if valid and (payload[33] != 63 or payload[34] != 63):
        raise ValueError('Reference valid без проверенных целей/момента 6/6')
    return dict(acquired_us=acquired, boot=boot, session=session, generation=generation,
        phase=PHASES[phase], error=error, error_name=RESULT_NAMES[error] if error<len(RESULT_NAMES) else 'UNKNOWN_ERROR',
        last_register=payload[35], failed_id=failed, reference_valid=valid,
        outstanding=bool(flags & 2), diagnostic=bool(flags & 4),
        torque_off_mask=payload[32], goal_verified_mask=payload[33], torque_on_mask=payload[34],
        raw_reference=list(struct.unpack_from('<6H', payload, 36)),
        raw_sent=list(struct.unpack_from('<6H', payload, 48)),
        captured_us=list(struct.unpack_from('<6Q', payload, 180)),
        lease_us=struct.unpack_from('<Q', payload, 228)[0],
        last_sent_us=struct.unpack_from('<Q', payload, 236)[0],
        received_monotonic_ns=now_ns, measured=False, estimated=valid,
        reference_source='operator_reference' if valid else 'unknown',
        state_source='command_estimate' if valid else 'unknown', **arrays)


def effective_calibration(nominal, reference):
    if not reference.get('reference_valid') or len(nominal) != 6:
        raise ValueError('Ручная опорная установка не завершена')
    result = []
    for index, cal in enumerate(nominal):
        raw = reference['raw_reference'][index]
        if type(raw) is not int or not cal.command_min <= raw <= cal.command_max:
            raise ValueError('Опорная координата вне сохранённого raw-диапазона: '+str(index+1))
        c = replace(cal, radians_per_tick=reference['slopes'][index],
            radians_at_raw_zero=reference['offsets'][index],
            lower=reference['lower'][index], upper=reference['upper'][index],
            recovery_min=0, recovery_max=0)
        # Sign and scale are inherited, never inferred from one reference pose.
        if abs(c.radians_per_tick-cal.radians_per_tick) > 1e-9:
            raise ValueError('Масштаб привода изменился')
        # Command-only firmware does not read servo position and therefore does
        # not infer a new electronic zero. ROS/raw calibration remains the
        # immutable nominal mapping; the operator reference only establishes
        # the current command state. Vendor degrees remain unchanged too:
        # gripper raw=2850 is about 159.5 degrees, never relabel it as 180.
        if (abs(c.radians_at_raw_zero-cal.radians_at_raw_zero)>1e-9 or
            abs(c.lower-cal.lower)>1e-9 or abs(c.upper-cal.upper)>1e-9):
            raise ValueError('Командная калибровка изменила сохранённое преобразование')
        result.append(c)
    return result


def reference_from_state(state, nominal, now_ns, expected_boot=None, expected_session=None):
    ref = state.get('manual_reference') or {}
    at = ref.get('received_monotonic_ns')
    if (not state.get('telemetry_fresh') or type(at) is not int or not 0 <= now_ns-at < 300_000_000):
        raise ValueError('Нет свежего состояния ручной привязки STM32')
    identity = state.get('identity') or {}
    controller = state.get('controller') or {}
    from command_arm_state import describe
    contract=describe(state,nominal,now_ns)
    if not contract['command_enabled']:raise ValueError(contract['blocked_by'] or 'CommandOnly arm not ready')
    if (ref.get('boot') != identity.get('boot') or ref.get('session') != controller.get('session') or
        expected_boot is not None and ref.get('boot') != expected_boot or
        expected_session is not None and ref.get('session') != expected_session):
        raise ValueError('Загрузка или сессия STM32 изменилась')
    cal = effective_calibration(nominal, ref)
    q = contract['q_command']
    physical = [((x-c.radians_at_raw_zero)/c.radians_per_tick)*c.physical_degrees_per_tick+
                c.physical_degrees_at_raw_zero for c, x in zip(cal, q)]
    return dict(at=time.time(), boot_id=ref['boot'], session=ref['session'],
        servo_deg=physical, position_rad=q, raw_ticks=list(ref['raw_sent']),
        outside_soft_limits=[i+1 for i, (c, x) in enumerate(zip(cal, q)) if not c.lower-1e-6 <= x <= c.upper+1e-6],
        measured=False, estimated_only=True, attained=False, source=contract['state_source'],
        reference_source='operator_reference', phase=ref['phase'],
        reference_generation=ref.get('reference_generation',ref['generation']),
        command_generation=ref['generation'], calibration=cal)


def manual_request_allowed(root, profile, identity, controller, request, now_ns):
    if profile.get('manual_reference_version') != 1:
        return False
    operation = request.get('operation')
    source = request.get('source_id', '')
    if operation not in OPERATIONS or not isinstance(source, str) or not source.startswith(('manual_', 'native_arm_')):
        return False
    if not identity or identity.get('source_sha256') != profile.get('firmware_source_sha256'):
        raise ValueError('Нет подтверждённой идентичности manual-reference прошивки')
    if operation in ('MR_BEGIN', 'MR_CAPTURE') and request.get('operator_supported') is not True:
        raise ValueError('Нужно явное подтверждение поддержки руки')
    # Stop remains available under bad power/feedback; it never enables torque.
    if operation in STOP_OPERATIONS:
        return True
    wheels = controller.get('wheels', [])
    if operation in ('MR_BEGIN','MR_CAPTURE','CALIBRATION') and (len(wheels) != 4 or any(w.get('pwm') != 0 or w.get('target_rad_s') != 0 or
            type(w.get('measured_rad_s')) not in (float, int) or
            not math.isfinite(w['measured_rad_s']) or abs(w['measured_rad_s']) > .1 for w in wheels)):
        raise ValueError('Для ручной установки/проверки руки шасси должно стоять')
    # Keep the existing power policy, including its CRITICAL latch. Do not
    # clear it, guess charge state or substitute nominal battery voltage.
    from controller_arm_commissioning import require_arm_test_power
    try:
        require_arm_test_power(root)
    except (OSError, ValueError, TypeError, AttributeError) as exc:
        raise ValueError('Недоступна пригодная телеметрия питания: '+str(exc)) from exc
    return True
