"""Documented servo packet decoder and explicitly sourced arm state. No serial IO."""
import math
import time


def decode_position(packet, expected_id, reply_header2=0xf5):
    """Yahboom USART3 position response; return raw ticks, NOT calibrated degrees."""
    if reply_header2 not in (0xf5,0xff) or len(packet) != 8 or packet[:2] != bytes((255,reply_header2)):
        raise ValueError('Invalid servo response header/length')
    if not 1 <= expected_id <= 6 or packet[2] != expected_id or packet[3] != 4:
        raise ValueError('Wrong joint ID or response length')
    if packet[4] != 0:
        raise ValueError('Servo returned an error')
    if (~sum(packet[2:7])) & 255 != packet[7]:
        raise ValueError('Invalid servo checksum')
    return dict(joint_id=expected_id, raw_ticks=(packet[5] << 8) | packet[6], units='ticks')


def describe(state, graph=None, now=None):
    now = time.time() if now is None else now
    arm=state.get('arm_state') or state.get('arm_measurements') or {}
    if arm.get('estimated_only'):
        fresh=not state.get('stale') and 0<=now-state.get('at',0)<1 and arm.get('reference_valid')
        values=arm.get('servo_deg') if fresh else None
        return dict(measured=dict(available=False,values=None,reason='COMMAND_ONLY_BY_DESIGN'),
            estimated=dict(values=values,source=arm.get('state_source') if fresh else 'unknown'),
            joints=[dict(id=i+1,estimated_deg=values[i] if values else None,measured_deg=None) for i in range(6)],
            state_source=arm.get('state_source') if fresh else 'unknown',phase=arm.get('phase'),
            command_echo_is_measurement=False,force_measured=False,
            next_step=arm.get('blocked_by') or 'Положение расчётное, контролируйте движение визуально.')
    command = state.get('arm_command_state', {})
    stamp = command.get('at')
    age = now-stamp if isinstance(stamp, (int, float)) and math.isfinite(stamp) else None
    raw = state.get('arm_feedback')
    transport=state.get('arm_feedback_transport',{})
    publishers=transport.get('publishers')
    received_age=state.get('sensor_age',{}).get('arm')
    fresh=isinstance(received_age,(float,int)) and 0<=received_age<.5
    measured = state.get('arm_measurements') or {}
    if measured.get('source') == 'stm32_uart3_readback':
        samples = measured.get('joints', [])
        valid = (fresh and not state.get('stale') and measured.get('all_fresh') and
                 len(samples) == 6 and all(s.get('joint') == i+1 and s.get('position_valid') and
                 s.get('fresh') and s.get('error') == 0 for i, s in enumerate(samples)))
        return dict(joint_ids=list(range(1,7)),
            commanded=dict(values=command.get('servo_deg'),units='degree',age_s=age,
                           phase=command.get('phase'),source='last_sent_command'),
            measured=dict(values=[s['physical_deg'] for s in samples] if valid else None,
                units='degree',available=bool(valid),age_s=received_age,
                reason='UART_READBACK' if valid else 'STALE_READBACK',calibration='vendor_nominal'),
            joints=[dict(id=s['joint'],raw_ticks=s.get('raw_ticks') if valid else None,
                measured_deg=s.get('physical_deg') if valid else None,
                joint_rad=s.get('position_rad') if valid else None,
                outside_soft_limit=s.get('outside_soft_limit'),fresh=bool(valid)) for s in samples],
            command_echo_is_measurement=False,force_measured=False,
            next_step='Показания получены от сервоприводов; калибровка механических нулей ещё не принята.')
    # Presence is not a measurement contract. None means semantics unverified,
    # False means that this sampled transport does not supply current values.
    available=None if raw is not None and fresh and publishers else False
    reason='NO_CURRENT_VALUES' if available is False else 'SOURCE_SEMANTICS_UNVERIFIED'
    if publishers==[]:reason='NO_PUBLISHER_IN_CURRENT_ROS_GRAPH'
    commanded=command.get('servo_deg')
    estimated=commanded if command.get('phase')=='command_elapsed_observation_required' else None
    joints=[dict(id=i,commanded_deg=commanded[i-1] if commanded else None,
                 measured_deg=None,estimated_deg=estimated[i-1] if estimated else None,
                 estimate_source='elapsed_command' if estimated else None,error_bound_deg=None)
            for i in range(1,7)]
    return dict(joint_ids=list(range(1, 7)), commanded=dict(values=command.get('servo_deg'),
        units='degree', age_s=age, phase=command.get('phase'), source='last_sent_command'),
        measured=dict(values=None, units='degree', age_s=None, available=available,
        reason=reason,hardware_readback_impossible=False),
        estimated=dict(values=estimated,source='elapsed_command',age_s=age,error_bound_deg=None),
        joints=joints,transport=transport,unverified_topic_values=raw,unverified_receive_age_s=received_age,graph=graph,
        servo_protocol=dict(bus='STM32 USART3', operation='UartServo_Get_Position',
                            address='0x38', response_data_bytes=2, raw_units='ticks'),
        next_step='Для измеренной позы требуется принятая полная прошивка и совместимый драйвер. Отправленная цель не является измерением.',
        command_echo_is_measurement=False, force_measured=False)
