"""Documented servo packet decoder and explicitly sourced arm state. No serial IO."""
import math
import time


def decode_position(packet, expected_id):
    """Yahboom USART3 position response; return raw ticks, NOT calibrated degrees."""
    if len(packet) != 8 or packet[:2] != b'\xff\xff':
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
    command = state.get('arm_command_state', {})
    stamp = command.get('at')
    age = now-stamp if isinstance(stamp, (int, float)) and math.isfinite(stamp) else None
    raw = state.get('arm_feedback')
    transport=state.get('arm_feedback_transport',{})
    publishers=transport.get('publishers')
    received_age=state.get('sensor_age',{}).get('arm')
    fresh=isinstance(received_age,(float,int)) and 0<=received_age<.5
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
        next_step='Текущий publisher отсутствует в проверенном графе; старый MILO V4 содержит отдельный publisher и UART-чтение. Прошивку не менять. Для текущей руки использовать наблюдение, не объявлять отправленные углы измеренными.',
        command_echo_is_measurement=False, force_measured=False)
