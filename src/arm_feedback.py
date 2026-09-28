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
    return dict(joint_ids=list(range(1, 7)), commanded=dict(values=command.get('servo_deg'),
        units='degree', age_s=age, phase=command.get('phase'), source='last_sent_command'),
        measured=dict(values=None, units='degree', age_s=None, available=False,
        reason='В активном micro-ROS интерфейсе нет подтверждённого пути запроса и ответа измерений'),
        estimated=None, unverified_topic_values=raw, graph=graph,
        servo_protocol=dict(bus='STM32 USART3', operation='UartServo_Get_Position',
                            address='0x38', response_data_bytes=2, raw_units='ticks'),
        next_step='Нужен документированный метод чтения через установленную прошивку либо согласованный отдельный интерфейс к шине сервоприводов; занятый порт не открывать',
        command_echo_is_measurement=False, force_measured=False)
