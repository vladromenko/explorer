"""Canonical semantic bindings for Explorer manual control."""

SCHEME_ID = "explorer-manual-v2"
ARM_MODES = ("cartesian", "joint")

BASE_KEYS = {
    "KeyW": "forward", "KeyS": "backward", "KeyA": "left", "KeyD": "right",
    "KeyQ": "turn_left", "KeyE": "turn_right", "KeyN": "grip_open", "KeyM": "grip_close",
}
CARTESIAN_KEYS = {
    "KeyI": "arm_x_forward", "KeyK": "arm_x_back", "KeyJ": "arm_y_left",
    "KeyL": "arm_y_right", "KeyU": "arm_z_up", "KeyO": "arm_z_down",
}
JOINT_KEYS = {
    "KeyJ": "joint1_decrease", "KeyL": "joint1_increase",
    "KeyI": "joint2_increase", "KeyK": "joint2_decrease",
    "KeyU": "joint3_increase", "KeyO": "joint3_decrease",
    "KeyY": "pitch_up", "KeyH": "pitch_down",
    "Comma": "wrist_left", "Period": "wrist_right",
}
MODIFIER_KEYS = frozenset(("ShiftLeft", "ShiftRight"))
MODE_KEYS = {"Digit1": "cartesian", "Digit2": "joint"}
ALL_KEY_CODES = frozenset(BASE_KEYS) | frozenset(CARTESIAN_KEYS) | frozenset(JOINT_KEYS) | MODIFIER_KEYS
ALL_ACTIONS = frozenset(BASE_KEYS.values()) | frozenset(CARTESIAN_KEYS.values()) | frozenset(JOINT_KEYS.values())
ARM_ACTIONS = frozenset(CARTESIAN_KEYS.values()) | frozenset(JOINT_KEYS.values())


def keyboard_inputs(keys, arm_mode="cartesian"):
    if arm_mode not in ARM_MODES:
        raise ValueError("Неизвестный режим руки")
    clean = {str(key) for key in keys}
    unknown = clean - ALL_KEY_CODES
    if unknown:
        raise ValueError("Неизвестная физическая клавиша teleop: " + ", ".join(sorted(unknown)))
    mapping = dict(BASE_KEYS)
    mapping.update(CARTESIAN_KEYS if arm_mode == "cartesian" else JOINT_KEYS)
    return {action: 1.0 for key, action in mapping.items() if key in clean}


def scheme():
    return {
        "id": SCHEME_ID,
        "arm_modes": list(ARM_MODES),
        "default_arm_mode": "cartesian",
        "frames": {"cartesian": "base_footprint", "x": "вперёд", "y": "влево", "z": "вверх"},
        "keyboard": {
            "common": dict(BASE_KEYS), "cartesian": dict(CARTESIAN_KEYS), "joint": dict(JOINT_KEYS),
            "precision": ["ShiftLeft", "ShiftRight"], "stop": ["Space", "Escape"],
            "resume": "Enter", "mode": dict(MODE_KEYS),
        },
        "gamepad": {
            "common": {
                "left_stick": "шасси: ход и mecanum-смещение", "l1": "поворот шасси влево",
                "r1": "поворот шасси вправо", "l2": "открывать захват",
                "r2": "закрывать захват", "x": "точный режим", "b": "STOP",
                "start": "возобновить", "select": "сменить режим руки",
            },
            "cartesian": {"right_stick_y": "X", "right_stick_x": "Y", "y_a": "Z"},
            "joint": {"right_stick_x": "J1", "right_stick_y": "J2", "y_a": "J3", "dpad_y": "J4", "dpad_x": "J5"},
        },
        "help": {
            "keyboard_common": "W/S — вперёд/назад; A/D — mecanum влево/вправо; Q/E — поворот; N/M — открыть/закрыть захват.",
            "keyboard_cartesian": "I/K — X вперёд/назад; J/L — Y влево/вправо; U/O — Z вверх/вниз.",
            "keyboard_joint": "J/L — J1−/+; I/K — J2+/−; U/O — J3+/−; Y/H — J4+/−; ,/. — J5−/+.",
            "gamepad_common": "Левый стик — ход и mecanum; L1/R1 — поворот; L2/R2 — открыть/закрыть захват.",
            "gamepad_cartesian": "Правый стик — X/Y; Y/A — Z вверх/вниз; крестовина не двигает руку.",
            "gamepad_joint": "Правый стик — J1/J2; Y/A — J3+/−; крестовина ↑/↓ — J4+/−, ←/→ — J5−/+.",
        },
    }
