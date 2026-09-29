#!/usr/bin/env python3
"""Read saved Explorer diagnostics. Never opens UART or sends robot commands."""
import hashlib
import json
import time
from pathlib import Path

ROOT = Path("/home/vlad/Explorer")


def as_dict(value):
    return value if isinstance(value, dict) else {}


def read_json(relative):
    try:
        value = json.loads((ROOT / relative).read_text())
        return value if isinstance(value, dict) else {"read_error": "Expected object"}
    except (OSError, ValueError) as exc:
        return {"read_error": str(exc)}


def pick(value, keys):
    return {key: value.get(key) for key in keys}


def main():
    profile = read_json("config/controller-profile.json")
    registry = read_json("config/controller-releases.json")
    state = read_json("data/controller-state.json")
    identity = as_dict(state.get("identity"))
    controller = as_dict(state.get("controller"))
    stamp = state.get("at")
    now_ns = time.monotonic_ns()
    report = {
        "profile": pick(profile, (
            "read_error", "transport", "device", "firmware_source_sha256",
            "calibration_sha256", "controller_board", "controller_release",
            "telemetry_only", "hardware_accepted", "blocking_reason_ru",
        )),
        "state": pick(state, (
            "read_error", "at", "monotonic_ns", "telemetry_fresh",
            "telemetry_only", "session_state", "fault", "parser_errors",
        )),
        "state_file_age_s": time.time() - stamp if isinstance(stamp, (int, float)) else None,
        "identity": pick(identity, (
            "source_sha256", "uid", "device", "revision", "flash_kib",
            "boot", "reset_flags",
        )),
        "controller": pick(controller, (
            "mode", "fault", "session", "sequence", "acquired_us",
            "arm_enabled", "arm_cancel_pending", "arm_sent_generation", "diagnostics",
        )),
        "staged_source_sha256": registry.get("staged_source_sha256"),
        "known_releases": {
            source: pick(as_dict(entry), ("version", "image_sha256", "board", "calibration_sha256"))
            for source, entry in as_dict(registry.get("entries")).items()
        },
        "joints": [],
    }
    samples = as_dict(state.get("arm")).get("joints", [])
    if not isinstance(samples, list):
        samples = []
    for value in samples:
        sample = as_dict(value)
        item = pick(sample, (
            "joint", "error", "device_error", "raw_ticks", "position_rad",
            "physical_deg", "outside_soft_limit", "raw_valid", "position_valid",
            "fresh", "reply_hex", "reply_header2",
        ))
        acquired = sample.get("acquired_monotonic_ns")
        item["measurement_age_ms"] = (
            (now_ns - acquired) / 1_000_000 if isinstance(acquired, int) else None
        )
        report["joints"].append(item)
    try:
        actual = hashlib.sha256((ROOT / "config/controller-calibration.json").read_bytes()).hexdigest()
        report["actual_calibration_sha256"] = actual
        report["calibration_hash_matches_profile"] = actual == profile.get("calibration_sha256")
    except OSError as exc:
        report["calibration_read_error"] = str(exc)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
