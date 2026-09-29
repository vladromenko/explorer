#!/usr/bin/env python3
"""Explicitly select the physically identified build, with motion disabled."""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root/'src'))
from controller_feedback import vendor_calibration
cal = root/'config/controller-calibration.json'
profile = root/'config/controller-profile.json'
if cal.exists() or profile.exists():
    raise SystemExit('Existing controller profile: review it instead of overwriting.')
cal.write_text(json.dumps(dict(schema=1, physically_calibrated=False,
    origin='vendor nominal coefficients; measured raw remains independent',
    joints=[asdict(c) for c in vendor_calibration()]), indent=2)+'\n')
profile.write_text(json.dumps(dict(transport='controller_v1', device='/dev/explorer_mcu',
    firmware_source_sha256='6c189a2d8d511c612465f4a7d098fb40212500e1b3a4a4ccee52984568eeaf3f',
    calibration_sha256=hashlib.sha256(cal.read_bytes()).hexdigest(),
    telemetry_only=True, hardware_accepted=False), indent=2)+'\n')
