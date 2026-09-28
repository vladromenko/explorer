#!/usr/bin/env python3
"""Avoid synchronizing MCU/ROS with the Jetson's initial 1970 clock.

Does not set the clock, change networking, or require an internet connection.
Control, watchdog, power and the web UI are not delayed by this guard.
"""
import time
import argparse
import subprocess

parser=argparse.ArgumentParser()
parser.add_argument('--network-grace',type=float,default=0.)
args=parser.parse_args()
deadline=time.monotonic()+min(60.,max(0.,args.network_grace))
# The controller takes its epoch when the agent connects. Give boot NTP a
# bounded chance to finish; offline boot is still possible after this window.
synced=False
while time.monotonic()<deadline and not synced:
    try:
        synced=subprocess.run(['timedatectl','show','-p','NTPSynchronized','--value'],
            capture_output=True,text=True,timeout=2).stdout.strip()=='yes'
    except (OSError,subprocess.TimeoutExpired):pass
    if not synced:time.sleep(1)

last_wall=time.time();last_mono=time.monotonic();stable=0
while stable<3:
    time.sleep(1)
    wall=time.time();mono=time.monotonic()
    sane=wall>=1704067200 and abs((wall-last_wall)-(mono-last_mono))<.1
    stable=stable+1 if sane else 0
    last_wall,last_mono=wall,mono
