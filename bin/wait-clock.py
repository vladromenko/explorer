#!/usr/bin/env python3
"""Avoid synchronizing MCU/ROS with the Jetson's initial 1970 clock.

Does not set the clock, change networking, or require an internet connection.
Control, watchdog, power and the web UI are not delayed by this guard.
"""
import time

last_wall=time.time();last_mono=time.monotonic();stable=0
while stable<3:
    time.sleep(1)
    wall=time.time();mono=time.monotonic()
    sane=wall>=1704067200 and abs((wall-last_wall)-(mono-last_mono))<.1
    stable=stable+1 if sane else 0
    last_wall,last_mono=wall,mono
