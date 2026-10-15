"""Read-only resource budget, including the macOS kernel pressure estimate."""
import os
import re
import subprocess
import sys

import psutil


def _mac_available(total):
    if sys.platform!='darwin':return None
    try:
        result=subprocess.run(['/usr/bin/memory_pressure'],capture_output=True,text=True,
            timeout=2,check=True,env=dict(os.environ,LC_ALL='C'))
        match=re.search(r'^System-wide memory free percentage:\s*(\d+)%\s*$',result.stdout,re.M)
        percent=int(match[1]) if match else -1
        if 0<=percent<=100:return int(total*percent/100.)
    except (OSError,ValueError,subprocess.SubprocessError):pass
    return None


def memory_budget():
    memory=psutil.virtual_memory();native=_mac_available(memory.total)
    return dict(total_bytes=memory.total,available_bytes=memory.available if native is None else native,
        psutil_available_bytes=memory.available,
        source='psutil' if native is None else 'macos_memory_pressure',
        minimum_available_fraction=.15,reserved_bytes_per_worker=int(1.7*2**30))
