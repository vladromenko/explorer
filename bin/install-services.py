#!/usr/bin/env python3
"""Install unprivileged boot services; no changes to old project units."""
from pathlib import Path
import subprocess

root=Path('/home/vlad/Explorer')
destination=Path.home()/'.config/systemd/user'
destination.mkdir(parents=True,exist_ok=True)
components=['mcu','watchdog','control','power','learning','state','ekf','camera','perception','llm','web','oled','gamepad','geometry','slam','mapview','planning','navigation']
boot_components=[n for n in components if n!='llm']
for name in components:
    limits=''
    if name=='llm':limits='Nice=10\nCPUWeight=20\nMemoryHigh=3000M\nMemoryMax=3400M\nOOMScoreAdjust=500\n'
    if name=='perception':limits='Nice=5\nCPUWeight=60\nMemoryMax=1500M\nOOMScoreAdjust=300\n'
    if name=='learning':limits='Nice=15\nCPUWeight=10\nMemoryMax=512M\nOOMScoreAdjust=500\n'
    after='After=explorer-mcu.service\n' if name=='control' else ''
    if name=='mcu':after='Before=explorer-control.service explorer-watchdog.service\n'
    if name=='watchdog':after='After=explorer-mcu.service\nBefore=explorer-control.service\n'
    unit=f'''[Unit]
Description=Explorer {name}
PartOf=explorer.target
StartLimitIntervalSec=60
StartLimitBurst=3
{after}
[Service]
Type=simple
WorkingDirectory={root}
ExecStart=/bin/bash {root}/bin/run.sh {name}
Restart=on-failure
RestartSec=3
TimeoutStopSec=8
KillMode=control-group
UMask=0077
{limits}
[Install]
WantedBy=explorer.target
'''
    (root/'systemd'/f'explorer-{name}.service').write_text(unit)
    (destination/f'explorer-{name}.service').write_text(unit)
target='[Unit]\nDescription=Explorer local robot runtime\nWants='+ ' '.join(f'explorer-{n}.service' for n in boot_components)+'\n\n[Install]\nWantedBy=default.target\n'
(root/'systemd/explorer.target').write_text(target)
(destination/'explorer.target').write_text(target)
for optional in ('train','speech','vpn','telegram','tailscale'):
    (destination/f'explorer-{optional}.service').write_text((root/f'systemd/explorer-{optional}.service').read_text())
# Only replace old transient units; leave existing persistent services running.
for name in components:
    transient=Path(f'/run/user/1000/systemd/transient/explorer-{name}.service')
    if transient.exists():
        subprocess.run(['systemctl','--user','stop',f'explorer-{name}'],check=True)
        transient.unlink(missing_ok=True)
subprocess.run(['systemctl','--user','daemon-reload'],check=True)
subprocess.run(['systemctl','--user','enable','--now','explorer.target'],check=True)
subprocess.run(['systemctl','--user','disable','explorer-llm.service'],check=False)
subprocess.run(['systemctl','--user','start']+[f'explorer-{n}' for n in boot_components],check=True)
