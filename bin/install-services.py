#!/usr/bin/env python3
"""Install only the local manual-control runtime."""
from pathlib import Path
import subprocess

root = Path('/home/vlad/Explorer')
destination = Path.home() / '.config/systemd/user'
destination.mkdir(parents=True, exist_ok=True)
keep = ('mcu', 'watchdog', 'control', 'power', 'camera', 'camera-cache', 'web')
remove = ('learning', 'state', 'ekf', 'perception', 'llm', 'oled', 'gamepad', 'geometry',
          'slam', 'mapview', 'planning', 'navigation', 'train', 'speech', 'vpn', 'telegram',
          'tailscale', 'observer', 'nano-camera-cache')

for name in keep:
    source = root / 'systemd' / f'explorer-{name}.service'
    (destination / source.name).write_text(source.read_text())
(destination / 'explorer.target').write_text((root / 'systemd/explorer.target').read_text())

subprocess.run(['systemctl', '--user', 'daemon-reload'], check=True)
for name in remove:
    subprocess.run(['systemctl', '--user', 'disable', '--now', f'explorer-{name}.service'], check=False)
subprocess.run(['systemctl', '--user', 'enable', 'explorer.target'], check=True)
subprocess.run(['systemctl', '--user', 'restart', 'explorer.target'], check=True)
subprocess.run(['systemctl', '--user', 'start', *[f'explorer-{name}.service' for name in keep]], check=True)
