#!/usr/bin/env python3
"""Freeze a reviewed candidate; this script never connects to hardware."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
root=Path(__file__).resolve().parents[1]
first=Path('/private/tmp/explorer-sensor-startup-20260929/build-stm32')
repeat=Path('/private/tmp/explorer-sensor-startup-repeat-20260929/build-stm32')
dest=root/'releases/Explorer-STM32-0.2.1-sensor-startup-candidate'
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
for name in ('explorer.bin','explorer.elf','explorer.hex'):
    assert sha(first/name)==sha(repeat/name), name
assert not dest.exists()
dest.mkdir()
for name in ('explorer.bin','explorer.elf','explorer.hex','explorer.map','build-result.json'):
    shutil.copy2(first/name,dest/name)
shutil.copytree(root/'firmware',dest/'source',ignore=shutil.ignore_patterns('build-*','__pycache__'))
for name in ('core', 'runtime', 'servo_bus'):
    pass
for folder in ('evidence','recovery'):
    (dest/folder).mkdir()
shutil.copytree(root/'data/controller-acceptance-20260929',dest/'evidence/live-before')
shutil.copy2(first/'build.log',dest/'evidence/build.log')
shutil.copy2(repeat/'build.log',dest/'evidence/repeat-build.log')
host=Path('/private/tmp/explorer-sensor-startup-20260929/build-host')
with (dest/'evidence/host-tests.log').open('w') as log:
    subprocess.run(['ctest','--test-dir',str(host),'--output-on-failure'],stdout=log,stderr=subprocess.STDOUT,check=True)
old=root/'releases/Explorer-STM32-0.2.0-bench1'
previous=(old/'explorer-sector0.bin').read_bytes()
assert hashlib.sha256(previous).hexdigest()=='a69d32fb7faa8e74eb6a39c2c048fd3bbd58b8af8f8d69eeb8764a3c81950fb1'
app=(first/'explorer.bin').read_bytes()
assert len(app)<131072
(dest/'explorer-sector0.bin').write_bytes(app+previous[len(app):])
(dest/'recovery/installed-0.2.0-sector0.bin').write_bytes(previous)
shutil.copy2(old/'recovery/CURRENT-sector0.bin',dest/'recovery/factory-working-sector0.bin')
sys.path.insert(0,str(root/'firmware/tools'))
from package import ihex, verify_elf
(dest/'explorer-sector0.hex').write_text(ihex((dest/'explorer-sector0.bin').read_bytes()))
manifest=dict(status='candidate_not_authorized_for_write',hardware_accepted=False,
    image_sha256=sha(dest/'explorer-sector0.bin'),
    source_sha256=(first/'generated/source.sha256').read_text().strip(),
    measured_device=dict(device_id='0x450',revision='0x2003',flash_kib=1024),
    first_sector_address='0x08000000',first_sector_bytes=131072,
    banks=[dict(address='0x08000000',bytes=524288),dict(address='0x08100000',bytes=524288)],
    elf_load_segments=verify_elf(dest/'explorer.elf'),
    files={p.relative_to(dest).as_posix():sha(p) for p in sorted(dest.rglob('*')) if p.is_file()})
(dest/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps(dict(directory=str(dest),image_sha256=manifest['image_sha256'],
                     source_sha256=manifest['source_sha256']),indent=2))
