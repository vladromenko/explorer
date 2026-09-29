#!/usr/bin/env python3
"""Build an OFFLINE candidate. Never opens serial, flashes or resets a board."""
from pathlib import Path
import argparse,hashlib,json,struct,subprocess
P=Path(__file__).resolve().parent
p=argparse.ArgumentParser();p.add_argument('--toolchain',required=True);p.add_argument('--base',required=True);p.add_argument('--output',required=True)
a=p.parse_args();tc=Path(a.toolchain);out=Path(a.output).resolve();out.mkdir(parents=True,exist_ok=True)
base=Path(a.base).read_bytes()
sha=lambda b:hashlib.sha256(b).hexdigest()
assert sha(base)=='e41e3cf0f8a406e354624cbf6ea15937e4da83d0ec0b9d04f76e16354b08b94c'
assert len(base)==1048576
# All literal references to the borrowed word are inside the replaced function.
references=[i for i in range(0,len(base)-3) if base[i:i+4]==struct.pack('<I',0x24001a64)]
assert references==[0xe2c], references
# The replaced original function must contain the only accesses through this literal.
assert base[0xdfc:0xe2c].hex()=='80b500af0a4b1b680133094a1360084b1b680a2b0add064b00221a6000f0c6fd0346002b02d0034b00221a7000bf80bd'
flags=['-mcpu=cortex-m7','-mthumb','-mfpu=fpv5-d16','-mfloat-abi=hard','-Os','-ffreestanding','-fno-builtin','-fno-unwind-tables','-fno-asynchronous-unwind-tables','-fno-stack-protector','-Wall','-Wextra','-Werror']
for src in ('watchdog.c','hooks.S'):
 subprocess.run([str(tc/'arm-none-eabi-gcc'),*flags,'-c',str(P/src),'-o',str(out/(src+'.o'))],check=True)
elf=out/'patch.elf'
subprocess.run([str(tc/'arm-none-eabi-gcc'),*flags,'-nostdlib','-Wl,--build-id=none','-Wl,-Map='+str(out/'patch.map'),'-T',str(P/'patch.ld'),str(out/'watchdog.c.o'),str(out/'hooks.S.o'),'-o',str(elf)],check=True)
subprocess.run([str(tc/'arm-none-eabi-objdump'),'-d',str(elf)],stdout=(out/'patch.asm').open('w'),check=True)
image=bytearray(base);changes=[]
for section,address,expected in [('.twist_hook',0xb65c,bytes.fromhex('b0b588b0')),('.tick_hook',0xdfc,bytes.fromhex('80b500af')),('.patch',0x57000,None)]:
 b=out/(section[1:]+'.bin')
 subprocess.run([str(tc/'arm-none-eabi-objcopy'),'-O','binary','--only-section='+section,str(elf),str(b)],check=True)
 data=b.read_bytes();before=base[address:address+len(data)]
 if expected is None:assert 0<len(data)<4096 and before==b'\xff'*len(data)
 else:assert before==expected and len(data)==4
 image[address:address+len(data)]=data
 changes.append(dict(offset=address,length=len(data),purpose=section,before=before.hex(),after=data.hex()))
# Keep active braking when factory ROS/application tasks terminate.
# Both original instructions are directly before a proved Motion_Stop call.
for address,expected_call in [(0xd68,'03f021fb'),(0xc41c,'f7f7c7ff')]:
 assert base[address:address+2]==b'\x00\x20'
 assert base[address+2:address+6].hex()==expected_call
 image[address:address+2]=b'\x01\x20'
 changes.append(dict(offset=address,length=2,purpose='factory shutdown brake',before='0020',after='0120'))
assert len(image)==len(base)
assert image.count(b'robotio')==base.count(b'robotio')==4
raw=bytes(image);(out/'M3PRO_FACTORY_WATCHDOG_1.1-rc1_NOT_ACCEPTED.bin').write_bytes(raw)
manifest=dict(schema=1,status='OFFLINE_CANDIDATE_NOT_INSTALLED_NOT_PHYSICALLY_ACCEPTED',image_bytes=len(raw),base_sha256=sha(base),image_sha256=sha(raw),changes=changes,
 limitations=['Twist has no source timestamp; queued zero followed by nonzero cannot be age-validated by this protocol','Recovery and brake behavior still require a secured physical bench','No new servo feedback; factory arm protocol unchanged'])
(out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps({k:v for k,v in manifest.items() if k!='changes'},indent=2))
