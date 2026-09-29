#!/usr/bin/env python3
"""Execute the actual patched Thumb bytes; board functions mocked at boundaries."""
import math,struct,sys
from pathlib import Path
from unicorn import Uc,UC_ARCH_ARM,UC_MODE_THUMB,UC_MODE_MCLASS,UC_HOOK_CODE,UC_HOOK_MEM_WRITE
from unicorn.arm_const import *
image=Path(sys.argv[1]).read_bytes()
u=Uc(UC_ARCH_ARM,UC_MODE_THUMB)
u.ctl_set_cpu_model(UC_CPU_ARM_CORTEX_A15)
u.mem_map(0x08000000,0x100000);u.mem_write(0x08000000,image)
u.mem_map(0x24000000,0x80000);u.mem_map(0xe000e000,0x2000)
u.mem_write(0xe000ed88,struct.pack('<I',0xf<<20))
u.reg_write(UC_ARM_REG_C1_C0_2,0xf<<20)
u.reg_write(UC_ARM_REG_FPEXC,0x40000000)
stop=0x080ff000
calls=[];low=False;bad_writes=[];irq=0
def code(u,addr,size,data):
 global irq
 # Unicorn's M7 model rejects FP64 comparison/conversion used by this actual
 # STM32H7 image. Execute FP64 on its A15 model; model only M-profile PRIMASK
 # instructions here. This is a function-level test, NOT an MCU/IRQ simulation.
 op=bytes(u.mem_read(addr,size))
 if size==4 and op[:3]==bytes.fromhex('eff310') and op[3]&0xf0==0x80:
  u.reg_write(UC_ARM_REG_R0+(op[3]&15),irq);u.reg_write(UC_ARM_REG_PC,(addr+4)|1);return
 if size==4 and op[1:]==bytes.fromhex('f31088') and op[0]&0xf0==0x80:
  irq=u.reg_read(UC_ARM_REG_R0+(op[0]&15));u.reg_write(UC_ARM_REG_PC,(addr+4)|1);return
 if op==bytes.fromhex('72b6'):
  irq=1;u.reg_write(UC_ARM_REG_PC,(addr+2)|1);return
 if addr==0x080043b0:
  assert u.reg_read(UC_ARM_REG_R0)==1
  assert irq==1 or '--reproduce-1.0' in sys.argv
  calls.append(('stop',1));u.reg_write(UC_ARM_REG_PC,u.reg_read(UC_ARM_REG_LR))
 elif addr==0x080049fc:
  values=[struct.unpack('<f',struct.pack('<I',u.reg_read(r)))[0] for r in (UC_ARM_REG_S0,UC_ARM_REG_S1,UC_ARM_REG_S2)]
  assert irq==1 or '--reproduce-1.0' in sys.argv
  calls.append(('command',values));u.reg_write(UC_ARM_REG_PC,u.reg_read(UC_ARM_REG_LR))
 elif addr==0x080019a8:
  assert irq==0
  calls.append(('battery',low));u.reg_write(UC_ARM_REG_R0,int(low));u.reg_write(UC_ARM_REG_PC,u.reg_read(UC_ARM_REG_LR))
def write(u,access,addr,size,value,data):
 allowed=(addr==0x24001a64 and size==4) or (addr==0x24001a5c and size==1) or (0x2406f000<=addr<0x24070000)
 if not allowed:bad_writes.append((hex(addr),size));raise AssertionError(bad_writes)
u.hook_add(UC_HOOK_CODE,code);u.hook_add(UC_HOOK_MEM_WRITE,write)
def reset():
 global low,irq
 low=False;calls.clear();u.mem_write(0x24001a64,b'\0'*4);u.mem_write(0x24001a5c,b'\1');irq=0
def call(address,arg=0):
 sp=0x24070000;u.reg_write(UC_ARM_REG_SP,sp);u.reg_write(UC_ARM_REG_LR,stop|1);u.reg_write(UC_ARM_REG_R0,arg)
 try:u.emu_start(address|1,stop,count=20000)
 except Exception:
  print("emulator PC",hex(u.reg_read(UC_ARM_REG_PC)));raise
 assert u.reg_read(UC_ARM_REG_PC)==stop
 assert u.reg_read(UC_ARM_REG_SP)==sp
 assert irq==0
 defstate=struct.unpack('<I',u.mem_read(0x24001a64,4))[0]
 assert defstate & ~0x1ffff==0
 return defstate
def twist(x=0,y=0,z=0):
 u.mem_write(0x24010000,struct.pack('<6d',x,y,0,0,0,z));return call(0x0800b65c,0x24010000)
def tick():return call(0x08000dfc)
if '--reproduce-1.0' in sys.argv:
 reset()
 for i in range(1100):twist(.04);tick()
 count=sum(c[0]=='battery' for c in calls)
 assert count==0,count
 print('REPRODUCED installed 1.0 defect: 1100 ticks + continuous Twist => zero battery checks')
 raise SystemExit(0)
reset();twist(.04);assert not calls
twist();twist(.04,.02,.15);assert len(calls)==2
assert all(abs(a-b)<1e-7 for a,b in zip(calls[-1][1],[.04,.02,.15]))
for i in range(25):tick()
assert not any(c[0]=='stop' for c in calls)
assert not tick()&0x10000
prior=len(calls);twist(.04);assert len(calls)==prior
reset();twist()
for i in range(1100):twist(.04);tick()
assert sum(c[0]=='battery' for c in calls)==100
assert sum(c[0]=='stop' for c in calls)==0
for value in (float('nan'),float('inf'),-float('inf'),1e300):
 assert not twist(value)&0x10000
reset();twist();low=True
for i in range(11):tick()
assert bytes(u.mem_read(0x24001a5c,1))==b'\0'
assert not bad_writes
print('PASS Thumb functions (M-profile PRIMASK modeled): hooks, hard-float ABI, stack, PRIMASK, zero rearm, expiry, battery cadence, NaN/Inf, RAM write allowlist')
