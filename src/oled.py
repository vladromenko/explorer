"""Minimal SSD1306 status display; fails without disturbing control."""
import json
import socket
import time
from pathlib import Path
from PIL import Image, ImageDraw
from smbus2 import SMBus, i2c_msg

ROOT=Path('/home/vlad/Explorer')
bus=SMBus(7)
def send(control, values):
    bus.i2c_rdwr(i2c_msg.write(0x3c, bytes([control]+list(values))))
send(0,[0xae,0xd5,0x80,0xa8,0x3f,0xd3,0,0x40,0x8d,0x14,0x20,0,0xa1,0xc8,0xda,0x12,0x81,0x7f,0xd9,0xf1,0xdb,0x40,0xa4,0xa6,0xaf])
while True:
    try:
        s=json.loads((ROOT/'data/status.json').read_text())
        age=s.get('sensor_age',{})
        ip=socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
        ip.connect(('192.0.2.1',1))
        addr=ip.getsockname()[0]
        ip.close()
        im=Image.new('1',(128,64))
        draw=ImageDraw.Draw(im)
        lines=['EXPLORER '+('STOP' if s['stop_latched'] else 'CHECK'),addr,
               f"BAT {s['battery'] or 0:.2f} MCU {'OK' if age.get('odom',99)<1 else '--'}",
               'LIDAR '+str(sum(age.get(k,99)<1 for k in ['scan0','scan1']))+'/2  ARM CAL',
               s['reason'][:21]]
        for i,line in enumerate(lines): draw.text((0,i*12),line,fill=1)
        data=[sum((1<<b) if im.getpixel((x,p*8+b)) else 0 for b in range(8)) for p in range(8) for x in range(128)]
        send(0,[0x21,0,127,0x22,0,7])
        for offset in range(0,len(data),32):send(0x40,data[offset:offset+32])
    except (OSError,ValueError,KeyError):
        pass
    time.sleep(2)
