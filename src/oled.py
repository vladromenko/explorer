"""M3 Pro 128x32 SSD1306 status display; independent of motion control."""
import json
import socket
import time
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from smbus2 import SMBus, i2c_msg

ROOT=Path('/home/vlad/Explorer')
bus=SMBus(7)
def send(control, values):
    bus.i2c_rdwr(i2c_msg.write(0x3c, bytes([control]+list(values))))
# Vendor bsp_ssd1306.h specifies 128x32, multiplex 0x1f, COM pins 0x02.
send(0,[0xae,0xd5,0x80,0xa8,0x1f,0xd3,0,0x40,0x8d,0x14,0x20,2,0xa1,0xc8,0xda,0x02,0x81,0x7f,0xd9,0xf1,0xdb,0x40,0x2e,0xa4,0xa6,0xaf])
font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf',10)
while True:
    try:
        s=json.loads((ROOT/'data/status.json').read_text())
        age=s.get('sensor_age',{})
        ip=socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
        ip.connect(('192.0.2.1',1))
        addr=ip.getsockname()[0]
        ip.close()
        im=Image.new('1',(128,32))
        draw=ImageDraw.Draw(im)
        fresh=time.time()-s['at']<2
        lines=['EXPLORER '+('STALE' if not fresh else 'STOP' if s['stop_latched'] else s['mode']),
               f"{s['battery'] or 0:.2f}V MCU {'OK' if fresh and age.get('odom',99)<1 else '--'}",
               addr]
        for i,line in enumerate(lines): draw.text((0,i*11-2),line[:21],font=font,fill=1)
        # Explicit page addressing avoids relying on retained controller state.
        for p in range(4):
            data=[sum((1<<b) if im.getpixel((x,p*8+b)) else 0 for b in range(8)) for x in range(128)]
            send(0,[0xb0+p,0,0x10])
            for offset in range(0,128,32):send(0x40,data[offset:offset+32])
    except (OSError,ValueError,KeyError):
        pass
    time.sleep(2)
