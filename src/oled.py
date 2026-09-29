"""Milo mini face and battery reserve on the M3 Pro 128x32 SSD1306."""
import json
import time
from pathlib import Path
from PIL import Image,ImageDraw,ImageFont
from smbus2 import SMBus,i2c_msg
from battery_gauge import battery_summary

ROOT=Path('/home/vlad/Explorer')
SMALL='/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf'
BOLD='/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf'

def render(status,now=None):
    now=time.monotonic() if now is None else now
    image=Image.new('1',(128,32));draw=ImageDraw.Draw(image)
    # Monochrome miniature of Milo's orange pixel-art cat face.
    draw.polygon([(3,8),(7,0),(13,6),(25,6),(31,0),(35,8),(35,18),(31,22),(7,22),(3,18)],outline=1,fill=0)
    draw.line([(7,5),(10,3),(12,6)],fill=1);draw.line([(26,6),(29,3),(32,5)],fill=1)
    blink=int(now*2)%17==0
    if blink:
        draw.line((10,12,15,12),fill=1);draw.line((23,12,28,12),fill=1)
    else:
        draw.rectangle((11,10,14,14),fill=1);draw.rectangle((24,10,27,14),fill=1)
        draw.point((13,11),fill=0);draw.point((26,11),fill=0)
    draw.polygon([(18,15),(21,15),(19,17)],fill=1)
    draw.line((19,17,16,19),fill=1);draw.line((20,17,23,19),fill=1)
    draw.line((9,16,1,15),fill=1);draw.line((9,18,1,20),fill=1)
    draw.line((29,16,38,15),fill=1);draw.line((29,18,38,20),fill=1)
    small=ImageFont.truetype(SMALL,7);medium=ImageFont.truetype(BOLD,9)
    fresh=0<=time.time()-status.get('at',0)<3
    voltage=status.get('battery') if fresh else None
    gauge=status.get('battery_gauge',{}) if fresh else {}
    percent=gauge.get('percent')
    if percent is None:percent=battery_summary(voltage,status.get('sensor_age',{}).get('battery'))['percent']
    charge='--%' if percent is None else '~%d%%'%percent
    volts='--.--V' if type(voltage) not in (int,float) else '%.2fV'%voltage
    # Charge sits directly below Milo; the right side keeps the name and voltage.
    draw.text((7,23),charge,font=small,fill=1)
    draw.text((47,-1),'MILO',font=medium,fill=1)
    draw.text((47,9),volts,font=medium,fill=1)
    draw.rectangle((47,21,125,29),outline=1);draw.rectangle((126,23,127,27),fill=1)
    if percent is not None:
        fill=max(0,min(76,round(76*percent/100)))
        if fill:draw.rectangle((49,23,48+fill,27),fill=1)
    return image

def send(bus,control,values):bus.i2c_rdwr(i2c_msg.write(0x3c,bytes([control]+list(values))))

def display(bus,image):
    for page in range(4):
        data=[sum((1<<bit) if image.getpixel((x,page*8+bit)) else 0 for bit in range(8)) for x in range(128)]
        send(bus,0,[0xb0+page,0,0x10])
        for offset in range(0,128,32):send(bus,0x40,data[offset:offset+32])

def main():
    bus=SMBus(7)
    # Vendor bsp_ssd1306.h: 128x32, multiplex 0x1f, COM pins 0x02.
    send(bus,0,[0xae,0xd5,0x80,0xa8,0x1f,0xd3,0,0x40,0x8d,0x14,0x20,2,0xa1,0xc8,0xda,0x02,0x81,0xcf,0xd9,0xf1,0xdb,0x40,0x2e,0xa4,0xa6,0xaf])
    while True:
        try:display(bus,render(json.loads((ROOT/'data/status.json').read_text())))
        except (OSError,ValueError,KeyError):pass
        time.sleep(1)

if __name__=='__main__':main()
