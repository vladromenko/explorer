"""Small-display and WS2812 appearance policy; never controls motion."""
import json
import math
import os
from pathlib import Path

MODES=('auto','off','headlights','work','search','success','error','water','marquee','breathe','gradient','sparkle','battery')
EFFECTS={'off':100,'water':101,'marquee':102,'breathe':103,'gradient':104,'sparkle':105,'battery':106}

def read(root):
    try:value=json.loads((Path(root)/'config/appearance.json').read_text())
    except (OSError,ValueError):value={}
    mode=value.get('rgb_mode')
    if mode not in MODES:
        legacy=value.get('rgb_effect',100)
        mode=next((name for name,effect in EFFECTS.items() if effect==legacy),'off')
    brightness=value.get('headlight_brightness',.8)
    if type(brightness) not in (int,float) or not math.isfinite(brightness):brightness=.8
    return dict(rgb_mode=mode,headlight_brightness=max(.1,min(1.,float(brightness))))

def write(root,mode,brightness=None):
    if mode not in MODES:raise ValueError('Unknown light mode')
    current=read(root)
    if brightness is not None:
        if type(brightness) not in (int,float) or not math.isfinite(brightness) or not .1<=brightness<=1:
            raise ValueError('Brightness must be 0.1..1.0')
        current['headlight_brightness']=float(brightness)
    current['rgb_mode']=mode
    path=Path(root)/'config/appearance.json';temporary=path.with_suffix('.tmp')
    temporary.write_text(json.dumps(current,ensure_ascii=False,indent=2)+'\n')
    os.replace(temporary,path)
    return current

def automatic(status):
    power=status.get('power_state','UNKNOWN')
    if power in ('CRITICAL','UNKNOWN'):return 'error'
    velocity=status.get('velocity') or (0,0,0)
    if status.get('mission') or status.get('arm_active') or any(abs(float(v))>.001 for v in velocity):return 'work'
    return 'off'

def messages(config,status,now):
    """Return ColorRGBA fields. a=255 addresses the complete Yahboom strip."""
    requested=config['rgb_mode'];mode=automatic(status) if requested=='auto' else requested
    if mode in EFFECTS:return mode,[dict(r=0.,g=0.,b=0.,a=float(EFFECTS[mode]))]
    brightness=config['headlight_brightness']
    colours={
        'headlights':(brightness,brightness,brightness),
        'work':(.05,.35,.8),
        'success':(.05,.8,.15),
        'search':(.9,.32,.02),
        'error':(.9,0.,0.) if int(now*2)%2==0 else (0.,0.,0.),
    }
    colour=colours.get(mode,(0.,0.,0.))
    return mode,[dict(r=colour[0],g=colour[1],b=colour[2],a=255.)]
