import tempfile,time
from pathlib import Path
from appearance import read,write,messages,automatic
from oled import render

def test_modes_and_auto_activity():
    assert automatic({'power_state':'OK','velocity':[0,0,0]})=='off'
    assert automatic({'power_state':'OK','velocity':[.1,0,0]})=='work'
    assert automatic({'power_state':'CRITICAL','velocity':[0,0,0]})=='error'
    mode,frames=messages({'rgb_mode':'headlights','headlight_brightness':.7},{},0)
    assert mode=='headlights' and frames==[{'r':.7,'g':.7,'b':.7,'a':255.}]
    assert messages({'rgb_mode':'gradient','headlight_brightness':.7},{},0)[1][0]['a']==104

def test_configuration_is_bounded_and_atomic():
    with tempfile.TemporaryDirectory() as folder:
        root=Path(folder);(root/'config').mkdir()
        assert write(root,'headlights',.6)=={'rgb_mode':'headlights','headlight_brightness':.6}
        assert read(root)['rgb_mode']=='headlights'
        try:write(root,'laser')
        except ValueError:pass
        else:raise AssertionError('unknown mode accepted')

def test_oled_contains_pixels_for_face_and_battery_without_network_text():
    status={'at':time.time(),'battery':11.7,'sensor_age':{'battery':.1},'battery_gauge':{'percent':50}}
    image=render(status,now=1)
    assert image.size==(128,32) and sum(image.getdata())>150
