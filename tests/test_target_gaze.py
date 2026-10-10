import math
import threading
import numpy as np
from target_gaze import gaze_rates

class State:
    def __init__(self, model):self.model=model
    def get_global_link_transform(self, name):
        yaw=math.radians(self.model.q[0]-90)
        pitch=math.radians(self.model.q[3]-90)
        ry=np.array([[math.cos(yaw),0,math.sin(yaw)],[0,1,0],[-math.sin(yaw),0,math.cos(yaw)]])
        rx=np.array([[1,0,0],[0,math.cos(pitch),-math.sin(pitch)],[0,math.sin(pitch),math.cos(pitch)]])
        t=np.eye(4);t[:3,:3]=ry@rx;return t
class Model:
    def __init__(self):self.lock=threading.RLock();self.state=State(self)
    def set_state(self, pose, grip):self.q=pose

def test_gaze_correction_uses_kinematics_and_keeps_grip():
    model=Model();k=np.array([[400.,0,320],[0,400.,240],[0,0,1.]])
    rate=gaze_rates(model,np.eye(4),[90]*6,[400,280],[640,480],k)
    assert 0<rate[0]<=6
    assert -6<=rate[3]<0
    assert rate[1:3]==[0.,0.]
    assert rate[4:]==[0.,0.]
    assert gaze_rates(model,np.eye(4),[90]*6,[320,240],[640,480],k)==[0.]*6
