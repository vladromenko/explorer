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


def test_gaze_accepts_selected_frame_without_counting_repeated_frame_as_five_observations(tmp_path):
    import json
    import time
    from types import SimpleNamespace
    from target_gaze import TargetGaze
    (tmp_path/"config").mkdir()
    (tmp_path/"data").mkdir()
    (tmp_path/"config/handeye-accepted.json").write_text(json.dumps({"execution_authorized":True,
        "reference_mount":"arm4","camera_to_mount_reference":np.eye(4).tolist()}))
    (tmp_path/"data/status.json").write_text(json.dumps({"mode":"MANUAL","stop_latched":True}))
    (tmp_path/"data/arm-state.json").write_text(json.dumps({"servo_deg":[90]*6}))
    observation = {"phase":"selected","target_id":"selected-target","image_stamp":time.time(),
        "center_px":[320,240],"image_size":[640,480],"camera_info_version":"v1",
        "intrinsic_k":[400.,0,320,0,400.,240,0,0,1.],"distortion_d":[0]*5}
    commands = []
    arm = SimpleNamespace(reference=lambda *args:{"servo_deg":[90]*6},streaming_active=False,
        stream_velocity=lambda *args,**kwargs:commands.append(args[0]),stop_stream=lambda *args:None)
    gaze = TargetGaze(tmp_path,SimpleNamespace(tick=lambda:observation),arm,lambda:Model())
    gaze.cancelled = SimpleNamespace(clear=lambda:None,is_set=lambda:False,wait=lambda *args:None)
    result = gaze.run({"target_id":"selected-target","duration_s":.02},
        SimpleNamespace(id="task",permit=lambda:None))
    assert result["evidence"]["phase"] == "duration_complete"
    assert len(commands) >= 2
    assert result["evidence"]["measured_joint_positions"] is False
