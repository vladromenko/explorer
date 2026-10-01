#!/usr/bin/env python3
"""Record synchronized RGB-D bracketed by stationary, measured joint feedback."""
import json,time,uuid
from pathlib import Path
import numpy as np
import rclpy
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image,CameraInfo
from cv_bridge import CvBridge
from arm_commissioning import stationary_status
from arm_model import ArmModel
from controller_feedback import load_calibration
from native_arm import measured_reference
from capture_geometry import stationary_exposure
ROOT=Path('/home/vlad/Explorer')

def main():
    status=json.loads((ROOT/'data/status.json').read_text());stationary_status(status,time.time())
    calibration=load_calibration(ROOT/'config/controller-calibration.json')
    profile=json.loads((ROOT/'config/controller-profile.json').read_text())
    def measured():
        live=json.loads((ROOT/'data/controller-state.json').read_text())
        if profile.get('manual_reference_version')==1:
            from manual_reference_host import reference_from_state
            value=reference_from_state(live,calibration,time.monotonic_ns())
            value['acquired_monotonic']=[live['manual_reference']['received_monotonic_ns']/1e9]*6
            return value
        return measured_reference(live,calibration,time.monotonic_ns())
    before=measured()
    rclpy.init();node=rclpy.create_node('explorer_capture_handeye');data={};bridge=CvBridge()
    def receive(key,msg):data[key]=msg
    for key,typ,topic in [('rgb',Image,'/camera/color/image_raw'),('depth',Image,'/camera/depth/image_raw'),('info',CameraInfo,'/camera/color/camera_info')]:
        node.create_subscription(typ,topic,lambda msg,k=key:receive(k,msg),qos_profile_sensor_data)
    def stamp(msg):return msg.header.stamp.sec+msg.header.stamp.nanosec/1e9
    deadline=time.monotonic()+5;ready=False
    while not ready and time.monotonic()<deadline:
        rclpy.spin_once(node,timeout_sec=.05)
        ready=len(data)==3 and abs(stamp(data['rgb'])-stamp(data['depth']))<.035
    if not ready or not 0<=time.time()-stamp(data['rgb'])<.5:raise ValueError('No fresh synchronized RGB-D')
    if data['rgb'].header.frame_id!=data['depth'].header.frame_id:raise ValueError('Unregistered depth')
    rgb=bridge.imgmsg_to_cv2(data['rgb'],'bgr8');depth=bridge.imgmsg_to_cv2(data['depth']).astype(np.float32)
    if data['depth'].encoding in ('16UC1','mono16'):depth*=.001
    if depth.shape!=rgb.shape[:2]:raise ValueError('Depth dimensions differ')
    exposure=time.monotonic()-(time.time()-stamp(data['rgb']))
    later=measured()
    deadline=time.monotonic()+.3
    while min(later['acquired_monotonic'])<exposure and time.monotonic()<deadline:
        time.sleep(.01);later=measured()
    state=stationary_exposure(before,later,exposure)
    stationary_status(json.loads((ROOT/'data/status.json').read_text()),time.time())
    model=ArmModel();model.set_state(state['servo_deg'][:5],-.3)
    transform=model.state.get_global_link_transform('Gripping')
    folder=ROOT/'data/handeye';folder.mkdir(exist_ok=True)
    name=str(int(time.time()))+'-'+uuid.uuid4().hex[:6]
    path=folder/(name+'.npz')
    np.savez_compressed(path,rgb=rgb,depth=depth,k=np.array(data['info'].k).reshape(3,3),
                        d=np.array(data['info'].d),stamp=stamp(data['rgb']),
                        servo_deg=state['servo_deg'],base_tool=transform,
                        measured_joint_positions=state['measured_joint_positions'],joint_state_source=state['joint_state_source'],controller_boot_id=str(state['controller_boot_id']),
                        controller_session=state['controller_session'],
                        exposure_monotonic=state['exposure_monotonic'],
                        base_mount=model.state.get_global_link_transform('arm4'),mount_frame='arm4',
                        base_pose=np.array(list(status['raw_pose'][k] for k in ('x','y','yaw'))))
    print(json.dumps(dict(path=str(path),servo_deg=state['servo_deg'],measured_joints=state['measured_joint_positions'])))
    node.destroy_node();rclpy.shutdown()

if __name__=='__main__':main()
