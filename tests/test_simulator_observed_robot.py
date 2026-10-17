"""Live adapter uses only sensor samples and robot CAD; no real transport."""
import copy
import time
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest

from simulator.observed_robot import ObservedRobotEnv
from simulator.rgbd_observation import observe_environment
from src.observed_skill_execution import FactoryObservedTransport,FullScanCache,ObservedSkillExecution


class Sensors:
    def __init__(self):
        self.value=dict(acquired_monotonic=time.monotonic(),servo_deg=[90,90,90,90,90,30],body_velocity=[0,0,0],
            boot_id='boot',map_epoch='map',calibration_sha256='calibration',joint_state_source='accepted_command_estimate',
            floor_transform_validated=True,base_to_world=np.eye(4).tolist())
        self.actions=[];self.stops=0;self.frame=None
    def state(self):return dict(self.value,acquired_monotonic=time.monotonic())
    def permit(self):pass
    def apply(self,action,**kw):self.actions.append(action);self.value['servo_deg']=action[:6].tolist();return self.state()
    def capture(self,**kw):return copy.deepcopy(self.frame)
    def stop(self):self.stops+=1


def test_live_environment_never_steps_physics_or_moves_on_construction(monkeypatch):
    monkeypatch.setattr(mujoco,'mj_step',lambda *a:pytest.fail('Live adapter cannot simulate hardware feedback'))
    s=Sensors();env=ObservedRobotEnv(s)
    assert not s.actions and len(env.model.names)>0
    before=env.jaw_center().copy();action=env.command.copy();action[0]+=.5
    env.step(action)
    assert len(s.actions)==1 and not np.allclose(before,env.jaw_center())
    assert not hasattr(env,'task_success') and not hasattr(env,'sock_id')
    env.close();assert s.stops==1


@pytest.mark.parametrize('bad',[{'boot_id':'changed'},{'floor_transform_validated':False},{'joint_state_source':'fake'},{'acquired_monotonic':0},{'map_epoch':''}])
def test_untrusted_or_stale_live_state_is_refused(bad):
    env=ObservedRobotEnv(Sensors());value=env.transport.state();value.update(bad)
    with pytest.raises(ValueError):env._sync(value)


def frame(env):
    camera=env.model.camera('wrist_depth').id;t=np.eye(4)
    t[:3,3]=env.data.cam_xpos[camera];t[:3,:3]=env.data.cam_xmat[camera].reshape(3,3)@np.diag([1,-1,-1])
    return dict(frame_id='one',registered=True,arm_stationary=True,depth_unit='metre',
        acquired_monotonic=time.monotonic(),robot_state=env.transport.state(),
        rgb=np.zeros((12,16,3),np.uint8),depth=np.full((12,16),np.nan),
        intrinsics=dict(width=16,height=12,fx=16,fy=16,cx=8,cy=6),optical_to_world=t.tolist())


def test_actual_rgbd_and_intrinsics_are_used_without_rendering_or_hidden_targets(monkeypatch):
    s=Sensors();env=ObservedRobotEnv(s);s.frame=frame(env)
    raw={};view=observe_environment(env,raw_output=raw)
    assert not view['simulation_only'] and view['source']=='live_registered_RGBD'
    assert np.array_equal(raw['rgb'],s.frame['rgb']) and raw['robot_depth'].shape==(12,16)
    assert not s.actions
    with pytest.raises(ValueError,match='advance'):env.capture_registered_frame()


@pytest.mark.parametrize('change',[{'registered':False},{'depth_unit':'millimetre'},{'arm_stationary':False},{'acquired_monotonic':0}])
def test_sensor_contract_cannot_silently_use_wrong_units_or_timing(change):
    s=Sensors();env=ObservedRobotEnv(s);s.frame=frame(env);s.frame.update(change)
    with pytest.raises(ValueError):env.capture_registered_frame()


def test_full_lidar_is_not_downsampled_and_invalid_returns_are_not_free_space():
    cache=FullScanCache(clock=lambda:10.)
    message=SimpleNamespace(header=SimpleNamespace(stamp=SimpleNamespace(sec=100,nanosec=0),frame_id='laser0'),
        range_min=.1,range_max=10.,ranges=[.2,float('inf'),0.,2.]*180,angle_min=-3.14,angle_increment=.008)
    cache.receive(0,message,100.1);cache.receive(1,message,100.1)
    a,b=cache.snapshot();assert len(a['ranges'])==720 and a['ranges'][1:3]==[None,None]
    assert a['acquired_monotonic']==pytest.approx(9.9)


def test_body_actions_use_existing_motion_owner_and_local_odometry_frame():
    calls=[];owner=SimpleNamespace(robot=SimpleNamespace(settings={'floor_plane_base':[0,0,1,.12]},
        _move=lambda x:calls.append(x),missions=SimpleNamespace(emit=lambda *a,**kw:None)),
        _lease=lambda:None,session='x',target=None,target_at=0.,emit=lambda *a,**kw:None)
    owner.robot.permit=lambda _:None
    transport=FactoryObservedTransport(owner)
    a=transport._base({'raw_pose':dict(x=4.,y=5.,yaw=1.)})
    assert a[:2,3]==pytest.approx([0,0]) and a[2,3]==pytest.approx(.12)
    transport.state=lambda:dict(ok=True)
    transport.apply(np.array([91,90,90,90,90,30,0,0,0]),reference=np.array([90]*5+[30,0,0,0]),period=0)
    assert calls==[[91,90,90,90,90,30]]
    with pytest.raises(ValueError,match='разделены'):
        transport.apply(np.array([92,90,90,90,90,30,.04,0,0]),reference=np.array([90]*5+[30,0,0,0]),period=0)


def test_expired_observation_lease_cannot_authorize_commands(tmp_path):
    owner=ObservedSkillExecution(tmp_path,SimpleNamespace(),lambda *a,**k:None,lambda:[],lambda x:None)
    with pytest.raises(ValueError,match='Наблюдение'):owner._lease()
    with pytest.raises(ValueError,match='оператор'):owner.start('a'*64,False)


def test_all_existing_owners_receive_stop_even_if_one_fails(tmp_path):
    stopped=[]
    def fail():stopped.append('trajectory');raise ValueError('failed')
    robot=SimpleNamespace(trajectory=SimpleNamespace(stop=fail),arm=SimpleNamespace(stop=lambda:stopped.append('arm')))
    owner=ObservedSkillExecution(tmp_path,robot,lambda *a,**kw:stopped.append('base'),lambda:[],lambda x:None)
    owner.lock.acquire()
    try:
        result=owner.stop()
        assert stopped==['base','trajectory','arm'] and result['stop_errors']==['failed']
    finally:owner.lock.release()
