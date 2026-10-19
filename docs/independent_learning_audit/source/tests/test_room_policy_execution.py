"""Actor clock matches issued physics steps; pauses are not hidden actions."""
from types import SimpleNamespace
import json
import numpy as np
import pytest

from simulator.room_cleanup_port import RoomCleanupPort
from simulator.room_policy import RoomActor


def test_observation_checks_do_not_insert_twelve_manual_ticks_into_act(tmp_path):
    port=RoomCleanupPort.__new__(RoomCleanupPort)
    command=np.array([90.,90.,90.,90.,90.,30.,0.,0.,0.])
    env=SimpleNamespace(command=command,steps=0,data=SimpleNamespace(time=0.))
    env.render_rgb=lambda camera:np.zeros((240,320,3),np.uint8)
    applied=[]
    def step(action,*,applied=False):
        assert applied
        env.command=np.asarray(action).copy();env.steps+=1;env.data.time+=.05
    env.step=step;port.env=env;port.folder=tmp_path
    port.geometry=SimpleNamespace(target=None)
    decisions=[]
    class Actor:
        sha='a'*64
        def begin(self,*args):pass
        def command(self,state,rgb,residual,*,execution_step):
            decisions.append(execution_step)
            self.command_evidence=dict(replanned=False,elapsed_s=6+execution_step*.05)
            return state.copy()
    port.actor=Actor();port.pose=lambda:[0.,0.,0.];port._ctx=lambda check=None:None
    port.progress=lambda value:None;port._transport_ready=lambda:True;port.held=False
    holds=[]
    def stop():
        holds.append(env.steps)
        env.steps+=12;env.data.time+=.6
    port._stop_base=stop
    observations=[]
    def proof(phase,ctx):
        observations.append(env.data.time)
        return dict(outcome='success' if len(decisions)>=21 else 'unknown')
    port._proof=proof
    result=port.learned_manipulation('pick',{},check=lambda:None)
    assert result['completed'] and decisions==list(range(21))
    assert observations==pytest.approx([.05,.55,1.05])
    assert holds==[21]  # only the phase-end settle, never interleaved with ACT
    records=[json.loads(line) for line in next(tmp_path.glob('policy-*/actions.jsonl')).read_text().splitlines()]
    assert sum(row.get('status')=='applied' for row in records)==21
    assert len(list(tmp_path.glob('policy-*/*.png')))==21


def test_passive_control_steps_refresh_queue_and_clock_without_doubling_inference():
    calls=[]
    class Actor:
        elapsed_s=0.
        last_action_was_queued=False
        last_action_elapsed_s=0.
        def reset(self):calls.append('reset');self.elapsed_s=0.
        def __call__(self,state,rgb):
            calls.append(self.elapsed_s);self.last_action_elapsed_s=self.elapsed_s;self.elapsed_s+=.05
            return state.copy()
    actor=RoomActor.__new__(RoomActor);actor.actor=Actor();actor.profile=dict(pick_elapsed_s=6.)
    state=np.array([90.,90.,90.,90.,90.,30.,0.,0.,0.]);image=np.zeros((240,320,3),np.uint8)
    actor.begin('pick',{},state)
    actor.command(state,image,None,execution_step=0)
    actor.command(state,image,None,execution_step=1)
    actor.command(state,image,None,execution_step=12)
    assert calls==['reset',6.,6.05,'reset',6.6]
    assert actor.command_evidence['resynchronized_after_passive_steps']
    with pytest.raises(ValueError,match='backwards'):actor.command(state,image,None,execution_step=11)


def test_base_settle_holds_one_fixed_arm_goal_instead_of_chasing_transient_state():
    port=RoomCleanupPort.__new__(RoomCleanupPort)
    port.env=SimpleNamespace(command=np.array([90.,90.,90.,90.,90.,30.,0.,0.,0.]))
    targets=[]
    def tick(arm,*,ctx):
        targets.append(np.asarray(arm).copy())
        port.env.command=port.env.command+np.r_[1.,np.zeros(8)]
    port.geometry=SimpleNamespace(tick=tick);port._ctx=lambda:None
    port._stop_base()
    assert len(targets)==12 and all(x[0]==90. for x in targets)


def test_trace_png_is_exact_and_video_uses_recorded_intervals(tmp_path,monkeypatch):
    from simulator.policy_trace import PolicyTrace,encode_trace
    from PIL import Image
    trace=PolicyTrace(tmp_path,'pick','a'*64);image=np.zeros((240,320,3),np.uint8);image[7,8]=[12,34,56]
    trace.observation(0,np.zeros(9),image,4.,0)
    trace.observation(1,np.zeros(9),image,4.55,11)
    np.testing.assert_array_equal(np.asarray(Image.open(trace.folder/'000000.png')),image)
    monkeypatch.setattr('shutil.which',lambda _: '/test/ffmpeg')
    def encode(args,**kwargs):Path(args[-1]).write_bytes(b'fixture video; no policy acceptance')
    from pathlib import Path
    monkeypatch.setattr('subprocess.run',encode)
    value=encode_trace(trace.folder)
    assert value['frames']==2 and not value['continuous_camera_video']
    assert 'duration 0.549999' in (trace.folder/'video-input.ffconcat').read_text()
