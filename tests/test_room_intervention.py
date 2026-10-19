"""Student states inform labels, but a student's actions never become targets."""
import json
from types import SimpleNamespace

import numpy as np
import pytest

from simulator.learning import inspect_episode,expert_segments
from simulator.room_learning import PhaseRecorder
from simulator.room_intervention import student_prefix,prefix_budget,reacquire_pick


def setup_prefix(tmp_path,refuse_at=None,inference_failure=False):
    env=SimpleNamespace(command=np.array([90.]*5+[30.,0.,0.,0.]),steps=0,
        data=SimpleNamespace(time=0.),domain_variation={})
    env.render_rgb=lambda _:np.full((240,320,3),round(env.command[0]),np.uint8)
    def step(command,applied=False):
        assert applied
        env.command=np.asarray(command).copy();env.steps+=1;env.data.time+=.05
    env.step=step
    class Actor:
        sha='a'*64
        def begin(self,*args):pass
        def command(self,state,rgb,residual,*,execution_step):
            if execution_step==refuse_at:
                key='inference_refusal' if inference_failure else 'refusal'
                self.command_evidence={key:'test refusal', 'queued':False}
                raise ValueError('test refusal')
            self.command_evidence={'queued':False}
            return state+np.r_[1.,np.zeros(8)]
    port=SimpleNamespace(env=env,actor=Actor(),folder=tmp_path/'observed',
        _ctx=lambda check:None,_proof=lambda *a:dict(outcome='unknown'))
    recorder=PhaseRecorder(env,tmp_path/'recording',dict(pick_elapsed_s=6.),42)
    recorder.begin('pick',{})
    return port,recorder


def test_actual_student_state_is_retained_and_excluded_from_expert_segments(tmp_path):
    port,recorder=setup_prefix(tmp_path)
    receipt=student_prefix(port,recorder,{},20,lambda:None)
    assert receipt['applied_steps']==20 and port.env.command[0]==110.
    recorder.targets(True)
    for _ in range(4):port.env.step(port.env.command-np.r_[1.,np.zeros(8)],applied=True)
    paths=recorder.finish(True,{'success':True})
    episode,rows,fps,_=inspect_episode(paths[0]);segments=expert_segments(rows,fps)
    assert len(rows)==24 and sum(r['training_target_valid'] for r in rows)==4
    assert len(segments)==1 and len(segments[0])==4
    assert segments[0][0][0]==pytest.approx(7.)
    assert segments[0][0][1]['observation_state'][0]==110.
    assert segments[0][0][1]['applied_action'][0]==109.
    assert episode['learner_intervention']['model_sha256']=='a'*64
    assert episode['learner_intervention']['student_targets_used'] is False
    assert episode['learner_intervention']['expert_target_frames']==4
    assert episode['learner_intervention']['expert_motion_frames']==4


def test_rejected_fresh_action_is_saved_without_actuation_or_label(tmp_path):
    port,recorder=setup_prefix(tmp_path,refuse_at=3)
    receipt=student_prefix(port,recorder,{},20,lambda:None)
    assert receipt['applied_steps']==3 and port.env.steps==3
    assert receipt['refusal']['queued'] is False
    recorder.finish(False,{})
    from pathlib import Path
    trace=[json.loads(s) for s in (Path(receipt['trace'])/'actions.jsonl').read_text().splitlines()]
    assert trace[-1]['status']=='refused' and trace[-1]['step']==3
    rows=[json.loads(s) for s in next((tmp_path/'recording').glob('*/samples.jsonl')).read_text().splitlines()]
    assert len(rows)==3 and not any(r['training_target_valid'] for r in rows)


def test_inference_failure_and_physics_error_are_not_hidden_by_teacher(tmp_path):
    port,recorder=setup_prefix(tmp_path,refuse_at=0,inference_failure=True)
    with pytest.raises(ValueError,match='test refusal'):student_prefix(port,recorder,{},20,lambda:None)
    assert port.env.steps==0
    recorder.finish(False,{})
    port,recorder=setup_prefix(tmp_path/'physics')
    def stop(*a,**k):raise RuntimeError('physics guard')
    port.env.step=stop
    with pytest.raises(RuntimeError,match='physics guard'):student_prefix(port,recorder,{},20,lambda:None)
    recorder.finish(False,{})


def test_program_varies_correction_states_and_keeps_unperturbed_controls():
    diagnosis=dict(findings=[dict(command_refusal=dict(step=760))])
    assert [prefix_budget(diagnosis,i) for i in range(8)]==[0,100,0,200,0,300,0,400]
    assert prefix_budget(dict(findings=[]),1)==50
    assert prefix_budget(dict(causes=['placement'],findings=[dict(last_stage='policy_place_running')]),1)==0


def test_reacquisition_uses_new_semantic_observation_or_rejects(tmp_path):
    calls=[];fresh=dict(kind='sock',center=[1.,2.,.01])
    port=SimpleNamespace(look_at=lambda target,**k:calls.append(target),observe=lambda:dict(objects=[fresh]))
    assert reacquire_pick(port,{'old':True},lambda:None) is fresh
    assert calls==[{'old':True}]
    port.observe=lambda:dict(objects=[])
    with pytest.raises(ValueError,match='reacquire'):reacquire_pick(port,{},lambda:None)
