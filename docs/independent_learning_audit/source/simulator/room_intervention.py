"""Training-only student states followed by observed expert corrections.

Never imported by the qualification port. A rejected network command is saved,
never actuated or used as an expert label. Physics and cancellation errors exit.
"""
import numpy as np

from .learning import write_json
from .policy_trace import PolicyTrace
from .environment import PERIOD


def prefix_budget(diagnosis,index):
    """Keep control demonstrations and cover progressively later student states."""
    if 'causes' in diagnosis and not any(row.get('last_stage') in
            ('policy_pick_running','policy_pick','verify_hold') for row in diagnosis['findings']):return 0
    if index%2==0:return 0
    refusals=[r.get('command_refusal',{}).get('step') for r in diagnosis['findings']]
    observed=[n for n in refusals if type(n) is int and n>0]
    horizon=min(400,max(observed,default=200))
    return max(20,round(horizon*(.25,.5,.75,1.)[(index//2)%4]))


def student_prefix(port,recorder,target,steps,check):
    """The real frozen student drives bounded commands; all its targets mask out."""
    if type(steps) is not int or not 1<=steps<=400:
        raise ValueError('Room intervention requires 1..400 bounded student steps')
    recorder.targets(False)
    actor=port.actor;actor.begin('pick',target,port.env.command)
    trace=PolicyTrace(port.folder,'training-pick-prefix',actor.sha)
    first=port.env.steps;residual=np.zeros(6);applied=0;refusal=None;held=False
    for index in range(steps):
        check();rgb=port.env.render_rgb('wrist')
        if rgb.shape!=(240,320,3):
            import cv2
            rgb=cv2.resize(rgb,(320,240),interpolation=cv2.INTER_AREA)
        execution=int(port.env.steps-first)
        trace.observation(index,port.env.command,rgb,float(port.env.data.time),execution)
        try:command=actor.command(port.env.command,rgb,residual,execution_step=execution)
        except ValueError:
            decision=actor.command_evidence
            trace.result(index,'refused',decision=decision)
            # An explicit action-contract rejection is a student state to
            # correct. An inference/interface failure requires engineering.
            if not decision.get('refusal') or decision.get('inference_refusal'):raise
            refusal=decision
            break
        port.env.step(command,applied=True);applied+=1
        trace.result(index,'applied',issued_command=command.tolist(),
            applied_command=port.env.command.tolist(),simulation_time=float(port.env.data.time),
            decision=actor.command_evidence)
        if index%10==0:
            proof=port._proof('pick',port._ctx(check))
            if proof.get('outcome')=='success' and port._transport_ready():held=True;break
    value=dict(schema='explorer_room_student_intervention_v1',model_sha256=actor.sha,
        requested_steps=steps,applied_steps=applied,refusal=refusal,observed_transport_hold=held,
        trace=str(trace.folder),student_targets_used=False,qualification=False,
        clock_s=recorder.active['clock']+recorder.active['index']*PERIOD)
    write_json(recorder.active['folder']/'intervention.json',value)
    recorder.active['intervention']=value
    return value


def reacquire_pick(port,target,check):
    """New labels start from newly observed material, never the old hidden pose."""
    port.look_at(target,check=check,held=False)
    seen=port.observe()
    candidates=[row for row in seen['objects'] if row.get('kind')=='sock']
    if len(candidates)!=1:
        raise ValueError('Student intervention could not reacquire one observed sock')
    return candidates[0]
