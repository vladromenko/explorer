import time
from types import SimpleNamespace
import pytest
from gesture_control import GestureControl


def context(observing=True):
    return SimpleNamespace(id="operator-gesture-session",observing=observing,permit=lambda:None,event=lambda *args:None)


def test_stale_or_unarmed_gesture_never_dispatches():
    control=GestureControl(None,None)
    for row,ctx in (({"session_id":"other"},context()),({"session_id":context().id,"expires_at":time.time()+1,"image_stamp":time.time()},context(False))):
        with pytest.raises(ValueError,match="expired"):control.execute_proposal(row,ctx)


def test_fixed_action_dispatches_only_existing_owner():
    calls=[]
    ports=SimpleNamespace(look=lambda args,ctx:calls.append(args) or {"state":"succeeded"})
    proposal={"session_id":context().id,"expires_at":time.time()+1,"image_stamp":time.time(),"action":"look_left"}
    assert GestureControl(None,ports).execute_proposal(proposal,context())["state"]=="succeeded"
    assert calls==[{"view":"left"}]


def test_wave_preserves_other_joints_and_does_not_invent_feedback():
    goals=[]
    ports=SimpleNamespace(arm=SimpleNamespace(reference=lambda:{"servo_deg":[90,90,60,15,90,90]}),
        move=lambda args,ctx:goals.append(args["goal_deg"]) or {"state":"unknown","evidence":{"command_completed":True}},
        outcome=lambda evidence,state:{"state":state,"evidence":evidence})
    proposal={"session_id":context().id,"expires_at":time.time()+1,"image_stamp":time.time(),"action":"wave"}
    result=GestureControl(None,ports).execute_proposal(proposal,context())
    assert [goal[0] for goal in goals]==[95,85,90]
    assert all(goal[1:]==[90,60,15,90,90] for goal in goals)
    assert result["state"]=="unknown"
