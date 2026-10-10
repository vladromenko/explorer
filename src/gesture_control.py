"""One explicitly armed gesture action through the existing skill executors."""
import time
from human_perception import GestureGate


class GestureControl:
    def __init__(self, perception, ports):
        self.perception=perception
        self.ports=ports

    def execute_proposal(self, proposal, context):
        context.permit()
        if (proposal.get("session_id")!=context.id or not time.time()<proposal.get("expires_at",0)
                or not 0<=time.time()-proposal.get("image_stamp",0)<.7 or not context.observing):
            raise ValueError("Gesture authorization or observation expired")
        action=proposal.get("action")
        context.event("gesture_dispatch",proposal)
        if action in ("look_left","look_right"):
            return self.ports.look({"view":action.removeprefix("look_")},context)
        if action in ("open_gripper","close_gripper"):
            return self.ports.gripper({},context,action=="open_gripper")
        if action in ("wave","nod","look_up","look_down"):
            q=list(self.ports.arm.reference()["servo_deg"])
            axis=0 if action=="wave" else 3
            deltas=(5,-5,0) if action in ("wave","nod") else (-5,) if action=="look_up" else (5,)
            results=[]
            for delta in deltas:
                goal=list(q);goal[axis]+=delta
                result=self.ports.move({"goal_deg":goal},context)
                results.append(result)
                if result["state"]=="failed":return result
                if result["state"]=="unknown" and result["evidence"].get("command_completed") is not True:
                    return result
                context.permit()
            return self.ports.outcome({"action":action,"results":results,"measured":False,"physical_verification":False},"unknown")
        raise ValueError("Gesture action is not an executable skill")

    def run(self, args, context):
        if not self.perception.lock.acquire(False):raise ValueError("Human perception is already in use")
        try:return self._run(args,context)
        finally:self.perception.lock.release()

    def _run(self, args, context):
        context.permit()
        gate=GestureGate()
        duration=args.get("duration_s",20)
        action=args["action"]
        gate.begin(context.id,{args["gesture"]:action},args["operator_roi"],duration,
            physical_authorized=context.observing)
        end=time.monotonic()+duration
        try:
            self.perception.prepare(("hand",))
            while time.monotonic()<end:
                context.permit()
                observation=self.perception.observe(("hand",))
                if observation.get("errors") or observation.get("stale"):
                    raise ValueError("Fresh gesture landmarks unavailable")
                result=gate.update(observation)
                context.event("gesture_observation",{key:result[key] for key in ("state","reason","drawing_2d","frame_id") if key in result})
                if result.get("proposal"):
                    return self.execute_proposal(result["proposal"],context)
                time.sleep(.03)
            context.permit()
            if action=="trace_2d":return self.ports.outcome({"drawing_2d":list(gate.stroke),"executed":False,"frame":"normalized_image"})
            return self.ports.outcome({"reason":"No stable authorized gesture within session","executed":False},"waiting")
        finally:
            gate.cancel()
            self.perception.unload()
