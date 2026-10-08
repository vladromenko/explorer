"""Private planning IPC only. Existing ArmPlanner never publishes actuators."""
import argparse
import json
import socket
from arm_planner_client import MAX_MESSAGE


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--fd",type=int,required=True);args=parser.parse_args()
    connection=socket.socket(fileno=args.fd)
    with connection,connection.makefile("rwb") as stream:
        planner=None
        while True:
            line=stream.readline(MAX_MESSAGE+1)
            if not line:return
            if len(line)>MAX_MESSAGE or not line.endswith(b"\n"):raise ValueError("Oversized planning message")
            request=json.loads(line);identifier=request["id"]
            try:
                if planner is None:
                    from arm_planner import ArmPlanner
                    planner=ArmPlanner()
                result=planner.plan(request["start_deg"],request["goal_deg"],request["gripper_rad"],request["obstacles"])
                response=dict(id=identifier,result=result)
            except Exception as exc:response=dict(id=identifier,error=str(exc))
            stream.write(json.dumps(response,allow_nan=False).encode()+b"\n");stream.flush()


if __name__=="__main__":main()
