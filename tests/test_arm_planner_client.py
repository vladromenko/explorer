import json
from pathlib import Path
import tempfile
import threading
import time
import unittest
from arm_planner_client import ArmPlannerClient


WORKER="""import argparse,json,socket,time
p=argparse.ArgumentParser();p.add_argument("--fd",type=int);a=p.parse_args()
with socket.socket(fileno=a.fd) as s,s.makefile("rwb") as f:
 while True:
  line=f.readline()
  if not line:break
  r=json.loads(line)
  if r["goal_deg"][0]==99:time.sleep(10)
  result={"planned":True,"executed":False,"start":r["start_deg"],"goal":r["goal_deg"]}
  f.write(json.dumps({"id":r["id"],"result":result}).encode()+b"\\n");f.flush()
"""


class PlannerClientTests(unittest.TestCase):
    def setUp(self):
        self.folder=tempfile.TemporaryDirectory();self.addCleanup(self.folder.cleanup)
        root=Path(self.folder.name);worker=root/"worker.py";worker.write_text(WORKER)
        self.client=ArmPlannerClient(root,worker,timeout=2.)
        self.addCleanup(self.client.cancel)

    def test_worker_preserves_request_and_is_reused_without_actuation(self):
        result=self.client.plan([90]*5,[80]*5,0.)
        self.assertEqual(result["goal"],[80]*5);self.assertFalse(result["executed"])
        pid=self.client.connection["process"].pid
        self.client.plan([80]*5,[85]*5,0.)
        self.assertEqual(self.client.connection["process"].pid,pid)

    def test_cancel_unblocks_reader_and_next_plan_uses_new_worker(self):
        errors=[]
        def plan():
            try:self.client.plan([90]*5,[99]*5,0.)
            except ValueError as exc:errors.append(str(exc))
        thread=threading.Thread(target=plan);thread.start();deadline=time.monotonic()+2
        while self.client.connection is None and time.monotonic()<deadline:time.sleep(.01)
        before=time.monotonic();self.client.cancel();thread.join(2)
        self.assertLess(time.monotonic()-before,2);self.assertFalse(thread.is_alive())
        self.assertEqual(len(errors),1)
        self.assertTrue(self.client.plan([90]*5,[80]*5,0.)["planned"])

    def test_timeout_does_not_return_a_stale_plan(self):
        self.client.timeout=.1
        with self.assertRaisesRegex(ValueError,"MoveIt worker"):
            self.client.plan([90]*5,[99]*5,0.)
        self.assertIsNone(self.client.connection)

    def test_dead_idle_worker_is_replaced_without_socket_leak(self):
        self.client.plan([90]*5,[80]*5,0.)
        previous=self.client.connection
        previous["process"].kill();previous["process"].wait(2)
        self.assertTrue(self.client.plan([90]*5,[81]*5,0.)["planned"])
        self.assertIsNot(self.client.connection,previous)
        self.assertTrue(previous["stream"].closed)
