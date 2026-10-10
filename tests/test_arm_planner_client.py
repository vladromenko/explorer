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
 count=0
 while True:
  line=f.readline()
  if not line:break
  r=json.loads(line)
  count+=1
  if r.get("operation")=="warmup":
   time.sleep(.08)
   result={"ready":True,"executed":False,"requests":count}
  else:
   if r["goal_deg"][0]==99:time.sleep(10)
   result={"planned":True,"executed":False,"start":r["start_deg"],"goal":r["goal_deg"],"requests":count}
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

    def test_explicit_warmup_does_not_send_joint_targets_and_reuses_worker(self):
        initialized=self.client.warmup()
        self.assertTrue(initialized["ready"])
        self.assertNotIn("goal",initialized)
        self.assertFalse(initialized["executed"])
        self.assertFalse(initialized["planner_timing"]["worker_reused"])
        result=self.client.plan([90]*5,[85]*5,0.)
        self.assertEqual(result["goal"],[85]*5)
        self.assertTrue(result["planner_timing"]["worker_reused"])
        self.assertEqual(result["requests"],2)

    def test_async_warmup_is_single_and_never_blocks_caller(self):
        before=time.monotonic()
        first=self.client.warmup_async()
        thread=self.client.warmup_thread
        self.client.warmup_async()
        self.assertLess(time.monotonic()-before,.05)
        self.assertEqual(first["phase"],"initializing")
        self.assertIs(self.client.warmup_thread,thread)
        thread.join(2)
        self.assertFalse(thread.is_alive())
        self.assertTrue(self.client.status()["ready"])
        self.assertTrue(self.client.status()["worker_alive"])
        self.client.warmup_async()
        self.assertIs(self.client.warmup_thread,thread)
        self.assertEqual(self.client.plan([90]*5,[80]*5,0.)["requests"],2)

    def test_warmup_timeout_does_not_leave_a_ready_worker(self):
        self.client.timeout=.03
        self.client.warmup_async()
        self.client.warmup_thread.join(2)
        self.assertEqual(self.client.status()["phase"],"failed")
        self.assertFalse(self.client.status()["ready"])
        self.assertFalse(self.client.status()["worker_alive"])

    def test_cancel_before_queued_warmup_prevents_worker_creation(self):
        self.client.lock.acquire()
        try:
            self.client.warmup_async()
            self.client.cancel()
        finally:self.client.lock.release()
        self.client.warmup_thread.join(2)
        self.assertIsNone(self.client.connection)
        self.assertFalse(self.client.status()["ready"])
        self.assertIn("cancelled",self.client.status()["last_warmup_error"])
