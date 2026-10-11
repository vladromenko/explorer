"""Exercise the real STOP function without importing ROS or hardware owners."""
import ast
from pathlib import Path
from types import SimpleNamespace
import unittest


def load_stop(components, emit):
    path=Path(__file__).resolve().parents[1]/"src/web.py"
    tree=ast.parse(path.read_text())
    function=next(node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name=="stop_all")
    namespace={"emit":emit,**components}
    exec(compile(ast.Module(body=[function],type_ignores=[]),str(path),"exec"),namespace)
    return namespace["stop_all"]


class StopDispatchTests(unittest.TestCase):
    def test_base_latch_precedes_every_owner_and_each_is_cancelled_once(self):
        calls=[]
        def cancelled(name):
            def operation(*args,**kwargs):
                self.assertEqual(calls[0],"stop")
                calls.append(name)
            return operation
        names=("trajectory_execution","manual_arm","policy_execution","mobile_policy_execution",
            "navigation_tasks","target_gaze","onboard_agent","experiments",
            "autonomy_supervisor","delivery_task")
        components={name:SimpleNamespace(cancel=cancelled(name),stop=cancelled(name),
            cancel_all=cancelled(name),status=lambda:{"busy":True},
            lock=SimpleNamespace(locked=lambda:True)) for name in names}
        queued={"queued":True,"id":"stop-request"}
        stop=load_stop(components,lambda op:calls.append(op) or queued)
        self.assertIs(stop(),queued)
        self.assertEqual(calls,["stop",*names])

    def test_broken_status_and_failed_cancel_do_not_skip_other_owners(self):
        calls=[]
        def fail_agent(reason):
            calls.append("agent_failed")
            raise OSError("memory database unavailable")
        def fail_status():
            calls.append("status_failed")
            raise ValueError("navigation worker unavailable")
        components={"onboard_agent":SimpleNamespace(cancel_all=fail_agent),
            "navigation_tasks":SimpleNamespace(status=fail_status),
            "manual_arm":SimpleNamespace(stop=lambda:calls.append("arm_stop")),
            "trajectory_execution":SimpleNamespace(stop=lambda:calls.append("trajectory_stop"))}
        stop=load_stop(components,lambda op:calls.append(op) or {"queued":True,"id":"stop-request"})
        result=stop()
        self.assertEqual(calls,["stop","trajectory_stop","arm_stop","status_failed","agent_failed"])
        self.assertTrue(result["queued"])
        self.assertEqual([error["owner"] for error in result["cancellation_errors"]],
            ["navigation_tasks","onboard_agent"])
        self.assertTrue(all(error["physical_stop_confirmed"] is False for error in result["cancellation_errors"]))

    def test_direct_publishers_are_invalidated_before_agent_memory_can_block(self):
        calls=[]
        def agent_cancel(reason):
            # This is the synchronous boundary at which task memory can wait.
            # Both mechanisms and learned publishers must already be stopped.
            self.assertEqual(calls,["stop","trajectory_stop","arm_stop","policy_stop",
                "mobile_policy_stop","navigation_cancel"])
            calls.append("agent_memory_boundary")
        components={"onboard_agent":SimpleNamespace(cancel_all=agent_cancel),
            "trajectory_execution":SimpleNamespace(stop=lambda:calls.append("trajectory_stop")),
            "manual_arm":SimpleNamespace(stop=lambda:calls.append("arm_stop")),
            "policy_execution":SimpleNamespace(lock=SimpleNamespace(locked=lambda:True),
                stop=lambda:calls.append("policy_stop")),
            "mobile_policy_execution":SimpleNamespace(lock=SimpleNamespace(locked=lambda:True),
                stop=lambda:calls.append("mobile_policy_stop")),
            "navigation_tasks":SimpleNamespace(status=lambda:{"busy":True},
                cancel=lambda:calls.append("navigation_cancel"))}
        result=load_stop(components,lambda op:calls.append(op) or {"queued":True})()
        self.assertEqual(calls[-1],"agent_memory_boundary")
        self.assertEqual(result,{"queued":True})

    def test_idle_policy_and_navigation_are_not_started_or_cancelled(self):
        def unexpected():
            self.fail("Idle worker should not be called")
        components={"navigation_tasks":SimpleNamespace(status=lambda:{"busy":False},cancel=unexpected),
            "policy_execution":SimpleNamespace(lock=SimpleNamespace(locked=lambda:False),stop=unexpected)}
        calls=[]
        result=load_stop(components,lambda op:calls.append(op) or {"queued":True})()
        self.assertEqual(calls,["stop"])
        self.assertEqual(result,{"queued":True})


if __name__=="__main__":
    unittest.main()
