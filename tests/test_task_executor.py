import unittest

from task_executor import Observation, STAGES, TaskExecutor


class Port:
    def __init__(self):
        self.now = 100.
        self.sequence = 0
        self.calls = []
        self.observations = []
        self.cancelled = False
        self.stops = []
        self.operations = {name: self.operation(name) for name in STAGES}

    def operation(self, name):
        def run(context):
            context.check()
            self.calls.append((name, context.base_revision, context.target))
            if name in {"find", "reobserve"}:
                return {"outcome": "success", "target": {"x": .2, "sequence": context.observation.sequence}}
            if name.startswith("verify_"):
                return {"outcome": "success", "evidence": {"confirmed": True, "kind": "measured"}}
            return {"outcome": "success"}
        return run

    def observe(self, context):
        self.sequence += 1
        value = Observation(self.sequence, self.now, context.base_revision, {"target_visible": True})
        self.observations.append(value)
        return value

    def executor(self, **options):
        return TaskExecutor(self.operations, self.observe, clock=lambda: self.now,
                            cancelled=lambda: self.cancelled, stop=lambda c: self.stops.append(c.stage), **options)


class TaskExecutorTests(unittest.TestCase):
    def test_complete_task_requires_both_measured_verifications(self):
        port = Port()
        result = port.executor().execute({"object": "sock"})
        self.assertEqual(result.state, "succeeded")
        self.assertTrue(result.held_verified)
        self.assertTrue(result.placed_verified)
        self.assertEqual([row[0] for row in port.calls], list(STAGES))
        self.assertEqual(port.stops, [])

    def test_unknown_hold_stops_before_carry_without_regrasp(self):
        port = Port()
        port.operations["verify_hold"] = lambda c: {"outcome": "unknown"}
        result = port.executor(max_grasp_attempts=3).execute({})
        self.assertEqual(result.state, "failed")
        self.assertNotIn("carry", [row[0] for row in port.calls])
        self.assertEqual(port.stops, ["verify_hold"])
        self.assertFalse(result.held_verified)

    def test_ack_or_truthy_evidence_never_confirms_success(self):
        for output in ({"accepted": True}, {"outcome": "success"},
                       {"outcome": "success", "evidence": {"confirmed": 1}}):
            with self.subTest(output=output):
                port = Port()
                port.operations["verify_place"] = lambda c: output
                result = port.executor().execute({})
                self.assertEqual(result.state, "failed")
                self.assertTrue(result.held_verified)
                self.assertFalse(result.placed_verified)

    def test_each_base_correction_invalidates_target_and_reobserves(self):
        port = Port()
        approach_calls = []
        def approach(context):
            approach_calls.append((context.target, context.observation.base_revision))
            return {"outcome": "retry" if len(approach_calls) == 1 else "success", "base_moved": True}
        port.operations["approach"] = approach
        result = port.executor().execute({})
        self.assertEqual(result.state, "succeeded")
        self.assertEqual(len(approach_calls), 2)
        self.assertEqual(approach_calls[1][1], 1)
        self.assertGreater(approach_calls[1][0]["sequence"], approach_calls[0][0]["sequence"])
        self.assertTrue(any(row.base_revision == 2 for row in port.observations))
        self.assertIsNotNone(next(row for row in port.calls if row[0] == "pregrasp")[2])

    def test_alignment_retry_cannot_run_forever(self):
        port = Port()
        port.operations["approach"] = lambda c: {"outcome": "retry", "base_moved": True}
        result = port.executor(max_alignment_attempts=2).execute({})
        self.assertEqual(result.state, "failed")
        self.assertEqual(sum(row.stage == "approach" for row in result.events), 2)
        self.assertNotIn("pregrasp", [row[0] for row in port.calls])

    def test_approach_retry_without_measured_base_motion_stops(self):
        port = Port()
        port.operations["approach"] = lambda c: {"outcome": "retry"}
        result = port.executor().execute({})
        self.assertEqual(result.state, "failed")
        self.assertEqual(sum(row.stage == "approach" for row in result.events), 1)

    def test_find_retries_are_bounded(self):
        port = Port()
        port.operations["find"] = lambda c: {"outcome": "retry"}
        result = port.executor(max_find_attempts=2).execute({})
        self.assertEqual(result.state, "failed")
        self.assertEqual(len(result.events), 2)
        self.assertEqual(result.stage, "find")

    def test_reused_observation_sequence_stops_after_base_movement(self):
        port = Port()
        port.operations["approach"] = lambda c: {"outcome": "success", "base_moved": True}
        executor = port.executor()
        def observe(context):
            if context.base_revision:
                return Observation(port.sequence, port.now, context.base_revision, {})
            return port.observe(context)
        executor.observe = observe
        result = executor.execute({})
        self.assertEqual(result.state, "failed")
        self.assertIn("Stale observation", result.reason)
        self.assertNotIn("pregrasp", [row[0] for row in port.calls])

    def test_cached_image_or_wrong_base_revision_is_rejected(self):
        for observation in (Observation(1, 99., 0, {}), Observation(1, 100., 1, {})):
            port = Port()
            executor = port.executor()
            executor.observe = lambda c: observation
            result = executor.execute({})
            self.assertEqual(result.state, "failed")
            self.assertEqual(port.calls, [])

    def test_cancellation_inside_operation_discards_late_success(self):
        port = Port()
        def grasp(context):
            port.cancelled = True
            return {"outcome": "success"}
        port.operations["grasp"] = grasp
        result = port.executor().execute({})
        self.assertEqual(result.state, "cancelled")
        self.assertFalse(result.held_verified)
        self.assertEqual(port.stops, ["grasp"])
        self.assertNotIn("lift", [row[0] for row in port.calls])

    def test_port_interruption_is_cancellation_and_stops_motion(self):
        port = Port()
        def interrupted(context):
            raise InterruptedError("operator pause")
        port.operations["grasp"] = interrupted
        result = port.executor().execute({})
        self.assertEqual(result.state, "cancelled")
        self.assertEqual(result.reason, "operator pause")
        self.assertEqual(port.stops, ["grasp"])
        self.assertNotIn("lift", [row[0] for row in port.calls])

    def test_stage_deadline_discards_late_result(self):
        port = Port()
        def operation(context):
            port.now += 2.
            return {"outcome": "success"}
        port.operations["pregrasp"] = operation
        result = port.executor(stage_timeouts={"pregrasp": 1.}).execute({})
        self.assertEqual(result.state, "failed")
        self.assertIn("Stage deadline", result.reason)
        self.assertNotIn("grasp", [row[0] for row in port.calls])

    def test_total_deadline_spans_multiple_stages(self):
        port = Port()
        for name in ("grasp", "lift"):
            def operation(context):
                port.now += 2.
                return {"outcome": "success"}
            port.operations[name] = operation
        result = port.executor(total_timeout=3.).execute({})
        self.assertEqual(result.state, "failed")
        self.assertIn("Task deadline", result.reason)

    def test_explicit_safe_recovery_required_for_second_grasp(self):
        port = Port()
        holds = iter(("failure", "success"))
        port.operations["verify_hold"] = lambda c: {"outcome": next(holds), "evidence": {"confirmed": True}}
        port.operations["recover_grasp"] = port.operation("recover_grasp")
        result = port.executor(max_grasp_attempts=2).execute({})
        self.assertEqual(result.state, "succeeded")
        names = [row[0] for row in port.calls]
        recovery = names.index("recover_grasp")
        self.assertEqual(names[recovery + 1: recovery + 5], ["reobserve", "pregrasp", "grasp", "lift"])

    def test_failure_without_recovery_port_never_repeats_grasp(self):
        port = Port()
        port.operations["verify_hold"] = lambda c: {"outcome": "failure"}
        result = port.executor(max_grasp_attempts=2).execute({})
        self.assertEqual(result.state, "failed")
        self.assertEqual(sum(row[0] == "grasp" for row in port.calls), 1)

    def test_guard_prevents_late_action_and_stop_failure_is_reported(self):
        port = Port()
        executor = port.executor()
        def guard(context):
            if context.stage == "release":
                raise ValueError("destination unsafe")
        executor.guard = guard
        executor.stop = lambda c: (_ for _ in ()).throw(OSError("stop transport unavailable"))
        result = executor.execute({})
        self.assertEqual(result.state, "failed")
        self.assertIn("destination unsafe", result.reason)
        self.assertIn("stop transport unavailable", result.stop_error)
        self.assertNotIn("release", [row[0] for row in port.calls])

    def test_exception_releases_single_run_lock(self):
        port = Port()
        executor = port.executor()
        executor.operations["find"] = lambda c: (_ for _ in ()).throw(ValueError("bad sensor"))
        self.assertEqual(executor.execute({}).state, "failed")
        executor.operations["find"] = port.operation("find")
        self.assertEqual(executor.execute({}).state, "succeeded")

    def test_target_must_be_explicit(self):
        port = Port()
        port.operations["find"] = lambda c: {"outcome": "success"}
        result = port.executor().execute({})
        self.assertEqual(result.state, "failed")
        self.assertIn("not measured", result.reason)


if __name__ == "__main__":
    unittest.main()
