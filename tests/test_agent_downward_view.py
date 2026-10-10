from types import SimpleNamespace
import unittest

from agent_intent import prepare_intent


class DownwardIntentTests(unittest.TestCase):
    def test_downward_view_uses_normal_validated_skill_without_llm(self):
        runtime = SimpleNamespace(validate_plan=lambda plan: plan,
            ports={"look_direction": SimpleNamespace(physical=True)})
        for command in ("Посмотри вниз", "осмотри пол перед рукой", "поверни камеру вниз"):
            result = prepare_intent(command, runtime)
            self.assertEqual(result["plan"]["steps"], [
                {"skill": "look_direction", "args": {"view": "down"}}])


if __name__ == "__main__":
    unittest.main()
