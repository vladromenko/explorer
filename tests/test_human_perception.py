import os
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from human_perception import GestureGate, HumanPerception, anchors, hand_gesture, verify_model, cv46_artifact, _cv46_bytes, fields, message, encode_int


def observation(stamp, gesture="open_palm", count=1):
    hand = {"confidence": .99, "landmarks": [[30, 30, 0] for _ in range(21)],
            "gesture": gesture, "index_tip_px": [40, 20], "direction_image": "up"}
    return {"frame_id": str(stamp), "image_stamp": stamp, "stale": False, "image_size": [100, 100],
            "mode_results": {"hand": [hand.copy() for _ in range(count)]}}


class HumanPerceptionTests(unittest.TestCase):
    def gate(self, authorized=False):
        gate = GestureGate()
        gate.begin("operator-session", {"open_palm": "wave", "point": "trace_2d"}, [0, 0, 100, 100], physical_authorized=authorized, now=10)
        return gate

    def test_context_required_for_any_proposal(self):
        result = GestureGate().update(observation(10), now=10)
        self.assertFalse(result["motion_authorized"])
        self.assertIsNone(result["proposal"])

    def test_explicit_context_never_grants_itself_motion(self):
        gate = self.gate()
        gate.update(observation(10), now=10)
        result = gate.update(observation(10.3), now=10.3)
        self.assertEqual(result["proposal"]["action"], "wave")
        self.assertFalse(result["motion_authorized"])
        self.assertTrue(result["proposal"]["requires_executor_validation"])

    def test_deduplicates_until_neutral(self):
        gate = self.gate(True)
        gate.update(observation(10), now=10)
        self.assertTrue(gate.update(observation(10.3), now=10.3)["motion_authorized"])
        self.assertIsNone(gate.update(observation(10.6), now=10.6)["proposal"])
        gate.update(observation(10.7, "neutral"), now=10.7)
        gate.update(observation(10.8), now=10.8)
        self.assertIsNotNone(gate.update(observation(11.1), now=11.1)["proposal"])

    def test_expiry_and_absence_reset_candidate(self):
        gate = self.gate(True)
        gate.update(observation(10), now=10)
        self.assertIsNone(gate.update(observation(10.3, count=0), now=10.3)["proposal"])
        self.assertIsNone(gate.update(observation(10.4), now=10.4)["proposal"])
        self.assertIsNone(gate.update(observation(31), now=31)["proposal"])
        self.assertIsNone(gate.context)

    def test_no_random_or_ambiguous_person_authority(self):
        gate = self.gate(True)
        self.assertIsNone(gate.update(observation(10, count=2), now=10)["proposal"])
        gate.context["region"] = [50, 50, 100, 100]
        self.assertIsNone(gate.update(observation(10.3), now=10.3)["proposal"])

    def test_stale_reordered_frames_cannot_trigger(self):
        gate = self.gate(True)
        gate.update(observation(10), now=10)
        self.assertIsNone(gate.update(observation(10), now=10.3)["proposal"])
        self.assertIsNone(gate.update(observation(10.3), now=11.3)["proposal"])
        self.assertIsNone(gate.update(observation(11.4), now=11.4)["proposal"])

    def test_drawing_is_only_a_bounded_normalized_image_trace(self):
        gate = self.gate(True)
        for index in range(300):
            result = gate.update(observation(10 + index * .03, "point"), now=10 + index * .03)
        self.assertEqual(result["state"], "drawing")
        self.assertEqual(len(result["drawing_2d"]), 256)
        self.assertEqual(result["drawing_2d"][-1], [.4, .2])
        self.assertFalse(result["motion_authorized"])
        self.assertIsNone(result["proposal"])

    def test_invalid_context_does_not_fall_back(self):
        for mapping in ({"open_palm": "shell"}, {"unknown": "wave"}):
            with self.assertRaises(ValueError):
                GestureGate().begin("operator-session", mapping, [0, 0, 100, 100])
        with self.assertRaises(ValueError):
            GestureGate().begin("operator-session", {"fist": "wave"}, [0, 0, 0, 100])

    def test_models_are_not_loaded_on_construction_or_cancel(self):
        perception = HumanPerception("/unavailable")
        self.assertIsNone(perception.cv)
        self.assertFalse(perception.networks)
        perception.unload()
        self.assertFalse(perception.networks)

    def test_wrong_model_hash_and_missing_weights_are_explicit(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(FileNotFoundError, "palm_detection"):
                verify_model(directory, "palm")
            name = "palm_detection_mediapipe_2023feb.onnx"
            (Path(directory) / name).write_bytes(b"not-a-model")
            with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                verify_model(directory, "palm")

    def test_anchor_shapes_and_order_match_pinned_models(self):
        palms = anchors(((24, 2), (12, 6)))
        people = anchors(((28, 2), (14, 2), (7, 6)))
        self.assertEqual(len(palms), 2016)
        self.assertEqual(len(people), 2254)
        self.assertEqual(palms[0], palms[1])
        self.assertEqual(palms[0], [.5 / 24, .5 / 24])
        self.assertEqual(palms[-1], [11.5 / 12, 11.5 / 12])

    def test_no_skin_or_bounding_box_pretends_to_be_landmarks(self):
        self.assertEqual(hand_gesture([[1, 1]] * 4)["gesture"], "neutral")
        self.assertEqual(hand_gesture([[1, 1]] * 21)["reason"], "hand_too_small")
        self.assertEqual(hand_gesture([[float("nan"), 1]] * 21)["reason"], "invalid_landmarks")

    def test_stale_rejected_before_loading_models(self):
        perception = HumanPerception("/unavailable")
        with self.assertRaisesRegex(ValueError, "stale"):
            perception.analyze(None, 1, now=5)
        self.assertIsNone(perception.cv)

    def test_invalid_landmarks_never_trigger_action(self):
        gate = self.gate(True)
        sample = observation(10)
        sample["mode_results"]["hand"][0]["landmarks"][1][0] = float("nan")
        self.assertIsNone(gate.update(sample, now=10)["proposal"])

    def test_drawing_outside_image_has_no_path(self):
        gate = self.gate(True)
        sample = observation(10, "point")
        sample["mode_results"]["hand"][0]["index_tip_px"] = [-1, 20]
        self.assertEqual(gate.update(sample, now=10)["reason"], "drawing_tip_outside_image")
        self.assertFalse(gate.stroke)

    def test_cv46_converter_is_pinned_and_never_overwrites_source(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FileNotFoundError):
                cv46_artifact(directory, "hand", create=True)

    def test_protobuf_preserves_unchanged_fields(self):
        sample = encode_int(8) + encode_int(130) + message(2, b"payload")
        decoded = fields(sample)
        self.assertEqual(b"".join(field[3] for field in decoded), sample)
        self.assertEqual(decoded[0][:3], (1, 0, 130))

    @unittest.skipUnless(os.environ.get("TEST_HUMAN_MODEL_DIR"), "Set TEST_HUMAN_MODEL_DIR for official model CPU forward tests")
    def test_real_pinned_models_and_runtime(self):
        directory = os.environ["TEST_HUMAN_MODEL_DIR"]
        for name in ("hand", "person", "pose"):
            cv46_artifact(directory, name, create=True)
        perception = HumanPerception(".", directory)
        report = perception.compatibility()
        self.assertTrue(all(row["available"] for row in report["components"].values()), report)
        import numpy as np
        import time
        result = perception.analyze(np.zeros((480, 640, 3), np.uint8), time.time(), ("hand", "pose", "face"), "synthetic")
        self.assertEqual(result["mode_results"]["hand"], [])
        self.assertEqual(result["mode_results"]["pose"], [])
        self.assertFalse(result["motion_authorized"])
        self.assertIn("insufficient contrast", result["errors"]["face"])


if __name__ == "__main__":
    unittest.main()


def test_observe_loads_before_acquiring_exposure(tmp_path, monkeypatch):
    import hashlib,json,time
    import cv2,numpy as np
    directory=tmp_path/"data";directory.mkdir()
    perception=HumanPerception(tmp_path)
    calls=[]
    def prepared(modes):
        calls.append("prepare")
        jpeg=cv2.imencode(".jpg",np.zeros((80,80,3),np.uint8))[1].tobytes()
        (directory/"frame-raw.jpg").write_bytes(jpeg)
        (directory/"camera-frame.json").write_text(json.dumps({"image_stamp":time.time(),"frame_id":"a"*32,"jpeg_sha256":hashlib.sha256(jpeg).hexdigest()}))
    monkeypatch.setattr(perception,"prepare",prepared)
    def analyzed(image,stamp,modes,identifier):
        calls.append("analyze")
        assert time.time()-stamp<.1
        return {"frame_id":identifier}
    monkeypatch.setattr(perception,"analyze",analyzed)
    result=perception.observe(("hand",))
    assert calls==["prepare","analyze"]
    assert result["image_endpoint"].endswith("a"*32)
    assert hashlib.sha256(perception.latest[1]).hexdigest()==result["image_sha256"]
