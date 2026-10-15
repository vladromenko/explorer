import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image

from simulator.phase_learning import prepare_phase_dataset, fit_phase_residuals, predict_phase_residual, iter_phase_examples
from src.mobile_policy_contract import CONTRACT_SHA256


class PhaseLearningTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def episode(self, number, *, seed=None, success=True, provenance=True):
        folder = self.root / str(number)
        folder.mkdir()
        metadata = {"source": "simulation", "task_id": "sock_basket", "seed": number if seed is None else seed,
            "success": success, "attempt_completed": True, "fps": 20,
            "sample_contract_sha256": CONTRACT_SHA256, "action_alignment": "same_row_pre_action"}
        if provenance:
            metadata["model_provenance"] = {"geometry_source": "checksum-verified fixture geometry",
                "camera": {"camera_info_version": "fixture-calibration"}, "handeye": {"reference_mount": "arm4"}}
        (folder / "episode.json").write_text(json.dumps(metadata))
        rows = []
        for index in range(8):
            rgb = np.full((12, 16, 3), 40 + number + index, np.uint8)
            Image.fromarray(rgb).save(folder / f"{index}.png")
            state = [90. + number + index] * 6 + [0., 0., 0.]
            action = state.copy()
            action[0] += 1.
            rows.append({"at": 1700000000. + number * 10 + index * .05,
                "simulation_time": index * .05, "observation": state, "action": action,
                "wrist_rgb": f"{index}.png", "phase": "approach" if index < 4 else "close",
                "simulated_outcome": {"success": True, "future_reward": 999},
                "simulator_diagnostics": {"sock_center_m": [900, 800, 700]}})
        (folder / "samples.jsonl").write_text("\n".join(json.dumps(row) for row in rows))
        return folder

    def prepare(self):
        paths = [self.episode(i) for i in range(5)]
        path = self.root / "dataset.json"
        result = prepare_phase_dataset(paths, path)
        return path, result

    def test_current_frame_features_explicit_boundaries_and_honest_artifact(self):
        path, dataset = self.prepare()
        self.assertEqual([item["split"] for item in dataset["episodes"]], ["train"] * 3 + ["validation", "test"])
        first = dataset["episodes"][0]
        self.assertEqual(first["segments"], [{"phase": "approach", "start": 0, "stop": 4},
                                             {"phase": "close", "start": 4, "stop": 8}])
        self.assertNotIn("examples", first)
        sample = next(iter_phase_examples(first))
        self.assertEqual(len(sample["features"]), 45)
        self.assertEqual(sample["features"][:9], [90.] * 6 + [0.] * 3)
        np.testing.assert_allclose(sample["features"][9:], 40 / 255.)
        self.assertEqual(sample["residual"], [1.] + [0.] * 8)
        self.assertNotIn("simulator_diagnostics", sample)
        self.assertNotIn("simulated_outcome", sample)
        model = fit_phase_residuals(path, self.root / "model.json")
        self.assertFalse(model["ready_for_transfer"])
        self.assertFalse(model["closed_loop_verified"])
        self.assertFalse(model["test_groups_used"])
        self.assertEqual(model["data_source"], "preserved_successful_simulation_demonstrations")
        prediction = predict_phase_residual(model, "approach", [100.] * 6 + [0.] * 3,
                                            np.full((12, 16, 3), 60, np.uint8))
        np.testing.assert_allclose(prediction, [1.] + [0.] * 8, atol=1e-10)
        self.assertLess(model["models"]["approach"]["validation_mae_by_coordinate"][0], .01)

    def test_failed_trials_remain_excluded_and_unmodified(self):
        paths = [self.episode(i) for i in range(4)]
        bad = self.episode(4, success=False, provenance=False)
        original = (bad / "episode.json").read_bytes()
        result = prepare_phase_dataset(paths + [bad], self.root / "dataset.json")
        self.assertEqual(result["excluded"], [{"folder": str(bad.resolve()), "reason": "not_completed_success"}])
        self.assertEqual((bad / "episode.json").read_bytes(), original)

    def test_missing_visual_or_geometry_provenance_rejected(self):
        paths = [self.episode(i) for i in range(3)]
        manifest = paths[1] / "episode.json"
        metadata = json.loads(manifest.read_text())
        del metadata["model_provenance"]["camera"]
        manifest.write_text(json.dumps(metadata))
        with self.assertRaisesRegex(ValueError, "visual/calibration"):
            prepare_phase_dataset(paths, self.root / "dataset.json")
        del metadata["model_provenance"]
        manifest.write_text(json.dumps(metadata))
        with self.assertRaisesRegex(ValueError, "geometry provenance"):
            prepare_phase_dataset(paths, self.root / "dataset.json")

    def test_unknown_phase_cannot_silently_train_whole_episode(self):
        paths = [self.episode(i) for i in range(3)]
        sample = paths[0] / "samples.jsonl"
        rows = [json.loads(line) for line in sample.read_text().splitlines()]
        rows[0]["phase"] = "policy"
        sample.write_text("\n".join(json.dumps(row) for row in rows))
        with self.assertRaisesRegex(ValueError, "explicit phase"):
            prepare_phase_dataset(paths, self.root / "dataset.json")

    def test_seed_and_duplicate_recording_union_never_crosses_split(self):
        paths = [self.episode(i) for i in range(6)]
        metadata_path = paths[4] / "episode.json"
        metadata = json.loads(metadata_path.read_text())
        metadata["seed"] = 0
        metadata_path.write_text(json.dumps(metadata))
        # Copy samples verbatim while claiming an unrelated scene seed.
        (paths[5] / "samples.jsonl").write_bytes((paths[4] / "samples.jsonl").read_bytes())
        result = prepare_phase_dataset(paths, self.root / "dataset.json")
        self.assertEqual(result["episodes"][0]["split"], result["episodes"][4]["split"])
        self.assertEqual(result["episodes"][4]["split"], result["episodes"][5]["split"])

    def test_test_samples_never_affect_training_weights(self):
        path, dataset = self.prepare()
        first = fit_phase_residuals(path, self.root / "first.json")
        # A corrupt reserved test cache must not be opened during fitting or
        # validation, let alone influence normalization or regression weights.
        Path(dataset["episodes"][-1]["examples_path"]).write_text("not JSON")
        second = fit_phase_residuals(path, self.root / "second.json")
        self.assertEqual(first["models"], second["models"])

    def test_changed_geometry_and_edited_split_rejected(self):
        paths = [self.episode(i) for i in range(3)]
        manifest = paths[2] / "episode.json"
        metadata = json.loads(manifest.read_text())
        metadata["model_provenance"]["geometry_source"] = "different geometry"
        manifest.write_text(json.dumps(metadata))
        with self.assertRaisesRegex(ValueError, "Incompatible geometry"):
            prepare_phase_dataset(paths, self.root / "dataset.json")
        metadata["model_provenance"]["geometry_source"] = "checksum-verified fixture geometry"
        manifest.write_text(json.dumps(metadata))
        path = self.root / "dataset.json"
        result = prepare_phase_dataset(paths, path)
        result["episodes"][-1]["split"] = "train"
        path.write_text(json.dumps(result))
        with self.assertRaisesRegex(ValueError, "independent scene split"):
            fit_phase_residuals(path, self.root / "bad.json")

    def test_fewer_than_three_groups_cannot_report_independence(self):
        with self.assertRaisesRegex(ValueError, "three independent"):
            prepare_phase_dataset([self.episode(1), self.episode(2, seed=1)], self.root / "dataset.json")

    def test_outputs_cannot_overwrite_existing_artifacts(self):
        path, dataset = self.prepare()
        with self.assertRaises(FileExistsError):
            prepare_phase_dataset([item["folder"] for item in dataset["episodes"]], path)

    def test_cancel_retains_completed_cache_and_resume_reuses_it(self):
        paths = [self.episode(i) for i in range(5)]
        destination = self.root / "dataset.json"
        messages = []
        def progress(value):
            messages.append(value)
        with self.assertRaises(InterruptedError):
            prepare_phase_dataset(paths, destination, progress=progress,
                cancelled=lambda: any(item.get("processed") == 1 and item.get("stage") == "prepared" for item in messages))
        self.assertFalse(destination.exists())
        self.assertEqual(len(list((self.root / "dataset.json.cache").glob("*/metadata.json"))), 1)
        messages.clear()
        result = prepare_phase_dataset(paths, destination, progress=progress)
        self.assertEqual(result["state"], "complete")
        self.assertEqual(result["sample_count"], 40)
        self.assertEqual([item["stage"] for item in messages if item.get("processed") == 1 and item["stage"] in ("cached", "prepared")], ["cached"])

    def test_cache_modification_is_detected_and_rebuilt_on_resume(self):
        paths = [self.episode(i) for i in range(5)]
        destination = self.root / "dataset.json"
        messages = []
        with self.assertRaises(InterruptedError):
            prepare_phase_dataset(paths, destination, progress=messages.append,
                cancelled=lambda: any(item.get("stage") == "prepared" for item in messages))
        cache_file = next((self.root / "dataset.json.cache").glob("*/examples.jsonl"))
        cache_file.write_text("partial or modified data")
        messages.clear()
        dataset = prepare_phase_dataset(paths, destination, progress=messages.append)
        self.assertEqual(dataset["sample_count"], 40)
        self.assertEqual([item["stage"] for item in messages if item.get("processed") == 1 and item["stage"] in ("cached", "prepared")], ["prepared"])
        path = Path(dataset["episodes"][0]["examples_path"])
        path.write_text("changed")
        with self.assertRaisesRegex(ValueError, "checksum mismatch"):
            fit_phase_residuals(destination, self.root / "bad.json")

    def test_cancelled_fit_does_not_create_model(self):
        path, _ = self.prepare()
        destination = self.root / "model.json"
        with self.assertRaises(InterruptedError):
            fit_phase_residuals(path, destination, cancelled=lambda: True)
        self.assertFalse(destination.exists())


if __name__ == "__main__":
    unittest.main()
