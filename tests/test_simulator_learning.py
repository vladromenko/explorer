import hashlib
import io
import json
from pathlib import Path
import sys
import tarfile
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from simulator.learning import inspect_episode, training_config, train_job, create_bundle, write_json, split_episodes, export_dataset
from simulator.transfer import export_package, import_package
from src.mobile_policy_contract import CONTRACT_SHA256, read_bundle, ACTOR_CLOCK, ACTOR_ACTION_TRANSFORM, decode_policy_action


class SimulatorLearningTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def episode(self):
        folder = self.root / "episode"
        folder.mkdir()
        write_json(folder / "episode.json", dict(id="episode", source="simulation", state="complete",
            fps=20, success=False, physics_validated=False, sample_contract_sha256=CONTRACT_SHA256))
        rows = []
        for index in range(4):
            (folder / f"{index}.jpg").write_bytes(b"image")
            rows.append(dict(at=1700000000 + index * .05, observation=[90] * 6 + [0] * 3,
                action=[90] * 6 + [.05, 0, 0], wrist_rgb=f"{index}.jpg",
                privileged_object_position=[1, 2, 3]))
        (folder / "samples.jsonl").write_text("\n".join(json.dumps(row) for row in rows))
        return folder

    def model(self):
        folder = self.root / "job"
        checkpoint = folder / "model/checkpoints/last/pretrained_model"
        checkpoint.mkdir(parents=True)
        (checkpoint / "model.safetensors").write_bytes(b"test weights")
        write_json(checkpoint / "config.json", dict(type="act", device="mps", n_action_steps=1,
            input_features={"observation.state": {"shape": [9]}, "observation.images.wrist": {"shape": [3, 240, 320]}},
            output_features={"action": {"shape": [9]}}))
        for name in ("policy_preprocessor.json", "policy_postprocessor.json"):
            write_json(checkpoint / name, {"steps": [{"config": {"device": "mps"}}]})
        write_json(folder / "train/explorer-provenance.json", {"dataset_fps": 20})
        state = dict(task="pick and place", episodes=["demo"], validation={"heldout_independent": False})
        write_json(folder / "job.json", state)
        create_bundle(folder, state)
        return folder

    def test_command_contract_and_actor_feature_config(self):
        _, rows, fps, inputs = inspect_episode(self.episode())
        self.assertEqual(fps, 20)
        self.assertEqual(rows[0]["observation_state"], [90] * 6 + [0] * 3)
        config = training_config(self.root, 3, "cpu", batch_size=1)
        self.assertEqual(set(config["policy"]["input_features"]), {"observation.state", "observation.images.wrist", "observation.environment_state"})
        self.assertEqual(config["policy"]["n_action_steps"], 20)
        self.assertEqual(config["policy"]["chunk_size"], 20)
        self.assertIsNone(config["policy"]["temporal_ensemble_coeff"])
        self.assertIn("samples.jsonl", inputs)

    def test_scene_seed_duplicates_never_leak_into_holdout(self):
        folders = [Path(name) for name in ("a", "b", "c", "d", "e")]
        metadata = [dict(source="simulation", seed=seed) for seed in (0, 10000, 10002, 0, 10000)]
        train, validation, independent = split_episodes(folders, metadata)
        self.assertTrue(independent)
        self.assertEqual(train, [folders[0], folders[2], folders[3]])
        self.assertEqual(validation, [folders[1], folders[4]])
        train, validation, independent = split_episodes(folders[:2], [metadata[0], metadata[0]])
        self.assertFalse(independent)
        self.assertEqual(train, validation)
        first = self.episode()
        second = self.root / "manual-copy"
        second.mkdir()
        (second / "samples.jsonl").write_bytes((first / "samples.jsonl").read_bytes())
        _, _, independent = split_episodes([first, second], [dict(source="simulator_manual")] * 2)
        self.assertFalse(independent)

    def test_optional_clock_transfers_and_mismatched_context_is_rejected(self):
        model = self.model()
        config_path = model / "model/checkpoints/last/pretrained_model/config.json"
        config = json.loads(config_path.read_text())
        config["input_features"][ACTOR_CLOCK["feature"]] = {"type": "ENV", "shape": [2]}
        config["pretrained_backbone_weights"] = "ResNet18_Weights.IMAGENET1K_V1"
        write_json(config_path, config)
        write_json(model / "train/explorer-provenance.json", {"dataset_fps":20, "actor_context":{**ACTOR_CLOCK, "period_s":.05}, "action_transform":ACTOR_ACTION_TRANSFORM})
        create_bundle(model, json.loads((model / "job.json").read_text()))
        archive = export_package(model, self.root / "clock.tar.gz")
        imported = import_package(archive["path"], self.root / "robot")
        bundle, checkpoint = read_bundle(imported["path"])
        self.assertEqual(bundle["actor_context"]["input_source"], "internal_elapsed_time")
        state = [90.]*6+[0.]*3
        delta = [1.]*6+[.1, .02, -.05]
        self.assertEqual(decode_policy_action(state, delta, bundle), [91.]*6+[.1, .02, -.05])
        self.assertEqual(decode_policy_action(state, delta, {}), delta)
        self.assertIsNone(json.loads((checkpoint / "config.json").read_text())["pretrained_backbone_weights"])
        bundle["actor_context"]["period_s"] = 0.
        write_json(Path(imported["path"]) / "bundle.json", bundle)
        with self.assertRaisesRegex(ValueError, "контекст времени"):
            read_bundle(imported["path"])

    def test_clock_remains_a_batched_vector_in_real_lerobot_dataset(self):
        import torch
        from PIL import Image
        from lerobot.datasets import LeRobotDataset
        episode = self.episode()
        for index in range(4):
            Image.new("RGB", (320, 240), (100, index, 50)).save(episode / f"{index}.jpg")
        export_dataset([episode], self.root / "dataset", "explorer/clock-test", "pick sock")
        dataset = LeRobotDataset("explorer/clock-test", root=self.root / "dataset", video_backend="pyav")
        batch = next(iter(torch.utils.data.DataLoader(dataset, batch_size=2)))
        self.assertEqual(tuple(batch[ACTOR_CLOCK["feature"]].shape), (2, 14))
        self.assertAlmostEqual(float(batch[ACTOR_CLOCK["feature"]][1, 0]), .05/80.)
        self.assertAlmostEqual(float(batch[ACTOR_CLOCK["feature"]][1, 1]), .05/80.)
        torch.testing.assert_close(batch["action"][:, :6], torch.full((2, 6),90.))
        self.assertAlmostEqual(float(batch["action"][0, 6]), .05)

    def test_bounded_temporal_chunks_transfer_with_one_action(self):
        model = self.model()
        config_path = model / "model/checkpoints/last/pretrained_model/config.json"
        config = json.loads(config_path.read_text())
        config.update(chunk_size=50, temporal_ensemble_coeff=.01, pretrained_path="/Mac/initial-policy")
        write_json(config_path, config)
        create_bundle(model, json.loads((model / "job.json").read_text()))
        archive = export_package(model, self.root / "temporal.tar.gz")
        imported = import_package(archive["path"], self.root / "robot")
        bundle, checkpoint = read_bundle(imported["path"])
        self.assertEqual(bundle["chunk_size"], 50)
        self.assertEqual(json.loads((checkpoint / "config.json").read_text())["n_action_steps"], 1)
        self.assertIsNone(json.loads((checkpoint / "config.json").read_text())["pretrained_path"])
        config.update(chunk_size=1000)
        write_json(config_path, config)
        create_bundle(model, json.loads((model / "job.json").read_text()))
        archive = export_package(model, self.root / "unbounded.tar.gz")
        with self.assertRaisesRegex(ValueError, "bounded"):
            import_package(archive["path"], self.root / "unbounded")

    def test_short_action_queue_is_declared_and_transfers_one_action_per_call(self):
        model = self.model()
        config_path = model / "model/checkpoints/last/pretrained_model/config.json"
        config = json.loads(config_path.read_text())
        config.update(chunk_size=20,n_action_steps=20,temporal_ensemble_coeff=None)
        write_json(config_path,config)
        create_bundle(model,json.loads((model / "job.json").read_text()))
        package = export_package(model,self.root / "queue.tar.gz")
        imported = import_package(package["path"],self.root / "robot")
        bundle, _ = read_bundle(imported["path"])
        self.assertEqual(bundle["inference_execution"]["actions_per_call"],1)
        bundle["inference_execution"]["actions_per_call"]=20
        write_json(Path(imported["path"]) / "bundle.json",bundle)
        with self.assertRaisesRegex(ValueError,"выполнение"):
            read_bundle(imported["path"])

    def test_incompatible_contract_and_unbounded_commands_rejected(self):
        folder = self.episode()
        metadata = json.loads((folder / "episode.json").read_text())
        metadata["sample_contract_sha256"] = "bad"
        write_json(folder / "episode.json", metadata)
        with self.assertRaisesRegex(ValueError, "contract"):
            inspect_episode(folder)
        metadata["sample_contract_sha256"] = CONTRACT_SHA256
        write_json(folder / "episode.json", metadata)
        rows = [json.loads(row) for row in (folder / "samples.jsonl").read_text().splitlines()]
        rows[0]["action"][6] = 5
        (folder / "samples.jsonl").write_text("\n".join(json.dumps(row) for row in rows))
        with self.assertRaisesRegex(ValueError, "exceed"):
            inspect_episode(folder)

    def test_failed_training_never_claims_trained(self):
        with patch("simulator.learning.choose_device", return_value="cpu"), \
             patch("importlib.metadata.version", return_value="0.6.1"), \
             patch("simulator.learning.export_dataset"), \
             patch("simulator.learning.subprocess.Popen", side_effect=OSError("learner unavailable")):
            state = train_job(self.root / "failed", [self.episode()], steps=3, device="cpu", diagnostic_only=True)
        self.assertEqual(state["state"], "failed")
        self.assertIn("learner unavailable", state["error"])
        self.assertFalse((self.root / "failed/bundle.json").exists())

    def test_cancelled_export_never_launches_a_learner(self):
        stopped = False
        def export(*args, **kwargs):
            nonlocal stopped
            stopped = True
        with patch("importlib.metadata.version", return_value="0.6.1"), \
             patch("simulator.learning.choose_device", return_value="cpu"), \
             patch("simulator.learning.export_dataset", side_effect=export), \
             patch("simulator.learning.subprocess.Popen") as learner:
            state = train_job(self.root / "cancelled", [self.episode()], steps=3,
                              diagnostic_only=True, cancelled=lambda: stopped)
        self.assertEqual(state["state"], "cancelled")
        learner.assert_not_called()

    def test_export_retargets_device_and_import_defaults_inactive(self):
        model = self.model()
        weights = model / "model/checkpoints/last/pretrained_model/model.safetensors"
        report = dict(validation_kind="closed_loop_policy", successes=0, episodes=2,
            model_weights_sha256=hashlib.sha256(weights.read_bytes()).hexdigest(), real_hardware_accepted=False,
            physics_validated=False)
        write_json(model / "physics-validation.json", report)
        before = (model / "model/checkpoints/last/pretrained_model/config.json").read_bytes()
        result = export_package(model, self.root / "policy.tar.gz")
        imported = import_package(result["path"], self.root / "robot")
        self.assertFalse(imported["registered"])
        self.assertFalse(imported["automatic_execution_allowed"])
        bundle, checkpoint = read_bundle(imported["path"])
        self.assertEqual(json.loads((checkpoint / "config.json").read_text())["device"], "cuda")
        self.assertEqual(json.loads((checkpoint / "policy_preprocessor.json").read_text())["steps"][0]["config"]["device"], "cuda")
        self.assertEqual(before, (model / "model/checkpoints/last/pretrained_model/config.json").read_bytes())
        self.assertFalse(bundle["physics_acceptance"])
        self.assertEqual(json.loads((Path(imported["path"]) / "physics-validation.json").read_text()), report)
        self.assertFalse((self.root / "robot/data/learning-jobs").exists())
        registered = import_package(result["path"], self.root / "robot", register=True)
        self.assertTrue(registered["registered"])
        self.assertEqual(registered["job"]["state"], "imported_simulation")
        self.assertFalse(registered["job"]["executed"])

    def test_validation_binds_execution_settings_through_device_retargeting(self):
        model = self.model()
        source, _ = read_bundle(model)
        report = dict(validation_kind="closed_loop_policy", successes=10, episodes=10,
            heldout_independent=True, model_weights_sha256=source["checkpoint_sha256"]["model.safetensors"],
            model_checkpoint_sha256=source["checkpoint_sha256"], real_hardware_accepted=False,
            physics_validated=False)
        write_json(model / "physics-validation.json", report)
        result = export_package(model, self.root / "bound-report.tar.gz")
        imported = import_package(result["path"], self.root / "robot")
        bundle, _ = read_bundle(imported["path"])
        saved = json.loads((Path(imported["path"]) / "physics-validation.json").read_text())
        self.assertEqual(saved["model_checkpoint_sha256"], source["checkpoint_sha256"])
        self.assertEqual(bundle["validation_checkpoint_sha256"], source["checkpoint_sha256"])
        self.assertEqual(saved["transfer_checkpoint_sha256"], bundle["checkpoint_sha256"])
        self.assertNotEqual(saved["model_checkpoint_sha256"]["config.json"], saved["transfer_checkpoint_sha256"]["config.json"])
        report["model_checkpoint_sha256"] = {**source["checkpoint_sha256"], "config.json": "different execution"}
        write_json(model / "physics-validation.json", report)
        with self.assertRaisesRegex(ValueError, "execution settings"):
            export_package(model, self.root / "mismatched-execution.tar.gz")

    def test_failed_attempts_are_not_default_task_training(self):
        with patch("importlib.metadata.version", return_value="0.6.1"), \
             patch("simulator.learning.choose_device", return_value="cpu"):
            state = train_job(self.root / "task-job", [self.episode()], steps=3)
        self.assertEqual(state["state"], "failed")
        self.assertIn("successful simulator demonstrations", state["error"])
        self.assertFalse((self.root / "task-job/bundle.json").exists())

    def test_export_rejects_stale_or_teacher_physics_reports(self):
        model = self.model()
        write_json(model / "physics-validation.json", dict(validation_kind="closed_loop_policy", model_weights_sha256="old"))
        with self.assertRaisesRegex(ValueError, "different"):
            export_package(model, self.root / "stale.tar.gz")
        write_json(model / "physics-validation.json", dict(validation_kind="teacher_physics_diagnostics"))
        with self.assertRaisesRegex(ValueError, "Teacher"):
            export_package(model, self.root / "teacher.tar.gz")

    def test_archive_rejects_traversal_links_duplicates_and_digest_changes(self):
        for kind in ("traversal", "symlink", "duplicate"):
            archive = self.root / (kind + ".tar.gz")
            with tarfile.open(archive, "w:gz") as stream:
                member = tarfile.TarInfo("../escape" if kind == "traversal" else "bundle.json")
                if kind == "symlink":
                    member.type = tarfile.SYMTYPE
                    member.linkname = "/tmp/escape"
                    stream.addfile(member)
                else:
                    member.size = 2
                    stream.addfile(member, io.BytesIO(b"{}"))
                    if kind == "duplicate":
                        stream.addfile(member, io.BytesIO(b"{}"))
            with self.assertRaises(ValueError):
                import_package(archive, self.root / kind)
        model = self.model()
        export_package(model, self.root / "valid.tar.gz")
        corrupted = self.root / "corrupted.tar.gz"
        with tarfile.open(self.root / "valid.tar.gz", "r:gz") as original, tarfile.open(corrupted, "w:gz") as output:
            for member in original.getmembers():
                content = original.extractfile(member).read()
                if member.name.endswith("model.safetensors"):
                    content = b"changed weights"
                member.size = len(content)
                output.addfile(member, io.BytesIO(content))
        with self.assertRaisesRegex(ValueError, "checksum"):
            import_package(corrupted, self.root / "robot")


if __name__ == "__main__":
    unittest.main()
