"""Isolated simulator data export and official LeRobot ACT training.

Actor inputs are the wrist RGB image and Explorer's nine commanded coordinates.
Physics diagnostics may describe a trajectory, but never become actor features.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import shutil
import signal
import sys
import time
import uuid

from src.mobile_policy_contract import ACTION_ORDER, UNITS, SAMPLE_CONTRACT, CONTRACT_SHA256, read_bundle


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False))
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _contained(folder, name):
    relative = Path(name)
    path = (folder / relative).resolve()
    if relative.is_absolute() or ".." in relative.parts or not path.is_relative_to(folder.resolve()):
        raise ValueError("Episode file is outside its folder")
    if not path.is_file():
        raise ValueError("Episode file is missing: " + str(name))
    return path


def _vector(value):
    if not isinstance(value, (list, tuple)) or len(value) != 9:
        raise ValueError("Expected nine commanded coordinates")
    vector = [float(item) for item in value]
    if not all(math.isfinite(item) for item in vector):
        raise ValueError("Non-finite trajectory coordinate")
    limits = [(0, 180)] * 4 + [(0, 270), (30, 180), (-.8, .8), (-.72, .72), (-1.67, 1.67)]
    if any(not low <= item <= high for item, (low, high) in zip(vector, limits)):
        raise ValueError("Trajectory commands exceed the Explorer contract")
    return vector


def inspect_episode(folder):
    """Read simulator/operator-simulation recording without assuming task success."""
    folder = Path(folder).resolve()
    episode = json.loads((folder / "episode.json").read_text())
    if episode.get("source") not in ("simulation", "simulator_teacher", "simulator_manual", "synthetic_demonstration"):
        raise ValueError("Only explicit simulator trajectory sources are accepted")
    if episode.get("state", "complete") != "complete":
        raise ValueError("Finish the trajectory before exporting")
    rows = [json.loads(line) for line in (folder / "samples.jsonl").read_text().splitlines() if line.strip()]
    if len(rows) < 3:
        raise ValueError("A trajectory needs at least three samples")
    period = float(episode.get("sample_period_s", .1))
    if not math.isfinite(period) or period <= 0:
        raise ValueError("Invalid simulator sample period")
    fps = float(episode.get("fps", episode.get("physical_sample_rate_hz", 1 / period)))
    if not math.isfinite(fps) or fps < 2 or fps > 60:
        raise ValueError("Invalid simulator recording frequency")
    inputs = {"episode.json": sha256(folder / "episode.json"), "samples.jsonl": sha256(folder / "samples.jsonl")}
    for row in rows:
        if row.get("sample_contract_sha256", episode.get("sample_contract_sha256")) != CONTRACT_SHA256:
            raise ValueError("Trajectory sample contract does not match Explorer")
        if row.get("joint_state_source", "command_estimate") != "command_estimate":
            raise ValueError("Simulator actor state must be the commanded estimate")
        row["observation_state"] = row.get("observation_state", row.get("observation", row.get("command")))
        row["applied_action"] = row.get("applied_action", row.get("action", row.get("command")))
        row["image"] = row.get("image", row.get("wrist_rgb"))
        row["image_stamp"] = row.get("image_stamp", row.get("at"))
        _vector(row["observation_state"])
        _vector(row["applied_action"])
        image = _contained(folder, row["image"])
        inputs[row["image"]] = sha256(image)
        stamp = float(row["image_stamp"])
        if not math.isfinite(stamp):
            raise ValueError("Invalid trajectory timestamp")
    for previous, current in zip(rows, rows[1:]):
        if "simulation_time" in previous and "simulation_time" in current:
            dt = float(current["simulation_time"]) - float(previous["simulation_time"])
        else:
            dt = float(current["image_stamp"]) - float(previous["image_stamp"])
        if not 0 < dt <= max(.75, 2 / fps):
            raise ValueError("Simulator trajectory has a discontinuity; record a new episode")
    return episode, rows, int(round(fps)), inputs


def export_dataset(episode_folders, destination, repo_id="explorer/simulator", task="pick and place", cancelled=None):
    """Write LeRobot data from pre-action state/image and that row's next command."""
    import numpy as np
    from PIL import Image
    from lerobot.datasets import LeRobotDataset

    folders = [Path(folder).resolve() for folder in episode_folders]
    if not folders:
        raise ValueError("No simulator trajectories were selected")
    episodes = [inspect_episode(folder) for folder in folders]
    rates = {episode[2] for episode in episodes}
    if len(rates) != 1:
        raise ValueError("Simulator trajectories must have the same recording frequency")
    destination = Path(destination).resolve()
    if destination.exists():
        raise ValueError("Dataset destination already exists")
    features = {
        "observation.state": {"dtype": "float32", "shape": (9,), "names": ACTION_ORDER},
        "action": {"dtype": "float32", "shape": (9,), "names": ACTION_ORDER},
        "observation.images.wrist": {"dtype": "image", "shape": (240, 320, 3),
                                     "names": ["height", "width", "channels"]},
    }
    fps = next(iter(rates))
    dataset = LeRobotDataset.create(repo_id=repo_id, root=destination, fps=fps,
        features=features, robot_type="rosmaster_m3pro_mobile_manipulation_simulated_command_estimate",
        use_videos=False, image_writer_threads=2)
    provenance = []
    try:
        for folder, (episode, rows, _, inputs) in zip(folders, episodes):
            alignment = episode.get("action_alignment", "pre_action_observation")
            if alignment not in ("pre_action_observation", "same_row_pre_action"):
                raise ValueError("Expected a pre-action observation and its next applied command")
            for current in rows:
                if cancelled and cancelled():
                    raise InterruptedError("Simulator training cancelled during dataset export")
                with Image.open(_contained(folder, current["image"])) as image:
                    rgb = np.asarray(image.convert("RGB").resize((320, 240))).copy()
                dataset.add_frame({
                    "observation.state": np.asarray(_vector(current.get("observation_state", current.get("command"))), dtype=np.float32),
                    "action": np.asarray(_vector(current["applied_action"]), dtype=np.float32),
                    "observation.images.wrist": rgb, "task": task or episode.get("name", "pick and place"),
                })
            dataset.save_episode()
            provenance.append({"episode": episode, "input_sha256": inputs,
                               "samples": len(rows), "action_alignment": "pre_action_observation"})
        dataset.finalize()
    finally:
        if dataset.writer is not None:
            dataset.writer.stop_image_writer()
    result = dict(format="explorer_simulator_dataset_v1", episodes=provenance, dataset_fps=fps,
        training_source="synthetic_only", domain="simulation", joint_state_source="command_estimate",
        sample_contract=SAMPLE_CONTRACT, sample_contract_sha256=CONTRACT_SHA256,
        actor_inputs=["observation.state", "observation.images.wrist"],
        original_timestamps="samples.jsonl:at (Unix seconds), simulation_time (simulation seconds)",
        dataset_timebase="fixed_simulation_step",
        privileged_actor_inputs=False, physics_acceptance=False, physical_success_verified=False,
        automatic_execution_allowed=False)
    write_json(destination / "explorer-provenance.json", result)
    return result


def choose_device(request="auto"):
    import torch
    if request not in ("auto", "cpu", "mps", "cuda"):
        raise ValueError("Unknown training device")
    if request == "auto":
        return "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    if request == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS is unavailable in this Python environment")
    if request == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable in this Python environment")
    return request


class LoadedPolicy:
    """Image/command-only actor callable for independent simulator rollouts."""
    def __init__(self, job_folder, device="cpu"):
        import torch
        from lerobot.policies.act import ACTPolicy
        from lerobot.policies.act.configuration_act import ACTConfig
        from lerobot.policies import make_pre_post_processors

        self.torch = torch
        self.device = choose_device(device)
        _, checkpoint = read_bundle(job_folder)
        config = ACTConfig.from_pretrained(checkpoint, local_files_only=True)
        config.device = self.device
        self.policy = ACTPolicy.from_pretrained(checkpoint, config=config, local_files_only=True)
        self.policy.to(self.device).eval()
        self.pre, self.post = make_pre_post_processors(self.policy.config, pretrained_path=str(checkpoint),
            preprocessor_overrides={"device_processor": {"device": self.device}},
            postprocessor_overrides={"device_processor": {"device": "cpu"}})
        self.reset()

    def reset(self):
        self.policy.reset()

    def __call__(self, state9, rgb):
        import numpy as np
        state = _vector(list(state9))
        image = np.asarray(rgb)
        if image.shape != (240, 320, 3) or image.dtype != np.uint8:
            raise ValueError("The actor requires a 320x240 RGB uint8 wrist frame")
        batch = {"observation.state": self.torch.tensor([state], dtype=self.torch.float32),
                 "observation.images.wrist": self.torch.from_numpy(image.copy()).permute(2, 0, 1).float().div(255).unsqueeze(0)}
        with self.torch.no_grad():
            action = self.post(self.policy.select_action(self.pre(batch))).cpu().numpy().reshape(-1)
        if action.shape != (9,) or not np.isfinite(action).all():
            raise ValueError("ACT produced invalid commands")
        return action


def load_policy(job_folder, device="cpu"):
    return LoadedPolicy(job_folder, device)


def training_config(folder, steps, device, batch_size=16, seed=1000):
    if type(steps) is not int or not 1 <= steps <= 200000:
        raise ValueError("Training steps must be between 1 and 200000")
    if type(batch_size) is not int or not 1 <= batch_size <= 32:
        raise ValueError("Batch size must be between 1 and 32")
    if type(seed) is not int or not 0 <= seed < 2 ** 32:
        raise ValueError("Training seed must be an unsigned 32-bit integer")
    folder = Path(folder).resolve()
    return dict(dataset=dict(repo_id="explorer/simulator-train", root=str(folder / "train"), video_backend="pyav"),
        policy=dict(type="act", device=device, push_to_hub=False, chunk_size=20, n_action_steps=1,
            temporal_ensemble_coeff=.01,
            pretrained_backbone_weights=None, dim_model=128, n_heads=4, dim_feedforward=512,
            n_encoder_layers=2, n_decoder_layers=1, n_vae_encoder_layers=2,
            use_vae=False, dropout=0., optimizer_lr=3e-4, optimizer_lr_backbone=1e-4,
            input_features={"observation.state": {"type": "STATE", "shape": [9]},
                "observation.images.wrist": {"type": "VISUAL", "shape": [3, 240, 320]}},
            output_features={"action": {"type": "ACTION", "shape": [9]}}),
        output_dir=str(folder / "model"), batch_size=batch_size, num_workers=2, steps=steps,
        save_freq=min(100, steps), log_freq=min(10, steps), env_eval_freq=0,
        wandb=dict(enable=False), seed=seed)


def validate_checkpoint(folder, device="cpu", maximum_samples=128, cancelled=None):
    import numpy as np
    import torch
    from lerobot.datasets import LeRobotDataset
    from lerobot.policies.act import ACTPolicy
    from lerobot.policies.act.configuration_act import ACTConfig
    from lerobot.policies import make_pre_post_processors

    folder = Path(folder)
    checkpoint = folder / "model/checkpoints/last/pretrained_model"
    config = ACTConfig.from_pretrained(checkpoint, local_files_only=True)
    config.device = device
    policy = ACTPolicy.from_pretrained(checkpoint, config=config, local_files_only=True)
    policy.to(device).eval()
    pre, post = make_pre_post_processors(policy.config, pretrained_path=str(checkpoint),
        preprocessor_overrides={"device_processor": {"device": device}},
        postprocessor_overrides={"device_processor": {"device": "cpu"}})
    dataset = LeRobotDataset("explorer/simulator-validation", root=folder / "validation", video_backend="pyav")
    if type(maximum_samples) is not int or maximum_samples < 1:
        raise ValueError("Validation sample limit must be positive")
    indices = np.unique(np.linspace(0, len(dataset) - 1, min(maximum_samples, len(dataset)), dtype=int)).tolist()
    selected = torch.utils.data.Subset(dataset, indices)
    errors, baseline = [], []
    with torch.no_grad():
        for batch in torch.utils.data.DataLoader(selected, batch_size=1, shuffle=False, num_workers=0):
            if cancelled and cancelled():
                raise InterruptedError("Simulator training cancelled during validation")
            expected = batch["action"].cpu().numpy().reshape(-1, 9)
            previous = batch["observation.state"].cpu().numpy().reshape(-1, 9)
            policy.reset()
            actual = post(policy.select_action(pre(batch))).cpu().numpy().reshape(-1, 9)
            errors.extend(np.abs(actual - expected).tolist())
            baseline.extend(np.abs(previous - expected).tolist())
    errors, baseline = np.asarray(errors), np.asarray(baseline)
    if not len(errors) or not np.isfinite(errors).all():
        raise ValueError("ACT validation returned no finite predictions")
    split = json.loads((folder / "split.json").read_text())
    scales = np.asarray([10.] * 6 + [.8, .72, 1.67])
    result = dict(samples=len(errors), heldout_independent=split["heldout_independent"],
        validation_total_frames=len(dataset), sampling="uniform_across_validation_frames",
        normalized_mae=float(np.mean(errors / scales)),
        normalized_hold_baseline_mae=float(np.mean(baseline / scales)),
        joint_mae_deg=float(np.mean(errors[:, :6])), base_mae_mixed=float(np.mean(errors[:, 6:])),
        improves_hold_baseline=bool(split["heldout_independent"] and np.mean(errors / scales) < np.mean(baseline / scales)),
        training_source="synthetic_only", domain="simulation", physics_acceptance=False,
        physical_success_evaluated=False, physical_success_verified=False, automatic_execution_allowed=False)
    write_json(folder / "validation.json", result)
    return result


def create_bundle(folder, state):
    folder = Path(folder)
    checkpoint = folder / "model/checkpoints/last/pretrained_model"
    files = {str(path.relative_to(checkpoint)): sha256(path) for path in checkpoint.rglob("*") if path.is_file()}
    required = {"model.safetensors", "config.json", "policy_preprocessor.json", "policy_postprocessor.json"}
    if not required.issubset(files):
        raise ValueError("The ACT checkpoint or its normalization processors are incomplete")
    config = json.loads((checkpoint / "config.json").read_text())
    bundle = dict(format="explorer_mobile_act_bundle_v2", created=time.time(), task=state["task"],
        skill_id=state.get("skill_id"), episodes=state["episodes"], dataset_kind="mobile_manipulation_9dof",
        dataset_fingerprint=state.get("dataset_fingerprint"),
        checkpoint_relative="model/checkpoints/last/pretrained_model", checkpoint_sha256=files,
        framework="lerobot", framework_version=state.get("runtime_versions", {}).get("lerobot", "0.6.1"),
        runtime_versions=state.get("runtime_versions", {}), policy="act", observation_order=ACTION_ORDER,
        action_order=ACTION_ORDER, units=UNITS, joint_state_source="command_estimate",
        sample_contract=SAMPLE_CONTRACT, sample_contract_sha256=CONTRACT_SHA256,
        camera_feature="observation.images.wrist", chunk_size=config.get("chunk_size", 1), action_steps=1,
        temporal_ensemble_coeff=config.get("temporal_ensemble_coeff"),
        dataset_fps=json.loads((folder / "train/explorer-provenance.json").read_text())["dataset_fps"],
        validation=state["validation"], training_source="synthetic_only", domain="simulation",
        training_goal=state.get("training_goal", "successful_simulation_imitation"),
        training_profile=state.get("training_profile", "compact_act_v1"),
        warm_start=state.get("warm_start"),
        physics_acceptance=False, physical_success_verified=False, autonomous_motion_on_registration=False)
    write_json(folder / "bundle.json", bundle)
    read_bundle(folder)
    return bundle


def split_episodes(folders, metadata):
    """Keep every recording of the held-out simulation scene out of training."""
    keys = []
    for folder, item in zip(folders, metadata):
        if item.get("source") == "simulation" and "seed" in item:
            if type(item["seed"]) is not int:
                raise ValueError("Simulator scene seed must be an integer")
            keys.append(("scene_seed", item["seed"]))
        else:
            # Copied manual recordings cannot masquerade as an independent
            # evaluation episode merely by being placed in another folder.
            keys.append(("recording", sha256(Path(folder) / "samples.jsonl")))
    # The final selected recording remains the held-out scene even when an
    # earlier recording of it appears before other unique scene seeds.
    groups = list(reversed(dict.fromkeys(reversed(keys))))
    if len(groups) < 2:
        return folders, folders, False
    heldout = set(groups[-max(1, len(groups) // 5):])
    return ([folder for folder, key in zip(folders, keys) if key not in heldout],
            [folder for folder, key in zip(folders, keys) if key in heldout], True)


def stage_warm_start(source_folder, destination, target_policy):
    """Preserve trained image/state weights while expanding bounded ACT queries."""
    from lerobot.policies.act import ACTPolicy
    from lerobot.policies.act.configuration_act import ACTConfig
    bundle, checkpoint = read_bundle(source_folder)
    source_config = ACTConfig.from_pretrained(checkpoint, local_files_only=True)
    source_config.device = "cpu"
    source = ACTPolicy.from_pretrained(checkpoint, config=source_config, local_files_only=True)
    config_values = dict(target_policy)
    config_values.pop("type")
    config_values["device"] = "cpu"
    # Typed feature objects are already verified by the existing bundle reader.
    config_values["input_features"] = source_config.input_features
    config_values["output_features"] = source_config.output_features
    target = ACTPolicy(ACTConfig(**config_values))
    source_weights, target_weights = source.state_dict(), target.state_dict()
    resized = []
    for key, value in target_weights.items():
        previous = source_weights.get(key)
        if previous is None:
            raise ValueError("Warm-start architecture does not match this ACT profile")
        if previous.shape == value.shape:
            target_weights[key] = previous
        elif key == "model.decoder_pos_embed.weight" and previous.shape[1:] == value.shape[1:]:
            # Distinct future queries retain the learned first query and the
            # new bounded horizon's positional differences.
            target_weights[key] = previous[:1].expand_as(value).clone() + .01 * (value - value[:1])
            resized.append(key)
        else:
            raise ValueError("Warm-start tensor shape mismatch: " + key)
    target.load_state_dict(target_weights, strict=True)
    target.save_pretrained(destination)
    for name in bundle["checkpoint_sha256"]:
        if name not in ("config.json", "model.safetensors"):
            copied = Path(destination) / name
            copied.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(checkpoint / name, copied)
    return dict(source_job=Path(source_folder).name,
                source_weights_sha256=bundle["checkpoint_sha256"]["model.safetensors"], resized_tensors=resized)


def train_job(job_folder, episode_folders, steps=3000, python_executable=None, device="auto", progress=None,
              task="pick and place", skill_id=None, cancelled=None, batch_size=16, seed=1000,
              diagnostic_only=False, warm_start=None):
    """Blocking worker; call from the simulator's background executor.

    The subprocess has no robot connection. A nonzero learner exit is a failed
    job, and never creates a bundle or reports a trained model.
    """
    from src.learning_environment import configure

    folder = Path(job_folder).resolve()
    folder.mkdir(parents=True, exist_ok=True)
    if (folder / "job.json").exists():
        raise ValueError("Training job folder already contains a job")
    folders = [Path(value).resolve() for value in episode_folders]
    state = dict(id=folder.name, created=time.time(), at=time.time(), state="queued", steps=steps,
        task=task, skill_id=skill_id, episodes=[item.name for item in folders], batch_size=batch_size, seed=seed,
        framework="lerobot", policy="act", dataset_kind="mobile_manipulation_9dof",
        training_source="synthetic_only", domain="simulation", physics_acceptance=False,
        training_goal="diagnostic_pipeline_smoke" if diagnostic_only else "successful_simulation_imitation",
        training_profile="temporal_compact_act_v3",
        physical_success_verified=False, automatic_execution=False, executed=False)
    child = None

    def update(phase, **values):
        state.update(state=phase, at=time.time(), **values)
        write_json(folder / "job.json", state)
        if progress:
            progress(dict(state))

    def check_cancelled():
        if cancelled and cancelled():
            raise InterruptedError("Simulator training cancelled")

    try:
        check_cancelled()
        if not folders:
            raise ValueError("No simulator trajectories were selected")
        if len(set(folders)) != len(folders) or len({item.name for item in folders}) != len(folders):
            raise ValueError("Duplicate trajectories cannot form independent train/validation episodes")
        training_config(folder, steps, "cpu", batch_size, seed)  # Validate request before a heavy import/export.
        update("exporting")
        from importlib.metadata import version
        state["runtime_versions"] = {name: version(name) for name in ("lerobot", "torch", "torchvision")}
        if state["runtime_versions"]["lerobot"] != "0.6.1":
            raise ValueError("Use the Explorer simulator's pinned LeRobot 0.6.1 environment")
        provenance = [inspect_episode(item) for item in folders]
        if not diagnostic_only and any(item[0].get("physics_verified_success", item[0].get("success")) is not True for item in provenance):
            raise ValueError("ACT task training requires successful simulator demonstrations; failed attempts remain diagnostic data")
        state["dataset_fingerprint"] = hashlib.sha256(json.dumps([item[3] for item in provenance], sort_keys=True).encode()).hexdigest()
        environment = configure(folder)
        state["device"] = choose_device(device)
        check_cancelled()
        training, validation, heldout = split_episodes(folders, [item[0] for item in provenance])
        export_dataset(training, folder / "train", "explorer/simulator-train", task, cancelled=cancelled)
        check_cancelled()
        export_dataset(validation, folder / "validation", "explorer/simulator-validation", task, cancelled=cancelled)
        check_cancelled()
        write_json(folder / "split.json", dict(train=[item.name for item in training],
            validation=[item.name for item in validation], heldout_independent=bool(heldout)))
        config = training_config(folder, steps, state["device"], batch_size, seed)
        if warm_start:
            check_cancelled()
            source_bundle, _ = read_bundle(warm_start)
            if source_bundle.get("dataset_fingerprint") != state["dataset_fingerprint"]:
                raise ValueError("Warm start requires the same verified dataset fingerprint and normalization source")
            state["warm_start"] = stage_warm_start(warm_start, folder / "initial-policy", config["policy"])
            config["policy"]["pretrained_path"] = str(folder / "initial-policy")
        write_json(folder / "train-config.json", config)
        write_json(folder / "train-config.environment.json", environment)
        interpreter = str(python_executable or sys.executable)
        env = dict(os.environ, HF_HUB_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1", WANDB_MODE="disabled",
                   PYTORCH_ENABLE_MPS_FALLBACK="1", OMP_NUM_THREADS="2", PYTHONUNBUFFERED="1")
        update("training")
        with (folder / "train.log").open("a") as log:
            check_cancelled()
            child = subprocess.Popen([interpreter, "-m", "lerobot.scripts.lerobot_train",
                "--config_path=" + str(folder / "train-config.json")], stdout=log, stderr=log, env=env,
                start_new_session=(os.name == "posix"))
            while child.poll() is None:
                if cancelled and cancelled():
                    raise InterruptedError("Simulator training cancelled")
                try:
                    with (folder / "train.log").open("rb") as tail:
                        tail.seek(max(0, (folder / "train.log").stat().st_size - 16000))
                        content = tail.read().decode("utf8", errors="replace")
                    # LeRobot abbreviates its metric steps as 1K/2K. tqdm's
                    # counter remains exact and must take precedence.
                    completed = re.findall(r"(\d+)/" + str(steps) + r"\s", content)
                    if completed:
                        state["steps_completed"] = max(state.get("steps_completed", 0), min(steps, int(completed[-1])))
                        state["progress"] = round(90 * state["steps_completed"] / steps, 1)
                except OSError:
                    pass
                update("training")
                time.sleep(.5)
            if child.returncode:
                raise RuntimeError("LeRobot training failed; see train.log (exit " + str(child.returncode) + ")")
        update("validating", steps_completed=steps, progress=95.)
        check_cancelled()
        state["validation"] = validate_checkpoint(folder, state["device"], cancelled=cancelled)
        check_cancelled()
        create_bundle(folder, state)
        update("validated_simulation" if state["validation"]["improves_hold_baseline"] else "trained_simulation_unvalidated", bundle="bundle.json", finished=time.time(), progress=100.)
    except (Exception, KeyboardInterrupt) as exc:
        update("cancelled" if isinstance(exc, (InterruptedError, KeyboardInterrupt)) else "failed",
               error=str(exc) or "Simulator training interrupted", finished=time.time())
    finally:
        if child and child.poll() is None:
            if os.name == "posix":
                try:
                    os.killpg(child.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
            else:
                child.terminate()
            try:
                child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                if os.name == "posix":
                    try:
                        os.killpg(child.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                else:
                    child.kill()
                child.wait()
    return state


def main():
    parser = argparse.ArgumentParser(description="Train compact ACT on isolated simulator trajectories")
    parser.add_argument("job_folder", type=Path)
    parser.add_argument("episodes", type=Path, nargs="+")
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--device", choices=("auto", "mps", "cpu", "cuda"), default="auto")
    parser.add_argument("--task", default="pick and place")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--warm-start", type=Path, help="Verified simulator ACT job to initialize the bounded temporal policy")
    parser.add_argument("--diagnostic-only", action="store_true", help="Allow failed/unverified trajectories for pipeline smoke; never task acceptance")
    args = parser.parse_args()
    state = train_job(args.job_folder, args.episodes, steps=args.steps, device=args.device, task=args.task,
                      batch_size=args.batch_size, seed=args.seed, diagnostic_only=args.diagnostic_only, warm_start=args.warm_start)
    print(json.dumps(state, ensure_ascii=False))
    return 0 if state["state"] in ("validated_simulation", "trained_simulation_unvalidated") else 1


if __name__ == "__main__":
    raise SystemExit(main())
