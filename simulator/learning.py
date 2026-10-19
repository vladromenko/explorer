"""Isolated simulator data export and official LeRobot ACT training.

Actor inputs are wrist RGB, nine commanded coordinates and a declared internal clock.
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

from src.mobile_policy_contract import ACTION_ORDER, UNITS, SAMPLE_CONTRACT, CONTRACT_SHA256, read_bundle, ACTOR_CLOCK, ACTOR_CLOCK_FOURIER, clock_features, ACTOR_ACTION_TRANSFORM, decode_policy_action


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
        if type(row.get('training_target_valid',True)) is not bool:
            raise ValueError('Invalid expert target mask')
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


def expert_segments(rows, fps):
    """Keep original clocks and prevent action chunks crossing a learner handoff."""
    segments=[];current=[]
    for index,row in enumerate(rows):
        elapsed=float(row.get('actor_elapsed_s',index/fps))
        if not math.isfinite(elapsed) or elapsed<0:
            raise ValueError('Invalid recorded actor clock')
        if index and abs(elapsed-float(rows[index-1].get('actor_elapsed_s',(index-1)/fps))-1/fps)>1e-5:
            raise ValueError('Actor clock is discontinuous')
        if row.get('training_target_valid',True):current.append((elapsed,row))
        else:
            if len(current)>=3:segments.append(current)
            current=[]
    if len(current)>=3:segments.append(current)
    if not segments:raise ValueError('No contiguous expert-labelled segment')
    return segments


def export_dataset(episode_folders, destination, repo_id="explorer/simulator", task="pick and place", cancelled=None, correction_replays=1):
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
        ACTOR_CLOCK["feature"]: {"dtype": "float32", "shape": (14,), "names": ACTOR_CLOCK_FOURIER["components"]},
    }
    fps = next(iter(rates))
    if type(correction_replays) is not int or not 1<=correction_replays<=8:
        raise ValueError('Invalid corrective replay weight')
    dataset = LeRobotDataset.create(repo_id=repo_id, root=destination, fps=fps,
        features=features, robot_type="rosmaster_m3pro_mobile_manipulation_simulated_command_estimate",
        use_videos=False, image_writer_threads=2)
    provenance = []
    try:
        for folder, (episode, rows, _, inputs) in zip(folders, episodes):
            alignment = episode.get("action_alignment", "pre_action_observation")
            if alignment not in ("pre_action_observation", "same_row_pre_action"):
                raise ValueError("Expected a pre-action observation and its next applied command")
            segments=expert_segments(rows,fps)
            repeats=correction_replays if episode.get('learner_intervention') or episode.get('corrective_demonstration') is True else 1
            for _ in range(repeats):
                for segment in segments:
                    for elapsed,current in segment:
                        if cancelled and cancelled():
                            raise InterruptedError("Simulator training cancelled during dataset export")
                        with Image.open(_contained(folder, current["image"])) as image:
                            rgb = np.asarray(image.convert("RGB").resize((320, 240))).copy()
                        observation_state = np.asarray(_vector(current.get("observation_state", current.get("command"))), dtype=np.float32)
                        training_action = np.asarray(_vector(current["applied_action"]), dtype=np.float32)
                        dataset.add_frame({
                            "observation.state": observation_state,
                            "action": training_action,
                            "observation.images.wrist": rgb, "task": task or episode.get("name", "pick and place"),
                            ACTOR_CLOCK["feature"]: np.asarray(clock_features({**ACTOR_CLOCK_FOURIER,"period_s":1/fps},elapsed), dtype=np.float32),
                        })
                    dataset.save_episode()
            provenance.append({"episode": episode, "input_sha256": inputs,
                               "samples": sum(len(s) for s in segments)*repeats,
                               "recorded_frames":len(rows),'expert_segments':len(segments),
                               'replay_weight':repeats,"action_alignment": "pre_action_observation"})
        dataset.finalize()
    finally:
        if dataset.writer is not None:
            dataset.writer.stop_image_writer()
    result = dict(format="explorer_simulator_dataset_v1", episodes=provenance, dataset_fps=fps,
        training_source="synthetic_only", domain="simulation", joint_state_source="command_estimate",
        sample_contract=SAMPLE_CONTRACT, sample_contract_sha256=CONTRACT_SHA256,
        actor_inputs=["observation.state", "observation.images.wrist", ACTOR_CLOCK["feature"]],
        actor_context={**ACTOR_CLOCK_FOURIER, "period_s": 1/fps},
        action_transform=None,
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


def training_budget(requested_steps, frames, batch_size, minimum_epochs=0,maximum_steps=200000):
    """Keep the requested floor while scaling automatic training with real data."""
    if (type(maximum_steps) is not int or not 200000<=maximum_steps<=1000000
            or type(requested_steps) is not int or not 1<=requested_steps<=maximum_steps
            or type(frames) is not int or frames<1 or type(batch_size) is not int or not 1<=batch_size<=32
            or type(minimum_epochs) is not int or not 0<=minimum_epochs<=20):
        raise ValueError('Invalid training exposure budget')
    required=math.ceil(frames*minimum_epochs/batch_size)
    effective=min(maximum_steps,max(requested_steps,required))
    return dict(requested_steps=requested_steps,effective_steps=effective,
                training_frames=frames,batch_size=batch_size,minimum_epochs=minimum_epochs,
                expected_dataset_passes=effective*batch_size/frames,
                step_cap_reached=required>maximum_steps)


class LoadedPolicy:
    """Image/command-only actor callable for independent simulator rollouts."""
    def __init__(self, job_folder, device="cpu"):
        import torch
        from lerobot.policies.act import ACTPolicy
        from lerobot.policies.act.configuration_act import ACTConfig
        from lerobot.policies import make_pre_post_processors

        self.torch = torch
        self.device = choose_device(device)
        self.bundle, checkpoint = read_bundle(job_folder)
        config = ACTConfig.from_pretrained(checkpoint, local_files_only=True)
        config.device = self.device
        config.pretrained_backbone_weights = None  # Saved checkpoint includes the entire backbone.
        self.policy = ACTPolicy.from_pretrained(checkpoint, config=config, local_files_only=True)
        self.policy.to(self.device).eval()
        self.pre, self.post = make_pre_post_processors(self.policy.config, pretrained_path=str(checkpoint),
            preprocessor_overrides={"device_processor": {"device": self.device}},
            postprocessor_overrides={"device_processor": {"device": "cpu"}})
        self.reset()

    def reset(self):
        self.policy.reset()
        self.elapsed_s = 0.
        self.last_action_was_queued=False
        self.last_action_elapsed_s=None

    def replan(self,state9,rgb):
        """Discard a refused queued tail; recompute this same decision once."""
        if not self.last_action_was_queued or self.last_action_elapsed_s is None:
            raise ValueError('Only a cached action may be replanned')
        self.policy.reset()
        self.elapsed_s=self.last_action_elapsed_s
        return self(state9,rgb)

    def __call__(self, state9, rgb):
        import numpy as np
        state = _vector(list(state9))
        image = np.asarray(rgb)
        if image.shape != (240, 320, 3) or image.dtype != np.uint8:
            raise ValueError("The actor requires a 320x240 RGB uint8 wrist frame")
        batch = {"observation.state": self.torch.tensor([state], dtype=self.torch.float32),
                 "observation.images.wrist": self.torch.from_numpy(image.copy()).permute(2, 0, 1).float().div(255).unsqueeze(0)}
        context = self.bundle.get("actor_context")
        if context:
            batch[context["feature"]] = self.torch.tensor([clock_features(context,self.elapsed_s)], dtype=self.torch.float32)
        with self.torch.no_grad():
            self.last_action_was_queued=bool(len(getattr(self.policy,'_action_queue',())))
            self.last_action_elapsed_s=self.elapsed_s
            action = self.post(self.policy.select_action(self.pre(batch))).cpu().numpy().reshape(-1)
        if action.shape != (9,) or not np.isfinite(action).all():
            raise ValueError("ACT produced invalid commands")
        self.elapsed_s += context["period_s"] if context else .05
        return np.asarray(decode_policy_action(state, action, self.bundle), dtype=np.float32)


def load_policy(job_folder, device="cpu"):
    return LoadedPolicy(job_folder, device)


def training_config(folder, steps, device, batch_size=16, seed=1000,maximum_steps=200000):
    if type(maximum_steps) is not int or not 200000<=maximum_steps<=1000000 or type(steps) is not int or not 1 <= steps <= maximum_steps:
        raise ValueError("Training steps exceed the explicit training budget")
    if type(batch_size) is not int or not 1 <= batch_size <= 32:
        raise ValueError("Batch size must be between 1 and 32")
    if type(seed) is not int or not 0 <= seed < 2 ** 32:
        raise ValueError("Training seed must be an unsigned 32-bit integer")
    folder = Path(folder).resolve()
    return dict(dataset=dict(repo_id="explorer/simulator-train", root=str(folder / "train"), video_backend="pyav"),
        policy=dict(type="act", device=device, push_to_hub=False, chunk_size=20, n_action_steps=20,
            temporal_ensemble_coeff=None,
            pretrained_backbone_weights="ResNet18_Weights.IMAGENET1K_V1", dim_model=128, n_heads=4, dim_feedforward=512,
            n_encoder_layers=2, n_decoder_layers=1, n_vae_encoder_layers=2,
            use_vae=False, dropout=0., optimizer_lr=3e-4, optimizer_lr_backbone=1e-4,
            input_features={"observation.state": {"type": "STATE", "shape": [9]},
                "observation.images.wrist": {"type": "VISUAL", "shape": [3, 240, 320]},
                ACTOR_CLOCK["feature"]: {"type": "ENV", "shape": [14]}},
            normalization_mapping={"VISUAL": "MEAN_STD", "STATE": "MEAN_STD", "ACTION": "MEAN_STD", "ENV": "IDENTITY"},
            output_features={"action": {"type": "ACTION", "shape": [9]}}),
        output_dir=str(folder / "model"), batch_size=batch_size, num_workers=2, steps=steps,
        save_freq=min(steps, max(100, min(1000,steps//10))), log_freq=min(10, steps), env_eval_freq=0,
        wandb=dict(enable=False), seed=seed)


def chunk_error_metrics(expected, predicted, previous, valid):
    """Score every non-padded future command, including the executed chunk tail."""
    import numpy as np
    expected, predicted, previous = map(lambda value: np.asarray(value, dtype=float),
                                        (expected, predicted, previous))
    valid = np.asarray(valid, dtype=bool)
    if (expected.ndim != 3 or expected.shape != predicted.shape or expected.shape[-1] != 9
            or previous.shape != (expected.shape[0],9) or valid.shape != expected.shape[:2]
            or not valid.any() or not all(np.isfinite(value).all() for value in (expected,predicted,previous))):
        raise ValueError('Invalid future-action validation batch')
    scales=np.asarray([10.]*6+[.8,.72,1.67])
    errors=abs(predicted-expected)
    hold=abs(previous[:,None,:]-expected)
    by_step=[]
    for index in range(expected.shape[1]):
        mask=valid[:,index]
        by_step.append(dict(step=index,valid_commands=int(mask.sum()),
            joint_mae_deg=float(errors[mask,index,:6].mean()) if mask.any() else None,
            normalized_mae=float((errors[mask,index,:]/scales).mean()) if mask.any() else None))
    return dict(prediction_horizon=expected.shape[1],valid_commands=int(valid.sum()),
        normalized_mae=float((errors[valid]/scales).mean()),
        normalized_hold_baseline_mae=float((hold[valid]/scales).mean()),
        improves_hold_baseline=bool((errors[valid]/scales).mean() < (hold[valid]/scales).mean()),
        joint_mae_deg=float(errors[valid,:6].mean()),
        per_channel_mae=errors[valid].mean(axis=0).tolist(),
        by_prediction_step=by_step)


def validate_checkpoint(folder, device="cpu", maximum_samples=128, cancelled=None, output_path=None, checkpoint=None):
    import numpy as np
    import torch
    from lerobot.datasets import LeRobotDataset
    from lerobot.policies.act import ACTPolicy
    from lerobot.policies.act.configuration_act import ACTConfig
    from lerobot.policies import make_pre_post_processors

    folder = Path(folder).resolve()
    checkpoint = Path(checkpoint).resolve() if checkpoint else (folder / "model/checkpoints/last/pretrained_model").resolve()
    if not checkpoint.is_relative_to(folder):
        raise ValueError('Validation checkpoint must belong to this job')
    config = ACTConfig.from_pretrained(checkpoint, local_files_only=True)
    config.device = device
    config.pretrained_backbone_weights = None
    policy = ACTPolicy.from_pretrained(checkpoint, config=config, local_files_only=True)
    policy.to(device).eval()
    pre, post = make_pre_post_processors(policy.config, pretrained_path=str(checkpoint),
        preprocessor_overrides={"device_processor": {"device": device}},
        postprocessor_overrides={"device_processor": {"device": "cpu"}})
    provenance = json.loads((folder / "train/explorer-provenance.json").read_text())
    fps=float(provenance['dataset_fps'])
    dataset = LeRobotDataset("explorer/simulator-validation", root=folder / "validation", video_backend="pyav",
                            delta_timestamps={'action':[index/fps for index in range(config.chunk_size)]})
    if type(maximum_samples) is not int or maximum_samples < 1:
        raise ValueError("Validation sample limit must be positive")
    indices = np.unique(np.linspace(0, len(dataset) - 1, min(maximum_samples, len(dataset)), dtype=int)).tolist()
    selected = torch.utils.data.Subset(dataset, indices)
    transform = provenance.get("action_transform")
    errors, baseline = [], []
    future_expected, future_actual, references, masks = [], [], [], []
    with torch.no_grad():
        for batch in torch.utils.data.DataLoader(selected, batch_size=1, shuffle=False, num_workers=0):
            if cancelled and cancelled():
                raise InterruptedError("Simulator training cancelled during validation")
            expected = batch["action"].cpu().numpy().reshape(-1,config.chunk_size,9)
            previous = batch["observation.state"].cpu().numpy().reshape(-1, 9)
            policy.reset()
            actual = post(policy.predict_action_chunk(pre(batch))).cpu().numpy().reshape(-1,config.chunk_size,9)
            if transform:
                expected[:, :, :6] += previous[:, None, :6]
                actual[:, :, :6] += previous[:, None, :6]
            errors.extend(np.abs(actual[:,0] - expected[:,0]).tolist())
            baseline.extend(np.abs(previous - expected[:,0]).tolist())
            future_expected.extend(expected.tolist())
            future_actual.extend(actual.tolist())
            references.extend(previous.tolist())
            masks.extend((~batch['action_is_pad']).cpu().numpy().reshape(-1,config.chunk_size).tolist())
    errors, baseline = np.asarray(errors), np.asarray(baseline)
    if not len(errors) or not np.isfinite(errors).all():
        raise ValueError("ACT validation returned no finite predictions")
    split = json.loads((folder / "split.json").read_text())
    scales = np.asarray([10.] * 6 + [.8, .72, 1.67])
    full_chunk=chunk_error_metrics(future_expected,future_actual,references,masks)
    result = dict(validation_kind='offline_action_prediction', samples=len(errors), heldout_independent=split["heldout_independent"],
        validation_total_frames=len(dataset), sampling="uniform_across_validation_frames",
        normalized_mae=float(np.mean(errors / scales)),
        normalized_hold_baseline_mae=float(np.mean(baseline / scales)),
        joint_mae_deg=float(np.mean(errors[:, :6])), base_mae_mixed=float(np.mean(errors[:, 6:])),
        full_chunk=full_chunk,
        applied_before_next_observation=config.n_action_steps,
        first_action_improves_hold_baseline=bool(split["heldout_independent"] and np.mean(errors / scales) < np.mean(baseline / scales)),
        improves_hold_baseline=bool(split["heldout_independent"] and full_chunk['improves_hold_baseline']),
        baseline_comparison='complete_prediction_horizon_excluding_padding',
        evaluated_checkpoint_relative=str(checkpoint.relative_to(folder)),
        training_source="synthetic_only", domain="simulation", physics_acceptance=False,
        physical_success_evaluated=False, physical_success_verified=False, automatic_execution_allowed=False)
    write_json(output_path or folder / "validation.json", result)
    return result


def select_checkpoint(folder, device='cpu', maximum_samples=128, cancelled=None, progress=None):
    """Select by independent recorded-action error, never final task test scenes."""
    folder=Path(folder).resolve()
    split=json.loads((folder/'split.json').read_text())
    checkpoints=folder/'model/checkpoints'
    candidates=sorted(path/'pretrained_model' for path in checkpoints.iterdir()
                      if path.is_dir() and path.name.isdecimal() and (path/'pretrained_model').is_dir())
    if split.get('heldout_independent') is not True or not candidates:
        candidates=[checkpoints/'last/pretrained_model']
    scores=[]
    best=None
    for index,checkpoint in enumerate(candidates):
        if cancelled and cancelled():
            raise InterruptedError('Simulator checkpoint selection cancelled')
        if progress:
            progress(f'Проверка checkpoint {index+1} из {len(candidates)}')
        report=validate_checkpoint(folder,device,maximum_samples,cancelled,
            output_path=folder/'checkpoint-validation'/f'{checkpoint.parent.name}.json', checkpoint=checkpoint)
        score=report['full_chunk']['normalized_mae']
        if not math.isfinite(score):
            raise ValueError('Non-finite checkpoint selection metric')
        relative=str(checkpoint.relative_to(folder))
        scores.append(dict(checkpoint_relative=relative,normalized_mae=score))
        if best is None or score<best[0]:
            best=(score,relative,report)
    result=dict(criterion='independent_recorded_full_chunk_normalized_mae',
                heldout_independent=split.get('heldout_independent') is True,
                final_task_test_used=False,candidates=scores,checkpoint_relative=best[1])
    write_json(folder/'checkpoint-selection.json',result)
    write_json(folder/'validation.json',best[2])
    return result,best[2]


def create_bundle(folder, state):
    folder = Path(folder)
    relative=Path(state.get('checkpoint_relative','model/checkpoints/last/pretrained_model'))
    checkpoint=(folder/relative).resolve()
    if relative.is_absolute() or '..' in relative.parts or not checkpoint.is_relative_to(folder.resolve()):
        raise ValueError('Selected checkpoint must belong to this job')
    files = {str(path.relative_to(checkpoint)): sha256(path) for path in checkpoint.rglob("*") if path.is_file()}
    required = {"model.safetensors", "config.json", "policy_preprocessor.json", "policy_postprocessor.json"}
    if not required.issubset(files):
        raise ValueError("The ACT checkpoint or its normalization processors are incomplete")
    config = json.loads((checkpoint / "config.json").read_text())
    bundle = dict(format="explorer_mobile_act_bundle_v2", created=time.time(), task=state["task"],
        skill_id=state.get("skill_id"), episodes=state["episodes"], dataset_kind="mobile_manipulation_9dof",
        dataset_fingerprint=state.get("dataset_fingerprint"),
        checkpoint_relative=str(relative), checkpoint_sha256=files,
        framework="lerobot", framework_version=state.get("runtime_versions", {}).get("lerobot", "0.6.1"),
        runtime_versions=state.get("runtime_versions", {}), policy="act", observation_order=ACTION_ORDER,
        action_order=ACTION_ORDER, units=UNITS, joint_state_source="command_estimate",
        sample_contract=SAMPLE_CONTRACT, sample_contract_sha256=CONTRACT_SHA256,
        camera_feature="observation.images.wrist", chunk_size=config.get("chunk_size", 1), action_steps=config.get("n_action_steps",1),
        inference_execution={"kind":"bounded_action_queue","actions_per_call":1,"queue_steps":config.get("n_action_steps",1)} if config.get("n_action_steps",1)>1 else None,
        temporal_ensemble_coeff=config.get("temporal_ensemble_coeff"),
        actor_context=json.loads((folder / "train/explorer-provenance.json").read_text()).get("actor_context"),
        action_transform=json.loads((folder / "train/explorer-provenance.json").read_text()).get("action_transform"),
        dataset_fps=json.loads((folder / "train/explorer-provenance.json").read_text())["dataset_fps"],
        validation=state["validation"], training_source="synthetic_only", domain="simulation",
        training_goal=state.get("training_goal", "successful_simulation_imitation"),
        training_profile=state.get("training_profile", "compact_act_v1"),
        warm_start=state.get("warm_start"),
        checkpoint_selection=state.get('checkpoint_selection'),
        training_budget=state.get('training_budget'),
        scene_profile=state.get('scene_profile','baseline'), scene_coverage=state.get('scene_coverage'),
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
    source_config.pretrained_backbone_weights = None
    source = ACTPolicy.from_pretrained(checkpoint, config=source_config, local_files_only=True)
    config_values = dict(target_policy)
    config_values.pop("type")
    config_values["device"] = "cpu"
    config_values["pretrained_backbone_weights"] = None
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


def inherited_split(folders, metadata, source_folder, force_training_seeds=()):
    """Keep source training scenes out of validation after dataset expansion."""
    source_folder=Path(source_folder)
    train=json.loads((source_folder/'train/explorer-provenance.json').read_text())
    validation=json.loads((source_folder/'validation/explorer-provenance.json').read_text())
    def identity(episode):
        seed=episode.get('seed')
        if episode.get('source')!='simulation' or type(seed) is not int:
            raise ValueError('Inherited simulation split requires explicit scene identities')
        return seed
    exposed={identity(item['episode']) for item in train['episodes']}
    heldout={identity(item['episode']) for item in validation['episodes']}
    if exposed & heldout:raise ValueError('Source training and validation overlap')
    forced=set(force_training_seeds)
    if any(type(seed) is not int for seed in forced) or forced&heldout:
        raise ValueError('Corrective training cannot reuse inherited validation scenes')
    training=[];testing=[]
    for folder,episode in zip(folders,metadata):
        seed=identity(episode)
        is_heldout=seed in heldout or seed not in exposed and seed not in forced and int(hashlib.sha256(str(seed).encode()).hexdigest(),16)%5==0
        (testing if is_heldout else training).append(folder)
    if not testing or not training:raise ValueError('Inherited split needs independent train and validation scenes')
    return training,testing,True


def completed_training_steps(content, total, resumed_step=0):
    """LeRobot's resumed tqdm counts remaining updates from zero."""
    full=re.findall(r'(\d+)/'+str(total)+r'\s',content)
    remaining=re.findall(r'(\d+)/'+str(total-resumed_step)+r'\s',content) if resumed_step else []
    if full:return min(total,int(full[-1]))
    if remaining:return min(total,resumed_step+int(remaining[-1]))
    return resumed_step


def resume_checkpoint(folder, episode_folders, skill_id):
    """Verify the saved optimizer checkpoint and the unchanged exported dataset."""
    folder=Path(folder).resolve()
    state=json.loads((folder/'job.json').read_text())
    if state.get('state') not in ('cancelled','failed','training'):
        raise ValueError('Only interrupted training can be resumed')
    if state.get('skill_id')!=skill_id or sorted(state['episodes'])!=sorted(Path(p).name for p in episode_folders):
        raise ValueError('Resume requires the same skill and demonstration set')
    checkpoint=(folder/'model/checkpoints/last').resolve()
    if not checkpoint.is_relative_to(folder/'model/checkpoints'):
        raise ValueError('Resume checkpoint is outside the training job')
    required=('pretrained_model/model.safetensors','pretrained_model/train_config.json',
              'training_state/optimizer_state.safetensors','training_state/optimizer_param_groups.json',
              'training_state/rng_state.safetensors','training_state/training_step.json')
    if any(not (checkpoint/name).is_file() or not (checkpoint/name).stat().st_size for name in required):
        raise ValueError('Resume requires model, optimizer and RNG checkpoint files')
    config=json.loads((checkpoint/'pretrained_model/train_config.json').read_text())
    step=json.loads((checkpoint/'training_state/training_step.json').read_text())['step']
    if (Path(config['output_dir']).resolve()!=folder/'model' or
            Path(config['dataset']['root']).resolve()!=folder/'train' or
            config['steps']!=state['steps'] or config['batch_size']!=state['batch_size'] or
            type(step) is not int or not 0<step<=state['steps']):
        raise ValueError('Resume configuration does not match the saved job')
    for part in ('train','validation'):
        if not (folder/part/'explorer-provenance.json').is_file():
            raise ValueError('Saved exported dataset is missing')
    pause=folder/'process-pause.json'
    if pause.is_file():
        evidence=json.loads(pause.read_text())
        if evidence.get('checkpoint_path')==str(checkpoint):
            for name,digest in evidence.get('checkpoint_sha256',{}).items():
                if sha256(_contained(checkpoint,name))!=digest:
                    raise ValueError('Saved checkpoint checksum mismatch')
    state.update(resumed_step=step,steps_completed=step,progress=round(90*step/state['steps'],1))
    for name in ('error','finished'):
        state.pop(name,None)
    return state,checkpoint/'pretrained_model/train_config.json'


def prepared_training_available(folder):
    """A completed export, paused before the first durable gradient step."""
    folder=Path(folder)
    try:
        state=json.loads((folder/'job.json').read_text())
        required=('train-config.json','split.json','initial-policy/model.safetensors',
            'train/explorer-provenance.json','validation/explorer-provenance.json',
            'train/meta/info.json','validation/meta/info.json')
        checkpoints=folder/'model/checkpoints'
        return (state.get('state') in ('cancelled','failed','training','verifying_prepared_export') and state.get('steps_completed',0)==0
            and not (folder/'bundle.json').exists() and not (folder/'model').exists()
            and not (checkpoints.exists() and any(checkpoints.iterdir()))
            and all((folder/name).is_file() for name in required))
    except (OSError,ValueError):return False


def resume_prepared_training(folder,episode_folders,skill_id,*,cancelled=None,progress=None):
    """Verify and reuse existing data/initial weights, never re-export or delete.

    This is not an optimizer resume: no gradient checkpoint existed yet. Every
    exported shard is hashed, partitions checked, and source record bytes bound
    to the original export provenance before a learner may start from step zero.
    """
    folder=Path(folder).resolve();folders=[Path(p).resolve() for p in episode_folders]
    if not prepared_training_available(folder):raise ValueError('No complete prepared training export to reuse')
    state=json.loads((folder/'job.json').read_text());config=json.loads((folder/'train-config.json').read_text())
    def report(**values):
        if progress:progress(dict(state,state='verifying_prepared_export',message='Проверяю сохранённый датасет и начальные веса перед продолжением',**values))
    report(verified_files=0)
    if state.get('skill_id')!=skill_id or sorted(state['episodes'])!=sorted(p.name for p in folders) or len({p.name for p in folders})!=len(folders):
        raise ValueError('Prepared training requires the same skill and demonstrations')
    if (Path(config['output_dir']).resolve()!=folder/'model' or Path(config['dataset']['root']).resolve()!=folder/'train'
            or Path(config['policy']['pretrained_path']).resolve()!=folder/'initial-policy'
            or config['steps']!=state['steps'] or config['batch_size']!=state['batch_size'] or config['seed']!=state['seed']):
        raise ValueError('Prepared training configuration changed')
    selected=json.loads((folder/'split.json').read_text());sources={p.name:p for p in folders};inputs={};seeds=[]
    for part in ('train','validation'):
        provenance=json.loads((folder/part/'explorer-provenance.json').read_text())
        info=json.loads((folder/part/'meta/info.json').read_text());rows=provenance['episodes']
        if (provenance.get('sample_contract_sha256')!=CONTRACT_SHA256 or provenance.get('action_transform') is not None
                or sorted(selected[part])!=sorted(row['episode']['id'] for row in rows)
                or info['total_frames']!=sum(row['samples'] for row in rows)):
            raise ValueError('Prepared exported partition or frame count changed')
        seeds.append({row['episode']['seed'] for row in rows})
        for row in rows:
            if cancelled and cancelled():raise InterruptedError('Prepared export verification paused')
            identity=row['episode']['id']
            if identity not in sources or identity in inputs:raise ValueError('Prepared partition overlaps or contains unknown source')
            for name in ('episode.json','samples.jsonl'):
                if sha256(_contained(sources[identity],name))!=row['input_sha256'][name]:
                    raise ValueError('Prepared source recording bytes changed: '+identity+'/'+name)
            inputs[identity]=row['input_sha256']
    if seeds[0]&seeds[1] or set(inputs)!=set(sources):raise ValueError('Prepared training partitions are not independent/complete')
    fingerprint=hashlib.sha256(json.dumps([inputs[p.name] for p in folders],sort_keys=True).encode()).hexdigest()
    if fingerprint!=state['dataset_fingerprint']:raise ValueError('Prepared dataset fingerprint changed')
    files={}
    for part in ('train','validation','initial-policy'):
        for path in sorted((folder/part).rglob('*')):
            if cancelled and cancelled():raise InterruptedError('Prepared export verification paused')
            if path.is_file():
                files[str(path.relative_to(folder))]=sha256(path)
                report(verified_files=len(files))
    for name in ('split.json','train-config.json'):files[name]=sha256(folder/name)
    receipt=folder/'prepared-export-reuse.json'
    if receipt.exists() and json.loads(receipt.read_text())['files']!=files:
        raise ValueError('Prepared exported data or initial weights checksum changed')
    if not receipt.exists():write_json(receipt,dict(files=files,dataset_fingerprint=fingerprint,
        saved_job=state,at=time.time(),optimizer_checkpoint_existed=False))
    state.update(prepared_export_reused=True,export_reuse_receipt=str(receipt),steps_completed=0,resumed_step=0,progress=0.)
    for name in ('error','finished'):state.pop(name,None)
    return state,folder/'train-config.json'


def train_job(job_folder, episode_folders, steps=3000, python_executable=None, device="auto", progress=None,
              task="pick and place", skill_id=None, cancelled=None, batch_size=16, seed=1000,
              diagnostic_only=False, warm_start=None, minimum_epochs=0, resume=False, correction_replays=1, corrective_training_seeds=(),maximum_steps=200000,
              reuse_prepared=False,monitor_owner=None):
    """Blocking worker; call from the simulator's background executor.

    The subprocess has no robot connection. A nonzero learner exit is a failed
    job, and never creates a bundle or reports a trained model.
    """
    from src.learning_environment import configure

    folder = Path(job_folder).resolve()
    folder.mkdir(parents=True, exist_ok=True)
    if resume and reuse_prepared:raise ValueError('Choose checkpoint resume or prepared export reuse')
    if (folder / "job.json").exists() and not (resume or reuse_prepared):
        raise ValueError("Training job folder already contains a job")
    folders = [Path(value).resolve() for value in episode_folders]
    state = dict(id=folder.name, created=time.time(), at=time.time(), state="queued", steps=steps,
        task=task, skill_id=skill_id, episodes=[item.name for item in folders], batch_size=batch_size, seed=seed,
        framework="lerobot", policy="act", dataset_kind="mobile_manipulation_9dof",
        training_source="synthetic_only", domain="simulation", physics_acceptance=False,
        training_goal="diagnostic_pipeline_smoke" if diagnostic_only else "successful_simulation_imitation",
        training_profile="fourier_timed_visual_act_checkpoint_selection_v7",
        physical_success_verified=False, automatic_execution=False, executed=False)
    resume_config = None
    if resume:
        state, resume_config = resume_checkpoint(folder, folders, skill_id)
    if reuse_prepared:
        def preparing(value):
            write_json(folder/'job.json',value)
            if progress:progress(value)
        try:state,_=resume_prepared_training(folder,folders,skill_id,cancelled=cancelled,progress=preparing)
        except Exception as error:
            saved=json.loads((folder/'job.json').read_text())
            saved.update(state='cancelled' if isinstance(error,InterruptedError) else 'failed',error=str(error))
            write_json(folder/'job.json',saved);raise
    child = None
    observer = None

    def update(phase, **values):
        state.update(state=phase, at=time.time(), **values)
        write_json(folder / "job.json", state)
        if progress:
            progress(dict(state))

    def check_cancelled():
        if cancelled and cancelled():
            raise InterruptedError("Simulator training cancelled")

    def stop_observer():
        if observer is not None and observer.poll() is None:
            from .live_room_diagnostic import stop_owned
            observer.terminate()
            try:observer.wait(timeout=10)
            except subprocess.TimeoutExpired:stop_owned(observer)

    try:
        check_cancelled()
        if resume or reuse_prepared:
            steps = state['steps']
            environment = configure(folder)
        else:
            if not folders:
                raise ValueError("No simulator trajectories were selected")
            if len(set(folders)) != len(folders) or len({item.name for item in folders}) != len(folders):
                raise ValueError("Duplicate trajectories cannot form independent train/validation episodes")
            training_config(folder, steps, "cpu", batch_size, seed,maximum_steps)  # Validate request before a heavy import/export.
            update("exporting")
            from importlib.metadata import version
            state["runtime_versions"] = {name: version(name) for name in ("lerobot", "torch", "torchvision")}
            if state["runtime_versions"]["lerobot"] != "0.6.1":
                raise ValueError("Use the Explorer simulator's pinned LeRobot 0.6.1 environment")
            provenance = [inspect_episode(item) for item in folders]
            profiles = {item[0].get('scene_profile', 'baseline') for item in provenance}
            if len(profiles) != 1:
                raise ValueError('Do not mix legacy and randomized scene profiles in one model')
            state['scene_profile'] = next(iter(profiles))
            if state['scene_profile'] == 'sock_variants_v1':
                from .domain_randomization import scene_coverage
                state['scene_coverage'] = scene_coverage([item[0] for item in provenance])
            if not diagnostic_only and any(item[0].get("physics_verified_success", item[0].get("success")) is not True for item in provenance):
                raise ValueError("ACT task training requires successful simulator demonstrations; failed attempts remain diagnostic data")
            state["dataset_fingerprint"] = hashlib.sha256(json.dumps([item[3] for item in provenance], sort_keys=True).encode()).hexdigest()
            # Reuse the verified Torch cache across jobs on the external volume.
            # Keep LeRobot's job-local environment and dataset provenance intact.
            shared_backbone = Path(os.environ.get("EXPLORER_SIM_VOLUME", "/Volumes/ExplorerSim")) / "cache/torch/hub/checkpoints/resnet18-f37072fd.pth"
            environment = configure(folder)
            if shared_backbone.is_file():
                backbone_cache = Path(environment["TORCH_HOME"]) / "hub/checkpoints"
                backbone_cache.mkdir(parents=True, exist_ok=True)
                shutil.copy2(shared_backbone, backbone_cache / shared_backbone.name)
            state["device"] = choose_device(device)
            check_cancelled()
            if corrective_training_seeds and not warm_start:
                raise ValueError('Corrective replay requires inherited independent validation')
            split_function=(lambda folders,metadata:inherited_split(folders,metadata,warm_start,corrective_training_seeds)) if warm_start else split_episodes
            training, validation, heldout = split_function(folders, [item[0] for item in provenance])
            if type(correction_replays) is not int or not 1<=correction_replays<=8:
                raise ValueError('Invalid corrective replay weight')
            frame_counts={path:sum(len(s) for s in expert_segments(item[1],item[2]))*
                (correction_replays if item[0].get('learner_intervention') or item[0].get('corrective_demonstration') is True else 1)
                for path,item in zip(folders,provenance)}
            budget=training_budget(steps,sum(frame_counts[path] for path in training),batch_size,minimum_epochs,maximum_steps)
            if minimum_epochs and budget['expected_dataset_passes']<minimum_epochs:
                raise ValueError('Объём данных превышает бюджет полных проходов; качество обучения не снижено')
            steps=budget['effective_steps']
            state.update(steps=steps,training_budget=budget)
            state['correction_replays']=correction_replays
            export_dataset(training, folder / "train", "explorer/simulator-train", task, cancelled=cancelled,correction_replays=correction_replays)
            check_cancelled()
            export_dataset(validation, folder / "validation", "explorer/simulator-validation", task, cancelled=cancelled)
            check_cancelled()
            write_json(folder / "split.json", dict(train=[item.name for item in training],
                validation=[item.name for item in validation], heldout_independent=bool(heldout),corrective_training_seeds=list(corrective_training_seeds)))
            config = training_config(folder, steps, state["device"], batch_size, seed,maximum_steps)
            if warm_start:
                check_cancelled()
                source_bundle, _ = read_bundle(warm_start)
                if (source_bundle.get('skill_id'),source_bundle.get('scene_profile','baseline'))!=(skill_id,state['scene_profile']):
                    raise ValueError('Warm start requires the same task and scene profile')
                context = json.loads((folder / "train/explorer-provenance.json").read_text()).get("actor_context")
                if source_bundle.get("actor_context") != context:
                    raise ValueError("Warm start requires the same actor time context")
                if source_bundle.get("action_transform") is not None:
                    raise ValueError("Warm start requires the same action transformation")
                state["warm_start"] = stage_warm_start(warm_start, folder / "initial-policy", config["policy"])
                state['warm_start'].update(normalization='preserved_source_processors',
                    source_dataset_fingerprint=source_bundle.get('dataset_fingerprint'),
                    expanded_dataset_fingerprint=state['dataset_fingerprint'],validation_partition='inherited_scene_identities')
                config["policy"]["pretrained_path"] = str(folder / "initial-policy")
                config['policy'].update(optimizer_lr=5e-5,optimizer_lr_backbone=1e-5)
            write_json(folder / "train-config.json", config)
            write_json(folder / "train-config.environment.json", environment)
        interpreter = str(python_executable or sys.executable)
        env = dict(os.environ, HF_HUB_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1", WANDB_MODE="disabled",
                   PYTORCH_ENABLE_MPS_FALLBACK="1", OMP_NUM_THREADS="2", PYTHONUNBUFFERED="1")
        env.pop('EXPLORER_FROZEN_NORMALIZATION',None)
        env.pop('EXPLORER_CHECKPOINT_RESUME',None)
        env['EXPLORER_UPDATE_MEASUREMENT']=str(folder/('update-measurement-'+str(time.time_ns())+'.json'))
        if (state.get('warm_start') or {}).get('normalization')=='preserved_source_processors':
            env['EXPLORER_FROZEN_NORMALIZATION']=str(folder/'initial-policy')
        if resume_config:env['EXPLORER_CHECKPOINT_RESUME']='1'
        update("training")
        training_started=time.monotonic()
        starting_step=state.get('resumed_step',0)
        log_start = (folder/'train.log').stat().st_size if resume and (folder/'train.log').is_file() else 0
        update('training',learner_started_at=time.time(),training_log_start_bytes=log_start)
        with (folder / "train.log").open("a") as log:
            check_cancelled()
            entry='simulator.train_entry'
            arguments = [interpreter, "-m", entry,
                         "--config_path=" + str(resume_config or folder / "train-config.json")]
            if resume_config:
                arguments.append("--resume=true")
            child = subprocess.Popen(arguments, stdout=log, stderr=log, env=env,
                start_new_session=(os.name == "posix"))
            if monitor_owner is not None:
                from .checkpoint_monitor import start_for_training
                try:observer=start_for_training(folder,monitor_owner,env,interpreter)
                except Exception as error:
                    update('training',checkpoint_observer_error=str(error))
            while child.poll() is None:
                if cancelled and cancelled():
                    raise InterruptedError("Simulator training cancelled")
                try:
                    with (folder / "train.log").open("rb") as tail:
                        tail.seek(max(log_start, (folder / "train.log").stat().st_size - 16000))
                        content = tail.read().decode("utf8", errors="replace")
                    # LeRobot abbreviates its metric steps as 1K/2K. tqdm's
                    # counter remains exact and must take precedence.
                    completed = completed_training_steps(content,steps,state.get('resumed_step',0))
                    state["steps_completed"] = max(state.get("steps_completed", 0), completed)
                    state["progress"] = round(90 * state["steps_completed"] / steps, 1)
                    elapsed=time.monotonic()-training_started
                    updates=state['steps_completed']-starting_step
                    if updates>=100:
                        state['eta_s']=round((steps-state['steps_completed'])*elapsed/updates)
                except OSError:
                    pass
                update("training")
                time.sleep(.5)
            if child.returncode:
                raise RuntimeError("LeRobot training failed; see train.log (exit " + str(child.returncode) + ")")
        stop_observer()
        update("validating", steps_completed=steps, progress=95.)
        check_cancelled()
        selection, state["validation"] = select_checkpoint(folder,state["device"],cancelled=cancelled,
            progress=lambda message: update('validating',message=message))
        state['checkpoint_selection']=selection
        state['checkpoint_relative']=selection['checkpoint_relative']
        check_cancelled()
        create_bundle(folder, state)
        update("trained_simulation_unvalidated", bundle="bundle.json", finished=time.time(), progress=100.,
               weight_updates_complete=True,offline_prediction_checked=True,
               simulation_qualified=False,robot_verified=False)
    except (Exception, KeyboardInterrupt) as exc:
        update("cancelled" if isinstance(exc, (InterruptedError, KeyboardInterrupt)) else "failed",
               error=str(exc) or "Simulator training interrupted", finished=time.time())
    finally:
        stop_observer()
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
