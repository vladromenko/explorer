"""Offline, phase-local residual baseline; never a transferable robot policy.

Teacher phase labels route separate regressors. They are not inferred from task
truth, and the caller must supply an independently observed phase at runtime.
Only the current commanded state and current wrist RGB enter the regressors.
"""
from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path

import numpy as np
from PIL import Image

from .learning import inspect_episode, sha256, _vector, _contained
from src.mobile_policy_contract import ACTION_ORDER, CONTRACT_SHA256

PHASES = frozenset(("inspect_floor", "align_base", "navigate_source", "approach",
    "lower", "close", "lift", "transport", "navigate_destination", "place",
    "release_retreat", "release_unhook"))
FEATURE_SPEC = {"state": "current_command_estimate_9", "vision": "current_wrist_rgb_4x3_area_mean",
    "phase": "external_controller_routing_only", "future_or_truth_inputs": False}


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def _write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)


def _visual(rgb):
    image = np.asarray(rgb)
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise ValueError("Expected current wrist RGB uint8 image")
    return np.asarray(Image.fromarray(image).resize((4, 3), Image.Resampling.BOX), float).ravel() / 255.


def _features(state, rgb):
    return np.r_[np.asarray(_vector(state), float), _visual(rgb)]


def _provenance(episode):
    model = episode.get("model_provenance")
    if not isinstance(model, dict) or not model.get("geometry_source"):
        raise ValueError("Missing geometry provenance; cannot prepare phase data")
    camera = model.get("camera")
    if not isinstance(camera, dict) or not camera.get("camera_info_version") or not model.get("handeye"):
        raise ValueError("Missing wrist visual/calibration provenance")
    if episode.get("action_alignment") not in ("pre_action_observation", "same_row_pre_action"):
        raise ValueError("Explicit pre-action alignment is required")
    return _digest(model)


def _groups(episodes):
    """Union by scene seed OR recording hash, including copies under new names."""
    parent = list(range(len(episodes)))
    def root(i):
        while parent[i] != i:
            i = parent[i]
        return i
    seen = {}
    for i, item in enumerate(episodes):
        for key in (("seed", item["seed"]), ("recording", item["input_sha256"]["samples.jsonl"])):
            if key in seen:
                parent[root(i)] = root(seen[key])
            else:
                seen[key] = i
    keys = [root(i) for i in range(len(episodes))]
    groups = list(dict.fromkeys(keys))
    if len(groups) < 3:
        raise ValueError("At least three independent seed/episode groups are required")
    count = max(1, len(groups) // 5)
    test, validation = set(groups[-count:]), set(groups[-2 * count:-count])
    return ["test" if key in test else "validation" if key in validation else "train" for key in keys]


def _check_cancelled(cancelled):
    if cancelled is not None and cancelled():
        raise InterruptedError("Phase preparation/fitting cancelled; completed caches retained")


def _notify(progress, **values):
    if progress is not None:
        progress(values)


def _cache_episode(folder, cache_root, manifest, *, cancelled=None, progress=None, processed=0, total=0):
    input_hashes = {name: sha256(folder / name) for name in ("episode.json", "samples.jsonl")}
    key = _digest({"folder": str(folder), "inputs": input_hashes, "feature_spec": FEATURE_SPEC,
                   "cache_schema": "phase_episode_cache_v1"})
    cache = cache_root / key
    meta_path, data_path = cache / "metadata.json", cache / "examples.jsonl"
    if meta_path.is_file() and data_path.is_file():
        meta = json.loads(meta_path.read_text())
        if (meta.get("complete") is True and meta.get("examples_sha256") == sha256(data_path)
                and meta.get("input_sha256", {}).get("episode.json") == input_hashes["episode.json"]
                and meta.get("input_sha256", {}).get("samples.jsonl") == input_hashes["samples.jsonl"]):
            valid = True
            for index, (name, digest) in enumerate(meta["input_sha256"].items()):
                _check_cancelled(cancelled)
                if sha256(_contained(folder, name)) != digest:
                    valid = False
                    break
                if index % 64 == 0:
                    _notify(progress, processed=processed, total=total, stage="verify_cache",
                            episode=str(folder), checked_files=index)
            if valid:
                return meta, meta_path, data_path, True
    _check_cancelled(cancelled)
    episode, rows, fps, hashes = inspect_episode(folder)
    _check_cancelled(cancelled)
    cache.mkdir(parents=True, exist_ok=True)
    partial = cache / "examples.partial.jsonl"
    segments = []
    try:
        with partial.open("w") as stream:
            for index, row in enumerate(rows):
                _check_cancelled(cancelled)
                phase = row.get("phase")
                if phase not in PHASES:
                    raise ValueError("Missing or unsupported explicit phase boundary: " + str(phase))
                if row.get("action_alignment", episode["action_alignment"]) not in ("same_row_pre_action", "pre_action_observation"):
                    raise ValueError("Inconsistent row action alignment")
                image_path = folder / row["image"]
                image_bytes = image_path.read_bytes()
                if hashlib.sha256(image_bytes).hexdigest() != hashes[row["image"]]:
                    raise ValueError("Source wrist image changed during phase preparation")
                with Image.open(io.BytesIO(image_bytes)) as image:
                    if image.mode != "RGB":
                        raise ValueError("Recorded wrist image must be RGB")
                    features = _features(row["observation_state"], np.asarray(image))
                if not segments or segments[-1]["phase"] != phase:
                    segments.append({"phase": phase, "start": index, "stop": index + 1})
                else:
                    segments[-1]["stop"] = index + 1
                example = {"row": index, "phase": phase, "features": features.tolist(),
                    "residual": (np.asarray(row["applied_action"]) - features[:9]).tolist(),
                    "image": str(image_path), "image_sha256": hashes[row["image"]]}
                stream.write(json.dumps(example, allow_nan=False) + "\n")
                if index % 32 == 0:
                    _notify(progress, processed=processed, total=total, stage="extract",
                            episode=str(folder), frame=index + 1, frames=len(rows))
        _check_cancelled(cancelled)
        # Detect concurrent modification before accepting the cache. RGB hashes
        # refer to inspect_episode's immutable source snapshot above.
        if any(sha256(folder / name) != value for name, value in input_hashes.items()):
            raise ValueError("Source episode changed during phase preparation")
        partial.replace(data_path)
        meta = {"schema": "phase_episode_cache_v1", "complete": True, "seed": episode["seed"],
            "fps": fps, "sample_count": len(rows), "segments": segments, "input_sha256": hashes,
            "examples_sha256": sha256(data_path), "geometry_camera_sha256": _provenance(manifest)}
        temporary = cache / "metadata.partial.json"
        temporary.write_text(json.dumps(meta, allow_nan=False))
        temporary.replace(meta_path)
        return meta, meta_path, data_path, False
    finally:
        partial.unlink(missing_ok=True)


def iter_phase_examples(episode, *, cancelled=None):
    """Stream a checksum-verified completed cache; bounded memory per episode."""
    _check_cancelled(cancelled)
    path = Path(episode["examples_path"])
    if sha256(path) != episode["examples_sha256"]:
        raise ValueError("Phase cache checksum mismatch")
    count = 0
    with path.open() as stream:
        for line in stream:
            _check_cancelled(cancelled)
            example = json.loads(line)
            if example.get("phase") not in PHASES:
                raise ValueError("Unsupported cached phase")
            yield example
            count += 1
    if count != episode["sample_count"]:
        raise ValueError("Incomplete phase cache")


def prepare_phase_dataset(episode_folders, destination, *, task_id="sock_basket", cancelled=None, progress=None):
    """Prepare immutable phase caches with cancellation and cross-run resume.

    The destination is a small manifest; its sibling '<name>.cache' holds one
    checksummed JSONL per episode. Only finished caches can be reused. Failed
    trajectories remain excluded, and no original recording is changed. Current
    episode work may repeat after interruption; earlier cached episodes survive.
    """
    destination = Path(destination).resolve()
    if destination.exists():
        raise FileExistsError(destination)
    folders = list(dict.fromkeys(Path(value).resolve() for value in episode_folders))
    cache_root = destination.with_name(destination.name + ".cache")
    episodes, excluded = [], []
    signature = None
    for processed, folder in enumerate(folders):
        _check_cancelled(cancelled)
        manifest = json.loads((folder / "episode.json").read_text())
        if manifest.get("success") is not True or manifest.get("attempt_completed") is not True:
            excluded.append({"folder": str(folder), "reason": "not_completed_success"})
        elif manifest.get("task_id") != task_id or manifest.get("source") != "simulation":
            excluded.append({"folder": str(folder), "reason": "incompatible_task_or_source"})
        else:
            if type(manifest.get("seed")) is not int:
                raise ValueError("Simulation scene seed is required")
            geometry = _provenance(manifest)
            if signature is not None and signature != geometry:
                raise ValueError("Incompatible geometry/camera provenance")
            signature = geometry
            meta, meta_path, data_path, reused = _cache_episode(folder, cache_root, manifest,
                cancelled=cancelled, progress=progress, processed=processed, total=len(folders))
            episodes.append({"folder": str(folder), "seed": meta["seed"], "fps": meta["fps"],
                "input_sha256": {key: meta["input_sha256"][key] for key in ("episode.json", "samples.jsonl")},
                "segments": meta["segments"], "sample_count": meta["sample_count"],
                "examples_path": str(data_path), "metadata_path": str(meta_path),
                "examples_sha256": meta["examples_sha256"]})
            _notify(progress, processed=processed + 1, total=len(folders), stage="cached" if reused else "prepared",
                    episode=str(folder), sample_count=meta["sample_count"])
    if len({episode["fps"] for episode in episodes}) > 1:
        raise ValueError("Incompatible recording rates for one-step residuals")
    splits = _groups(episodes)
    for episode, split in zip(episodes, splits):
        episode["split"] = split
    result = {"schema": "explorer_phase_dataset_v1", "state": "complete", "task_id": task_id,
        "data_source": "preserved_successful_simulation_demonstrations",
        "geometry_camera_sha256": signature, "sample_contract_sha256": CONTRACT_SHA256,
        "actor_feature_spec": FEATURE_SPEC, "episodes": episodes, "excluded": excluded,
        "sample_count": sum(episode["sample_count"] for episode in episodes),
        "split_unit": "entire_scene_seed_and_duplicate_recording_union",
        "ready_for_transfer": False, "real_hardware_accepted": False}
    _check_cancelled(cancelled)
    _write_new(destination, result)
    _notify(progress, processed=len(folders), total=len(folders), stage="complete", sample_count=result["sample_count"])
    return result


def _matrix(examples):
    x = np.asarray([row["features"] for row in examples], float)
    y = np.asarray([row["residual"] for row in examples], float)
    if x.shape != (len(examples), 45) or y.shape != (len(examples), 9) or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("Invalid phase training/validation matrix")
    return x, y


def fit_phase_residuals(dataset_path, destination, *, regularization=1., cancelled=None, progress=None):
    """Fit separate ridge models by streaming sufficient statistics per phase.

    Train-only normalization and regression statistics use bounded memory. Test
    caches are not opened. Offline command error is not closed-loop success.
    """
    if not np.isfinite(regularization) or regularization <= 0:
        raise ValueError("Regularization must be finite and positive")
    _check_cancelled(cancelled)
    dataset = json.loads(Path(dataset_path).read_text())
    if (dataset.get("schema") != "explorer_phase_dataset_v1" or dataset.get("state") != "complete"
            or dataset.get("actor_feature_spec") != FEATURE_SPEC
            or dataset.get("sample_contract_sha256") != CONTRACT_SHA256
            or not dataset.get("geometry_camera_sha256")):
        raise ValueError("Unsupported/incomplete phase dataset")
    expected = _groups(dataset["episodes"])
    if expected != [episode.get("split") for episode in dataset["episodes"]]:
        raise ValueError("Invalid independent scene split")
    stats = {}
    training = [episode for episode in dataset["episodes"] if episode["split"] == "train"]
    for index, episode in enumerate(training):
        _check_cancelled(cancelled)
        by_phase = {}
        for example in iter_phase_examples(episode, cancelled=cancelled):
            by_phase.setdefault(example["phase"], []).append(example)
        for phase, examples in by_phase.items():
            x, y = _matrix(examples)
            stat = stats.setdefault(phase, {"n": 0, "x": np.zeros(45), "y": np.zeros(9),
                "xx": np.zeros((45, 45)), "xy": np.zeros((45, 9)), "cap": np.zeros(9)})
            stat["n"] += len(x)
            stat["x"] += x.sum(axis=0)
            stat["y"] += y.sum(axis=0)
            stat["xx"] += np.einsum("ni,nj->ij", x, x)
            stat["xy"] += np.einsum("ni,nj->ij", x, y)
            stat["cap"] = np.maximum(stat["cap"], np.max(np.abs(y), axis=0))
        _notify(progress, processed=index + 1, total=len(training), stage="fit_statistics")
    models = {}
    for phase in sorted({segment["phase"] for episode in dataset["episodes"] for segment in episode["segments"]}):
        _check_cancelled(cancelled)
        stat = stats.get(phase)
        if stat is None or stat["n"] < 3:
            models[phase] = {"status": "insufficient_independent_data"}
            continue
        n = stat["n"]
        mean = stat["x"] / n
        variance = np.maximum(np.diag(stat["xx"]) / n - mean ** 2, 0.)
        scale = np.sqrt(variance)
        scale[scale < 1e-8] = 1.
        gram = np.zeros((46, 46))
        gram[:45, :45] = (stat["xx"] - n * np.outer(mean, mean)) / np.outer(scale, scale)
        gram[-1, -1] = n
        cross = np.r_[(stat["xy"] - np.outer(mean, stat["y"])) / scale[:, None], stat["y"][None, :]]
        penalty = np.eye(46) * regularization
        penalty[-1, -1] = 0.
        weights = np.linalg.solve(gram + penalty, cross)
        models[phase] = {"status": "awaiting_independent_validation", "mean": mean.tolist(), "scale": scale.tolist(),
            "weights": weights.tolist(), "residual_abs_cap": stat["cap"].tolist(),
            "train_samples": n, "validation_samples": 0,
            "validation_mae_by_coordinate": [0.] * 9, "zero_residual_baseline_mae_by_coordinate": [0.] * 9}
    validation = [episode for episode in dataset["episodes"] if episode["split"] == "validation"]
    for index, episode in enumerate(validation):
        by_phase = {}
        for example in iter_phase_examples(episode, cancelled=cancelled):
            by_phase.setdefault(example["phase"], []).append(example)
        for phase, examples in by_phase.items():
            model = models[phase]
            if "weights" not in model:
                continue
            x, y = _matrix(examples)
            design = np.c_[(x - model["mean"]) / model["scale"], np.ones(len(x))]
            cap = np.asarray(model["residual_abs_cap"])
            predicted = np.clip(np.einsum("ni,ij->nj", design, model["weights"]), -cap, cap)
            model["validation_samples"] += len(x)
            for name, value in (("validation_mae_by_coordinate", np.abs(predicted - y).sum(axis=0)),
                                ("zero_residual_baseline_mae_by_coordinate", np.abs(y).sum(axis=0))):
                model[name] = (np.asarray(model[name]) + value).tolist()
        _notify(progress, processed=index + 1, total=len(validation), stage="offline_validation")
    for phase, model in list(models.items()):
        count = model.get("validation_samples", 0)
        if not count:
            models[phase] = {"status": "insufficient_independent_data"}
            continue
        model["status"] = "offline_fitted"
        for name in ("validation_mae_by_coordinate", "zero_residual_baseline_mae_by_coordinate"):
            model[name] = (np.asarray(model[name]) / count).tolist()
    result = {"schema": "explorer_phase_residual_v1", "task_id": dataset["task_id"],
        "dataset_sha256": sha256(dataset_path), "geometry_camera_sha256": dataset["geometry_camera_sha256"],
        "sample_contract_sha256": CONTRACT_SHA256, "actor_feature_spec": FEATURE_SPEC,
        "data_source": dataset["data_source"], "models": models, "action_order": ACTION_ORDER,
        "regularization": regularization, "validation_kind": "offline_one_step_command_residual",
        "test_groups_used": False, "closed_loop_verified": False, "ready_for_transfer": False,
        "real_hardware_accepted": False,
        "limitations": ["coarse RGB baseline, not an object detector", "requires externally observed phase transitions",
            "no physical success inferred from offline error", "simulation dynamics remain unvalidated"]}
    _check_cancelled(cancelled)
    _write_new(destination, result)
    return result


def predict_phase_residual(artifact, phase, observation_state, wrist_rgb):
    """Experimental simulation inference; output is a residual, never motion I/O."""
    if artifact.get("schema") != "explorer_phase_residual_v1" or artifact.get("actor_feature_spec") != FEATURE_SPEC:
        raise ValueError("Unsupported phase residual artifact")
    model = artifact["models"].get(phase, {})
    if model.get("status") != "offline_fitted":
        raise ValueError("Phase has no fitted residual model")
    x = _features(observation_state, wrist_rgb)
    normalized = (x - np.asarray(model["mean"])) / np.asarray(model["scale"])
    residual = np.einsum("i,ij->j", np.r_[normalized, 1.], np.asarray(model["weights"]))
    cap = np.asarray(model["residual_abs_cap"])
    if not np.isfinite(residual).all():
        raise ValueError("Invalid phase residual prediction")
    return np.clip(residual, -cap, cap).tolist()
