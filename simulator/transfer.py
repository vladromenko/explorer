"""Portable, checksummed ACT packages; importing never opens robot interfaces."""
from __future__ import annotations

import argparse
import json
from pathlib import Path, PurePosixPath
import shutil
import tarfile
import tempfile
import time
import uuid

from src.mobile_policy_contract import read_bundle, CONTRACT_SHA256, ACTOR_CLOCK
from simulator.learning import sha256, write_json
from src.simulation_task_contract import simulation_qualification

MAX_PACKAGE_BYTES = 2 * 1024 ** 3
MAX_FILES = 10000


def _safe_name(name):
    if not isinstance(name, str) or not name or "\\" in name or "\x00" in name or ":" in name:
        raise ValueError("Unsafe package path")
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts or path.as_posix() != name or name == ".":
        raise ValueError("Unsafe package path: " + name)
    return path


def _device_fields(value, device):
    """Change saved processor placement while preserving normalization tensors."""
    if isinstance(value, dict):
        return {key: device if key == "device" and item in ("mps", "cuda", "cpu") else _device_fields(item, device)
                for key, item in value.items()}
    if isinstance(value, list):
        return [_device_fields(item, device) for item in value]
    return value


def _validate_processors(checkpoint, digests):
    for name in ("policy_preprocessor.json", "policy_postprocessor.json"):
        processor = json.loads((checkpoint / name).read_text())
        for step in processor.get("steps", []):
            state_file = step.get("state_file")
            if state_file is not None:
                _safe_name(state_file)
                if state_file not in digests:
                    raise ValueError("Processor normalization state is absent from checkpoint checksums")


def export_package(job_folder, destination, inference_device="cuda"):
    if inference_device not in ("cuda", "cpu"):
        raise ValueError("Jetson inference placement must be cuda or cpu")
    folder = Path(job_folder).resolve()
    bundle, checkpoint = read_bundle(folder)
    if bundle.get("training_source") != "synthetic_only" or bundle.get("physical_success_verified") is not False:
        raise ValueError("This exporter expects an explicitly unaccepted simulator model")
    required = {"policy_preprocessor.json", "policy_postprocessor.json"}
    if not required.issubset(bundle["checkpoint_sha256"]):
        raise ValueError("Checkpoint lacks saved normalization processors")
    _validate_processors(checkpoint, bundle["checkpoint_sha256"])
    physics_report = folder / "physics-validation.json"
    if physics_report.is_file():
        report = json.loads(physics_report.read_text())
        if report.get("validation_kind") != "closed_loop_policy":
            raise ValueError("Teacher diagnostics are not a trained policy validation report")
        if report.get("model_weights_sha256") != bundle["checkpoint_sha256"]["model.safetensors"]:
            raise ValueError("Physics validation report refers to a different or unidentified checkpoint")
        if report.get("model_checkpoint_sha256",bundle.get("validation_checkpoint_sha256",bundle["checkpoint_sha256"])) != bundle.get("validation_checkpoint_sha256",bundle["checkpoint_sha256"]):
            raise ValueError("Physics validation report refers to different execution settings or processors")
        if report.get("real_hardware_accepted") is not False or report.get("physics_validated") is not False:
            raise ValueError("Simulation rollout reports cannot grant hardware or dynamics acceptance")
    destination = Path(destination).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise ValueError("Package destination already exists")
    with tempfile.TemporaryDirectory(prefix="explorer-export-", dir=destination.parent) as temporary:
        stage = Path(temporary)
        target = stage / bundle["checkpoint_relative"]
        target.mkdir(parents=True)
        for name in bundle["checkpoint_sha256"]:
            _safe_name(name)
            source = checkpoint / name
            if source.is_symlink():
                raise ValueError("Symlink files are not portable checkpoint inputs")
            copied = target / name
            copied.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, copied)
            if name in ("config.json", "policy_preprocessor.json", "policy_postprocessor.json"):
                value = _device_fields(json.loads(copied.read_text()), inference_device)
                if name == "config.json":
                    value["device"] = inference_device
                    # The exported weights are self-contained. A training-time
                    # warm-start path on the Mac must never redirect a Jetson
                    # policy factory away from this verified checkpoint.
                    value["pretrained_path"] = None
                    value["pretrained_revision"] = None
                    value["pretrained_backbone_weights"] = None
                write_json(copied, value)
        for name in ("job.json", "validation.json", "split.json", "checkpoint-selection.json", "physics-validation.json", "preview-validation.json"):
            source = folder / name
            if source.is_file():
                shutil.copyfile(source, stage / name)
        for split in ("train", "validation"):
            source = folder / split / "explorer-provenance.json"
            if source.is_file():
                write_json(stage / (split + "-provenance.json"), json.loads(source.read_text()))
        source_digests = bundle.get("validation_checkpoint_sha256",bundle["checkpoint_sha256"])
        bundle = dict(bundle, validation_checkpoint_sha256=source_digests,
                      checkpoint_sha256={str(path.relative_to(target)): sha256(path)
                      for path in target.rglob("*") if path.is_file()},
                      transfer_inference_device=inference_device, physics_acceptance=False,
                      autonomous_motion_on_registration=False, physical_success_verified=False)
        write_json(stage / "bundle.json", bundle)
        packaged_report = stage / "physics-validation.json"
        if packaged_report.is_file():
            report = json.loads(packaged_report.read_text())
            if "model_checkpoint_sha256" in report:
                # Record the device-only retargeting separately from the
                # complete checkpoint actually exercised in simulation.
                report["transfer_checkpoint_sha256"] = bundle["checkpoint_sha256"]
                write_json(packaged_report,report)
        read_bundle(stage)
        files = {str(path.relative_to(stage)): sha256(path) for path in stage.rglob("*") if path.is_file()}
        manifest = dict(format="explorer_simulator_transfer_v1", created=time.time(), files=files,
            sample_contract_sha256=CONTRACT_SHA256, training_source="synthetic_only",
            physics_acceptance=False, physical_success_verified=False, automatic_execution_allowed=False,
            inference_device=inference_device, framework_version="0.6.1")
        manifest["training_goal"] = bundle.get("training_goal", "successful_simulation_imitation")
        write_json(stage / "transfer.json", manifest)
        temporary_archive = stage / "package.tar.gz"
        with tarfile.open(temporary_archive, "w:gz") as archive:
            for name in sorted([*files, "transfer.json"]):
                archive.add(stage / name, arcname=name, recursive=False)
        temporary_archive.replace(destination)
    return dict(path=str(destination), archive=str(destination), sha256=sha256(destination),
                bytes=destination.stat().st_size, inference_device=inference_device,
                training_source="synthetic_only", physics_acceptance=False,
                automatic_execution_allowed=False)


def _extract_verified(archive_path, destination):
    with tarfile.open(archive_path, "r:gz") as archive:
        members = archive.getmembers()
        if len(members) > MAX_FILES or sum(member.size for member in members) > MAX_PACKAGE_BYTES:
            raise ValueError("Package exceeds import limits")
        seen = set()
        for member in members:
            _safe_name(member.name)
            if member.name in seen or not member.isfile() or member.size < 0:
                raise ValueError("Packages accept unique regular files only; links/devices are forbidden")
            seen.add(member.name)
        if "transfer.json" not in seen or "bundle.json" not in seen:
            raise ValueError("Package manifest or policy bundle is missing")
        for member in members:
            path = destination / member.name
            path.parent.mkdir(parents=True, exist_ok=True)
            source = archive.extractfile(member)
            if source is None:
                raise ValueError("Unreadable package member")
            with source, path.open("xb") as target:
                shutil.copyfileobj(source, target)
    manifest = json.loads((destination / "transfer.json").read_text())
    if manifest.get("format") != "explorer_simulator_transfer_v1":
        raise ValueError("Unknown transfer format")
    if (manifest.get("sample_contract_sha256") != CONTRACT_SHA256 or
            manifest.get("training_source") != "synthetic_only" or
            manifest.get("physics_acceptance") is not False or
            manifest.get("physical_success_verified") is not False or
            manifest.get("automatic_execution_allowed") is not False):
        raise ValueError("Package provenance or acceptance claims are incompatible")
    digests = manifest.get("files")
    if not isinstance(digests, dict) or set(digests) != seen - {"transfer.json"}:
        raise ValueError("Manifest does not match the package contents")
    for name, digest in digests.items():
        _safe_name(name)
        if not isinstance(digest, str) or sha256(destination / name) != digest:
            raise ValueError("Package checksum mismatch: " + name)
    bundle, checkpoint = read_bundle(destination)
    if (bundle.get("training_source") != "synthetic_only" or
            bundle.get("physical_success_verified") is not False or
            bundle.get("physics_acceptance") is not False or
            bundle.get("autonomous_motion_on_registration") is not False):
        raise ValueError("Imported simulator model must remain physically unaccepted")
    report_path = destination / "physics-validation.json"
    if report_path.is_file():
        report = json.loads(report_path.read_text())
        if (report.get("validation_kind") != "closed_loop_policy" or
                report.get("model_weights_sha256") != bundle["checkpoint_sha256"]["model.safetensors"]):
            raise ValueError("Physics validation report does not match this policy checkpoint")
        if report.get("model_checkpoint_sha256",bundle.get("validation_checkpoint_sha256",bundle["checkpoint_sha256"])) != bundle.get("validation_checkpoint_sha256",bundle["checkpoint_sha256"]):
            raise ValueError("Physics validation report does not match execution settings or processors")
        if report.get("transfer_checkpoint_sha256",bundle["checkpoint_sha256"]) != bundle["checkpoint_sha256"]:
            raise ValueError("Retargeted checkpoint differs from the transfer report")
        if report.get("real_hardware_accepted") is not False or report.get("physics_validated") is not False:
            raise ValueError("Simulation rollout reports cannot grant hardware or dynamics acceptance")
    required = {"policy_preprocessor.json", "policy_postprocessor.json"}
    if not required.issubset(bundle["checkpoint_sha256"]):
        raise ValueError("ACT normalization processors are missing")
    _validate_processors(checkpoint, bundle["checkpoint_sha256"])
    config = json.loads((checkpoint / "config.json").read_text())
    features = {"observation.state", "observation.images.wrist"}
    if bundle.get("actor_context"):
        features.add(ACTOR_CLOCK["feature"])
    if set(config.get("input_features", {})) != features or set(config.get("output_features", {})) != {"action"}:
        raise ValueError("Simulator actor features differ from its verified observation contract")
    horizon = config.get("chunk_size", 1)
    steps = config.get("n_action_steps")
    if type(horizon) is not int or horizon not in (1, 20, 50) or type(steps) is not int or not 1<=steps<=20 or steps>horizon:
        raise ValueError("Simulator transfer requires bounded ACT chunks and exactly one applied action per call")
    if horizon > 1 and steps==1 and config.get("temporal_ensemble_coeff") != .01:
        raise ValueError("Bounded simulator ACT chunks require the tested temporal ensemble")
    return manifest, bundle, checkpoint


def import_package(archive, robot_root, register=False):
    """Unpack into quarantine or explicitly register an inactive simulation candidate.

    Only independently qualified imports can enter the observed trial selector.
    Commissioning and observed transfer validation remain separate.
    """
    if type(register) is not bool:
        raise ValueError("Registration must be explicitly true or false")
    archive = Path(archive).resolve()
    if not archive.is_file() or archive.stat().st_size > MAX_PACKAGE_BYTES:
        raise ValueError("Transfer archive is missing or too large")
    root = Path(robot_root).expanduser().resolve()
    base = root / "data" / ("learning-jobs" if register else "simulator-imports")
    base.mkdir(parents=True, exist_ok=True)
    identifier = sha256(archive)[:32]
    target = base / identifier
    if target.exists():
        raise ValueError("This package already has an import at " + str(target))
    with tempfile.TemporaryDirectory(prefix=".simulator-import-", dir=base) as temporary:
        stage = Path(temporary)
        manifest, bundle, _ = _extract_verified(archive, stage)
        record = dict(id=identifier, at=time.time(), created=time.time(), finished=time.time(),
            state="imported_simulation", source="simulator_transfer", task=bundle.get("task", "simulation"),
            skill_id=bundle.get("skill_id"), episodes=bundle.get("episodes", []),
            dataset_kind="mobile_manipulation_9dof", framework="lerobot", policy="act", bundle="bundle.json",
            validation=bundle.get("validation", {}), training_source="synthetic_only",
            training_goal=bundle.get("training_goal", "successful_simulation_imitation"),
            physics_acceptance=False, physical_success_verified=False, automatic_execution=False,
            executed=False, observed_transfer_validation_required=True,
            registered=register, transfer_sha256=sha256(archive))
        record['simulation_qualification'] = simulation_qualification(stage,bundle)
        write_json(stage / "job.json", record)
        # Keep the original manifest as provenance; job.json is now a local import record.
        (stage / "transfer.json").rename(stage / "source-transfer.json")
        stage.rename(target)
    return dict(path=str(target), job=record, registered=register,
                automatic_execution_allowed=False, physics_acceptance=False)


def export_curriculum(plan, destination):
    """One checksummed archive for all finished task policies, including failures."""
    from .curriculum import task_spec
    destination = Path(destination)
    if destination.exists():
        raise ValueError('Curriculum archive already exists')
    destination.parent.mkdir(parents=True,exist_ok=True)
    manifest = dict(format='explorer_curriculum_v1',batch_id=plan['id'],tasks=[],
                    created=time.time(),automatic_execution_allowed=False,files={})
    if plan.get('learning_goal'):
        source=plan['learning_goal']
        manifest['learning_goal']={key:source[key] for key in
            ('title','request','task_ids','fully_supported','missing','requested_handover',
             'automatic_hardware_execution','physical_ready')}
        _validate_learning_goal(manifest['learning_goal'],[task['task_id'] for task in plan['tasks']])
    with tempfile.TemporaryDirectory(dir=destination.parent) as name:
        stage = Path(name)
        for task in plan['tasks']:
            entry = dict(task=task_spec(task['task_id']),state=task['state'],error=task.get('error'))
            exported = task['stages'].get('export')
            if exported:
                package = Path(exported['path'])
                if sha256(package) != exported['sha256']:
                    raise ValueError('Task archive changed after export')
                relative = 'policies/'+task['task_id']+'.tar.gz'
                target = stage / relative
                target.parent.mkdir(exist_ok=True)
                shutil.copyfile(package,target)
                manifest['files'][relative] = sha256(target)
                entry['package'] = relative
                entry['validation'] = task['stages'].get('validate')
            manifest['tasks'].append(entry)
        write_json(stage/'curriculum.json',manifest)
        archive_path = stage/'result.tar.gz'
        with tarfile.open(archive_path,'w:gz') as archive:
            for relative in [*manifest['files'],'curriculum.json']:
                archive.add(stage/relative,arcname=relative,recursive=False)
        archive_path.replace(destination)
    return dict(path=str(destination),sha256=sha256(destination),tasks=len(manifest['tasks']),
                policies=len(manifest['files']),automatic_execution_allowed=False)


def _validate_learning_goal(goal, task_ids):
    """Intent metadata is inert; it cannot widen an executable task contract."""
    keys={'title','request','task_ids','fully_supported','missing','requested_handover',
          'automatic_hardware_execution','physical_ready'}
    if not isinstance(goal,dict) or set(goal)!=keys:
        raise ValueError('Invalid language goal metadata')
    if goal['physical_ready'] is not False or goal['automatic_hardware_execution'] is not False:
        raise ValueError('Language goals cannot grant hardware execution')
    if goal['task_ids']!=task_ids or type(goal['fully_supported']) is not bool:
        raise ValueError('Language goal differs from the packaged curriculum')
    if any(not isinstance(goal[key],str) or not 0<len(goal[key])<=limit for key,limit in (('title',120),('request',4000))):
        raise ValueError('Language request exceeds metadata limits')
    if not isinstance(goal['missing'],list) or len(goal['missing'])>16 or any(not isinstance(text,str) or len(text)>500 for text in goal['missing']):
        raise ValueError('Invalid missing skill description')
    if goal['requested_handover'] not in ('none','supported_surface','human_hand'):
        raise ValueError('Unknown handover contract')
    if goal['fully_supported'] and (goal['missing'] or goal['requested_handover']=='human_hand'):
        raise ValueError('Unimplemented handover cannot be marked complete')


def import_curriculum(archive, robot_root, register=False):
    """Verify every nested policy before registering any; import never moves robot."""
    from .curriculum import task_spec
    root = Path(robot_root).resolve()
    archive = Path(archive).resolve()
    with tempfile.TemporaryDirectory() as name:
        stage = Path(name)
        with tarfile.open(archive,'r:gz') as handle:
            members = handle.getmembers()
            names = [item.name for item in members]
            if len(names) != len(set(names)) or len(names)>100 or sum(m.size for m in members)>MAX_PACKAGE_BYTES:
                raise ValueError('Curriculum archive exceeds limits or contains duplicate files')
            for item in members:
                _safe_name(item.name)
                if not item.isfile():
                    raise ValueError('Curriculum accepts regular files only')
                target = stage/item.name
                target.parent.mkdir(parents=True,exist_ok=True)
                with handle.extractfile(item) as source, target.open('wb') as output:
                    shutil.copyfileobj(source,output)
        manifest = json.loads((stage/'curriculum.json').read_text())
        if manifest.get('format') != 'explorer_curriculum_v1' or set(names) != {'curriculum.json',*manifest['files']}:
            raise ValueError('Invalid curriculum manifest')
        for relative,digest in manifest['files'].items():
            _safe_name(relative)
            if sha256(stage/relative)!=digest:
                raise ValueError('Curriculum checksum mismatch')
        seen = set()
        nested_bytes = 0
        if manifest.get('learning_goal') is not None:
            _validate_learning_goal(manifest['learning_goal'],[task['task']['id'] for task in manifest['tasks']])
        for task in manifest['tasks']:
            identifier = task['task']['id']
            if task['task'] != task_spec(identifier) or identifier in seen:
                raise ValueError('Unknown or duplicated task contract')
            seen.add(identifier)
            if task.get('package'):
                if task['package'] not in manifest['files']:
                    raise ValueError('Missing task policy')
                checked = stage/('checked-'+identifier)
                checked.mkdir()
                with tarfile.open(stage/task['package'],'r:gz') as nested:
                    nested_bytes += sum(member.size for member in nested.getmembers())
                if nested_bytes > MAX_PACKAGE_BYTES:
                    raise ValueError('Combined expanded policies exceed import limits')
                _,bundle,_ = _extract_verified(stage/task['package'],checked)
                if bundle.get('skill_id') != identifier:
                    raise ValueError('Policy belongs to a different task')
        imported = []
        for task in manifest['tasks']:
            if task.get('package'):
                package = stage/task['package']
                existing=root/'data'/('learning-jobs' if register else 'simulator-imports')/sha256(package)[:32]
                if existing.exists():
                    record=json.loads((existing/'job.json').read_text())
                    read_bundle(existing)
                    if record.get('transfer_sha256')!=sha256(package) or record.get('state')!='imported_simulation':
                        raise ValueError('Conflicting import already exists')
                    result=dict(path=str(existing),job=record,registered=register,
                                automatic_execution_allowed=False,physics_acceptance=False)
                else:
                    result = import_package(package,root,register)
                imported.append(dict(task=task['task'],**result))
        if register:
            folder = root/'data/simulation-curricula'
            folder.mkdir(parents=True,exist_ok=True)
            write_json(folder/(sha256(archive)+'.json'),dict(manifest=manifest,imports=imported))
        return dict(tasks=imported,automatic_execution_allowed=False)


def import_archive(archive, robot_root, register=False):
    with tarfile.open(archive,'r:gz') as handle:
        curriculum = 'curriculum.json' in handle.getnames()
    return import_curriculum(archive,robot_root,register) if curriculum else import_package(archive,robot_root,register)


def main():
    parser = argparse.ArgumentParser(description="Export/import inactive Explorer simulator ACT candidates")
    commands = parser.add_subparsers(dest="command", required=True)
    export = commands.add_parser("export")
    export.add_argument("job_folder", type=Path)
    export.add_argument("destination", type=Path)
    export.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    importer = commands.add_parser("import")
    importer.add_argument("archive", type=Path)
    importer.add_argument("robot_root", type=Path)
    importer.add_argument("--register", action="store_true", help="Register an inactive simulation candidate only")
    args = parser.parse_args()
    result = export_package(args.job_folder, args.destination, args.device) if args.command == "export" else import_archive(args.archive, args.robot_root, args.register)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
