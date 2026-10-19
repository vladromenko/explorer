"""Read-only evidence inventory. Writes only under the explicit audit output.

No process signals, learner, simulator steps, ROS, SSH or dataset modification.
Checkpoint hashing is streaming; symlink aliases are listed but not rehashed.
"""
import argparse
import csv
import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import time


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + '\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--hash-checkpoints', action='store_true')
    args = parser.parse_args()
    root, out = args.root.resolve(), args.output.resolve()
    if out.is_relative_to(root):
        raise ValueError('Audit output must be outside the experiment data root')
    out.mkdir(parents=True, exist_ok=True)
    repo = Path(__file__).resolve().parents[3]
    revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip()
    modules = {'simulator.train_entry', 'simulator.neural_recovery', 'simulator.room_validation',
               'simulator.checkpoint_monitor', 'simulator.live_room_diagnostic'}
    import psutil
    processes = []
    for process in psutil.process_iter(['pid', 'ppid', 'cmdline', 'create_time']):
        try:
            command = process.info['cmdline'] or []
            if '-m' in command and command[command.index('-m') + 1] in modules:
                processes.append(dict(process.info, cwd=process.cwd()))
        except psutil.Error:
            pass
    states = {}
    for name in ['neural-recovery/924d1ee5ab01/state.json', 'room-learning/0c946789dbd7/state.json',
                 'runs/d4f59a63085b/job.json', 'room-learning/current.json',
                 'automatic-requests/queue-pause.json', 'diagnostics/early-room-current.json']:
        path = root / name
        if path.is_file():
            states[name] = dict(value=json.loads(path.read_text()), sha256=digest(path))
    snapshot = dict(at=datetime.datetime.now(datetime.timezone.utc).isoformat(), revision=revision,
                    experiment_root=str(root), active_compute_processes=processes, states=states,
                    operator_pause_preserved=True, new_training_started=False,
                    hardware_probed=False, audit_is_read_only_for_experiments=True)
    save(out / 'system_snapshot.json', snapshot)
    history = subprocess.check_output(['git', 'log', '--all', '--reverse',
        '--format=%H\t%aI\t%cI\t%s'], cwd=repo, text=True)
    (out / 'evidence/git_history.tsv').write_text('commit\tauthor_date\tcommit_date\tsubject\n' + history)
    (out / 'evidence/git_refs.txt').write_text(subprocess.check_output(
        ['git', 'show-ref'], cwd=repo, text=True))
    manifests, experiments, malformed = [], [], []
    for path in sorted((root / 'runs').glob('*/job.json')):
        try:
            job = json.loads(path.read_text())
            experiments.append({k: job.get(k) for k in ['id', 'state', 'created', 'at', 'steps',
                'steps_completed', 'batch_size', 'device', 'seed', 'task', 'training_profile',
                'resumed_step', 'warm_start', 'dataset_fingerprint']})
            for split in ['train', 'validation']:
                provenance = path.parent / split / 'explorer-provenance.json'
                if provenance.is_file():
                    value = json.loads(provenance.read_text())
                    episodes = value.get('episodes', [])
                    manifests.append(dict(run=path.parent.name, split=split, path=str(provenance),
                        sha256=digest(provenance), size=provenance.stat().st_size,
                        recording_count=len(episodes), seeds=len({r['episode'].get('seed') for r in episodes}),
                        recording_hashes=[r.get('input_sha256', {}).get('samples.jsonl') for r in episodes],
                        frames=sum(r.get('samples', 0) for r in episodes),
                        declared_successes=sum(r['episode'].get('success') is True for r in episodes),
                        declared_failures=sum(r['episode'].get('success') is False for r in episodes),
                        teacher_profiles=dict(__import__('collections').Counter(
                            r['episode'].get('teacher_profile', 'unspecified') for r in episodes))))
        except (ValueError, OSError, KeyError) as error:
            malformed.append(dict(path=str(path), error=str(error)))
    save(out / 'evidence/run_inventory.json', dict(runs=experiments, malformed=malformed))
    save(out / 'evidence/dataset_manifest_inventory.json', dict(datasets=manifests,
        limitation='Provenance inventory, not a byte-by-byte revalidation of every image'))
    fields = ['id', 'state', 'created', 'at', 'steps', 'steps_completed', 'batch_size', 'device',
              'seed', 'task', 'training_profile', 'resumed_step', 'warm_start', 'dataset_fingerprint']
    with (out / 'experiments.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(experiments)
    checkpoints, hashes = [], {}
    paths = sorted(root.glob('runs/*/model/checkpoints/*/pretrained_model/model.safetensors'))
    for index, path in enumerate(paths):
        resolved = path.resolve()
        if resolved not in hashes and args.hash_checkpoints:
            hashes[resolved] = digest(resolved)
        state = path.parent.parent / 'training_state'
        required = ['optimizer_state.safetensors', 'optimizer_param_groups.json',
                    'rng_state.safetensors', 'training_step.json']
        missing = [name for name in required if not (state / name).is_file()]
        row = dict(run=path.parts[-6], checkpoint=path.parts[-3], path=str(path),
            resolved_path=str(resolved), bytes=path.stat().st_size,
            sha256=hashes.get(resolved), hash_status='computed' if resolved in hashes else 'UNKNOWN',
            alias=path.parent.parent.is_symlink(), full_training_state_files_present=not missing,
            missing_training_files=';'.join(missing))
        checkpoints.append(row)
        if index % 50 == 0:
            print(json.dumps(dict(checkpoint_index=index, total=len(paths), at=time.time())), flush=True)
    with (out / 'checkpoints.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(checkpoints[0]) if checkpoints else ['path'])
        writer.writeheader()
        writer.writerows(checkpoints)
    save(out / 'evidence/checkpoint_inventory_summary.json', dict(entries=len(checkpoints),
        distinct_resolved_files=len({r['resolved_path'] for r in checkpoints}),
        distinct_hashes=len(set(hashes.values())), physical_bytes=sum(p.stat().st_size for p in hashes),
        completed_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        integrity_scope='model bytes SHA256; state file presence, not tensor semantic checks of all optimizers'))
    print('Inventory complete', flush=True)


if __name__ == '__main__':
    main()
