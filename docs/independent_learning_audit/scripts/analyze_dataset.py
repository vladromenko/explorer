"""Read-only recording/visual sample audit; no training or physical rollout."""
import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image


def percentile(values):
    a = np.asarray(values, float)
    return dict(n=len(a), minimum=np.min(a, axis=0).tolist(), p05=np.percentile(a, 5, axis=0).tolist(),
                median=np.median(a, axis=0).tolist(), p95=np.percentile(a, 95, axis=0).tolist(),
                maximum=np.max(a, axis=0).tolist()) if len(a) else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.repo))
    from simulator.episode_lineage import recording_index, trained_recording
    from simulator.learning import sha256
    root, out = args.root, args.output
    index, cache = recording_index(root), {}
    selections = []
    for split in ['train', 'validation']:
        source = root / 'runs/d4f59a63085b' / split / 'explorer-provenance.json'
        selections.extend(dict(row=row, split=split) for row in json.loads(source.read_text())['episodes'])
    summaries, visuals, bad, deltas, durations, starts = [], [], [], [], [], []
    target_state, next_mismatch = [], []
    meta = {'height': [], 'sock_position': [], 'basket_position': [], 'lighting': [],
            'sock_rgb': [], 'basket_rgb': [], 'mass': [], 'young': [], 'friction': []}
    phase = Counter()
    total, idle, masked, nonfinite, nonmonotonic = 0, 0, 0, 0, 0
    raw_files = []
    for pattern in ['demonstrations/*/episode.json',
                    'neural-recovery/*/round-*/correction-*/recovery-*/episode.json',
                    'room-learning/*/round-*/corrections/scene-*/room-expert-*/episode.json']:
        raw_files.extend(root.glob(pattern))
    raw_meta, raw_errors = [], []
    for path in raw_files:
        try:
            d = json.loads(path.read_text())
            raw_meta.append(dict(path=str(path), id=d.get('id'), seed=d.get('seed'), success=d.get('success'),
                                 task=d.get('task_id'), teacher_profile=d.get('teacher_profile'),
                                 frames=d.get('frames'), generator=d.get('generator')))
        except (OSError, ValueError) as exc:
            raw_errors.append(dict(path=str(path), error=str(exc)))
    for number, item in enumerate(selections):
        source, split = item['row'], item['split']
        try:
            folder = trained_recording(source, index, cache)
            rows = [json.loads(line) for line in (folder / 'samples.jsonl').read_text().splitlines()]
            ep = source['episode']; a = np.asarray([r['action'] for r in rows], float)
            o = np.asarray([r['observation'] for r in rows], float)
            t = np.asarray([r.get('actor_elapsed_s', r.get('simulation_time', i*.05)) for i, r in enumerate(rows)])
            valid = np.array([r.get('training_target_valid', True) for r in rows], bool)
            finite = np.isfinite(a).all(axis=1) & np.isfinite(o).all(axis=1)
            total += len(rows); masked += int((~valid).sum()); nonfinite += int((~finite).sum())
            nonmonotonic += int(np.sum(np.diff(t) <= 0))
            delta = a[valid & finite, :6] - o[valid & finite, :6]
            deltas.extend(delta.tolist()); target_state.extend(a[valid & finite].tolist())
            is_idle = (np.max(abs(a[:, :6]-o[:, :6]), axis=1) < .1) & (np.max(abs(a[:, 6:]), axis=1) < .005)
            idle += int((is_idle & valid).sum()); phase.update(r.get('phase', 'unknown') for r in rows)
            if len(rows) > 1:
                next_mismatch.extend(np.max(abs(o[1:]-a[:-1]), axis=1).tolist())
            durations.append(len(rows)*.05); starts.append(o[0].tolist())
            variation = ep.get('domain_variation', {}); parameters = variation.get('parameters', {})
            for name, value in [('height', parameters.get('bin', {}).get('height_m')),
                ('sock_position', variation.get('sock_position')), ('basket_position', variation.get('bin_position')),
                ('lighting', variation.get('lighting_scale')), ('sock_rgb', variation.get('sock_rgb')),
                ('basket_rgb', variation.get('basket_rgb')), ('mass', parameters.get('sock', {}).get('mass_kg')),
                ('young', parameters.get('sock', {}).get('young_pa')), ('friction', parameters.get('sock', {}).get('friction'))]:
                if value is not None:
                    meta[name].append(value)
            summary = dict(id=ep['id'], split=split, seed=ep.get('seed'), source=str(folder),
                samples_sha256=source['input_sha256']['samples.jsonl'], frames=len(rows),
                target_valid_frames=int(valid.sum()), masked_frames=int((~valid).sum()),
                idle_frames=int((is_idle & valid).sum()), declared_success=ep.get('success'),
                max_target_delta=np.max(abs(delta), axis=0).tolist() if len(delta) else None,
                teacher_profile=ep.get('teacher_profile', 'unspecified'))
            summaries.append(summary)
            wanted = [('initial', 0), ('close', next((i for i,r in enumerate(rows) if r.get('phase')=='close'),len(rows)//2)),
                      ('place', next((i for i,r in enumerate(rows) if r.get('phase')=='place'),len(rows)-1))]
            for label, i in wanted:
                row = rows[i]; name = row.get('wrist_rgb', row.get('image')); p = folder / name
                expected = source['input_sha256'].get(name)
                if not p.is_file():
                    bad.append(dict(path=str(p), issue='missing_sampled_image'))
                else:
                    current = sha256(p)
                    if expected and current != expected:
                        bad.append(dict(path=str(p), issue='sampled_image_hash_mismatch'))
                    rgb = np.asarray(Image.open(p).convert('RGB').resize((16,12)),float)/255
                    visuals.append(dict(id=ep['id'], split=split, phase=label, row=i, image=str(p),
                        image_sha256=current, target=row['action'], state=row['observation'],
                        clock_s=float(t[i]), rgb_descriptor=rgb.ravel().tolist()))
            if number % 100 == 0:
                print('recording', number, flush=True)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            bad.append(dict(id=source.get('episode',{}).get('id'), issue=str(exc)))
    nearest = []
    for label in ['initial','close','place']:
        group = [v for v in visuals if v['phase']==label]
        x = np.asarray([v['rgb_descriptor'] for v in group], np.float32)
        if len(x)>1:
            distance = np.maximum(0, (x*x).sum(1)[:,None]+(x*x).sum(1)[None,:]-2*x@x.T)/x.shape[1]
            np.fill_diagonal(distance, np.inf)
            for i, item in enumerate(group):
                cross=[j for j,v in enumerate(group) if v['split']!=item['split']]
                if cross:
                    j=min(cross,key=lambda j:float(distance[i,j])); other=group[j]
                    nearest.append(dict(phase=label, id=item['id'], other=other['id'],
                        rgb_thumbnail_rmse=float(np.sqrt(distance[i,j])),
                        state_arm_rmse_deg=float(np.sqrt(np.mean((np.asarray(item['state'])[:6]-np.asarray(other['state'])[:6])**2))),
                        target_arm_rmse_deg=float(np.sqrt(np.mean((np.asarray(item['target'])[:6]-np.asarray(other['target'])[:6])**2))),
                        clock_difference_s=abs(item['clock_s']-other['clock_s']),
                        source_image=item['image'], other_image=other['image']))
    hashes={split:{s['samples_sha256'] for s in summaries if s['split']==split} for split in ['train','validation']}
    seeds={split:{s['seed'] for s in summaries if s['split']==split} for split in ['train','validation']}
    result=dict(scope='All current run recordings; metadata inventory of all raw episode families; visual sample only',
        recording_count=len(summaries), raw_episode_manifests=len(raw_meta),
        raw_declared_outcomes=dict(Counter(str(r['success']) for r in raw_meta)),
        raw_parse_errors=raw_errors, recording_hash_overlap=sorted(hashes['train'] & hashes['validation']),
        seed_overlap=sorted(seeds['train'] & seeds['validation']), total_raw_rows=total,
        target_masked_rows=masked, nonfinite_rows=nonfinite, nonincreasing_clock_pairs=nonmonotonic,
        idle_definition='max arm target-current <0.1 degree AND max base velocity <0.005',
        idle_rows=idle, idle_fraction_of_unmasked=idle/max(1,total-masked),
        target_minus_state_degrees=percentile(deltas), target_distribution=percentile(target_state),
        durations_s=percentile(durations), initial_states=percentile(starts),
        next_state_minus_prior_action_absmax=percentile(next_mismatch),
        domain_distributions={k:percentile(v) for k,v in meta.items()}, phase_counts=dict(phase),
        visual_sample_count=len(visuals), sampled_image_problems=bad,
        nearest_cross_split_rgb_thumbnail=sorted(nearest,key=lambda x:x['rgb_thumbnail_rmse'])[:100],
        limitations=['Visual comparison uses 3 sampled frames per recording, not all images',
                    'RGB thumbnail similarity is not proof of scene identity or semantic leakage',
                    'Original full physics states and expert targets at all student states do not exist',
                    'All historical datasets are inventoried by provenance, not decoded in full again'])
    (out/'dataset_statistics.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    (out/'evidence/recording_statistics.json').write_text(json.dumps(summaries,ensure_ascii=False,indent=2)+'\n')
    (out/'evidence/raw_episode_inventory.json').write_text(json.dumps(raw_meta,ensure_ascii=False,indent=2)+'\n')
    (out/'evidence/visual_sample_descriptors.json').write_text(json.dumps(visuals,ensure_ascii=False)+'\n')
    print(json.dumps(dict(records=len(summaries), rows=total, bad=len(bad), idle_fraction=result['idle_fraction_of_unmasked'])))


if __name__=='__main__':
    main()
