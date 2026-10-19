#!/usr/bin/env python3
"""Read-only provenance and runtime audit. No training, hardware or qualification."""
import argparse
from collections import Counter
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import re
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from simulator.learning import sha256,write_json


def partition(path):
    value=json.loads(Path(path).read_text());rows=value['episodes']
    digests=[row['input_sha256']['samples.jsonl'] for row in rows]
    seeds=[row['episode'].get('seed') for row in rows]
    def histogram(key):return dict(Counter(str(row['episode'].get(key,'unspecified')) for row in rows))
    positions=[row['episode'].get('domain_variation',{}).get('sock_position') for row in rows]
    heights=[row['episode'].get('domain_variation',{}).get('parameters',{}).get('bin',{}).get('height_m') for row in rows]
    raw_frames=sum(row['samples']//row.get('replay_weight',1) for row in rows)
    weighted_frames=sum(row['samples'] for row in rows)
    return dict(episodes=len(rows),unique_episode_ids=len({row['episode']['id'] for row in rows}),
        unique_recording_hashes=len(set(digests)),unique_scene_seeds=len(set(seeds)),
        samples=raw_frames,effective_replay_frames=weighted_frames,
        replay_weights=dict(Counter(str(row.get('replay_weight',1)) for row in rows)),
        sources=histogram('source'),teacher_profiles=histogram('teacher_profile'),scene_profiles=histogram('scene_profile'),
        basket_heights=dict(Counter(str(x) for x in heights)),
        positive_physics_labels=sum(row['episode'].get('physics_verified_success') is True for row in rows),
        whole_room_outcome_labels=sum(row['episode'].get('room_cleanup_success') is True for row in rows),
        room_label_caveat='Absence of room_cleanup_success is not proof of absence; inspect recorded task stages.',
        unique_sock_placements=len({json.dumps(x) for x in positions if x is not None}),
        privileged_actor_inputs=value.get('privileged_actor_inputs'),
        actor_inputs=value.get('actor_inputs'),actor_context=value.get('actor_context'),
        joint_state_source=value.get('joint_state_source'),action_transform=value.get('action_transform'),
        provenance_sha256=sha256(path)),set(digests),set(seeds)


def log_metrics(path):
    with Path(path).open('rb') as stream:
        stream.seek(0,2);size=stream.tell();stream.seek(max(0,size-200000));tail=stream.read().decode(errors='replace')
    pattern=r'step:(\d+).*?loss:([\d.e+-]+).*?grdn:([\d.e+-]+).*?updt_s:([\d.e+-]+).*?data_s:([\d.e+-]+)'
    rows=[dict(step=int(m[0]),loss=float(m[1]),gradient_norm=float(m[2]),update_s=float(m[3]),data_s=float(m[4]))
          for m in re.findall(pattern,tail)]
    return dict(size_bytes=size,modified_at=Path(path).stat().st_mtime,recent_windows=rows[-12:])


def audit(root,training):
    import psutil
    folder=Path(root)/'runs'/training;job=json.loads((folder/'job.json').read_text())
    config=json.loads((folder/'train-config.json').read_text())
    train,th,ts=partition(folder/'train/explorer-provenance.json')
    validation,vh,vs=partition(folder/'validation/explorer-provenance.json')
    memory=psutil.virtual_memory();swap=psutil.swap_memory()
    processes=[]
    for process in psutil.process_iter(['pid','ppid','cmdline','memory_info','cpu_times']):
        args=process.info['cmdline'] or []
        if any(x in args for x in ['simulator.train_entry','simulator.neural_recovery','simulator.live_room_diagnostic','simulator.server']):
            info=process.info;mem=info['memory_info']
            processes.append(dict(pid=info['pid'],ppid=info['ppid'],command=args,rss_bytes=mem.rss,
                rss_caveat='RSS includes shared mapped dataset pages; do not sum as private allocations.',
                cpu_seconds=sum(info['cpu_times'][:2]),cwd=process.cwd()))
    return dict(schema='explorer_learning_audit_v1',at=time.time(),host=platform.node(),platform=platform.platform(),
        architecture=platform.machine(),code_revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        run_id=training,state=job['state'],steps_completed=job.get('steps_completed'),
        steps=config['steps'],batch_size=config['batch_size'],minimum_expected_passes=config['steps']*config['batch_size']/train['effective_replay_frames'],
        accumulation_steps=1,num_workers=config['num_workers'],prefetch_factor=config.get('prefetch_factor'),
        configured_device=config['policy']['device'],device_placement_proven=False,
        actor_config=config['policy'],train=train,validation=validation,
        shared_recording_hashes=sorted(th&vh),shared_scene_seeds=sorted(ts&vs),
        unique_total_recordings=len(th|vh),unique_total_scene_seeds=len(ts|vs),
        near_duplicate_trajectories_checked=False,whole_room_policy_success_proven=False,robot_verified=False,
        versions={name:importlib.metadata.version(name) for name in ['torch','torchvision','lerobot','mujoco','datasets','pyarrow','accelerate']},
        config_sha256=sha256(folder/'train-config.json'),split_sha256=sha256(folder/'split.json'),
        memory=dict(total_bytes=memory.total,available_bytes=memory.available,swap_used_bytes=swap.used),
        processes=processes,training_log=log_metrics(folder/'train.log'))


def wilson(successes,total):
    if not total:return [0.,1.]
    z=1.95996398454;p=successes/total;d=1+z*z/total
    center=(p+z*z/(2*total))/d
    radius=z*math.sqrt(p*(1-p)/total+z*z/(4*total*total))/d
    return [center-radius,center+radius]


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,default=Path('/Volumes/ExplorerSim/ExplorerSim'))
    parser.add_argument('--training',required=True);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    if not re.fullmatch('[0-9a-f]{12}',args.training):raise ValueError('Invalid training identity')
    if args.output.exists():raise ValueError('Choose a new audit path; previous evidence is retained')
    value=audit(args.root,args.training);write_json(args.output,value)
    print(json.dumps(dict(output=str(args.output),steps_completed=value['steps_completed'],
        unique_recordings=value['unique_total_recordings'],train_validation_shared_scenes=value['shared_scene_seeds'])))


if __name__=='__main__':main()
