#!/usr/bin/env python3
"""Isolated room mission inference; hardware is accessible only via owned RPC."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'src')]


def run(folder):
    from src.learning_environment import configure
    configure(ROOT)
    from simulator.room_transfer import validate_room_folder
    from simulator.observed_robot import ObservedRobotEnv
    from simulator.room_cleanup_port import RoomCleanupPort
    from src.room_cleanup_rpc import ClientTransport,save
    from src.room_cleanup import RoomCleanup
    config=json.loads((folder/'config.json').read_text());package=Path(config['package'])
    checked=validate_room_folder(package,check_runtime=True)
    transport=ClientTransport(folder,config['session']);env=None;port=None
    latest={}
    def record(value):latest.update(value);save(folder/'progress.json',latest)
    def motion(value):latest['motion']=value;save(folder/'progress.json',latest)
    try:
        env=ObservedRobotEnv(transport,mesh_dir=config['mesh_dir'])
        port=RoomCleanupPort(env,folder/'mission',package/'model',checked['profile'],config['semantic_root'],
            cancelled=transport.cancelled,device='cuda',progress=motion)
        if port.contract!=checked['mission_contract_sha256']:raise ValueError('Mission runtime differs from qualification')
        result=RoomCleanup(port,record,cancelled=transport.cancelled).run()
        save(folder/'result.json',result)
    finally:
        if port is not None:port.close()
        if env is not None:env.close()


def inspect(package,output,device='cuda'):
    """Load actual transferred weights and infer once, with no hardware imports."""
    from src.learning_environment import configure
    configure(ROOT)
    import importlib.metadata
    import numpy as np
    import torch
    import mujoco,gymnasium,scipy
    from simulator.room_transfer import validate_room_folder
    from simulator.room_policy import RoomActor
    from simulator.learning import write_json
    checked=validate_room_folder(package,check_runtime=True)
    if device=='cuda' and not torch.cuda.is_available():raise ValueError('CUDA unavailable; hardware runtime is not admitted')
    actor=RoomActor(package/'model',checked['profile'],device=device)
    actor.actor.elapsed_s=checked['profile']['pick_elapsed_s']
    action=actor(np.asarray(checked['profile']['initial_command']),np.zeros((240,320,3),np.uint8))
    if action.shape!=(9,) or not np.isfinite(action).all():raise ValueError('Transferred ACT inference is invalid')
    value=dict(ready=True,model_sha256=actor.sha,mission_contract_sha256=checked['mission_contract_sha256'],
        device=device,versions={name:importlib.metadata.version(name) for name in ('lerobot','torch','mujoco','gymnasium','scipy')},
        hardware_accessed=False,physical_ready=False,automatic_execution_allowed=False)
    write_json(output,value);return value


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('folder',type=Path);parser.add_argument('--inspect-output',type=Path)
    args=parser.parse_args()
    if args.inspect_output:print(json.dumps(inspect(args.folder,args.inspect_output)))
    else:run(args.folder)
