"""Separate full-mission qualification and inactive Jetson packages.

A successful local ACT report cannot qualify room cleanup. Imports contain no
executable installer and never change STOP, activate a model or contact hardware.
"""
import json
from pathlib import Path
import shutil
import tarfile
import tempfile
import time

from .learning import read_bundle,sha256,write_json
from .room_policy import contract_sha,runtime_files,validate_profile
from .transfer import export_package,_extract_verified,_safe_name,MAX_PACKAGE_BYTES,MAX_FILES


def qualification(groups,profile,bundle):
    if not isinstance(groups,list) or len(groups)!=2:raise ValueError('Two independent room groups required')
    expected=bundle.get('validation_checkpoint_sha256',bundle['checkpoint_sha256'])
    if profile.get('model_checkpoint_sha256')!=expected:raise ValueError('Phase handoff is bound to different weights/processors')
    seen=set();contracts=set();counts=set()
    for group in groups:
        rows=group.get('results',[])
        if len(rows)<10 or group.get('episodes')!=len(rows) or group.get('successes')!=len(rows):
            raise ValueError('At least ten successful full-room scenes per group required')
        counts.add(len(rows))
        user_basket=False;heights=set()
        for row in rows:
            if len(rows)>=50 and row.get('physical_outcome',{}).get('checked_after_actions_stopped') is not True:
                raise ValueError('Fresh final placement outcome required for the 100-scene protocol')
            seed=row.get('seed')
            if type(seed) is not int or seed in seen:raise ValueError('Room acceptance scenes overlap')
            seen.add(seed)
            if (row.get('success') is not True or row.get('state')!='succeeded'
                    or row.get('physical_outcome',{}).get('success') is not True
                    or row.get('room_layout') is not True or row.get('collision_sampling')!='each_physics_substep'
                    or row.get('immutable_source_verified') is not True or not isinstance(row.get('source_revision'),str)
                    or len(row['source_revision'])!=40
                    or row.get('collision_free') is not True or row.get('actor_truth_inputs') is not False
                    or row.get('teacher_fallback') is not False or row.get('physical_ready') is not False
                    or row.get('simulation_only') is not True or row.get('placed_verified') is not True
                    or row.get('model_sha256')!=expected['model.safetensors']):
                raise ValueError('Room result lacks independent physical and observed success')
            stages={e.get('stage') for e in row.get('events',[])}
            if not {'search','approach_sock','policy_pick','verify_hold','approach_basket','policy_place','verify_place'}<=stages:
                raise ValueError('Local placement is not a completed room mission')
            contracts.add(row.get('mission_contract_sha256'))
            scene=row['domain_variation'];bin_spec=scene['parameters']['bin']
            import math
            if math.dist(scene['sock_position'][:2],scene['bin_position'][:2])<1.:
                raise ValueError('Room targets were not separated for navigation')
            heights.add(bin_spec['height_m'])
            user_basket|=(bin_spec['shape']=='round' and abs(bin_spec['height_m']-.30)<1e-6
                and all(abs(x-.125)<1e-6 for x in bin_spec['inner_half_size_m']))
        if not user_basket:raise ValueError('Each room group must include the 25 cm by 30 cm user basket')
        if heights!={.08,.12,.16,.20,.30}:raise ValueError('Room validation lacks a required basket height')
    if len(counts)!=1:raise ValueError('Room final groups must have equal sizes')
    if len(contracts)!=1 or not next(iter(contracts)):raise ValueError('Room execution contract differs between trials')
    return seen,next(iter(contracts))


def validate_room_folder(folder,*,check_runtime=False):
    folder=Path(folder);manifest=json.loads((folder/'room-transfer.json').read_text())
    if (manifest.get('format')!='explorer_room_cleanup_v1' or manifest.get('simulation_qualified') is not True
            or manifest.get('physical_ready') is not False or manifest.get('automatic_execution_allowed') is not False):
        raise ValueError('Room transfer cannot grant physical acceptance or automatic motion')
    files=manifest.get('files',{})
    actual={str(p.relative_to(folder)) for p in folder.rglob('*') if p.is_file()}-{'room-transfer.json'}
    if not files or set(files)!=actual:raise ValueError('Room manifest differs from package contents')
    for name,digest in files.items():
        _safe_name(name);p=folder/name
        if p.is_symlink() or sha256(p)!=digest:raise ValueError('Room file checksum mismatch: '+name)
    bundle,_=read_bundle(folder/'model');profile=json.loads((folder/'handoff.json').read_text())
    validate_profile(profile)
    provenance=json.loads((folder/'model/train-provenance.json').read_text())
    sources={r['input_sha256']['samples.jsonl'] for r in provenance['episodes']}
    if any(r.get('source_sha256') not in sources for r in profile['placement_proposals']):
        raise ValueError('Phase proposals are not derived from this model training data')
    groups=[json.loads((folder/f'group-{i}.json').read_text()) for i in (1,2)]
    seeds,contract=qualification(groups,profile,bundle)
    if seeds & set(manifest['excluded_seeds']):raise ValueError('Room validation overlaps training or model selection')
    if contract!=manifest['mission_contract_sha256']:raise ValueError('Room qualification contract mismatch')
    code=manifest['runtime_files']
    if not code or any(files.get('runtime/'+name)!=digest for name,digest in code.items()):raise ValueError('Runtime manifest is incomplete')
    if contract_sha(profile,[folder/'runtime'/name for name in code])!=contract:raise ValueError('Qualified runtime or handoff changed')
    if check_runtime:
        installed={str(p.relative_to(Path(__file__).resolve().parents[1])):p for p in runtime_files()}
        if set(installed)!=set(code):raise ValueError('Installed mission dependency list differs')
        for name,path in installed.items():
            if not path.is_file() or sha256(path)!=code[name]:raise ValueError('Installed mission code differs: '+name)
    assets=json.loads((folder/'runtime/config/arm_model.json').read_text())['asset_sha256']
    for name,digest in assets.items():
        if name.endswith('.STL') and files.get('assets/meshes/'+Path(name).name)!=digest:raise ValueError('Room robot CAD differs')
    return dict(manifest,profile=profile,bundle=bundle)


def export_room_package(model,profile_path,group_paths,destination,*,excluded_seeds=()):
    from .neural_recovery import training_seeds,persisted_seeds
    from .explorer_model import resolve_mesh_dir
    model,profile_path,destination=Path(model),Path(profile_path),Path(destination)
    bundle,_=read_bundle(model);profile=json.loads(profile_path.read_text())
    groups=[json.loads(Path(p).read_text()) for p in group_paths];seeds,contract=qualification(groups,profile,bundle)
    excluded=training_seeds(model)|set(excluded_seeds)
    for name in ('split.json','checkpoint-selection.json','physics-validation.json','qualification-evidence.json'):
        p=model/name
        if p.exists():excluded|=persisted_seeds(json.loads(p.read_text()))
    if seeds&excluded:raise ValueError('Room scenes overlap existing training/selection')
    if contract_sha(profile,runtime_files())!=contract:raise ValueError('Code changed after room tests')
    destination.parent.mkdir(parents=True,exist_ok=True)
    if destination.exists():
        # Resume archival after a deliberate pause without overwriting or
        # silently adopting a package for another model/composition.
        with tempfile.TemporaryDirectory(prefix='.room-reopen-',dir=destination.parent) as temporary:
            checked=extract_room(destination,Path(temporary)/'package')
            if (checked['profile']!=profile or checked['mission_contract_sha256']!=contract
                    or [json.loads((Path(temporary)/'package'/f'group-{i}.json').read_text()) for i in (1,2)]!=groups):
                raise ValueError('Existing room archive belongs to a different qualification')
        return dict(path=str(destination),sha256=sha256(destination),physical_ready=False)
    with tempfile.TemporaryDirectory(prefix='.room-export-',dir=destination.parent) as temporary:
        stage=Path(temporary);package=stage/'package';package.mkdir()
        inner=stage/'component.tar.gz';export_package(model,inner)
        target=package/'model';target.mkdir();_extract_verified(inner,target)
        shutil.copyfile(profile_path,package/'handoff.json')
        for index,group in enumerate(groups,1):write_json(package/f'group-{index}.json',group)
        root=Path(__file__).resolve().parents[1];runtime={}
        for p in runtime_files():
            name=str(p.relative_to(root));runtime[name]=sha256(p)
            copy=package/'runtime'/name;copy.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,copy)
        meshes=resolve_mesh_dir()
        for name,digest in json.loads((root/'config/arm_model.json').read_text())['asset_sha256'].items():
            if not name.endswith('.STL'):continue
            p=meshes/Path(name).name
            if sha256(p)!=digest:raise ValueError('Robot mesh changed')
            copy=package/'assets/meshes'/p.name;copy.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,copy)
        files={str(p.relative_to(package)):sha256(p) for p in package.rglob('*') if p.is_file()}
        write_json(package/'room-transfer.json',dict(format='explorer_room_cleanup_v1',created=time.time(),
            simulation_qualified=True,physical_ready=False,automatic_execution_allowed=False,
            qualification_episodes=len(seeds),qualification_successes=len(seeds),
            mission_contract_sha256=contract,runtime_files=runtime,files=files,excluded_seeds=sorted(excluded)))
        validate_room_folder(package,check_runtime=True)
        archive=stage/'room.tar.gz'
        with tarfile.open(archive,'w:gz') as stream:
            for p in sorted(package.rglob('*')):
                if p.is_file():stream.add(p,arcname=str(p.relative_to(package)),recursive=False)
        # Reopen the entire archive before advertising it, including all hashes.
        verified=stage/'verified';extract_room(archive,verified);validate_room_folder(verified,check_runtime=True)
        archive.replace(destination)
    return dict(path=str(destination),sha256=sha256(destination),physical_ready=False)


def extract_room(archive,destination):
    archive,destination=Path(archive),Path(destination)
    if archive.stat().st_size>MAX_PACKAGE_BYTES:raise ValueError('Room package too large')
    destination.mkdir(parents=True,exist_ok=False)
    with tarfile.open(archive,'r:gz') as stream:
        members=stream.getmembers();seen=set()
        if len(members)>MAX_FILES or sum(m.size for m in members)>MAX_PACKAGE_BYTES:raise ValueError('Room package exceeds limits')
        for member in members:
            _safe_name(member.name)
            if member.name in seen or not member.isfile() or member.size<0:raise ValueError('Room archive accepts unique regular files only')
            seen.add(member.name)
        for member in members:
            p=destination/member.name;p.parent.mkdir(parents=True,exist_ok=True)
            with stream.extractfile(member) as source,p.open('xb') as target:shutil.copyfileobj(source,target)
    return validate_room_folder(destination)


def import_room_package(archive,robot_root):
    base=Path(robot_root)/'data/room-cleanup-candidates';base.mkdir(parents=True,exist_ok=True)
    identifier=sha256(archive);destination=base/identifier
    if destination.exists():raise ValueError('Room package already imported')
    with tempfile.TemporaryDirectory(prefix='.room-import-',dir=base) as temporary:
        stage=Path(temporary)/'package';checked=extract_room(archive,stage)
        stage.replace(destination)
    return dict(candidate=identifier,path=str(destination),mission_contract_sha256=checked['mission_contract_sha256'],
        registered=True,active=False,physical_ready=False)


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('archive',type=Path);parser.add_argument('--robot-root',type=Path,required=True)
    args=parser.parse_args();print(json.dumps(import_room_package(args.archive,args.robot_root),ensure_ascii=False,indent=2))
