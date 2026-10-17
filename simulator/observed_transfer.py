"""Portable observed-skill candidates, with independently rechecked evidence.

Import is inert. A candidate is not a physical acceptance record and cannot
clear STOP, grant a mission or register a hardware execution permission.
"""
import copy
import hashlib
import json
from pathlib import Path,PurePosixPath
import shutil
import tarfile
import tempfile
import time
import subprocess

FORMAT='explorer_observed_skill_transfer_v1'
MAX_BYTES=512*1024**2
MAX_FILES=2048


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def name(value):
    p=PurePosixPath(value)
    if (not value or '\\' in value or ':' in value or '\x00' in value or p.is_absolute() or
            '..' in p.parts or p.as_posix()!=value or value=='.'):raise ValueError('Unsafe observed package path')
    return value


def read(path):return json.loads(Path(path).read_text())


def write(path,value):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    Path(path).write_text(json.dumps(value,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False)+'\n')


def _dimensions(recipe,row):
    request=recipe['request'];actual=row.get('domain_variation',{}).get('parameters',{}).get('bin',{})
    if request.get('basket_height_cm') is not None:
        if abs(actual.get('height_m',-1)-request['basket_height_cm']/100.)>1e-8:raise ValueError('User basket height not tested')
    if request.get('basket_diameter_cm') is not None:
        size=actual.get('inner_half_size_m',[])
        if actual.get('shape')!='round' or len(size)!=2 or any(abs(v-request['basket_diameter_cm']/200.)>1e-8 for v in size):
            raise ValueError('User round basket diameter not tested')


def validate(folder):
    from .skill_parameters import parameters
    from .failure_repair import program
    from .scene_recipes import validate_recipe
    from .curriculum import report_passes
    folder=Path(folder);manifest=read(folder/'observed-transfer.json')
    if (manifest.get('format')!=FORMAT or manifest.get('physical_ready') is not False or
            manifest.get('automatic_execution_allowed') is not False or manifest.get('simulation_ready') is not True):
        raise ValueError('Observed package claims incompatible readiness')
    files=manifest.get('files',{})
    actual={str(p.relative_to(folder)) for p in folder.rglob('*') if p.is_file()}
    if actual!={*files,'observed-transfer.json'}:raise ValueError('Observed manifest differs from package files')
    if len(files)>MAX_FILES:raise ValueError('Observed package exceeds file limit')
    for relative,digest in files.items():
        name(relative);p=folder/relative
        if p.is_symlink() or sha(p)!=digest:raise ValueError('Observed checksum mismatch: '+relative)
    policy=read(folder/'policy.json');qualification=read(folder/'qualification.json')
    if policy.get('algorithm')!='observed_causal_program_repair_v4' or policy.get('simulation_ready') is not True or policy.get('physical_ready') is not False:
        raise ValueError('Not a qualified observed policy')
    recipe=validate_recipe(policy['scene_recipe'])
    if recipe['task_id']!=policy.get('task_id'):raise ValueError('Task recipe differs from policy')
    if policy.get('parameters')!=parameters(policy.get('parameters')):raise ValueError('Incomplete learned policy parameters')
    if policy.get('controller_program')!=program(policy.get('controller_program')):raise ValueError('Invalid controller program')
    controller=qualification['controller'];code=controller['files']
    digest=hashlib.sha256(json.dumps(code,sort_keys=True).encode()).hexdigest()
    if digest!=controller.get('sha256') or digest!=policy.get('controller_sha256'):raise ValueError('Different qualified controller')
    for relative,digest in code.items():
        name(relative)
        if files.get('runtime/'+relative)!=digest:raise ValueError('Missing checked execution source: '+relative)
    for relative,digest in qualification.get('runtime_files',{}).items():
        name(relative)
        if files.get('runtime/'+relative)!=digest:raise ValueError('Different frozen runtime source')
    assets=read(folder/'runtime/config/arm_model.json')['asset_sha256']
    for relative,digest in assets.items():
        if relative.endswith('.STL') and files.get('assets/meshes/'+Path(relative).name)!=digest:
            raise ValueError('Qualified robot CAD asset missing or different')
    groups=qualification['validation'];expected=3 if len(recipe['request'])>1 else 2
    if len(groups)!=expected:raise ValueError('Independent validation groups are missing')
    protected=set(qualification.get('training_seeds',[]))|set(qualification.get('experience_source_seeds',[]))|set(qualification.get('quarantined_holdouts',[]))
    used=set()
    for index,group in enumerate(groups):
        rows=group.get('results',[])
        if len(rows)!=10 or not report_passes(group):raise ValueError('Independent ten-scene group did not pass')
        for row in rows:
            seed=row.get('seed')
            if type(seed) is not int or seed in used or seed in protected:raise ValueError('Validation scene overlap')
            used.add(seed)
            if row.get('status')!='complete' or row.get('success') is not True:raise ValueError('Unfinished validation episode')
            if index==2:_dimensions(recipe,row)
            proof=name(row['report']);report=read(folder/proof)
            execution=report.get('execution',{});truth=report.get('independent_physics_outcome',{})
            if (report.get('success') is not True or report.get('seed')!=seed or report.get('task_id')!=recipe['task_id'] or
                    execution.get('state')!='succeeded' or execution.get('held_verified') is not True or
                    execution.get('placed_verified') is not True or truth.get('success') is not True):
                raise ValueError('Observed and physical validation evidence did not both pass')
            if (report.get('execution_fingerprint')!=controller or report.get('policy_parameters')!=policy['parameters'] or
                    report.get('controller_program')!=policy['controller_program'] or report.get('scene_recipe')!=recipe):
                raise ValueError('Validation evidence refers to a different execution policy')
    experience=read(folder/'actor-experience.json')
    if set(experience)-{'sha256','ik_seeds'}:raise ValueError('Historical outcomes cannot enter actor input')
    if experience.get('sha256')!=policy.get('experience_sha256'):raise ValueError('Different historical joint prior')
    if len(experience.get('ik_seeds',[]))>64:raise ValueError('Unbounded historical joint priors')
    for pose in experience.get('ik_seeds',[]):
        if (len(pose)!=5 or any(type(v) not in (int,float) or not 0<=v<=upper for v,upper in zip(pose,[180]*4+[270]))):
            raise ValueError('Invalid historical joint pose')
    return dict(manifest=manifest,policy=policy,qualification=qualification,experience=experience,
        simulation_ready=True,physical_ready=False,hardware_trial_required=True)


def export_skill(training_folder,destination):
    """Export only a FINISHED independently qualified controller, not a preview."""
    folder=Path(training_folder).resolve();state=read(folder/'state.json');destination=Path(destination)
    if state.get('state')!='simulation_qualified' or state.get('simulation_ready') is not True:
        raise ValueError('Skill is not independently qualified')
    if destination.exists():raise ValueError('Transfer destination already exists')
    policy=read(folder/'policy.json')
    if sha(folder/'policy.json')!=state.get('policy_sha256'):raise ValueError('Qualified policy changed')
    archive=state.get('archive',{})
    if not archive.get('verified') or sha(archive['path'])!=archive.get('sha256'):raise ValueError('Full learning archive not verified')
    destination.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.observed-export-',dir=destination.parent) as temporary:
        stage=Path(temporary);write(stage/'policy.json',policy)
        runtime=dict(state.get('runtime_files',{}),**state['controller']['files'])
        for relative,digest in runtime.items():
            name(relative);source=folder/'accepted-source'/relative
            if sha(source)!=digest:raise ValueError('Qualified execution source changed')
            target=stage/'runtime'/relative;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,target)
        from .explorer_model import resolve_mesh_dir
        meshes=resolve_mesh_dir()
        for relative,digest in read(stage/'runtime/config/arm_model.json')['asset_sha256'].items():
            if not relative.endswith('.STL'):continue
            source=meshes/Path(relative).name
            if sha(source)!=digest:raise ValueError('Robot CAD changed before export')
            target=stage/'assets/meshes'/source.name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,target)
        groups=copy.deepcopy(state['validation'])
        for gi,group in enumerate(groups):
            for ri,row in enumerate(group['results']):
                source=Path(row['report']).resolve()
                if not source.is_relative_to(folder):raise ValueError('Validation proof outside training storage')
                relative=f'proofs/{gi}-{ri}.json';target=stage/relative;target.parent.mkdir(exist_ok=True)
                shutil.copyfile(source,target);row['report']=relative
        write(stage/'qualification.json',dict(controller=state['controller'],runtime_files=runtime,validation=groups,
            training_seeds=sorted({r['seed'] for r in state['episodes'] if r.get('split','train')=='train'}),
            experience_source_seeds=state.get('experience',{}).get('source_seeds',[]),
            quarantined_holdouts=[s for h in state.get('validation_history',[]) for g in h['seeds'] for s in g],
            full_archive_sha256=archive['sha256'],full_archive_kind=archive.get('kind')))
        experience=state.get('experience',{})
        write(stage/'actor-experience.json',dict(sha256=experience.get('sha256'),ik_seeds=experience.get('ik_seeds',[])))
        manifest=dict(format=FORMAT,created=time.time(),simulation_ready=True,physical_ready=False,
            automatic_execution_allowed=False,hardware_trial_required=True,task_id=state['task_id'],
            files={str(p.relative_to(stage)):sha(p) for p in stage.rglob('*') if p.is_file()})
        write(stage/'observed-transfer.json',manifest);validate(stage)
        output=destination.with_suffix('.pending')
        with tarfile.open(output,'w:gz') as package:
            for relative in sorted([*manifest['files'],'observed-transfer.json']):package.add(stage/relative,arcname=relative,recursive=False)
        with tempfile.TemporaryDirectory(prefix='.observed-check-',dir=destination.parent) as checked:
            extract(output,Path(checked));validate(checked)
        output.replace(destination)
    return dict(path=str(destination),sha256=sha(destination),verified=True,kind=FORMAT,
        simulation_ready=True,physical_ready=False,hardware_trial_required=True)


def extract(archive,destination,required_manifest='observed-transfer.json'):
    """No links, traversal, executable hooks or unchecked extracted content."""
    destination=Path(destination)
    with tarfile.open(archive,'r:gz') as package:
        members=package.getmembers()
        if len(members)>MAX_FILES or sum(m.size for m in members)>MAX_BYTES:raise ValueError('Observed import exceeds size limit')
        seen=set()
        for item in members:
            name(item.name)
            if not item.isfile() or item.name in seen or item.size<0:raise ValueError('Observed package has links or duplicate entries')
            seen.add(item.name)
        if required_manifest not in seen:raise ValueError('Observed manifest missing')
        for item in members:
            target=destination/item.name;target.parent.mkdir(parents=True,exist_ok=True)
            with package.extractfile(item) as source,target.open('xb') as output:shutil.copyfileobj(source,output)


def import_skill(archive,robot_root,register=False):
    if type(register) is not bool:raise ValueError('Explicit candidate registration required')
    archive=Path(archive);root=Path(robot_root).resolve()
    if not archive.is_file() or archive.stat().st_size>MAX_BYTES:raise ValueError('Observed archive missing or oversized')
    digest=sha(archive);base=root/'data'/('observed-skill-candidates' if register else 'observed-skill-imports');base.mkdir(parents=True,exist_ok=True)
    target=base/digest
    if target.exists():
        checked=validate(target/'package');record=read(target/'candidate.json')
        if record.get('archive_sha256')!=digest:raise ValueError('Conflicting candidate')
        return dict(path=str(target),candidate=record,registered=register,physical_ready=False)
    with tempfile.TemporaryDirectory(prefix='.observed-import-',dir=base) as temporary:
        stage=Path(temporary);package=stage/'package';package.mkdir();extract(archive,package);checked=validate(package)
        mismatches=[]
        for relative,digest2 in checked['qualification']['controller']['files'].items():
            if relative.startswith('config/') and (not (root/relative).is_file() or sha(root/relative)!=digest2):mismatches.append(relative)
        record=dict(id=digest,archive_sha256=digest,task_id=checked['policy']['task_id'],registered=register,
            simulation_ready=True,physical_ready=False,automatic_execution_allowed=False,
            hardware_trial_required=True,calibration_mismatches=mismatches,at=time.time())
        write(stage/'candidate.json',record);stage.rename(target)
    return dict(path=str(target),candidate=record,registered=register,physical_ready=False)


def verified_archive(archive):
    archive=Path(archive)
    with tempfile.TemporaryDirectory(prefix='observed-verify-') as temporary:
        extract(archive,temporary);checked=validate(temporary)
    return dict(path=str(archive),sha256=sha(archive),verified=True,kind=FORMAT,
        policy=checked['policy'],simulation_ready=True,physical_ready=False,hardware_trial_required=True)


def import_curriculum(archive,robot_root,register=False):
    """Check every independent task before importing any inactive candidate."""
    with tempfile.TemporaryDirectory(prefix='observed-curriculum-') as temporary:
        stage=Path(temporary);extract(archive,stage,required_manifest='manifest.json')
        manifest=read(stage/'manifest.json')
        if (manifest.get('format')!='explorer_observed_curriculum_transfer_v1' or
                manifest.get('simulation_ready') is not True or manifest.get('physical_ready') is not False or
                manifest.get('automatic_hardware_execution') is not False):raise ValueError('Invalid observed curriculum')
        tasks=manifest.get('tasks',[]);names={name(t['file']) for t in tasks}
        if not tasks or len(tasks)>4 or len(names)!=len(tasks) or len({t['task_id'] for t in tasks})!=len(tasks):
            raise ValueError('Duplicate or missing curriculum task')
        if {str(p.relative_to(stage)) for p in stage.rglob('*') if p.is_file()}!=names|{'manifest.json'}:
            raise ValueError('Curriculum contains unchecked files')
        for task in tasks:
            path=stage/task['file']
            if sha(path)!=task['sha256']:raise ValueError('Curriculum child checksum mismatch')
            checked=verified_archive(path)
            if checked['policy']['task_id']!=task['task_id'] or checked['policy']['scene_recipe']!=task['recipe']:
                raise ValueError('Curriculum task differs from qualified child')
        results=[import_skill(stage/t['file'],robot_root,register) for t in tasks]
    return dict(tasks=results,simulation_ready=True,physical_ready=False,automatic_execution_allowed=False)
def runtime_snapshot():
    """Tracked install source only; never glob local tokens, data or secrets."""
    source=Path(__file__).resolve().parents[1]
    frozen=source/'source-manifest.json'
    if frozen.is_file():names=read(frozen)['files']
    else:
        names=subprocess.check_output(['git','ls-files','--','simulator','src','bin','config'],cwd=source,text=True).splitlines()
    files={relative:sha(source/relative) for relative in names if relative.startswith(('simulator/','src/','bin/','config/')) and (source/relative).is_file()}
    # The checked execution set is mandatory even in a development fixture.
    from .skill_workshop import execution_fingerprint
    files.update(execution_fingerprint()['files'])
    return files
