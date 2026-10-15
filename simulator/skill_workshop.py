"""Finite, persisted skill preparation and observed execution diagnostics.

Preparation never schedules another training round and never qualifies a model.
Sensor-driven task outcomes and independent simulator truth are saved separately.
"""
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import time

from .learning import write_json

STAGE_TITLES=dict(find='Поиск предмета',approach='Подъезд',reobserve='Новое измерение',
    pregrasp='Подход руки',grasp='Захват',lift='Подъём',verify_hold='Проверка удержания',
    carry='Поиск корзины и перенос',find_destination='Поиск корзины',support='Размещение над корзиной',
    release='Отпускание',withdraw='Отвод руки',verify_place='Проверка размещения')


def execution_fingerprint():
    root=Path(__file__).resolve().parents[1]
    files={name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in
        ('src/task_executor.py','simulator/skill_port.py','simulator/rgbd_observation.py',
         'simulator/placement_evidence.py','simulator/inspection_planning.py')}
    return dict(files=files,sha256=hashlib.sha256(json.dumps(files,sort_keys=True).encode()).hexdigest())


def list_workshops(root):
    rows=[]
    for path in (Path(root)/'skill-workshops').glob('*/state.json'):
        try:
            value=json.loads(path.read_text())
            if (not isinstance(value,dict) or type(value.get('created')) not in (float,int)
                or not math.isfinite(value['created']) or not isinstance(value.get('id'),str)):continue
            rows.append(value)
        except (OSError,ValueError):continue
    return sorted(rows,key=lambda row:row['created'],reverse=True)


def prepare(root,identifier,task_id,cancelled=None,progress=None):
    from .phase_learning import prepare_phase_dataset,fit_phase_residuals
    root=Path(root);folder=root/'skill-workshops'/identifier
    folder.mkdir(parents=True,exist_ok=True)
    saved=json.loads((folder/'state.json').read_text()) if (folder/'state.json').is_file() else None
    if saved and (saved.get('task_id')!=task_id or saved.get('controller')!='observed_hierarchical_skills_v1'):
        raise ValueError('Подготовка относится к другой задаче или версии исполнителя')
    if saved and saved.get('state')=='prepared':return saved
    state=dict(id=identifier,task_id=task_id,created=saved['created'] if saved else time.time(),state='preparing',
        controller='observed_hierarchical_skills_v1',simulation_ready=False,physical_ready=False,
        automatic_execution_allowed=False,learned_residuals_executed=False)
    write_json(folder/'state.json',state)
    try:
        records=[]
        for path in (root/'demonstrations').glob('*/episode.json'):
            if cancelled and cancelled():raise InterruptedError('Подготовка навыков остановлена')
            try:meta=json.loads(path.read_text())
            except (OSError,ValueError):continue
            if meta.get('task_id')==task_id and meta.get('success') is True and meta.get('attempt_completed') is True:
                records.append((path.parent,meta))
        records.sort(key=lambda row:row[1].get('finished_at',0))
        if not records:raise ValueError('Нет завершённых успешных показов для этой задачи')
        signature=records[-1][1].get('model_provenance')
        compatible=[path for path,meta in records if meta.get('model_provenance')==signature
                    and meta.get('generator')=='ground_truth_teacher']
        if saved and saved.get('selected_episodes'):compatible=[Path(path) for path in saved['selected_episodes']]
        state['selected_episodes']=[str(path) for path in compatible]
        state.update(compatible_episodes=len(compatible),excluded_geometry_versions=len(records)-len(compatible))
        write_json(folder/'state.json',state)
        updated_at=0.
        def update(value):
            nonlocal updated_at
            now=time.monotonic()
            if now-updated_at<.5:return
            updated_at=now
            state['preparation']=value;write_json(folder/'state.json',state)
            if progress:progress(dict(message='Подготовка отдельных навыков',
                progress=100*value.get('processed',0)/max(1,value.get('total',1)),**value))
        if not (folder/'dataset.json').is_file():
            prepare_phase_dataset(compatible,folder/'dataset.json',task_id=task_id,cancelled=cancelled,progress=update)
        if cancelled and cancelled():raise InterruptedError('Подготовка навыков остановлена')
        if not (folder/'phase-residuals.json').is_file():
            fit_phase_residuals(folder/'dataset.json',folder/'phase-residuals.json',cancelled=cancelled,progress=update)
        state.update(state='prepared',message='Короткие навыки подготовлены для диагностики; исполнение и перенос ещё не приняты',
            dataset=str(folder/'dataset.json'),artifact=str(folder/'phase-residuals.json'))
    except Exception as error:
        state.update(state='paused' if isinstance(error,InterruptedError) else 'failed',reason=str(error))
        write_json(folder/'state.json',state);raise
    state['finished']=time.time();write_json(folder/'state.json',state)
    return state


def trial(env,root,identifier,task_id,seed,scene_profile='sock_variants_v1',cancelled=None,progress=None):
    from src.task_executor import TaskExecutor
    from .skill_port import SimulatorSkillPort
    from .explorer_model import model_metadata
    root=Path(root);folder=root/'skill-workshops'/identifier
    folder.mkdir(parents=True,exist_ok=False)
    report=dict(id=identifier,task_id=task_id,seed=seed,created=time.time(),state='running',
        controller='observed_hierarchical_skills_v1',phase='find',simulation_ready=False,physical_ready=False,
        automatic_execution_allowed=False,learned_residuals_executed=False,scene_profile=scene_profile,
        semantic_identity='unverified_single_geometric_target',environment_provenance=model_metadata())
    report['execution_fingerprint']=execution_fingerprint()
    write_json(folder/'state.json',report)
    port=None
    try:
        env.reset(seed=seed,options=dict(task_id=task_id,scene_profile=scene_profile))
        port=SimulatorSkillPort(env,folder/'execution',cancelled or (lambda:False),progress or (lambda value:None))
        executor=TaskExecutor(port.operations(),port.observe,guard=lambda ctx:port.check(),
            cancelled=cancelled,total_timeout=300.,stage_timeout=90.,max_find_attempts=1,
            stop=lambda ctx:port.stop_motion())
        result=executor.execute(dict(task_id=task_id))
        port.stop_motion()
        # Physics truth is read ONLY after the actor has stopped. It cannot
        # supply targets or change the executor's observed outcome.
        truth=dict(env.state()['task'])
        report.update(state='diagnosed' if result.state!='cancelled' else 'paused',phase=result.stage,
            execution=asdict(result),independent_physics_outcome=truth,
            success=result.state=='succeeded' and truth.get('success') is True,
            reason=result.reason,geometry_calibration='estimated',qualification=False)
        write_json(folder/'state.json',report)
    except Exception as error:
        report.update(state='paused' if isinstance(error,InterruptedError) else 'failed',reason=str(error))
        write_json(folder/'state.json',report);raise
    finally:
        if port is not None:port.close()
    return report
