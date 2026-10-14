import copy
import importlib.util
import json
from pathlib import Path

import pytest

from simulator.model_comparison import paired_result
from simulator.learning_cycle import LearningCycles
from simulator.learning import inherited_split
from simulator.train_entry import frozen_processor_call


def test_pairing_retains_reference_on_tie_and_rejects_incomplete_or_error_results():
    rows=[dict(model_id=model,seed=seed,success=seed%2==0,error=None)
          for model in ('old','new') for seed in range(10)]
    result=paired_result(rows,'new','old')
    assert result['improved'] is False and result['selected_model_id']=='old'
    rows[-1]['success']=True
    assert paired_result(rows,'new','old')['improved'] is True
    assert paired_result(rows,'new','old')['qualification'] is False
    with pytest.raises(ValueError):paired_result(rows[:-1],'new','old')
    rows[-1]['error']='physics failed'
    with pytest.raises(ValueError,match='Execution errors'):paired_result(rows,'new','old')


def test_no_improvement_stops_before_another_generation_or_training(tmp_path):
    store=LearningCycles(tmp_path);cycle=store.create(['toy_box'],models={'toy_box':'a'*12})
    cycle['tasks'][0]['model_id']='b'*12;cycle['tasks'][0]['iteration']=2;store.save(cycle)
    calls=[]
    def execute(stage,options,cancelled):
        calls.append(stage)
        assert options['reference_model_id']=='a'*12
        return {'improved':False,'candidate_successes':4,'reference_successes':6,'episodes':10}
    result=store.run(cycle['id'],execute,lambda:False,lambda value:None)
    assert result['state']=='needs_diagnosis' and calls==['compare']
    assert result['tasks'][0]['reference_model_id']=='a'*12
    assert result['physical_ready'] is False
    store.run(cycle['id'],execute,lambda:False,lambda value:None)
    assert calls==['compare']


def test_verified_teacher_correction_runs_once_from_reference(tmp_path):
    store=LearningCycles(tmp_path);cycle=store.create(['toy_box'],models={'toy_box':'a'*12})
    cycle['options']['teacher_profile']='planned_grasp_v3'
    task=cycle['tasks'][0]
    task.update(model_id='b'*12,iteration=2,corrective_teacher_evidence_sha256='c'*64)
    store.save(cycle)
    calls=[]
    def execute(stage,options,cancelled):
        calls.append((stage,options))
        if stage=='compare':return {'improved':False,'candidate_successes':4,'reference_successes':6,'episodes':10}
        if stage=='trials':
            assert options['teacher_profile']=='planned_grasp_v3'
            return {'episodes':100}
        if stage=='train':
            assert options['warm_start_model_id']=='a'*12
            return {'model_id':'d'*12}
        raise AssertionError(stage)
    def stop():return sum(stage=='train' for stage,_ in calls)>0
    result=store.run(cycle['id'],execute,stop,lambda value:None)
    corrected=result['tasks'][0]
    assert [stage for stage,_ in calls]==['compare','trials','train']
    assert corrected['rejected_model_id']=='b'*12
    assert corrected['corrective_round_used'] is True
    assert corrected['reference_model_id']=='a'*12


def test_inherited_split_never_relabels_a_seen_training_scene_as_validation(tmp_path):
    source=tmp_path/'source'
    def save(kind,seeds):
        path=source/kind;path.mkdir(parents=True)
        (path/'explorer-provenance.json').write_text(json.dumps({'episodes':[{'episode':{'source':'simulation','seed':seed}} for seed in seeds]}))
    save('train',[1,2,3]);save('validation',[4])
    seeds=[4,1,2,3,4,5,6,7,8,9,10]
    folders=[Path(str(index)) for index in range(len(seeds))]
    training,validation,independent=inherited_split(folders,[dict(seed=seed,source='simulation') for seed in seeds],source)
    assert independent
    assert all(folders[index] in training for index in (1,2,3))
    assert folders[0] in validation and folders[4] in validation
    training_seeds={seed for folder,seed in zip(folders,seeds) if folder in training}
    heldout_seeds={seed for folder,seed in zip(folders,seeds) if folder in validation}
    assert not training_seeds & heldout_seeds


def test_finetuning_preserves_saved_statistics_without_mutating_dataset_statistics(tmp_path):
    captured={}
    def make(*args,**kwargs):captured.update(kwargs);return 'processors'
    stats={'observation.state':{'mean':[1,2,3]}}
    overrides={'normalizer_processor':{'stats':stats,'features':'verified'},'device_processor':{'device':'cpu'}}
    original=copy.deepcopy(overrides)
    call=frozen_processor_call(make,tmp_path)
    assert call(pretrained_path=str(tmp_path),dataset_stats=stats,
        preprocessor_overrides=overrides,postprocessor_overrides={'unnormalizer_processor':{'stats':stats}})=='processors'
    assert 'dataset_stats' not in captured
    assert 'stats' not in captured['preprocessor_overrides']['normalizer_processor']
    assert captured['preprocessor_overrides']['normalizer_processor']['features']=='verified'
    assert overrides==original and stats=={'observation.state':{'mean':[1,2,3]}}
    with pytest.raises(ValueError):call(pretrained_path=str(tmp_path/'other'))


def test_corrective_install_boundary_waits_for_training_and_respects_operator_pause():
    path=Path(__file__).parents[1]/'bin/install-corrective-cycle.py'
    spec=importlib.util.spec_from_file_location('install_corrective_cycle',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    active={'jobs':[{'id':'a'*12,'status':'running'}]}
    finished={'jobs':[{'id':'a'*12,'status':'complete'}]}
    cycle={'pause_kind':None,'state':'running','tasks':[{'model_id':'a'*12}]}
    assert module.boundary(active,cycle,'a'*12)=='waiting_for_training'
    assert module.boundary(finished,cycle,'a'*12)=='ready'
    assert module.boundary({'jobs':[]},cycle,'a'*12,'validated_simulation')=='ready'
    assert module.boundary(finished,{'pause_kind':None,'state':'running','tasks':[{'model_id':'b'*12}]},'a'*12)=='waiting_for_training'
    assert module.boundary(finished,{'pause_kind':'operator'},'a'*12)=='operator_paused'
