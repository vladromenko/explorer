import copy
import json

from simulator.learning_cycle import LearningCycles
from simulator.feedback_parallel import MODES,choose_mode
from simulator.learning_status import compact_value,learning_status


def report(seed,successes=10):
    return dict(heldout_independent=True,episodes=10,successes=successes,
        results=[dict(success=i<successes,domain_variation={'seed':seed+i}) for i in range(10)])


def test_failed_final_test_adds_teaching_data_then_uses_new_independent_scenes(tmp_path):
    store=LearningCycles(tmp_path);cycle=store.create(['toy_box'],models={'toy_box':'a'*12},scene_profile='baseline')
    calls=[]
    def execute(stage,options,cancelled):
        calls.append((stage,copy.deepcopy(options)))
        if stage=='compare':return {'improved':True}
        if stage=='audit':return {'model_id':options['model_id'],'diagnostic_only':True}
        if stage in ('validate','final_validate'):
            seen=sum(name=='final_validate' for name,_ in calls)
            return report(options['validation_seed'],9 if stage=='final_validate' and seen==1 else 10)
        if stage=='train':return {'model_id':'b'*12}
        return {'episodes':100,'path':'package','sha256':'hash'}
    result=store.run(cycle['id'],execute,lambda:False,lambda value:None)
    assert result['state']=='ready_for_transfer' and result['physical_ready'] is False
    assert len([name for name,_ in calls if name=='final_validate'])==2
    scenes=[options['validation_seed'] for name,options in calls if name in ('validate','final_validate')]
    assert len(set(scenes))==4
    trial=next(options for name,options in calls if name=='trials')
    assert scenes[1]+9 in trial['scene_seeds']
    assert all(name!='export' for name,_ in calls[:5])
    assert next(options for name,options in calls if name=='train')['minimum_epochs']==8


def test_pause_preserves_finished_stages_and_training_identity(tmp_path):
    store=LearningCycles(tmp_path);cycle=store.create(['toy_box'],scene_profile='baseline')
    calls=[];paused_once=False
    def execute(stage,options,cancelled):
        nonlocal paused_once
        calls.append((stage,options['job_id']))
        if stage=='train' and not paused_once:
            paused_once=True;raise InterruptedError('pause')
        if stage in ('train','audit'):return {'model_id':'a'*12}
        if stage in ('validate','final_validate'):return report(options['validation_seed'])
        return {'path':'package','episodes':100}
    paused=store.run(cycle['id'],execute,lambda:False,lambda value:None)
    assert paused['state']=='paused' and 'trials' in paused['tasks'][0]['stages']
    pending=paused['tasks'][0]['pending']['job_id']
    resumed=LearningCycles(tmp_path).run(cycle['id'],execute,lambda:False,lambda value:None)
    assert resumed['state']=='ready_for_transfer'
    assert sum(name=='trials' for name,_ in calls)==1
    assert [identifier for name,identifier in calls if name=='train']==[pending,pending]


def test_operator_pause_starts_no_new_work_and_restart_requires_resume(tmp_path):
    store=LearningCycles(tmp_path);cycle=store.create(['toy_box'])
    cycle['state']='running';store.save(cycle)
    reboot=LearningCycles(tmp_path)
    assert reboot.get(cycle['id'])['state']=='interrupted'
    calls=[]
    paused=reboot.run(cycle['id'],lambda *args:calls.append(args),lambda:True,lambda value:None)
    assert paused['pause_kind']=='operator' and not calls


def test_tuning_cannot_grant_qualification_and_prefers_unchanged_mode_on_tie():
    rows=[dict(mode=mode,success=True,error=None) for mode in MODES]
    decision=choose_mode(rows)
    assert decision['mode']=='queue20' and decision['qualification'] is False
    rows[0]['success']=False
    assert choose_mode(rows)['mode']=='queue10'


def test_persisted_results_remain_visible_without_live_jobs(tmp_path):
    batches=[dict(id='a'*12,state='complete',tasks=[dict(task_id='toy_box',state='complete',simulation_qualified=False,
                stages={'validate':report(100,6),'train':{'steps_completed':100,'training_budget':{'effective_steps':100}}})])]
    snapshot=dict(jobs=[],night_batches=batches,learning_cycles=[],model={'id':'m','successes':6,'episodes':10})
    result=learning_status(tmp_path,snapshot)
    assert result['state']=='needs_more_training' and result['successes']==6 and result['steps_completed']==100
    full={'summaries':[1]*1000,'episodes':['sample']*1000,'successes':6,'training_budget':{'effective_steps':100}}
    small=compact_value(full)
    assert small['episodes']==1000 and 'summaries' not in small
    assert len(full['episodes'])==1000 and small['successes']==6


def test_progress_api_compacts_large_persisted_results_and_blocks_unqualified_download(tmp_path):
    from test_simulator_server import client
    from simulator.curriculum import NightQueue
    queue=NightQueue(tmp_path);plan=queue.create(['toy_box'],100,1000,16,8,1)
    plan['state']='complete';plan['tasks'][0].update(state='complete',simulation_qualified=False,
        stages={'validate':report(100,6),'trials':{'episodes':100,'summaries':[{'id':str(i)} for i in range(1000)]}})
    (queue.folder/(plan['id']+'.json')).write_text(json.dumps(plan))
    with client(tmp_path) as connection:
        compact=connection.get('/api/state?compact=true').json()
        assert compact['learning_status']['successes']==6
        assert compact['learning_status']['state']=='needs_more_training'
        assert 'summaries' not in compact['night_batches'][0]['tasks'][0]['stages']['trials']
        assert connection.get('/api/state').json()['night_batches'][0]['tasks'][0]['stages']['trials']['summaries']
        assert connection.get('/api/cycles/'+plan['id']+'/package').status_code==404
