import json
from pathlib import Path

import pytest

from simulator.learning import completed_training_steps, resume_checkpoint, sha256, train_job, write_json
from simulator.learning import prepared_training_available,resume_prepared_training


def prepared_run(root):
    import hashlib
    from src.mobile_policy_contract import CONTRACT_SHA256
    folder=root/'prepared';sources=[];inputs=[]
    for i,part in enumerate(('train','validation')):
        source=root/('source-'+part);source.mkdir()
        write_json(source/'episode.json',dict(id=source.name,seed=i))
        (source/'samples.jsonl').write_text('original samples')
        hashes={name:sha256(source/name) for name in ('episode.json','samples.jsonl')}
        write_json(folder/part/'explorer-provenance.json',dict(sample_contract_sha256=CONTRACT_SHA256,
            action_transform=None,episodes=[dict(episode=dict(id=source.name,seed=i),input_sha256=hashes,samples=4)]))
        write_json(folder/part/'meta/info.json',dict(total_frames=4))
        (folder/part/'data').mkdir();(folder/part/'data/shard.parquet').write_bytes(b'verified export')
        sources.append(source);inputs.append(hashes)
    (folder/'initial-policy').mkdir();(folder/'initial-policy/model.safetensors').write_bytes(b'initial model')
    write_json(folder/'split.json',dict(train=[sources[0].name],validation=[sources[1].name]))
    write_json(folder/'train-config.json',dict(output_dir=str(folder/'model'),dataset=dict(root=str(folder/'train')),
        policy=dict(pretrained_path=str(folder/'initial-policy')),steps=200,batch_size=16,seed=7))
    write_json(folder/'job.json',dict(id=folder.name,state='cancelled',skill_id='sock_basket',steps_completed=0,
        episodes=[p.name for p in sources],steps=200,batch_size=16,seed=7,
        dataset_fingerprint=hashlib.sha256(json.dumps(inputs,sort_keys=True).encode()).hexdigest()))
    return folder,sources


def test_prepared_export_reuses_same_identity_without_rewriting_data(tmp_path):
    folder,sources=prepared_run(tmp_path);before={str(p):p.read_bytes() for p in folder.rglob('*') if p.is_file()}
    assert prepared_training_available(folder)
    state,config=resume_prepared_training(folder,sources,'sock_basket')
    assert state['prepared_export_reused'] and state['steps']==200 and config==folder/'train-config.json'
    assert all(Path(p).read_bytes()==content for p,content in before.items())
    resume_prepared_training(folder,sources,'sock_basket')
    (folder/'train/data/shard.parquet').write_bytes(b'tampered')
    with pytest.raises(ValueError,match='checksum'):
        resume_prepared_training(folder,sources,'sock_basket')


def test_prepared_export_rejects_changed_source_or_optimizer_work(tmp_path):
    folder,sources=prepared_run(tmp_path)
    (sources[0]/'samples.jsonl').write_text('changed samples')
    with pytest.raises(ValueError,match='recording bytes'):
        resume_prepared_training(folder,sources,'sock_basket')
    assert not (folder/'prepared-export-reuse.json').exists()
    checkpoint=folder/'model/checkpoints/000100';checkpoint.mkdir(parents=True)
    assert not prepared_training_available(folder)


def test_prepared_training_path_starts_learner_without_exporting_again(tmp_path,monkeypatch):
    from types import SimpleNamespace
    folder,sources=prepared_run(tmp_path);calls=[]
    monkeypatch.setattr('src.learning_environment.configure',lambda _: {})
    monkeypatch.setattr('simulator.learning.export_dataset',lambda *a,**k:pytest.fail('must reuse export'))
    def spawn(args,**kwargs):calls.append(args);return SimpleNamespace(poll=lambda:0,returncode=0)
    monkeypatch.setattr('simulator.learning.subprocess.Popen',spawn)
    monkeypatch.setattr('simulator.learning.select_checkpoint',lambda *a,**k:({'checkpoint_relative':'fake'},{'improves_hold_baseline':True}))
    monkeypatch.setattr('simulator.learning.create_bundle',lambda *a:None)
    result=train_job(folder,sources,skill_id='sock_basket',reuse_prepared=True)
    assert result['prepared_export_reused'] and result['steps_completed']==200
    assert '--config_path='+str(folder/'train-config.json') in calls[0]
    assert '--resume=true' not in calls[0]


def test_progress_counts_remaining_tqdm_updates_after_resume():
    assert completed_training_steps('Training: 118/58297 [00:34',145741,87444)==87562
    assert completed_training_steps('Training: 145741/145741 [',145741,87444)==145741
    assert completed_training_steps('Training: 118/145741 [',145741)==118
    assert completed_training_steps('step:88K',145741,87444)==87444


def saved_run(root):
    folder=root/'run'
    checkpoint=folder/'model/checkpoints/000100'
    for name in ('pretrained_model/model.safetensors','training_state/optimizer_state.safetensors',
                 'training_state/optimizer_param_groups.json','training_state/rng_state.safetensors'):
        path=checkpoint/name
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_bytes(b'saved state')
    write_json(checkpoint/'pretrained_model/train_config.json',{
        'output_dir':str(folder/'model'),'dataset':{'root':str(folder/'train')},
        'steps':200,'batch_size':16})
    write_json(checkpoint/'training_state/training_step.json',{'step':100})
    (checkpoint.parent/'last').symlink_to(checkpoint.name,target_is_directory=True)
    write_json(folder/'job.json',{'id':'run','state':'cancelled','steps':200,'batch_size':16,
        'skill_id':'sock_basket','episodes':['one','two'],'steps_completed':110,'error':'cancelled'})
    for part in ('train','validation'):
        write_json(folder/part/'explorer-provenance.json',{})
    return folder,checkpoint


def test_resume_restores_durable_step_and_keeps_original_dataset(tmp_path):
    folder,checkpoint=saved_run(tmp_path)
    original=(folder/'job.json').read_bytes()
    state,config=resume_checkpoint(folder,[tmp_path/'one',tmp_path/'two'],'sock_basket')
    assert state['steps_completed']==state['resumed_step']==100
    assert 'error' not in state
    assert config==checkpoint/'pretrained_model/train_config.json'
    assert (folder/'job.json').read_bytes()==original


def test_resume_rejects_missing_optimizer_or_changed_data_without_overwriting_job(tmp_path):
    folder,checkpoint=saved_run(tmp_path)
    original=(folder/'job.json').read_bytes()
    with pytest.raises(ValueError,match='same skill'):
        train_job(folder,[tmp_path/'one'],skill_id='sock_basket',resume=True)
    assert (folder/'job.json').read_bytes()==original
    (checkpoint/'training_state/rng_state.safetensors').unlink()
    with pytest.raises(ValueError,match='RNG'):
        resume_checkpoint(folder,[tmp_path/'one',tmp_path/'two'],'sock_basket')


def test_resume_rejects_modified_checkpoint_and_outside_symlink(tmp_path):
    folder,checkpoint=saved_run(tmp_path)
    model=checkpoint/'pretrained_model/model.safetensors'
    write_json(folder/'process-pause.json',{'checkpoint_path':str(checkpoint),
        'checkpoint_sha256':{'pretrained_model/model.safetensors':sha256(model)}})
    model.write_bytes(b'changed')
    with pytest.raises(ValueError,match='checksum'):
        resume_checkpoint(folder,[tmp_path/'one',tmp_path/'two'],'sock_basket')
    last=checkpoint.parent/'last'
    last.unlink(); last.symlink_to(tmp_path,target_is_directory=True)
    with pytest.raises(ValueError,match='outside'):
        resume_checkpoint(folder,[tmp_path/'one',tmp_path/'two'],'sock_basket')
