import json
from pathlib import Path

import pytest

from simulator.learning import completed_training_steps, resume_checkpoint, sha256, train_job, write_json


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
