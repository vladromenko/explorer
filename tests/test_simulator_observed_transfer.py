"""Synthetic evidence here is a unit-test fixture, never a qualified robot skill."""
import copy
import io
import json
from pathlib import Path
import shutil
import tarfile

import pytest

from simulator.observed_transfer import export_skill,import_skill,sha,write,validate,extract
from simulator.scene_recipes import compile_recipes,reset_options
from simulator.skill_training import SkillTraining


@pytest.fixture
def qualified_fixture(tmp_path):
    root=tmp_path/'simulation';root.mkdir()
    recipe=compile_recipes(dict(fully_supported=True,task_ids=['sock_basket']),
        [dict(task_id='sock_basket',basket_height_cm=30,basket_diameter_cm=25)])['recipes'][0]
    trainer=SkillTraining(root,'ab'*6);state=trainer.create(attempts=8,scene_recipe=recipe)
    state.update(state='simulation_qualified',simulation_ready=True,validation=[])
    policy=dict(schema=1,algorithm=state['algorithm'],task_id='sock_basket',scene_recipe=recipe,
        parameters=state['incumbent'],controller_program=state['controller_program'],
        controller_sha256=state['controller']['sha256'],experience_sha256=None,
        simulation_ready=True,physical_ready=False)
    write(trainer.folder/'policy.json',policy);state['policy_sha256']=sha(trainer.folder/'policy.json')
    state['episodes']=[dict(seed=123,status='complete',split='train')]
    for gi,base in enumerate([1300160000,1400160000,1500160000]):
        rows=[]
        for i in range(10):
            seed=base+i;proof=trainer.folder/f'episodes/{gi}-{i}/skill-workshops/attempt/state.json'
            write(proof,dict(unit_test_fixture=True,task_id='sock_basket',seed=seed,success=True,
                execution=dict(state='succeeded',held_verified=True,placed_verified=True),
                independent_physics_outcome=dict(success=True),execution_fingerprint=state['controller'],
                policy_parameters=policy['parameters'],controller_program=policy['controller_program'],scene_recipe=recipe))
            variation=reset_options(recipe,seed,validation=gi<2)['generated_scene']
            rows.append(dict(seed=seed,status='complete',success=True,split='validation' if gi<2 else 'user_validation',
                domain_variation=variation,report=str(proof)))
        state['validation'].append(dict(heldout_independent=True,episodes=10,successes=10,
            scene_profile='sock_variants_v1' if gi<2 else 'user_constraints',results=rows))
        state['episodes']+=rows
    state['archive']=trainer._archive(state);trainer.save(state)
    return trainer,state


def test_new_package_roundtrip_is_inert_and_keeps_proofs_and_policy(qualified_fixture,tmp_path):
    trainer,state=qualified_fixture
    archive=tmp_path/'learned.tar.gz';result=export_skill(trainer.folder,archive)
    assert result['verified'] and result['physical_ready'] is False
    robot=tmp_path/'jetson';robot.mkdir()
    with tarfile.open(state['archive']['path']) as f:
        assert 'episodes/0-0/skill-workshops/attempt/state.json' in f.getnames()
    imported=import_skill(archive,robot,register=True)
    assert imported['candidate']['simulation_ready'] and not imported['candidate']['automatic_execution_allowed']
    assert imported['candidate']['calibration_mismatches']
    checked=validate(Path(imported['path'])/'package')
    assert checked['policy']['parameters']==state['incumbent']
    assert len(checked['qualification']['validation'])==3
    assert import_skill(archive,robot,True)['path']==imported['path']


def test_unqualified_or_changed_policy_cannot_be_exported(qualified_fixture,tmp_path):
    trainer,state=qualified_fixture;state['state']='running';trainer.save(state)
    with pytest.raises(ValueError,match='not independently'):export_skill(trainer.folder,tmp_path/'bad.tar.gz')
    state['state']='simulation_qualified';trainer.save(state)
    policy=json.loads((trainer.folder/'policy.json').read_text());policy['parameters']['pick_fraction']=.3
    write(trainer.folder/'policy.json',policy)
    with pytest.raises(ValueError,match='policy changed'):export_skill(trainer.folder,tmp_path/'bad.tar.gz')


def test_tampered_validation_and_overlap_fail_even_with_recomputed_file_checksums(qualified_fixture,tmp_path):
    trainer,_=qualified_fixture;archive=tmp_path/'learned.tar.gz';export_skill(trainer.folder,archive)
    folder=tmp_path/'unpacked';folder.mkdir();extract(archive,folder)
    report=json.loads((folder/'proofs/0-0.json').read_text());report['independent_physics_outcome']['success']=False
    write(folder/'proofs/0-0.json',report)
    m=json.loads((folder/'observed-transfer.json').read_text());m['files']['proofs/0-0.json']=sha(folder/'proofs/0-0.json');write(folder/'observed-transfer.json',m)
    with pytest.raises(ValueError,match='both pass'):validate(folder)


def test_user_basket_dimensions_are_rechecked_on_import(qualified_fixture,tmp_path):
    trainer,_=qualified_fixture;archive=tmp_path/'learned.tar.gz';export_skill(trainer.folder,archive)
    folder=tmp_path/'unpacked';folder.mkdir();extract(archive,folder)
    q=json.loads((folder/'qualification.json').read_text())
    q['validation'][2]['results'][0]['domain_variation']['parameters']['bin']['height_m']=.2
    write(folder/'qualification.json',q)
    m=json.loads((folder/'observed-transfer.json').read_text());m['files']['qualification.json']=sha(folder/'qualification.json');write(folder/'observed-transfer.json',m)
    with pytest.raises(ValueError,match='basket height'):validate(folder)


@pytest.mark.parametrize('member',['../escape','/absolute','x/../../escape'])
def test_path_traversal_is_rejected_before_extracting_files(tmp_path,member):
    archive=tmp_path/'bad.tar.gz'
    with tarfile.open(archive,'w:gz') as f:
        info=tarfile.TarInfo(member);info.size=1;f.addfile(info,io.BytesIO(b'x'))
    dest=tmp_path/'out';dest.mkdir()
    with pytest.raises(ValueError,match='Unsafe'):extract(archive,dest)
    assert not list(dest.iterdir())


def test_installer_default_plan_is_inert_and_preserves_physical_calibration(qualified_fixture,tmp_path,monkeypatch):
    import importlib.util
    trainer,state=qualified_fixture;archive=tmp_path/'learned.tar.gz';export_skill(trainer.folder,archive)
    robot=tmp_path/'jetson';robot.mkdir();imported=import_skill(archive,robot,True)
    script=Path(__file__).resolve().parents[1]/'bin/prepare-observed-runtime.py'
    spec=importlib.util.spec_from_file_location('observed_setup',script);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    monkeypatch.setattr(module.subprocess,'run',lambda *a,**kw:pytest.fail('Inspection cannot install or execute'))
    value=module.plan(imported['path'],robot)
    assert not value['automatic_execution_allowed'] and not value['physical_ready']
    assert 'src/observed_skill_execution.py' in value['source_files']
    path=robot/'config/handeye-accepted.json';path.parent.mkdir();path.write_text('{}')
    value=module.plan(imported['path'],robot)
    assert 'config/handeye-accepted.json' not in value['source_files']
    assert any('handeye-accepted' in reason for reason in value['blocked_by'])
