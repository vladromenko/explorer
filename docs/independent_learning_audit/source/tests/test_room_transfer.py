"""Dummy weights and synthetic reports exercise packaging, never acceptance."""
import io
import json
from pathlib import Path
import tarfile

import numpy as np
import pytest

from simulator.learning import create_bundle,write_json,sha256
from simulator.room_policy import contract_sha,runtime_files
from simulator.room_transfer import export_room_package,import_room_package,validate_room_folder,extract_room
from test_room_cleanup import room_groups


@pytest.fixture
def room_fixture(tmp_path):
    model=tmp_path/'model';checkpoint=model/'model/checkpoints/last/pretrained_model';checkpoint.mkdir(parents=True)
    (checkpoint/'model.safetensors').write_bytes(b'unit test only; not learned weights')
    write_json(checkpoint/'config.json',dict(type='act',device='mps',n_action_steps=1,
        input_features={'observation.state':{'shape':[9]},'observation.images.wrist':{'shape':[3,240,320]}},
        output_features={'action':{'shape':[9]}}))
    for name in ('policy_preprocessor.json','policy_postprocessor.json'):write_json(checkpoint/name,dict(steps=[]))
    for index,kind in enumerate(('train','validation')):
        write_json(model/kind/'explorer-provenance.json',dict(dataset_fps=20,episodes=[dict(
            episode=dict(seed=index+1),input_sha256={'samples.jsonl':'a'*64})]))
    state=dict(task='sock basket fixture',episodes=[],validation=dict(heldout_independent=False))
    write_json(model/'job.json',state);bundle=create_bundle(model,state)
    profile=dict(schema='explorer_room_act_handoff_v1',model_checkpoint_sha256=bundle['checkpoint_sha256'],
        initial_command=[90]*5+[30]+[0]*3,pick_elapsed_s=6.,placement_proposals=[dict(height_m=.3,
        half_size_m=[.125,.125],elapsed_s=45.,observed_command=[90]*5+[170]+[0]*3,source_sha256='a'*64)],
        navigation_owner='observed_lidar_and_map_planner',actor_execution='arm_only_after_observed_alignment',
        actuator_quantization='integer_degree_error_diffusion_2deg_step',
        transport_height_m=.40,transport_maximum_reach_m=.28,source_standoff_m=.20,teacher_fallback=False,
        actor_inputs_unchanged=True,execution_changed=True,requires_new_room_qualification=True,physical_ready=False)
    handoff=tmp_path/'handoff.json';write_json(handoff,profile)
    contract=contract_sha(profile,runtime_files());groups=room_groups();paths=[]
    for index,group in enumerate(groups):
        for row in group['results']:row.update(mission_contract_sha256=contract,model_sha256=bundle['checkpoint_sha256']['model.safetensors'])
        p=tmp_path/f'group-{index}.json';write_json(p,group);paths.append(p)
    return model,handoff,paths


def test_whole_archive_reopens_checksums_and_import_is_inactive(room_fixture,tmp_path):
    archive=tmp_path/'room.tar.gz';value=export_room_package(*room_fixture,archive)
    assert export_room_package(*room_fixture,archive)==value
    assert value['sha256']==sha256(archive) and value['physical_ready'] is False
    imported=import_room_package(archive,tmp_path/'robot')
    assert imported['registered'] and not imported['active'] and not imported['physical_ready']
    checked=validate_room_folder(imported['path'],check_runtime=True)
    assert checked['simulation_qualified'] and not checked['automatic_execution_allowed']
    assert checked['bundle']['transfer_inference_device']=='cuda'
    p=Path(imported['path'])/'handoff.json';p.write_text('{}')
    with pytest.raises(ValueError,match='checksum'):validate_room_folder(imported['path'])


def test_training_overlap_and_wrong_controller_cannot_export(room_fixture,tmp_path):
    model,handoff,groups=room_fixture
    with pytest.raises(ValueError,match='overlap'):
        export_room_package(model,handoff,groups,tmp_path/'bad.tar.gz',excluded_seeds={1600000010})
    value=json.loads(groups[0].read_text());value['results'][0]['mission_contract_sha256']='changed';write_json(groups[0],value)
    with pytest.raises(ValueError,match='contract'):export_room_package(model,handoff,groups,tmp_path/'bad.tar.gz')


@pytest.mark.parametrize('name,type_', [('../escape',tarfile.REGTYPE),('/absolute',tarfile.REGTYPE),('linked',tarfile.SYMTYPE)])
def test_room_archive_rejects_escape_and_links(tmp_path,name,type_):
    archive=tmp_path/'bad.tar.gz'
    with tarfile.open(archive,'w:gz') as stream:
        member=tarfile.TarInfo(name);member.type=type_;member.size=1
        stream.addfile(member,io.BytesIO(b'x'))
    with pytest.raises(ValueError):extract_room(archive,tmp_path/'stage')


def test_live_transport_converts_map_tf_instead_of_treating_odom_as_map():
    from types import SimpleNamespace
    from src.room_cleanup_execution import FactoryRoomTransport,RoomCleanupExecution
    pose=dict(x=10.,y=20.,yaw=np.pi/2,age_s=0.,localization_verified=True)
    calls=[]
    maps=SimpleNamespace(pose=lambda:pose,preview=lambda x,y:(calls.append((x,y)) or dict(points=[[10.,20.],[10.,21.]])))
    robot=SimpleNamespace(settings=dict(floor_plane_base=[0.,0.,1.,0.]),missions=SimpleNamespace(maps=maps))
    owner=SimpleNamespace(robot=robot)
    t=FactoryRoomTransport(owner);t.permit=lambda:None
    first=t._base(dict(raw_pose=dict(x=0.,y=0.,yaw=0.)))
    np.testing.assert_allclose(first,np.eye(4),atol=1e-12)
    t.state=lambda:None
    plan=t.plan(dict(x=1.,y=0.,yaw=0.))
    np.testing.assert_allclose(calls[0],(10.,21.));np.testing.assert_allclose(plan['points'],[[0,0],[1,0]],atol=1e-12)
    pose['x']+=.2
    with pytest.raises(ValueError,match='Локализация'):t._base(dict(raw_pose=dict(x=0.,y=0.,yaw=0.)))


def test_room_owner_inspection_does_not_actuate_or_admit_local_only_candidate(tmp_path):
    from types import SimpleNamespace
    from src.room_cleanup_execution import RoomCleanupExecution
    events=[];owner=RoomCleanupExecution(tmp_path,SimpleNamespace(),lambda *a,**k:events.append(a),lambda:[],lambda _:None)
    check=owner.preflight('a'*64)
    assert not events and not check['ready'] and not check['simulation_ready']
    with pytest.raises(ValueError):owner.start('a'*64,True)
    assert not owner.lock.locked() and not events


def test_whole_cycle_archive_preserves_failed_evidence_and_checks_every_member(tmp_path):
    from simulator.room_archive import archive_cycle,verify
    cycle=tmp_path/'cycle';cycle.mkdir();write_json(cycle/'failure.json',dict(success=False,reason='missed'))
    transfer=tmp_path/'transfer.tar.gz';transfer.write_bytes(b'unit fixture only')
    output=archive_cycle(cycle,dict(path=str(transfer),sha256=sha256(transfer)),tmp_path/'cycle.tar.gz')
    manifest=verify(output['path'])
    assert manifest['files']['evidence/failure.json']==sha256(cycle/'failure.json') and not manifest['physical_ready']
    assert output['verified'] and output['sha256']==sha256(output['path'])


def test_whole_cycle_archival_can_pause_without_removing_originals(tmp_path):
    from simulator.room_archive import archive_cycle
    cycle=tmp_path/'cycle';cycle.mkdir();(cycle/'frame.bin').write_bytes(b'raw frame')
    transfer=tmp_path/'transfer.tar.gz';transfer.write_bytes(b'fixture')
    with pytest.raises(InterruptedError):archive_cycle(cycle,dict(path=str(transfer),sha256=sha256(transfer)),tmp_path/'out.tar.gz',cancelled=lambda:True)
    assert (cycle/'frame.bin').read_bytes()==b'raw frame' and not (tmp_path/'out.tar.gz').exists()


def test_only_frozen_unchanged_source_can_supply_acceptance_evidence(tmp_path,monkeypatch):
    import simulator.room_validation as validation
    import simulator.room_policy as policy
    source=tmp_path/'source';(source/'simulator').mkdir(parents=True)
    module=source/'simulator/room_validation.py';module.write_text('unit fixture')
    monkeypatch.setattr(validation,'__file__',str(module))
    monkeypatch.setattr(policy,'RUNTIME_FILES',('simulator/room_validation.py',))
    assert validation.frozen_source() is None
    write_json(source/'source-revision.json',dict(revision='a'*40,files={'simulator/room_validation.py':sha256(module)}))
    assert validation.frozen_source()=='a'*40
    module.write_text('changed implementation')
    with pytest.raises(ValueError,match='source changed'):validation.frozen_source()


def test_hundred_scene_protocol_preserves_all_physical_gates(room_fixture):
    import copy
    from simulator.room_transfer import qualification
    model,handoff,paths=room_fixture
    bundle=json.loads((model/'bundle.json').read_text());profile=json.loads(handoff.read_text())
    groups=[json.loads(path.read_text()) for path in paths]
    for index,group in enumerate(groups):
        originals=group['results'];expanded=[]
        for repeat in range(5):
            for original in originals:
                row=copy.deepcopy(original);row['seed']+=repeat*1000+index*10000
                row['physical_outcome']['checked_after_actions_stopped']=True
                expanded.append(row)
        group.update(results=expanded,episodes=50,successes=50)
    seeds,_=qualification(groups,profile,bundle)
    assert len(seeds)==100
    groups[1]['results'][-1]['collision_free']=False
    with pytest.raises(ValueError,match='physical and observed success'):qualification(groups,profile,bundle)
