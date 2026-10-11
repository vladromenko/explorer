"""Curriculum geometry reaches the physics, never the learned actor inputs."""
import copy
import json

import numpy as np
import pytest

from simulator.domain_randomization import PROFILE, sample_sock_scene, scene_coverage
from simulator.environment import ExplorerEnv
from simulator.explorer_model import load_model, model_config
from simulator.curriculum import NightQueue, report_passes
from src.simulation_task_contract import simulation_qualification


def test_reproducible_variants_cover_five_heights_and_distinct_materials():
    scenes = [sample_sock_scene(seed) for seed in range(100)]
    assert sample_sock_scene(7) == sample_sock_scene(7)
    assert len({scene['parameters']['sock']['length_m'] for scene in scenes}) == 100
    assert len({scene['parameters']['sock']['friction'] for scene in scenes}) == 100
    coverage = scene_coverage([dict(domain_variation=scene) for scene in scenes])
    assert coverage['basket_heights_m'] == [.08, .12, .16, .20, .30]
    assert coverage['all_height_categories_present']
    assert coverage['physical_parameters_measured'] is False
    assert coverage['user_basket_25cm_diameter_30cm_rim_episodes'] == 10


def test_compiled_geometry_and_cloth_mass_match_sample_without_changing_defaults():
    before = model_config()
    scene = sample_sock_scene(3)
    model, _ = load_model(scene='sock', scene_parameters=scene['parameters'])
    assert model.geom('bin_left').size[2] * 2 == pytest.approx(.20)
    assert model.geom('bin_bottom').size[0] == pytest.approx(
        scene['parameters']['bin']['inner_half_size_m'][0] + .008)
    mass = sum(model.body_mass[index] for index in range(model.nbody)
               if model.body(index).name.startswith('sock_'))
    assert mass == pytest.approx(scene['parameters']['sock']['mass_kg'])
    assert model_config() == before
    parameters = copy.deepcopy(scene['parameters'])
    parameters['bin']['height_m'] = .65
    with pytest.raises(ValueError, match='bounds'):
        load_model(scene='sock', scene_parameters=parameters)


def test_reset_applies_profile_and_keeps_truth_out_of_actor():
    env = ExplorerEnv(render=False)
    try:
        observation, _ = env.reset(seed=2, options=dict(scene_profile=PROFILE))
        assert env.bin_height == pytest.approx(.16)
        assert set(observation) == {'state', 'wrist'}
        assert env.domain_variation['profile'] == PROFILE
        previous = env.domain_variation
        env.reset(seed=2, options=dict(scene_profile=PROFILE))
        assert env.domain_variation == previous
        assert np.isfinite(env.data.qpos).all()
    finally:
        env.close()


def test_new_model_qualification_requires_results_across_all_basket_heights(tmp_path):
    checkpoints = {'model.safetensors': 'weights', 'config.json': 'config'}
    bundle = dict(skill_id='sock_basket', scene_profile=PROFILE, checkpoint_sha256=checkpoints)
    report = dict(validation_kind='closed_loop_policy', task_id='sock_basket',
        heldout_independent=True, episodes=10, successes=10, physics_validated=False,
        real_hardware_accepted=False, model_weights_sha256='weights',
        model_checkpoint_sha256=checkpoints,
        results=[dict(success=True, domain_variation=sample_sock_scene(seed)) for seed in range(10)])
    path = tmp_path / 'physics-validation.json'
    path.write_text(json.dumps(report))
    assert simulation_qualification(tmp_path, bundle)['simulation_qualified']
    report['scene_profile'] = PROFILE
    assert report_passes(report)
    for result in report['results']:
        result['domain_variation'] = sample_sock_scene(4)
    path.write_text(json.dumps(report))
    assert not simulation_qualification(tmp_path, bundle)['simulation_qualified']
    assert not report_passes(report)


def test_user_round_basket_has_physical_wall_and_radius_not_square_corners():
    env = ExplorerEnv(render=False)
    try:
        env.reset(seed=4, options=dict(scene_profile=PROFILE))
        assert env.bin_height == pytest.approx(.30)
        assert env.bin_shape == 'round'
        np.testing.assert_allclose(env.bin_half_size, [.125, .125])
        import mujoco
        assert env.model.geom('bin_bottom').type == mujoco.mjtGeom.mjGEOM_CYLINDER
        assert sum(env.model.geom(index).name.startswith('bin_wall_') for index in range(env.model.ngeom)) == 23
        assert not env.state()['task']['success']
    finally:
        env.close()


def test_night_resume_preserves_scene_profile_and_attempt_budget(tmp_path):
    queue = NightQueue(tmp_path)
    task = queue.create(['sock_basket'],100,10000,16,8,500,scene_profile=PROFILE)
    seen = []
    def execute(stage,task,options,cancelled):
        seen.append(options)
        raise InterruptedError('pause before generation')
    queue.run(task['id'],execute,lambda:False,lambda text:None)
    assert queue.get(task['id'])['state'] == 'paused'
    queue.run(task['id'],execute,lambda:False,lambda text:None)
    assert len(seen) == 2
    assert all(options['scene_profile'] == PROFILE and options['count'] == 100 for options in seen)
