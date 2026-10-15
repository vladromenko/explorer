"""Geometric sensor boundaries and conservative target acceptance."""
import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from simulator.rgbd_observation import (ObservationRefused, deproject_depth,
    estimate_target, extract_candidates, observe_environment, render_robot_depth)


class RgbdObservationTests(unittest.TestCase):
    def setUp(self):
        self.k = dict(fx=100., fy=100., cx=63.5, cy=63.5, width=128, height=128)
        self.pose = np.diag([1., -1., -1., 1.])
        self.pose[2, 3] = 1.

    def observation(self, candidates):
        return dict(frame_id='camera:1', simulation_time=1.,
                    captured_monotonic=time.monotonic(), **candidates)

    def test_optical_depth_deprojection_and_world_transform(self):
        k = dict(fx=2., fy=4., cx=1., cy=1.)
        points = deproject_depth(np.full((3, 3), 2.), k, self.pose)
        np.testing.assert_allclose(points[1, 1], [0., 0., -1.])
        np.testing.assert_allclose(points[2, 2], [1., -.5, -1.])
        bad = np.eye(4); bad[0, 0] = -1
        with self.assertRaises(ValueError):
            deproject_depth(np.ones((3, 3)), k, bad)

    def test_invalid_depth_is_not_an_object(self):
        depth = np.ones((128, 128))
        depth[:15] = np.nan; depth[15:20] = 0.; depth[20:25] = np.inf
        result = extract_candidates(depth, self.k, self.pose)
        self.assertEqual(result['low_objects'], [])
        self.assertEqual(result['containers'], [])

    def test_low_cluster_is_metric_and_semantically_unverified(self):
        depth = np.ones((128, 128)); depth[55:65, 70:85] = .98
        result = extract_candidates(depth, self.k, self.pose)
        target = estimate_target(self.observation(result), 'sock', now_simulation_time=1.)
        self.assertAlmostEqual(target['center'][2], .02)
        self.assertAlmostEqual(target['center'][0], (77-63.5)*.98/100, places=3)
        self.assertEqual(target['semantic_identity'], 'unverified')
        self.assertFalse(target['evidence']['semantic_identity_proven'])

    def test_two_visible_clusters_refuse_guessing(self):
        depth = np.ones((128, 128)); depth[55:65, 70:85] = .98
        depth[25:35, 30:45] = .98
        result = extract_candidates(depth, self.k, self.pose)
        self.assertEqual(len(result['low_objects']), 2)
        with self.assertRaises(ObservationRefused):
            estimate_target(self.observation(result), 'object', now_simulation_time=1.)

    def test_stale_future_and_border_observations_refused(self):
        depth = np.ones((128, 128)); depth[:10, 60:75] = .98
        observation = self.observation(extract_candidates(depth, self.k, self.pose))
        for stamp in (1., 2., .5):
            with self.assertRaises(ObservationRefused):
                estimate_target(observation, 'sock', now_simulation_time=stamp)

    def test_raised_rim_with_observed_opening_supports_container_candidate(self):
        depth = np.ones((128, 128))
        depth[40:88, 40:44] = .7; depth[40:88, 84:88] = .7
        depth[40:44, 40:88] = .7; depth[84:88, 40:88] = .7
        depth[42:44, 40:88] = .8
        target = estimate_target(self.observation(extract_candidates(depth, self.k, self.pose)),
                                 'basket', now_simulation_time=1.)
        self.assertAlmostEqual(target['evidence']['rim_height_m'], .3)
        self.assertEqual(target['evidence']['rim_sectors'], 4)
        self.assertTrue(target['evidence']['bounds_are_visible_estimate'])

    def test_visible_vertical_panel_does_not_prove_basket(self):
        depth = np.ones((128, 128))
        depth[40:80, 70:90] = np.linspace(.65, .9, 40)[:, None]
        self.assertEqual(extract_candidates(depth, self.k, self.pose)['containers'], [])

    def test_generic_elevated_geometry_is_not_hold_proof(self):
        depth = np.ones((128, 128)); depth[55:65, 70:85] = .72
        result = extract_candidates(depth, self.k, self.pose)
        self.assertEqual(len(result['low_objects']), 0)
        self.assertEqual(len(result['object_candidates']), 1)
        self.assertFalse(result['object_candidates'][0]['evidence']['hold_proof'])
        self.assertEqual(result['hold_status'], 'unknown')

    def test_robot_cad_depth_removes_self_and_preserves_separated_material(self):
        depth = np.ones((128, 128)); depth[55:65, 70:85] = .72
        predicted = np.full(depth.shape, 8.); predicted[55:65, 70:85] = .72
        self.assertEqual(extract_candidates(depth, self.k, self.pose,
            robot_depth=predicted)['object_candidates'], [])
        predicted[55:65, 70:85] = .74
        result = extract_candidates(depth, self.k, self.pose, robot_depth=predicted,
                                    rgb=np.full((128, 128, 3), [128, 32, 64], np.uint8))
        target = result['object_candidates'][0]
        self.assertTrue(target['robot_mask_resolved'])
        self.assertFalse(target['evidence']['hold_proof'])
        np.testing.assert_allclose(target['appearance']['rgb_median'], np.array([128,32,64])/255.)

    def test_footprint_exclusion_retains_geometry_held_above_base(self):
        depth = np.ones((128, 128)); depth[55:65, 60:75] = .72
        result = extract_candidates(depth, self.k, self.pose, robot_exclusion=(0, 0, .19))
        self.assertEqual(len(result['object_candidates']), 1)
        depth[55:65, 60:75] = .98
        result = extract_candidates(depth, self.k, self.pose, robot_exclusion=(0, 0, .19))
        self.assertEqual(result['object_candidates'], [])

    def test_valid_cad_reference_does_not_clip_free_floor_near_base(self):
        depth = np.ones((128, 128)); depth[55:65, 76:86] = .98
        result = extract_candidates(depth, self.k, self.pose,
            robot_depth=np.full(depth.shape, 8.), robot_exclusion=(0, 0, .19))
        self.assertEqual(len(result['low_objects']), 1)
        target = result['low_objects'][0]
        self.assertLess(np.linalg.norm(target['center'][:2]), .19)
        self.assertTrue(target['robot_mask_resolved'])

    def test_cad_prediction_uses_only_robot_group_and_restores_renderer(self):
        class Renderer:
            def __init__(self, fail=False):
                self.mode = False; self.options = []; self.fail = fail
            def enable_depth_rendering(self): self.mode = True
            def disable_depth_rendering(self): self.mode = False
            def update_scene(self, data, **kwargs): self.options.append(kwargs['scene_option'])
            def render(self):
                if self.fail: raise RuntimeError('camera unavailable')
                return np.ones((10, 10))
        for fail in (False, True):
            renderer = Renderer(fail)
            original = object()
            env = SimpleNamespace(renderer=renderer, data=object(), render_options=original)
            prediction = render_robot_depth(env)
            self.assertEqual(prediction is None, fail)
            self.assertFalse(renderer.mode)
            self.assertIs(renderer.options[-1], original)
            np.testing.assert_array_equal(renderer.options[0].geomgroup, [0, 0, 1, 0, 0, 0])
            self.assertFalse(np.any(renderer.options[0].flexgroup))
            self.assertFalse(np.any(renderer.options[0].skingroup))
            import mujoco
            for flag in (mujoco.mjtVisFlag.mjVIS_FLEXSKIN, mujoco.mjtVisFlag.mjVIS_SKIN,
                         mujoco.mjtVisFlag.mjVIS_FLEXFACE):
                self.assertFalse(renderer.options[0].flags[flag])

    def test_capture_only_reads_sensor_and_robot_pose_and_tracks_view(self):
        camera = SimpleNamespace(id=0)
        data = SimpleNamespace(time=1., cam_xmat=np.eye(3).reshape(1, 9),
                               cam_xpos=np.array([[0., 0., 1.]]), qpos=np.array([0., 0.]))
        # No object truth methods/attributes exist on this sensor-only fake.
        env = SimpleNamespace(render_depth=lambda: np.ones((128, 128)),
                              model=SimpleNamespace(camera=lambda name: camera), data=data)
        cfg = dict(self.k, intrinsics_status='test measured')
        with patch('simulator.explorer_model.model_config', return_value={'camera': cfg}):
            first = observe_environment(env)
            data.cam_xpos[0, 0] += .01
            second = observe_environment(env)
        self.assertNotEqual(first['frame_id'], second['frame_id'])
        self.assertEqual(first['calibration']['floor_status'], 'synthetic_room_prior')
        np.testing.assert_allclose(np.array(first['optical_to_world'])[:3, :3], np.diag([1., -1., -1.]))


if __name__ == '__main__':
    unittest.main()
