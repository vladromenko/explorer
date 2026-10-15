from types import SimpleNamespace

import numpy as np
import pytest

from simulator.environment import ExplorerEnv
from simulator.inspection_planning import plan_camera_view,CameraViewUnavailable


def test_feasible_view_satisfies_each_constraint_and_preserves_state():
    env=ExplorerEnv(render=False)
    before=env.data.qpos.copy();command=env.command.copy()
    try:
        view=plan_camera_view(env,[.4,.24,.15],gripper=170.,distance_m=.4,min_camera_height_m=.3)
        assert view.lateral_error_m<=.04 and view.range_error_m<=.04
        assert view.position[2]>=.292 and view.command[5]==170.
        np.testing.assert_array_equal(before,env.data.qpos)
        np.testing.assert_array_equal(command,env.command)
    finally:env.close()


def test_unreachable_camera_height_is_not_accepted_as_an_approximate_view():
    env=ExplorerEnv(render=False)
    try:
        with pytest.raises(CameraViewUnavailable,match='нехватка высоты'):
            plan_camera_view(env,[.4,.24,.15],gripper=30.,distance_m=.4,min_camera_height_m=1.0)
    finally:env.close()


def test_optimizer_success_cannot_hide_wrong_pose(monkeypatch):
    env=ExplorerEnv(render=False)
    monkeypatch.setattr('scipy.optimize.least_squares',lambda fun,start,**kwargs:
        SimpleNamespace(x=np.asarray(start),fun=np.zeros(8),success=True))
    try:
        with pytest.raises(CameraViewUnavailable):
            plan_camera_view(env,[.4,.24,.15],gripper=30.,distance_m=.4,min_camera_height_m=1.0)
    finally:env.close()


def test_cancelled_planning_does_not_change_robot():
    env=ExplorerEnv(render=False);before=env.data.qpos.copy()
    def stop():raise InterruptedError('operator pause')
    try:
        with pytest.raises(InterruptedError):
            plan_camera_view(env,[.4,.24,.15],gripper=30.,distance_m=.4,min_camera_height_m=.3,check=stop)
        np.testing.assert_array_equal(before,env.data.qpos)
    finally:env.close()
