import numpy as np
import pytest

from simulator.observed_navigation import TranslationFeedback


def test_odometry_feedback_reaches_a_goal_despite_rolling_dead_zone():
    servo=TranslationFeedback();position=np.zeros(2);goal=np.array([.14,.025]);previous=np.zeros(2)
    for _ in range(160):
        error=goal-position
        if np.linalg.norm(error)<.008:break
        velocity=servo.update(error,0.,.05)
        assert np.linalg.norm(velocity)<=.06+1e-10
        assert np.linalg.norm(velocity-previous)<=.005+1e-10
        # Independent simple plant: finite rolling resistance, not the PI rule.
        speed=np.linalg.norm(velocity)
        actual=velocity*(max(0.,speed-.012)/max(speed,1e-10))*.75
        position+=actual*.05;previous=velocity
    assert np.linalg.norm(goal-position)<.008


def test_world_target_is_transformed_into_current_robot_axes():
    a=TranslationFeedback();b=TranslationFeedback()
    world=a.update([.1,0.],0.,.05)
    turned=b.update([.1,0.],np.pi/2,.05)
    np.testing.assert_allclose(turned,[world[1],-world[0]],atol=1e-10)


def test_invalid_odometry_cannot_generate_a_drive_command():
    servo=TranslationFeedback()
    with pytest.raises(ValueError):servo.update([np.nan,0.],0.,.05)
