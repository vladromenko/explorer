import itertools

import numpy as np
import pytest

from simulator.observed_free_space import DepthFreeSpace,certify_boxes


def box(center=(0.,0.,.5),half=.02):
    return np.asarray(center)+np.asarray(list(itertools.product((-half,half),repeat=3)))


def view(depth=None,transform=None,identifier='one'):
    return DepthFreeSpace(np.ones((60,80)) if depth is None else depth,
        dict(fx=60.,fy=60.,cx=39.5,cy=29.5),np.eye(4) if transform is None else transform,identifier,1.)


def test_whole_box_in_front_of_observed_surface_has_a_witness():
    assert view().certifies_box(box())
    assert certify_boxes([box()],[view()])['confirmed']


@pytest.mark.parametrize('center',[(0,0,1.1),(1,0,.5),(0,0,-.2),(0,0,.98)])
def test_behind_surface_outside_image_behind_camera_or_without_margin_is_unknown(center):
    assert not view().certifies_box(box(center))


@pytest.mark.parametrize('pixel',[np.nan,np.inf,0.,3.,.4])
def test_one_missing_or_nearer_pixel_cannot_be_filled_across_an_occlusion(pixel):
    depth=np.ones((60,80));depth[30,40]=pixel
    assert not view(depth).certifies_box(box())


def test_different_views_can_witness_separate_cells_without_inventing_an_unseen_cell():
    first=view();transform=np.eye(4);transform[0,3]=1.
    second=view(transform=transform,identifier='two')
    result=certify_boxes([box(),box((1.,0.,.5))],[first,second])
    assert result['confirmed'] and result['frames']==['one','two']
    assert not certify_boxes([box(),box((2.,0.,.5))],[first,second])['confirmed']
    assert not certify_boxes([],[first])['confirmed']


def test_world_to_optical_transform_and_original_depth_are_not_mutated():
    from scipy.spatial.transform import Rotation
    transform=np.eye(4);transform[:3,:3]=Rotation.from_euler('y',90.,degrees=True).as_matrix()
    transform[:3,3]=[2.,3.,4.];depth=np.ones((60,80))
    sensor=view(depth,transform)
    world=box()@transform[:3,:3].T+transform[:3,3]
    assert sensor.certifies_box(world)
    depth[:]=0.;transform[:]=0.
    assert sensor.certifies_box(world)


def test_invalid_camera_rotation_cannot_certify_space():
    transform=np.eye(4);transform[0,0]=2.
    with pytest.raises(ValueError):view(transform=transform)
