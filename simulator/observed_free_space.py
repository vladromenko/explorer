"""Conservative local free-space witnesses from recorded metric depth rays.

This certifies a supplied convex box only in front of actually measured depth.
An unseen cavity, a missing pixel, a crop or a surface behind an occluder is
unknown. It is not a whole-scene map or a certificate for a moving payload.
"""
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DepthFreeSpace:
    depth: np.ndarray
    intrinsics: dict
    optical_to_world: np.ndarray
    frame_id: str
    simulation_time: float

    def __post_init__(self):
        depth=np.array(self.depth,dtype=float,copy=True)
        transform=np.array(self.optical_to_world,dtype=float,copy=True)
        intrinsics={k:float(self.intrinsics[k]) for k in ('fx','fy','cx','cy')}
        if (depth.ndim!=2 or min(depth.shape)<2 or transform.shape!=(4,4)
            or not np.isfinite(transform).all() or not np.allclose(transform[3],[0.,0.,0.,1.])
            or not np.isfinite(list(intrinsics.values())).all()
            or intrinsics['fx']<=0. or intrinsics['fy']<=0.
            or not isinstance(self.frame_id,str) or not self.frame_id
            or not np.isfinite(self.simulation_time)):
            raise ValueError('Invalid calibrated depth view')
        rotation=transform[:3,:3]
        if not np.allclose(rotation.T@rotation,np.eye(3),atol=1e-5) or np.linalg.det(rotation)<.99:
            raise ValueError('Invalid camera rotation')
        depth.flags.writeable=False;transform.flags.writeable=False
        object.__setattr__(self,'depth',depth)
        object.__setattr__(self,'optical_to_world',transform)
        object.__setattr__(self,'intrinsics',intrinsics)

    def certifies_box(self,corners_world,*,margin_m=.012):
        return self.box_witness(corners_world,margin_m=margin_m) is not None

    def box_witness(self,corners_world,*,margin_m=.012):
        """Require the whole projected box before every covered depth pixel.

        The projection of a convex box in front of the camera lies within the
        corner projection rectangle. Requiring the farthest box Z before the
        nearest surface in that rectangle overestimates obstruction, never
        substitutes interpolation across an occlusion or a missing depth ray.
        """
        corners=np.asarray(corners_world,float)
        if (corners.shape!=(8,3) or not np.isfinite(corners).all() or
            not np.isfinite(margin_m) or margin_m<.012):
            return None
        transform=self.optical_to_world
        optical=np.einsum('vi,ij->vj',corners-transform[:3,3],transform[:3,:3])
        z=optical[:,2]
        if z.min()<.035 or z.max()+margin_m>2.5:return None
        k=self.intrinsics
        pixels=optical[:,:2]/z[:,None]*[k['fx'],k['fy']]+[k['cx'],k['cy']]
        lower=np.floor(pixels.min(0)-.5).astype(int)
        upper=np.ceil(pixels.max(0)+.5).astype(int)
        height,width=self.depth.shape
        if np.any(lower<0) or upper[0]>=width or upper[1]>=height:return None
        region=self.depth[lower[1]:upper[1]+1,lower[0]:upper[0]+1]
        if (not np.isfinite(region).all() or np.any(region<.035) or np.any(region>2.5)):
            return None
        if float(z.max())+margin_m>=float(region.min()):return None
        return dict(frame_id=self.frame_id,covered_depth_pixels=int(region.size),
            required_surface_margin_m=margin_m,measured_surface_margin_m=float(region.min()-z.max()))


def certify_boxes(boxes,views,*,margin_m=.012):
    """Every supplied cell needs one complete depth-ray witness.

    Cells must cover the robot volume under test; accepting a few vertices
    alone does not establish clearance for mesh faces or a swept trajectory.
    """
    witnesses=[]
    for box in boxes:
        view=next((view for view in views if view.certifies_box(box,margin_m=margin_m)),None)
        if view is None:return dict(confirmed=False,reason='unobserved_or_obstructed_volume')
        witnesses.append(view.frame_id)
    if not witnesses:return dict(confirmed=False,reason='empty_volume')
    return dict(confirmed=True,source='calibrated_depth_ray_free_space',frames=sorted(set(witnesses)))
