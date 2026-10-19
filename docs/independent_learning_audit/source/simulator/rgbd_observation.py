"""Unprivileged metric wrist-depth observations for closed-loop skills.

Only camera pixels, camera calibration and robot camera kinematics enter here.
Candidates are geometry, never proof of an object's semantic class. Floor z=0
is an explicit simulation prior; a real robot needs a measured floor transform.
"""
from __future__ import annotations

import hashlib
import time
from typing import Any

import numpy as np
from scipy.ndimage import label


class ObservationRefused(RuntimeError):
    """The camera evidence cannot support an unambiguous fresh target."""


def held_material_patches(record,jaw_xyz,reference_rgb):
    """Measure local non-self material near the pads, not a cloth centroid.

    A deforming cloth's whole silhouette may include a tail on the floor.
    That centroid cannot track a grasp. These are explicitly LOCAL patches;
    they prove neither that the whole item is clear nor its semantic identity.
    The unchanged hold motion/freshness/pixel/floor gates run at the caller.
    """
    if record.get('rgb') is None:return None
    frame=record['frame'];depth=np.asarray(record['depth'],float)
    robot=np.asarray(record['robot_depth'],float);rgb=np.asarray(record['rgb'])
    if robot.shape!=depth.shape or rgb.shape!=(*depth.shape,3):return []
    reference=np.asarray(reference_rgb,float);jaw=np.asarray(jaw_xyz,float)
    if reference.shape!=(3,) or jaw.shape!=(3,) or not np.isfinite(np.r_[reference,jaw]).all():return []
    points=deproject_depth(depth,frame['intrinsics'],frame['optical_to_world'])
    colors=rgb.astype(float)/255.;chromatic=colors/np.maximum(colors.sum(-1,keepdims=True),.05)
    reference=reference/max(.05,float(reference.sum()))
    selected=(np.isfinite(points).all(-1)&(depth>=.035)&(depth<=2.5)&
        np.all(abs(points-jaw)<[.04,.04,.035],axis=-1)&
        (np.linalg.norm(chromatic-reference,axis=-1)<.08)&
        ~(np.isfinite(robot)&(robot>0)&(abs(depth-robot)<=.008)))
    result=[]
    for cloud,pixels in _clusters(points,selected,min_pixels=100):
        lo,hi=np.quantile(cloud,[.02,.98],axis=0)
        result.append(dict(center=np.median(cloud,axis=0).tolist(),bounds_min=lo.tolist(),bounds_max=hi.tolist(),
            uncertainty_m=.012,appearance=dict(rgb_median=np.median(colors[tuple(pixels.T)],axis=0).tolist()),
            evidence=dict(visible_pixels=len(pixels),self_occlusion_not_resolved=False,
                source='registered_RGBD_local_material_patch',whole_object_clearance_proven=False,
                semantic_identity_proven=False)))
    return result


def deproject_depth(depth, intrinsics, optical_to_world):
    """Return HxWx3 world points; depth is metric optical Z, not ray length."""
    depth = np.asarray(depth, dtype=float)
    matrix = np.asarray(optical_to_world, dtype=float)
    if depth.ndim != 2 or matrix.shape != (4, 4):
        raise ValueError('Expected HxW metric depth and a 4x4 camera transform')
    if not np.isfinite(matrix).all() or not np.allclose(matrix[3], [0, 0, 0, 1]):
        raise ValueError('Invalid camera transform')
    rotation = matrix[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5) or np.linalg.det(rotation) < .99:
        raise ValueError('Camera rotation must be proper and orthonormal')
    fx, fy = float(intrinsics['fx']), float(intrinsics['fy'])
    if fx <= 0 or fy <= 0:
        raise ValueError('Focal lengths must be positive')
    rows, columns = np.indices(depth.shape)
    optical = np.stack(((columns-float(intrinsics['cx']))*depth/fx,
                        (rows-float(intrinsics['cy']))*depth/fy, depth), axis=-1)
    with np.errstate(invalid='ignore', over='ignore'):
        points = optical @ rotation.T + matrix[:3, 3]
    points[~np.isfinite(depth) | (depth <= 0)] = np.nan
    return points


def _clusters(points, selected, *, min_pixels=12, cell_m=.012):
    """Connected world XY voxels tolerate image holes without merging far objects."""
    ij = np.argwhere(selected)
    if len(ij) < min_pixels:
        return []
    samples = points[selected]
    origin = samples[:, :2].min(axis=0)
    cells = np.floor((samples[:, :2]-origin)/cell_m).astype(int)
    shape = cells.max(axis=0)+1
    if np.prod(shape) > 2_000_000:
        return []
    occupied = np.zeros(tuple(shape), dtype=bool)
    occupied[cells[:, 0], cells[:, 1]] = True
    labels, count = label(occupied, structure=np.ones((3, 3)))
    membership = labels[cells[:, 0], cells[:, 1]]
    return [(samples[membership == index], ij[membership == index])
            for index in range(1, count+1) if np.count_nonzero(membership == index) >= min_pixels]


def extract_candidates(depth, intrinsics, optical_to_world, *, floor_z_m=0.,
                       min_depth_m=.035, max_depth_m=2.5, robot_exclusion=None, robot_depth=None, rgb=None):
    """Extract low geometry and open raised-wall geometry, conservatively.

    robot_exclusion may be (base_x, base_y, radius_m), obtained from robot
    kinematics. This low-floor fallback is used only without CAD depth; a valid
    CAD prediction preserves visible free floor even close to the robot base.
    Occluded or incomplete containers are intentionally not actionable.
    """
    depth = np.asarray(depth, float)
    points = deproject_depth(depth, intrinsics, optical_to_world)
    valid = np.isfinite(points).all(axis=-1) & (depth >= min_depth_m) & (depth <= max_depth_m)
    height = points[..., 2]-floor_z_m
    robot_mask_resolved = robot_depth is not None
    if robot_mask_resolved:
        predicted = np.asarray(robot_depth, float)
        if predicted.shape != depth.shape:
            raise ValueError('Robot depth must use the same camera resolution')
        matching = np.isfinite(predicted) & (predicted > 0) & (np.abs(depth-predicted) <= .008)
        valid &= ~matching
    if robot_exclusion is not None and not robot_mask_resolved:
        bx, by, radius = robot_exclusion
        valid &= (np.hypot(points[..., 0]-bx, points[..., 1]-by) > radius) | (height >= .15)
    low_mask = valid & (height >= .004) & (height <= .075)
    wall_mask = valid & (height >= .075) & (height <= .55)
    h, w = depth.shape
    containers = []
    hypotheses = []
    observed_rims = []
    for sample, pixels in _clusters(points, wall_mask, min_pixels=30, cell_m=.02):
        lo, hi = np.quantile(sample, [.02, .98], axis=0)
        span = hi-lo
        if max(span[:2]) >= .06 and span[2] >= .025:
            hypotheses.append(_candidate((lo+hi)/2, lo, hi, .06, .3,
                {'shape': 'partial_raised_geometry_for_camera_aim_only',
                 'visible_pixels': len(sample), 'actionable': False,
                 'semantic_identity_proven': False}))
        if not (.09 <= span[0] <= .65 and .09 <= span[1] <= .65 and span[2] >= .04):
            continue
        rim = sample[sample[:, 2] >= hi[2]-.025]
        if len(rim) < 20:
            continue
        center = (lo+hi)/2
        half = np.maximum(span[:2]/2, .001)
        relative = (rim[:, :2]-center[:2])/half
        # At least three outer sectors and a visible depth discontinuity into
        # the opening are needed. One observed vertical panel is not a bin.
        angles = np.arctan2(relative[:, 1], relative[:, 0])
        sectors = len(np.unique(np.floor((angles+np.pi)/(np.pi/2)).astype(int).clip(0, 3)))
        # Completeness concerns the top boundary, not a 25 mm strip of the
        # vertical wall below it. A cropped lower wall must not erase a fully
        # visible top rim; a cropped top rim still remains unknown.
        top=sample[:,2]>=hi[2]-.008
        top_rim=sample[top];rim_pixels=pixels[top]
        top_relative=(top_rim[:,:2]-center[:2])/half
        top_angles=np.arctan2(top_relative[:,1],top_relative[:,0])
        top_sectors=len(np.unique(np.floor((top_angles+np.pi)/(np.pi/2)).astype(int).clip(0,3)))
        rim_border=bool(np.any((rim_pixels[:,0]<=1)|(rim_pixels[:,0]>=h-2)|
                              (rim_pixels[:,1]<=1)|(rim_pixels[:,1]>=w-2)))
        if len(top_rim)>=20 and top_sectors==4 and not rim_border:
            # A complete visible rim can guide later inspection, but does not
            # establish an empty opening or permission to release an object.
            observed_rims.append(_candidate(center,lo,hi,.06,.6,
                {'shape':'complete_rim_without_opening_proof','actionable':False,
                 'rim_sectors':top_sectors,'rim_pixels':len(top_rim),
                 'boundary_band_m':.008,
                 'touches_image_border':rim_border,
                 'rim_points':top_rim[::max(1,len(top_rim)//256)].tolist(),
                 'semantic_identity_proven':False}))
        inside = valid & (np.abs(points[..., 0]-center[0]) < half[0]*.55) & (np.abs(points[..., 1]-center[1]) < half[1]*.55)
        opening = inside & (points[..., 2] < hi[2]-.035)
        border = bool(np.any((pixels[:, 0] <= 1) | (pixels[:, 0] >= h-2) |
                             (pixels[:, 1] <= 1) | (pixels[:, 1] >= w-2)))
        if sectors < 3 or np.count_nonzero(opening) < 8 or border:
            continue
        uncertainty = max(.012, float(np.max(span[:2]))*.07)
        containers.append(_candidate(center, lo, hi, uncertainty, .75,
            {'shape': 'raised_walls_with_visible_opening', 'rim_sectors': sectors,
             'opening_pixels': int(np.count_nonzero(opening)), 'visible_pixels': len(sample),
             'rim_height_m': float(hi[2]), 'bounds_are_visible_estimate': True}))
    lows = []
    for sample, pixels in _clusters(points, low_mask):
        lo, hi = np.quantile(sample, [.02, .98], axis=0)
        span = hi-lo
        if not (.012 <= max(span[:2]) <= .4 and min(span[:2]) >= .006):
            continue
        center = np.median(sample, axis=0)
        if any(np.all(center[:2] >= np.array(c['bounds_min'])[:2]-.025) and
               np.all(center[:2] <= np.array(c['bounds_max'])[:2]+.025) for c in containers):
            continue
        border = bool(np.any((pixels[:, 0] <= 1) | (pixels[:, 0] >= h-2) |
                             (pixels[:, 1] <= 1) | (pixels[:, 1] >= w-2)))
        uncertainty = max(.006, float(np.median(depth[tuple(pixels.T)]))*.01)
        lows.append(_candidate(center, lo, hi, uncertainty, .45 if border else .75,
            {'shape': 'low_nonfloor_cluster', 'visible_pixels': len(sample),
             'touches_image_border': border, 'semantic_identity_proven': False}))
    objects = []
    for sample, pixels in _clusters(points, valid & (height >= .004) & (height <= .65)):
        lo, hi = np.quantile(sample, [.02, .98], axis=0)
        span = hi-lo
        if max(span) > .4 or max(span[:2]) < .012 or min(span[:2]) < .006:
            continue
        border = bool(np.any((pixels[:, 0] <= 1) | (pixels[:, 0] >= h-2) |
                             (pixels[:, 1] <= 1) | (pixels[:, 1] >= w-2)))
        objects.append(_candidate(np.median(sample, axis=0), lo, hi, .012,
            .45 if border else .72, {'shape': 'compact_nonfloor_geometry',
            'visible_pixels': len(sample), 'semantic_identity_proven': False,
            'self_occlusion_not_resolved': not robot_mask_resolved, 'hold_proof': False}))
    for candidate in lows+containers+objects+observed_rims:
        candidate['robot_mask_resolved'] = robot_mask_resolved
        candidate['evidence']['robot_mask_resolved'] = robot_mask_resolved
        if rgb is not None:
            colors = np.asarray(rgb)
            if colors.shape == (*depth.shape, 3):
                lo, hi = np.array(candidate['bounds_min']), np.array(candidate['bounds_max'])
                region = valid & np.all(points >= lo, axis=-1) & np.all(points <= hi, axis=-1)
                if np.count_nonzero(region) >= 6:
                    samples = colors[region].astype(float)/255.
                    candidate['appearance'] = {'rgb_median': np.median(samples, axis=0).tolist(),
                        'rgb_spread': np.std(samples, axis=0).tolist(),
                        'source': 'registered_wrist_rgb', 'semantic_identity_proven': False}
    visible = points[valid]
    stride = max(1, len(visible)//2000)
    return {'low_objects': lows, 'containers': containers, 'container_hypotheses': hypotheses,
            'observed_rims':observed_rims,
            'object_candidates': objects,
            'surface_points': visible[::stride].tolist(),
            'hold_status': 'unknown', 'placement_status': 'unknown'}


def _candidate(center, lo, hi, uncertainty, confidence, evidence):
    return {'center': np.asarray(center).tolist(), 'bounds_min': lo.tolist(),
            'bounds_max': hi.tolist(), 'uncertainty_m': uncertainty,
            'confidence': confidence, 'evidence': evidence,
            'semantic_identity': 'unverified'}


def render_robot_depth(env):
    """Predict only known robot CAD surfaces, using the current camera FK.

    Group 2 is the robot visual CAD group declared in explorer_model.py. This
    uses the same known mesh used for self collision filtering; no environment
    labels, object names or segmentation are queried. Scene options and renderer
    depth mode are restored even when prediction fails.
    """
    try:
        import mujoco
        renderer = env.renderer
        if renderer is None:
            return None
        options = mujoco.MjvOption()
        options.geomgroup[:] = 0
        options.geomgroup[2] = 1
        options.sitegroup[:] = 0
        options.flexgroup[:] = 0
        options.skingroup[:] = 0
        for flag in (mujoco.mjtVisFlag.mjVIS_FLEXFACE, mujoco.mjtVisFlag.mjVIS_FLEXEDGE,
                     mujoco.mjtVisFlag.mjVIS_FLEXVERT, mujoco.mjtVisFlag.mjVIS_FLEXSKIN,
                     mujoco.mjtVisFlag.mjVIS_SKIN):
            options.flags[flag] = False
        try:
            renderer.enable_depth_rendering()
            renderer.update_scene(env.data, camera='wrist_depth', scene_option=options)
            result = renderer.render().copy()
        finally:
            renderer.disable_depth_rendering()
            renderer.update_scene(env.data, camera='wrist_depth', scene_option=env.render_options)
        return result
    except (AttributeError, RuntimeError, ImportError, ValueError):
        return None


def observe_environment(env,*,raw_output=None) -> dict[str, Any]:
    """Capture the wrist sensor; do not inspect scene objects or segmentation.

    MuJoCo camera pose is robot forward kinematics, equivalent to a calibrated
    wrist extrinsic plus joint state. No environment body or geometry is read.
    """
    if callable(getattr(env,'capture_registered_frame',None)):
        return env.capture_registered_frame(raw_output=raw_output)
    from .explorer_model import model_config
    depth = env.render_depth()
    robot_depth = render_robot_depth(env)
    try:
        rgb = env.render_rgb('wrist')
    except (AttributeError, RuntimeError):
        rgb = None
    camera = env.model.camera('wrist_depth')
    index = camera.id
    matrix = np.eye(4)
    matrix[:3, :3] = env.data.cam_xmat[index].reshape(3, 3) @ np.diag([1., -1., -1.])
    matrix[:3, 3] = env.data.cam_xpos[index]
    config = model_config()['camera']
    h, w = depth.shape
    intrinsics = {'width': w, 'height': h,
                  'fx': config['fx']*w/config['width'], 'fy': config['fy']*h/config['height'],
                  'cx': (config['cx']+.5)*w/config['width']-.5,
                  'cy': (config['cy']+.5)*h/config['height']-.5}
    # Root pose is robot kinematics and only masks self-observations.
    base = np.asarray(env.data.qpos[:2], float)
    candidates = extract_candidates(depth, intrinsics, matrix,
                                    robot_exclusion=(float(base[0]), float(base[1]), .19),
                                    robot_depth=robot_depth, rgb=rgb)
    stamp = float(env.data.time)
    digest = hashlib.sha256(matrix.tobytes()+depth.tobytes()).hexdigest()[:16]
    if raw_output is not None:
        raw_output.update(depth=depth.copy(),robot_depth=None if robot_depth is None else robot_depth.copy(),rgb=rgb.copy())
    return dict(schema=1, frame_id=f'wrist_depth:{stamp:.9f}:{digest}',
                simulation_time=stamp, captured_monotonic=time.monotonic(),
                intrinsics=intrinsics, optical_to_world=matrix.tolist(),
                robot_self_filter={'available': robot_depth is not None,
                    'source': 'robot_only_CAD_depth_group2_and_camera_FK',
                    'depth_tolerance_m': .008,
                    'near_surface_material_may_be_unobservable': True,
                    'semantic_identity_proven': False},
                calibration={'intrinsics_status': config['intrinsics_status'],
                             'extrinsics_status': 'robot_forward_kinematics_with_recorded_handeye',
                             'floor_prior_z_m': 0., 'floor_status': 'synthetic_room_prior',
                             'scope': 'simulation_only_real_handeye_and_dynamics_require_validation'},
                **candidates)


def combine_container_views(views):
    """Bound a static opening from two separate complete RGBD viewpoints.

    This measures cross-view consistency rather than reducing the acceptance
    threshold. The 12 mm uncertainty floor is explicit simulation uncertainty;
    it does not qualify the physical camera or unknown real basket geometry.
    Partial hypotheses never enter this function or become action targets.
    """
    for latest in reversed(views):
        current = latest['candidate']
        for previous in reversed(views):
            if previous['frame_id'] == latest['frame_id']:
                continue
            if np.linalg.norm(np.array(previous['camera_position'])-latest['camera_position']) < .02:
                continue
            earlier = previous['candidate']
            pair = (earlier, current)
            if any(c.get('confidence', 0.) < .7 or
                   c.get('evidence', {}).get('shape') != 'raised_walls_with_visible_opening'
                   for c in pair):
                continue
            centers = np.array([c['center'] for c in pair])
            lower = np.array([c['bounds_min'] for c in pair])
            upper = np.array([c['bounds_max'] for c in pair])
            disagreement = max(float(np.max(np.abs(lower[0]-lower[1]))),
                               float(np.max(np.abs(upper[0]-upper[1]))))
            if np.linalg.norm(centers[0]-centers[1]) > .025 or disagreement > .025:
                continue
            lo, hi = lower.mean(axis=0), upper.mean(axis=0)
            target = dict(current, center=((lo+hi)/2).tolist(), bounds_min=lo.tolist(),
                          bounds_max=hi.tolist(), uncertainty_m=max(.012, .006+disagreement/2),
                          frame_id=latest['frame_id'], simulation_time=latest['simulation_time'])
            target['evidence'] = dict(current['evidence'],
                multiview_consistent=True, source_frames=[previous['frame_id'], latest['frame_id']],
                camera_baseline_m=float(np.linalg.norm(np.array(previous['camera_position'])-latest['camera_position'])),
                bound_disagreement_m=disagreement,
                uncertainty_source='two_view_bound_consistency_with_12mm_simulation_floor',
                rim_height_m=float(hi[2]), real_calibration_validated=False)
            return target
    raise ObservationRefused('Два полных независимых обзора корзины не согласованы')


def estimate_target(observation, kind, *, now_simulation_time=None, max_age_s=.25):
    """Select exactly one fresh geometric candidate; reject uncertainty."""
    key = {'object': 'low_objects', 'sock': 'low_objects', 'basket': 'containers',
           'container': 'containers', 'elevated_object': 'object_candidates', 'low_objects': 'low_objects', 'containers': 'containers'}.get(kind)
    if key is None:
        raise ValueError('Unknown geometric target kind')
    if not observation.get('frame_id') or not np.isfinite(observation.get('simulation_time', np.nan)):
        raise ObservationRefused('Observation has no valid sensor timestamp')
    if now_simulation_time is not None:
        age = now_simulation_time-observation['simulation_time']
    else:
        age = time.monotonic()-observation.get('captured_monotonic', -np.inf)
    if age < -1e-6 or age > max_age_s:
        raise ObservationRefused('Camera observation is stale')
    candidates = observation.get(key, [])
    if len(candidates) != 1:
        raise ObservationRefused(f'Expected one geometric target, observed {len(candidates)}')
    target = candidates[0]
    if target['confidence'] < .7 or target['uncertainty_m'] > .04:
        raise ObservationRefused('Target geometry is incomplete or uncertain')
    return target
