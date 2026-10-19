"""Bounded, reproducible sock curriculum; parameters are estimates, not scans."""
from __future__ import annotations

import numpy as np

PROFILE = 'sock_variants_v1'
RANGES = {
    'length_m': [.12, .20], 'width_m': [.04, .065],
    'mass_kg': [.015, .035], 'friction': [.9, 1.2],
    'young_pa': [8000., 14000.], 'yaw_deg': [-20., 20.],
    'basket_heights_m': [.08, .12, .16, .20, .30],
    'basket_inner_width_m': [.20, .45],
    'basket_inner_depth_m': [.17, .40],
}


def sample_sock_scene(seed):
    rng = np.random.default_rng(seed)
    # Stratification covers every rim height in each group of five seeds.
    # Other parameters and image appearance are independent of that category.
    height = RANGES['basket_heights_m'][seed % len(RANGES['basket_heights_m'])]
    sock = {key: float(rng.uniform(*RANGES[key])) for key in
            ('length_m', 'width_m', 'mass_kg', 'friction', 'young_pa')}
    sock['yaw_rad'] = float(np.deg2rad(rng.uniform(*RANGES['yaw_deg'])))
    sock['initial_thickness_m'] = float(rng.uniform(.006, .012))
    shape = 'round' if seed % 2 == 0 else 'rectangular'
    half_size = [float(rng.uniform(.10, .225)), float(rng.uniform(.085, .20))]
    if shape == 'round':
        radius = .125 if seed % 10 == 4 else float(rng.uniform(.10, .20))
        half_size = [radius, radius]
    parameters = {'sock': sock, 'bin': dict(height_m=height, shape=shape,
        inner_half_size_m=half_size)}
    return dict(profile=PROFILE, seed=int(seed), parameters=parameters,
        sock_position=[float(rng.uniform(.28, .36)), float(rng.uniform(-.035, .035)), .015],
        bin_position=[float(rng.uniform(.43, .47)),
                      parameters['bin']['inner_half_size_m'][1]+float(rng.uniform(.18, .22)), 0.],
        sock_rgb=rng.uniform(.10, .85, 3).tolist(),
        basket_rgb=rng.uniform(.15, .75, 3).tolist(),
        lighting_scale=float(rng.uniform(.70, 1.15)),
        floor_rgb=rng.uniform(.12, .38, 3).tolist(),
        parameter_status='estimated',
        unsupported=['arbitrary folds or knots', 'basket rim above 0.30 m',
                     'unseen room layout', 'measured real cloth dynamics'])


def scene_coverage(episodes):
    scenes = [episode.get('domain_variation', {}) for episode in episodes if isinstance(episode, dict)]
    matched = [scene for scene in scenes if isinstance(scene, dict) and scene.get('profile') == PROFILE
               and type(scene.get('seed')) is int and isinstance(scene.get('parameters'), dict)
               and isinstance(scene['parameters'].get('bin'), dict)
               and scene['parameters']['bin'].get('height_m') in RANGES['basket_heights_m']]
    heights = sorted({scene['parameters']['bin']['height_m'] for scene in matched})
    user_basket = sum(scene['parameters']['bin'].get('shape') == 'round'
        and scene['parameters']['bin']['height_m'] == .30
        and scene['parameters']['bin'].get('inner_half_size_m') == [.125, .125]
        for scene in matched)
    return dict(profile=PROFILE, episodes=len(matched),
        distinct_seeds=len({scene['seed'] for scene in matched}),
        basket_heights_m=heights, required_heights_m=RANGES['basket_heights_m'],
        all_height_categories_present=heights == RANGES['basket_heights_m'],
        user_basket_25cm_diameter_30cm_rim_episodes=user_basket,
        ranges=RANGES, physical_parameters_measured=False,
        unsupported=matched[0].get('unsupported', []) if matched else [])
