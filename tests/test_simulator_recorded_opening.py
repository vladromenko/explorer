import json
import hashlib
from pathlib import Path

import numpy as np
import pytest

from simulator.multiview_container import combine_registered_container
from simulator.rgbd_observation import ObservationRefused


def recorded():
    folder=Path(__file__).parent/'fixtures/registered_high_basket'
    manifest=json.loads((folder/'manifest.json').read_text());records=[]
    for entry in manifest['records']:
        path=folder/entry['file']
        assert hashlib.sha256(path.read_bytes()).hexdigest()==entry['sha256']
        with np.load(path) as data:
            records.append(dict(frame=entry['frame'],depth=data['depth'].copy(),robot_depth=data['robot_depth'].copy()))
    return records


def test_recorded_30cm_basket_has_two_measured_empty_volume_witnesses_without_floor_assumption():
    records=recorded();now=max(r['frame']['simulation_time'] for r in records)
    result=combine_registered_container(records,now_simulation_time=now)
    proof=result['evidence']['opening_proof']
    assert proof['source']=='two_view_measured_free_volume'
    assert proof['center_depth_below_rim_m']>.035
    assert proof['unseen_floor_assumed'] is False
    assert all(v['covered_depth_pixels']>=8 and v['measured_surface_margin_m']>=.012
               and v['registered_rim_pixels']>=20 and v['registered_rim_sectors']>=2 for v in proof['views'])
    assert result['evidence']['rim_sectors']==4 and result['uncertainty_m']<=.02
    assert result['evidence']['opening_camera_baseline_m']>=.02


def test_recorded_geometry_does_not_bypass_freshness_or_missing_depth():
    records=recorded();now=max(r['frame']['simulation_time'] for r in records)
    with pytest.raises(ObservationRefused):combine_registered_container(records,now_simulation_time=now+1.)
    newest=max(records,key=lambda r:r['frame']['simulation_time']);newest['depth'][:]=np.nan
    with pytest.raises(ObservationRefused):combine_registered_container(records,now_simulation_time=now)
