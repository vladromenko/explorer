import json
import os

import pytest

from simulator.worker_views import publish, sources, capture_overview


def scene(root, job, index, value):
    folder=root/'demonstrations'/f'trial-{job}-{index:04d}'
    (folder/'frames').mkdir(parents=True)
    (folder/'frames'/'000001.jpg').write_bytes(value)
    return folder


def test_live_views_follow_distinct_attempts_and_keep_surviving_slots(tmp_path):
    root=tmp_path/'data';dest=tmp_path/'static';assignments={}
    for index in range(5):scene(root,'a'*12,index,str(index).encode())
    job=dict(id='a'*12,kind='trials',status='running',details={'active_attempts':[1,2,3,4]})
    output=publish(root,dest,{'jobs':[job]},assignments)
    assert len({view['identity'] for view in output['views']})==4
    assert all(view['live'] for view in output['views'])
    assert [(dest/f'worker-{slot}-wrist.jpg').read_bytes() for slot in range(4)]==[b'0',b'1',b'2',b'3']
    job['details']['active_attempts']=[2,3,4,5]
    publish(root,dest,{'jobs':[job]},assignments)
    assert assignments['a'*12+':2']==1
    assert assignments['a'*12+':5']==0


def test_training_shows_recordings_never_claims_live_workers(tmp_path):
    root=tmp_path/'data';dest=tmp_path/'static'
    for index in range(4):scene(root,'a'*12,index,b'example')
    jobs=[dict(id='a'*12,kind='trials',status='complete'),dict(id='b'*12,kind='train',status='running')]
    result=publish(root,dest,{'jobs':jobs},{})
    assert result['phase']=='train' and len(result['views'])==4
    assert not any(view['live'] for view in result['views'])
    assert json.loads((dest/'state.json').read_text())==result


def test_old_live_frame_is_reported_stale_and_no_frame_is_not_substituted(tmp_path):
    root=tmp_path/'data';dest=tmp_path/'static'
    folder=scene(root,'a'*12,0,b'frame')
    os.utime(folder/'frames'/'000001.jpg',(10,10))
    job=dict(id='a'*12,kind='trials',status='running',details={'active_attempts':[1,2]})
    result=publish(root,dest,{'jobs':[job]},{},now=30)
    assert len(result['views'])==1 and result['views'][0]['stale']
    assert result['views'][0]['captured_at']==10


def test_worker_folder_cannot_escape_storage(tmp_path):
    root=tmp_path/'data';(root/'demonstrations').mkdir(parents=True)
    outside=tmp_path/'outside';outside.mkdir()
    (root/'demonstrations'/('trial-'+'a'*12+'-0000')).symlink_to(outside,target_is_directory=True)
    job=dict(id='a'*12,kind='trials',status='running',details={'active_attempts':[1]})
    with pytest.raises(ValueError,match='outside'):
        publish(root,tmp_path/'static',{'jobs':[job]}, {})


def test_overview_capture_is_throttled_and_does_not_advance_physics(tmp_path):
    import numpy as np
    class Env:
        calls=0
        def render_rgb(self,camera):
            assert camera=='overview';self.calls+=1
            return np.zeros((20,20,3),dtype=np.uint8)
    env=Env()
    stamp=capture_overview(env,tmp_path,-1)
    assert capture_overview(env,tmp_path,stamp)==stamp and env.calls==1
    assert (tmp_path/'overview-preview.jpg').is_file()
