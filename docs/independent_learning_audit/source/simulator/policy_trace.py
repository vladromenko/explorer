"""Exact actor exposures and issued endpoints; no privileged policy inputs."""
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image


class PolicyTrace:
    def __init__(self,root,phase,model_sha256):
        self.folder=Path(root)/('policy-'+phase+'-'+str(time.time_ns()))
        self.folder.mkdir(parents=True,exist_ok=False)
        self.phase,self.sha=phase,model_sha256

    def append(self,value):
        with (self.folder/'actions.jsonl').open('a') as output:
            output.write(json.dumps(dict(phase=self.phase,model_sha256=self.sha,**value),allow_nan=False)+'\n')

    def observation(self,step,command,rgb,simulation_time,execution_step):
        image=np.asarray(rgb,dtype=np.uint8)
        name=f'{step:06d}.png';Image.fromarray(image).save(self.folder/name)
        self.append(dict(event='observation',step=step,command=np.asarray(command).tolist(),
            image=name,rgb_sha256=hashlib.sha256(image.tobytes()).hexdigest(),simulation_time=simulation_time,
            execution_step=execution_step,joint_state_source='command_estimate'))

    def result(self,step,status,**values):self.append(dict(event='result',step=step,status=status,**values))


def encode_trace(folder):
    """After actions stop, encode recorded exposures with their actual intervals."""
    import math
    import shutil
    import subprocess
    from .learning import sha256,write_json
    folder=Path(folder);trace=folder/'actions.jsonl';video=folder/'wrist.mp4'
    rows=[json.loads(line) for line in trace.read_text().splitlines()]
    exposures=[row for row in rows if row.get('event')=='observation']
    if not exposures:raise ValueError('No recorded policy exposures')
    if video.exists():raise ValueError('A prior video is retained; do not overwrite')
    records=[]
    for index,row in enumerate(exposures):
        name=row['image']
        if Path(name).name!=name or not name.endswith('.png'):raise ValueError('Invalid policy frame path')
        path=folder/name
        if not path.is_file() or path.is_symlink():raise ValueError('Missing or linked policy frame')
        delta=exposures[index+1]['simulation_time']-row['simulation_time'] if index+1<len(exposures) else .05
        if not math.isfinite(delta) or delta<=0:raise ValueError('Policy exposure timestamps did not increase')
        quoted=str(path.resolve()).replace("'","'\\''")
        records.extend(["file '"+quoted+"'",'duration '+str(delta)])
    records.append(records[-2])
    inputs=folder/'video-input.ffconcat';inputs.write_text('\n'.join(records)+'\n')
    executable=shutil.which('ffmpeg')
    if executable is None:raise RuntimeError('ffmpeg unavailable; all raw PNGs and timestamps retained')
    subprocess.run([executable,'-hide_banner','-loglevel','error','-nostdin','-n','-f','concat','-safe','0',
        '-i',str(inputs),'-fps_mode','vfr','-c:v','libx264','-threads','1','-pix_fmt','yuv420p',str(video)],
        check=True,capture_output=True,text=True,timeout=60)
    value=dict(path=str(video),sha256=sha256(video),trace_sha256=sha256(trace),frames=len(exposures),
        source='recorded_policy_RGB_exposures_with_last_frame_held_between_decisions',
        continuous_camera_video=False,teacher_fallback=False)
    write_json(folder/'video.json',value);return value
