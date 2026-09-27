"""On-demand object localization. Metric points stay in the optical camera frame."""
import json
from pathlib import Path
import re
import shutil
import subprocess
import threading
import time
import uuid
import cv2
import numpy as np

ALIASES={'носок':'sock','носки':'sock','бутылка':'bottle','бутылку':'bottle','рюкзак':'backpack',
         'чашка':'cup','чашку':'cup','кружка':'mug','кружку':'mug','мяч':'ball','игрушка':'toy',
         'игрушку':'toy','пульт':'remote control','книга':'book','книгу':'book','обувь':'shoe'}

def english_label(label):
    text=label.strip().lower()
    if text in ALIASES:return ALIASES[text]
    if re.fullmatch(r'[a-z][a-z -]{1,59}',text):return text
    raise ValueError('Для этого предмета введите английское название; например sock, bottle или backpack')

def depth_position(sample,box):
    b=np.asarray(box,dtype=float);depth=sample['depth'];h,w=depth.shape
    if b.shape!=(4,) or not np.isfinite(b).all() or b[2]<=b[0] or b[3]<=b[1]:return None
    u,v=(b[:2]+b[2:])/2;radius=max(3,min(12,int(min(b[2]-b[0],b[3]-b[1])/6)))
    x,y=int(np.clip(u,0,w-1)),int(np.clip(v,0,h-1))
    patch=depth[max(0,y-radius):min(h,y+radius+1),max(0,x-radius):min(w,x+radius+1)]
    valid=patch[np.isfinite(patch)&(patch>.15)&(patch<3.)]
    if len(valid)<20 or len(valid)<patch.size*.7:return None
    z=float(np.median(valid));spread=float(np.percentile(valid,90)-np.percentile(valid,10))
    if spread>.04:return None
    xy=cv2.undistortPoints(np.array([[[u,v]]],dtype=float),sample['k'],sample['d'])[0,0]
    return dict(x=float(xy[0]*z),y=float(xy[1]*z),z=z,frame=str(sample['frame']),
                depth_spread_m=spread,depth_is_object_verified=False)

class ObjectFinder:
    def __init__(self,root):
        self.root=Path(root);self.lock=threading.Lock();self.state=dict(phase='idle',executed=False);self.folder=None

    def status(self):return dict(self.state,busy=self.lock.locked())

    def start(self,label):
        translated=english_label(label)
        if not self.lock.acquire(blocking=False):raise ValueError('Поиск предмета уже идёт')
        try:
            power=json.loads((self.root/'data/power.json').read_text())
            if not 0<=time.time()-power['at']<4 or power['state'] not in ('NORMAL','IDLE'):
                raise ValueError('Поиск отложен из-за питания или телеметрии')
            source=self.root/'data/rgbd-snapshot.npz'
            with np.load(source,allow_pickle=False) as sample:
                if not 0<=time.time()-float(sample['stamp'])<2:raise ValueError('Нет свежего совмещённого RGB-D кадра')
            self.folder=self.root/'data/object-searches'/uuid.uuid4().hex;self.folder.mkdir(parents=True)
            shutil.copyfile(source,self.folder/'rgbd.npz')
            (self.folder/'request.json').write_text(json.dumps(dict(label=label,english_label=translated)))
            self.state=dict(phase='searching',label=label,executed=False,id=self.folder.name)
            threading.Thread(target=self.run,args=(self.folder,),daemon=True).start()
            return self.status()
        except Exception:self.lock.release();raise

    def run(self,folder):
        try:
            with (folder/'log.txt').open('w') as log:
                subprocess.run(['systemd-run','--user','--quiet','--wait','--pipe','--collect',
                    '--unit=explorer-grounding-'+folder.name,'--property=MemoryMax=3000M','--property=PartOf=explorer.target',
                    '--property=Nice=15','--property=CPUWeight=10','--property=RuntimeMaxSec=90',
                    str(self.root/'.venv-learning/bin/python'),str(self.root/'bin/ground-object.py'),str(folder)],
                    check=True,timeout=100,stdout=log,stderr=log)
            self.state.update(phase='ready',result=json.loads((folder/'result.json').read_text()))
        except (OSError,ValueError,subprocess.SubprocessError) as exc:self.state.update(phase='error',error=str(exc))
        finally:self.lock.release()
