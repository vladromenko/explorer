"""Latest-frame TensorRT perception, camera-frame depth association and SQLite memory."""
import ctypes as C
import json
import math
import os
from pathlib import Path
import sqlite3
import threading
import time
from collections import deque
import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
import tensorrt as trt

ROOT=Path('/home/vlad/Explorer')
LABELS='person,bicycle,car,motorcycle,airplane,bus,train,truck,boat,traffic light,fire hydrant,stop sign,parking meter,bench,bird,cat,dog,horse,sheep,cow,elephant,bear,zebra,giraffe,backpack,umbrella,handbag,tie,suitcase,frisbee,skis,snowboard,sports ball,kite,baseball bat,baseball glove,skateboard,surfboard,tennis racket,bottle,wine glass,cup,fork,knife,spoon,bowl,banana,apple,sandwich,orange,broccoli,carrot,hot dog,pizza,donut,cake,chair,couch,potted plant,bed,dining table,toilet,tv,laptop,mouse,remote,keyboard,cell phone,microwave,oven,toaster,sink,refrigerator,book,clock,vase,scissors,teddy bear,hair drier,toothbrush'.split(',')

class Engine:
    def __init__(self):
        self.cuda=C.CDLL('libcudart.so.13')
        self.cuda.cudaMalloc.argtypes=[C.POINTER(C.c_void_p),C.c_size_t]
        self.cuda.cudaMemcpy.argtypes=[C.c_void_p,C.c_void_p,C.c_size_t,C.c_int]
        self.cuda.cudaFree.argtypes=[C.c_void_p]
        self.cuda.cudaStreamCreate.argtypes=[C.POINTER(C.c_void_p)]
        self.cuda.cudaStreamSynchronize.argtypes=[C.c_void_p]
        self.log=trt.Logger(trt.Logger.WARNING)
        self.runtime=trt.Runtime(self.log)
        self.engine=self.runtime.deserialize_cuda_engine((ROOT/'models/yolo26n.engine').read_bytes())
        if not self.engine:raise RuntimeError('Invalid TensorRT engine')
        self.ctx=self.engine.create_execution_context()
        self.stream=C.c_void_p()
        self.check(self.cuda.cudaStreamCreate(C.byref(self.stream)))
        self.buffers={}
        for i in range(self.engine.num_io_tensors):
            name=self.engine.get_tensor_name(i)
            a=np.empty(self.engine.get_tensor_shape(name), dtype=trt.nptype(self.engine.get_tensor_dtype(name)))
            p=C.c_void_p()
            self.check(self.cuda.cudaMalloc(C.byref(p),a.nbytes))
            self.buffers[name]=(a,p)
            self.ctx.set_tensor_address(name,p.value)
            if self.engine.get_tensor_mode(name)==trt.TensorIOMode.INPUT:self.input=name
            else:self.output=name
    def check(self,code):
        if code:raise RuntimeError('CUDA error '+str(code))
    def infer(self,frame):
        h,w=frame.shape[:2]
        scale=min(640/w,640/h)
        nw,nh=round(w*scale),round(h*scale)
        px,py=(640-nw)//2,(640-nh)//2
        canvas=np.full((640,640,3),114,np.uint8)
        canvas[py:py+nh,px:px+nw]=cv2.resize(frame,(nw,nh))
        inp,ptr=self.buffers[self.input]
        inp[:]=canvas[:,:,::-1].transpose(2,0,1)[None]/255.
        self.check(self.cuda.cudaMemcpy(ptr,inp.ctypes.data,inp.nbytes,1))
        if not self.ctx.execute_async_v3(self.stream.value):raise RuntimeError('TensorRT execute failed')
        self.check(self.cuda.cudaStreamSynchronize(self.stream))
        out,p=self.buffers[self.output]
        self.check(self.cuda.cudaMemcpy(out.ctypes.data,p,out.nbytes,2))
        if out.shape != (1,300,6):raise RuntimeError('Unexpected model output '+str(out.shape))
        detections=[]
        for x1,y1,x2,y2,score,cls in out[0]:
            if score<.4:continue
            idx=int(cls)
            if not 0<=idx<len(LABELS):continue
            box=[float(np.clip((x1-px)/scale,0,w-1)),float(np.clip((y1-py)/scale,0,h-1)),
                 float(np.clip((x2-px)/scale,0,w-1)),float(np.clip((y2-py)/scale,0,h-1))]
            detections.append(dict(label=LABELS[idx],confidence=round(float(score),3),bbox=box))
        return detections

def stamp(msg):return msg.header.stamp.sec+msg.header.stamp.nanosec/1e9

class Perception(Node):
    def __init__(self):
        super().__init__('explorer_perception')
        self.bridge=CvBridge()
        self.rgb=deque(maxlen=8)
        self.depth=deque(maxlen=8)
        self.info=None
        self.lock=threading.Lock()
        self.create_subscription(Image,'/camera/color/image_raw',self.color_cb,qos_profile_sensor_data)
        self.create_subscription(Image,'/camera/depth/image_raw',self.depth_cb,qos_profile_sensor_data)
        self.create_subscription(CameraInfo,'/camera/color/camera_info',self.info_cb,qos_profile_sensor_data)
    def color_cb(self,msg):
        with self.lock:self.rgb.append((self.bridge.imgmsg_to_cv2(msg,'bgr8'),stamp(msg),msg.header.frame_id))
    def depth_cb(self,msg):
        scale=.001 if msg.encoding in ('16UC1','mono16') else 1.
        with self.lock:self.depth.append((self.bridge.imgmsg_to_cv2(msg),stamp(msg),msg.header.frame_id,scale))
    def info_cb(self,msg):
        with self.lock:self.info=msg

def run():
    rclpy.init()
    node=Perception()
    thread=threading.Thread(target=rclpy.spin,args=(node,),daemon=True)
    thread.start()
    engine=Engine()
    db=sqlite3.connect(ROOT/'data/world.sqlite3')
    db.execute('PRAGMA journal_mode=WAL')
    db.execute('CREATE TABLE IF NOT EXISTS observations(id INTEGER PRIMARY KEY, seen REAL, track INTEGER, label TEXT, confidence REAL, frame TEXT, x REAL, y REAL, z REAL)')
    tracks={}
    next_id=1
    previous=-1
    last_save=0
    while rclpy.ok():
        started=time.monotonic()
        with node.lock:rgbs,depths,info=list(node.rgb),list(node.depth),node.info
        rgb=depth=None
        # Match capture timestamps, not the latest two independently delivered frames.
        for candidate in reversed(rgbs):
            if candidate[1]<=previous:break
            match=min(depths,key=lambda d:abs(d[1]-candidate[1]),default=None)
            if match is not None and abs(match[1]-candidate[1])<=.05:
                rgb,depth=candidate,match
                break
        if rgb is None and rgbs and (not depths or time.time()-rgbs[-1][1]>.3):rgb=rgbs[-1]
        if rgb is None or rgb[1]<=previous:
            time.sleep(.01)
            continue
        raw,ts,frame_id=rgb
        frame=raw.copy()
        previous=ts
        ok,raw_jpg=cv2.imencode('.jpg',raw,[cv2.IMWRITE_JPEG_QUALITY,82])
        if ok:
            temp=ROOT/'data/frame-raw.tmp';temp.write_bytes(raw_jpg.tobytes());temp.replace(ROOT/'data/frame-raw.jpg')
        detections=engine.infer(frame)
        used=set()
        for d in detections:
            box=d['bbox']; center=np.array([(box[0]+box[2])/2,(box[1]+box[3])/2])
            matches=[(np.linalg.norm(center-t['center']),key) for key,t in tracks.items() if key not in used and t['label']==d['label'] and ts-t['seen']<1.]
            distance,tid=min(matches,default=(math.inf,None))
            if distance>70:tid=next_id;next_id+=1
            used.add(tid)
            tracks[tid]=dict(center=center,label=d['label'],seen=ts)
            d['track']=tid
            d['position']=None
            # Never index unregistered depth with RGB pixel coordinates.
            aligned=(depth is not None and info is not None and abs(ts-depth[1])<.1
                     and depth[0].shape==frame.shape[:2] and depth[2]==frame_id)
            if aligned and info.k[0]>0 and info.k[4]>0:
                u,v=map(int,center)
                patch=depth[0][max(0,v-8):v+9,max(0,u-8):u+9].astype(float)*depth[3]
                valid=patch[np.isfinite(patch)&(patch>.15)&(patch<5)]
                if valid.size>=10:
                    z=float(np.median(valid))
                    d['position']=dict(x=(u-info.k[2])*z/info.k[0],y=(v-info.k[5])*z/info.k[4],z=z,frame=frame_id)
            a,b,c,e=map(int,box)
            cv2.rectangle(frame,(a,b),(c,e),(66,225,137),2)
            cv2.putText(frame,f"{d['label']} {d['confidence']:.2f}",(a,max(15,b-5)),cv2.FONT_HERSHEY_SIMPLEX,.5,(66,225,137),1)
        tracks={k:v for k,v in tracks.items() if ts-v['seen']<2}
        if time.monotonic()-last_save>2:
            for d in detections:
                p=d['position'] or {}
                db.execute('INSERT INTO observations(seen,track,label,confidence,frame,x,y,z) VALUES(?,?,?,?,?,?,?,?)',
                           (time.time(),d['track'],d['label'],d['confidence'],p.get('frame',frame_id),p.get('x'),p.get('y'),p.get('z')))
            db.commit();last_save=time.monotonic()
        ms=(time.monotonic()-started)*1000
        state=dict(at=time.time(),image_stamp=ts,inference_ms=round(ms,2),objects=detections,
                   depth_sync_ms=round(abs(ts-depth[1])*1000,2) if depth else None,
                   frame=frame_id,world_coordinates_validated=False)
        tmp=ROOT/'data/perception.tmp';tmp.write_text(json.dumps(state));tmp.replace(ROOT/'data/perception.json')
        ok,jpg=cv2.imencode('.jpg',frame,[cv2.IMWRITE_JPEG_QUALITY,78])
        if ok:
            tmp=ROOT/'data/frame.tmp';tmp.write_bytes(jpg.tobytes());tmp.replace(ROOT/'data/frame.jpg')
        time.sleep(max(0,.10-(time.monotonic()-started)))

if __name__=='__main__':run()
