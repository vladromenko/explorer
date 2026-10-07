#!/usr/bin/env python3
"""Read-only CAD check for existing room-camera footprint; no actuator commands."""
import hashlib
import json
from pathlib import Path
import struct
import sys
import xml.etree.ElementTree as ET
import numpy as np
from scipy.spatial.transform import Rotation

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
import arm_model
from camera_views import VIEWS


def vertices(path):
    data=path.read_bytes()
    if len(data)<84:raise ValueError("Incomplete binary STL: "+str(path))
    count=struct.unpack_from("<I",data,80)[0]
    if len(data)!=84+count*50:raise ValueError("Unexpected STL encoding: "+str(path))
    dtype=np.dtype([("normal","<f4",(3,)),("vertices","<f4",(3,3)),("attribute","<u2")])
    result=np.frombuffer(data,dtype=dtype,count=count,offset=84)["vertices"].reshape(-1,3).astype(float)
    if not np.isfinite(result).all():raise ValueError("Non-finite mesh geometry")
    return result


def main():
    arm_model.ROOT=ROOT
    model=arm_model.ArmModel()  # Pins all mesh, URDF and SRDF hashes first.
    urdf=ET.parse(ROOT/"config/explorer.urdf").getroot()
    meshes=[]
    for link in urdf.findall("link"):
        for collision in link.findall("collision"):
            mesh=collision.find("geometry/mesh")
            if mesh is None:raise ValueError("Unsupported collision geometry")
            path=Path(mesh.attrib["filename"].removeprefix("file://"))
            path=ROOT/path.relative_to("/home/vlad/Explorer")
            points=vertices(path)*np.fromstring(mesh.get("scale","1 1 1"),sep=" ")
            origin=collision.find("origin")
            xyz=np.fromstring(origin.get("xyz","0 0 0") if origin is not None else "0 0 0",sep=" ")
            rpy=np.fromstring(origin.get("rpy","0 0 0") if origin is not None else "0 0 0",sep=" ")
            local=points@Rotation.from_euler("xyz",rpy).as_matrix().T+xyz
            meshes.append((link.attrib["name"],np.column_stack([local,np.ones(len(local))])))
    lower=np.full(2,np.inf);upper=np.full(2,-np.inf)
    for view in VIEWS.values():
        for jaw in np.linspace(-1.54,0.,33):
            model.set_state(view,float(jaw))
            for name,points in meshes:
                transformed=points@np.asarray(model.state.get_global_link_transform(name)).T
                lower=np.minimum(lower,transformed[:,:2].min(axis=0))
                upper=np.maximum(upper,transformed[:,:2].max(axis=0))
    profile=json.loads((ROOT/"config/navigation-footprint.json").read_text())
    polygon=np.asarray(profile["polygon"])
    margin=float(min(np.min(lower-polygon.min(axis=0)),np.min(polygon.max(axis=0)-upper)))
    result=dict(cad_bounds_xy_m=[lower.tolist(),upper.tolist()],minimum_margin_m=margin,
        jaw_range_rad=[-1.54,0.],jaw_samples=33,views=list(VIEWS),no_actuator_io=True,
        manifest_sha256=hashlib.sha256((ROOT/"config/arm_model.json").read_bytes()).hexdigest())
    print(json.dumps(result,indent=2))
    if margin<.025:raise ValueError("Camera footprint has insufficient CAD margin")


if __name__=="__main__":main()
