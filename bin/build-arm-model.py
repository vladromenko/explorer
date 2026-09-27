#!/usr/bin/env python3
"""Build a pinned reference description. Never infer live joint positions."""
from pathlib import Path
import xml.etree.ElementTree as ET
import json, hashlib
ROOT=Path(__file__).resolve().parents[1]
src=ROOT/'vendor/m3_reference/M3Pro.urdf'
r=ET.parse(src).getroot();r.set('name','explorer')
assets={}
for mesh in r.findall('.//mesh'):
    asset=ROOT/'vendor/m3_reference/meshes'/Path(mesh.get('filename')).name
    if not asset.is_file() or asset.stat().st_size<84:raise ValueError('Missing mesh: '+str(asset))
    assets[str(asset.relative_to(ROOT))]=hashlib.sha256(asset.read_bytes()).hexdigest()
    mesh.set('filename',asset.as_uri())
for link in r.findall('link'):
    for tag in ('visual','collision'):
        for el in link.findall(tag):
            if el.find('geometry') is None or link.get('name')=='base_footprint':link.remove(el)
for j in r.findall('joint'):
    if j.get('name')=='arm4_Joiint':j.set('name','arm4_Joint')
    limit=j.find('limit')
    if limit is not None and limit.get('lower')==limit.get('upper'):
        j.set('type','fixed')
        j.remove(limit)
        for tag in ('axis','dynamics'):
            el=j.find(tag)
            if el is not None:j.remove(el)
srdf=ET.Element('robot',name='explorer')
arm=ET.SubElement(srdf,'group',name='arm')
for i in range(1,6):ET.SubElement(arm,'joint',name=f'arm{i}_Joint')
grip=ET.SubElement(srdf,'group',name='gripper')
for j in r.findall('joint'):
    if j.get('type')!='fixed' and not j.get('name').startswith('arm'):ET.SubElement(grip,'joint',name=j.get('name'))
# Only rigidly connected bodies and immediate mechanical neighbours are exempt.
# No random-pose sampling is used to hide collisions in the reference model.
links=[x.get('name') for x in r.findall('link')];parent={x:x for x in links}
def root(x):
    while parent[x]!=x:x=parent[x]
    return x
for j in r.findall('joint'):
    a,b=j.find('parent').get('link'),j.find('child').get('link')
    ET.SubElement(srdf,'disable_collisions',link1=a,link2=b,reason='Adjacent')
    if j.get('type')=='fixed':parent[root(a)]=root(b)
for i,a in enumerate(links):
    for b in links[i+1:]:
        if root(a)==root(b):ET.SubElement(srdf,'disable_collisions',link1=a,link2=b,reason='Rigid assembly')
for name,xml in [('explorer.urdf',r),('explorer.srdf',srdf)]:
    ET.indent(xml);ET.ElementTree(xml).write(ROOT/'config'/name,encoding='unicode',xml_declaration=True)
for name in ('explorer.urdf','explorer.srdf'):
    assets['config/'+name]=hashlib.sha256((ROOT/'config'/name).read_bytes()).hexdigest()
manifest=dict(asset_sha256=assets,reference_only=True,calibration_verified=False,joint_state_source=None,
              source_revision='bdac682e57a8eeaf7b18eece44bfbfc54de98e8a',
              original_sha256=hashlib.sha256(src.read_bytes()).hexdigest(),
              joints={str(i):n for i,n in enumerate(['base','shoulder','elbow','wrist_pitch','wrist_rotation','gripper'],1)},
              notes='CAD zero and servo directions are reference data. Gripper linkage is reference-only; no servo-to-aperture calibration. No joint states may be fabricated from commands.')
(ROOT/'config/arm_model.json').write_text(json.dumps(manifest,indent=2)+'\n')
