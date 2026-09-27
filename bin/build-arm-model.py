#!/usr/bin/env python3
"""Build a pinned reference description. Never infer live joint positions."""
from pathlib import Path
import xml.etree.ElementTree as ET
import json, hashlib
ROOT=Path(__file__).resolve().parents[1]
src=ROOT/'vendor/m3_reference/M3Pro.urdf'
r=ET.parse(src).getroot();r.set('name','explorer')
# The pinned third-party conversion changed wrist roll to a pitch axis.
# Both official Orin and Nano URDFs use local +Z; restore that source geometry.
r.find("joint[@name='arm5_Joint']/axis").set('xyz','0 0 1')
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
# Fixed sublinks belong to the same physical body at a joint. In particular,
# Gripping is the fixed support plate on arm5, and llink3/rlink3 pivot through it.
# Propagate ONLY existing adjacency across fixed joints, not arbitrary contacts.
for j in r.findall('joint'):
    if j.get('type')!='fixed':
        a,b=j.find('parent').get('link'),j.find('child').get('link')
        for x in links:
            for y in links:
                if root(x)==root(a) and root(y)==root(b) and (x,y)!=(a,b):
                    ET.SubElement(srdf,'disable_collisions',link1=x,link2=y,reason='Adjacent rigid bodies at '+j.get('name'))
# The URDF is a tree, but the physical parallel jaws are four-bar mechanisms.
# These are actual pin connections omitted from the tree, plus the meshing gears.
# Opposing fingertips (rlink2/llink2) remain collision checked.
for a,b,why in [('rlink2','rlink3','Right four-bar closing pin'),
                ('llink2','llink3','Left four-bar closing pin'),
                ('rlink1','llink1','Designed meshing gear contact')]:
    ET.SubElement(srdf,'disable_collisions',link1=a,link2=b,reason=why)
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
