#!/usr/bin/env python3
"""Install repository-verified MoveIt packages in Explorer's private prefix.

No root access or system package changes. The apt solver supplies the exact
missing dependency closure; downloaded package versions and hashes are recorded.
"""
from pathlib import Path
import concurrent.futures
import hashlib
import json
import re
import subprocess

root=Path('/home/vlad/Explorer')
packages=['ros-jazzy-moveit-ros-move-group','ros-jazzy-moveit-planners-ompl',
          'ros-jazzy-moveit-kinematics','ros-jazzy-moveit-servo',
          'ros-jazzy-moveit-simple-controller-manager','ros-jazzy-moveit-py']
plan=subprocess.check_output(['apt-get','-s','install','--no-install-recommends',*packages],text=True)
items=re.findall(r'^Inst (\S+)(?: \[[^\]]+\])? \((\S+)',plan,re.M)
cache=root/'vendor/moveit_debs';prefix=root/'vendor/moveit_root'
cache.mkdir(exist_ok=True);prefix.mkdir(exist_ok=True)
def download(item):
    subprocess.run(['apt-get','download',item[0]+'='+item[1]],cwd=cache,check=True,stdout=subprocess.DEVNULL)
with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:list(pool.map(download,items))
archives=[]
for deb in sorted(cache.glob('*.deb')):
    subprocess.run(['dpkg-deb','-x',str(deb),str(prefix)],check=True)
    archives.append(dict(file=deb.name,sha256=hashlib.sha256(deb.read_bytes()).hexdigest()))
link=root/'vendor/ros/moveit'
if not link.exists():link.symlink_to(prefix/'opt/ros/jazzy',target_is_directory=True)
(root/'config/moveit_packages.json').write_text(json.dumps(dict(requested=packages,dependencies=items,archives=archives),indent=2))
print('MoveIt private prefix installed:',len(archives),'packages',prefix)
