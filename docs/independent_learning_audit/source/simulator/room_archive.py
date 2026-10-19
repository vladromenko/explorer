"""Complete room-cycle evidence, separate from the small Jetson package."""
import hashlib
import json
from pathlib import Path
import tarfile
import tempfile

from .learning import sha256,write_json
from .transfer import _safe_name


def verify(path,*,cancelled=lambda:False):
    with tarfile.open(path,'r:gz') as stream:
        members=stream.getmembers();names=[m.name for m in members]
        if len(names)!=len(set(names)) or any(not m.isfile() for m in members):raise ValueError('Cycle archive accepts unique regular files only')
        for name in names:_safe_name(name)
        manifest=json.load(stream.extractfile('cycle-archive.json'))
        if manifest.get('format')!='explorer_room_cycle_v1' or manifest.get('physical_ready') is not False:
            raise ValueError('Invalid room evidence archive')
        if set(manifest['files'])!=set(names)-{'cycle-archive.json'}:raise ValueError('Cycle archive manifest differs')
        for name,digest in manifest['files'].items():
            if cancelled():raise InterruptedError('Cycle verification paused')
            with stream.extractfile(name) as source:
                if hashlib.file_digest(source,'sha256').hexdigest()!=digest:raise ValueError('Cycle evidence checksum mismatch: '+name)
    return manifest


def archive_cycle(folder,transfer,destination,*,cancelled=lambda:False):
    folder,destination=Path(folder),Path(destination)
    if sha256(transfer['path'])!=transfer['sha256']:raise ValueError('Transfer package changed before cycle archive')
    if destination.exists():
        manifest=verify(destination,cancelled=cancelled)
        if manifest['transfer_sha256']!=transfer['sha256']:raise ValueError('Existing cycle archive contains a different transfer')
        return dict(path=str(destination),sha256=sha256(destination),verified=True)
    files={}
    for p in folder.rglob('*'):
        if p.is_symlink():raise ValueError('Cycle evidence must not contain links')
        if p.is_file() and p.suffix!='.lock' and not p.name.endswith('.tmp'):
            files['evidence/'+str(p.relative_to(folder))]=p
    files['transfer.tar.gz']=Path(transfer['path'])
    destination.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.room-cycle-',dir=destination.parent) as temporary:
        stage=Path(temporary);digests={}
        for name,p in files.items():
            if cancelled():raise InterruptedError('Cycle archival paused')
            digests[name]=sha256(p)
        manifest=stage/'cycle-archive.json';write_json(manifest,dict(format='explorer_room_cycle_v1',files=digests,
            physical_ready=False,transfer_sha256=transfer['sha256']))
        archive=stage/'cycle.tar.gz'
        with tarfile.open(archive,'w:gz') as stream:
            for name,p in files.items():
                if cancelled():raise InterruptedError('Cycle archival paused')
                stream.add(p,arcname=name,recursive=False)
            stream.add(manifest,arcname='cycle-archive.json',recursive=False)
        verify(archive,cancelled=cancelled);archive.replace(destination)
    return dict(path=str(destination),sha256=sha256(destination),verified=True)
