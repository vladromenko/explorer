"""Transactional episode/video manifests rooted on the Jetson."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
import uuid


class EpisodeStore:
    def __init__(self, root, quota_gb=40):
        self.root=Path(root);self.folder=self.root/'data/episodes';self.folder.mkdir(parents=True,exist_ok=True)
        self.quota=int(quota_gb*1024**3)

    def usage(self):
        files=[path for path in self.folder.glob('**/*') if path.is_file()]
        used=sum(path.stat().st_size for path in files)
        disk=shutil.disk_usage(self.root)
        return dict(root=str(self.folder),bytes=used,quota_bytes=self.quota,files=len(files),
                    free_bytes=disk.free,recording_allowed=used<self.quota and disk.free>2*1024**3,
                    automatic_deletion=False,automatic_cloud_upload=False)

    def begin(self, job_id, goal, scene_version, policy_version, reward_version, metadata=None):
        status=self.usage()
        if not status['recording_allowed']:raise ValueError('Episode storage quota or free-space reserve reached')
        identifier=uuid.uuid4().hex;folder=self.folder/identifier;folder.mkdir()
        manifest=dict(id=identifier,job_id=job_id,goal=goal,scene_version=scene_version,
                      policy_version=policy_version,reward_version=reward_version,state='recording',
                      started_at=time.time(),ended_at=None,outcome='unknown',artifacts=[],metadata=metadata or {})
        self.write(folder,manifest);return manifest

    def artifact(self, episode_id, path, kind, timestamps=None):
        folder=self.path(episode_id);source=Path(path).resolve()
        if not source.is_file():raise ValueError('Episode artifact not found')
        if self.root.resolve() not in source.parents:raise ValueError('Artifacts must remain under Explorer root')
        digest=hashlib.sha256(source.read_bytes()).hexdigest();manifest=self.read(episode_id)
        relative=str(source.relative_to(self.root));entry=dict(kind=kind,path=relative,bytes=source.stat().st_size,
                                                               sha256=digest,timestamps=timestamps or {})
        if not any(item['path']==relative for item in manifest['artifacts']):manifest['artifacts'].append(entry)
        self.write(folder,manifest);return entry

    def finish(self, episode_id, outcome, evidence):
        if outcome not in ('success','failure','unknown','cancelled','infrastructure_error'):
            raise ValueError('Invalid episode outcome')
        folder=self.path(episode_id);manifest=self.read(episode_id)
        if manifest['state']!='recording':raise ValueError('Episode already finalized')
        manifest.update(state='complete',ended_at=time.time(),outcome=outcome,evidence=evidence)
        self.write(folder,manifest);return manifest

    def recover(self):
        recovered=[]
        for path in self.folder.glob('*/episode.json'):
            try:manifest=json.loads(path.read_text())
            except (OSError,ValueError):manifest=None
            if manifest and manifest.get('state')=='recording':
                manifest.update(state='interrupted',ended_at=time.time(),outcome='unknown',
                                evidence={'reason':'Process restarted; unfinished motion was not replayed'})
                self.write(path.parent,manifest);recovered.append(manifest['id'])
        return recovered

    def path(self, identifier):
        if not isinstance(identifier,str) or len(identifier)!=32 or any(ch not in '0123456789abcdef' for ch in identifier):
            raise ValueError('Invalid episode identifier')
        folder=self.folder/identifier
        if not folder.is_dir():raise ValueError('Episode not found')
        return folder

    def read(self, identifier):return json.loads((self.path(identifier)/'episode.json').read_text())

    @staticmethod
    def write(folder, value):
        path=Path(folder)/'episode.json';temporary=path.with_suffix('.tmp')
        temporary.write_text(json.dumps(value,ensure_ascii=False,allow_nan=False,indent=2));temporary.replace(path)
