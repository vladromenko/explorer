"""Transactional episode/video manifests rooted on the Jetson."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
import uuid

SOURCES=('HUMAN_DEMONSTRATION','HUMAN_INTERVENTION','AUTONOMOUS','EVALUATION')
OUTCOMES=('success','failure','unknown','cancelled','infrastructure_error')


class EpisodeStore:
    def __init__(self, root, quota_gb=40):
        self.root=Path(root).resolve();self.folder=self.root/'data/episodes';self.folder.mkdir(parents=True,exist_ok=True)
        self.quota=int(quota_gb*1024**3)

    def usage(self):
        files=[path for path in self.folder.glob('**/*') if path.is_file()]
        used=sum(path.stat().st_size for path in files)
        disk=shutil.disk_usage(self.root)
        return dict(root=str(self.folder),bytes=used,quota_bytes=self.quota,files=len(files),
                    free_bytes=disk.free,recording_allowed=used<self.quota and disk.free>2*1024**3,
                    automatic_deletion=False,automatic_cloud_upload=False)

    def begin(self, job_id, goal, scene_version, policy_version, reward_version, metadata=None,
              source='AUTONOMOUS',task='',skill='',target=''):
        status=self.usage()
        if not status['recording_allowed']:raise ValueError('Episode storage quota or free-space reserve reached')
        if source not in SOURCES:raise ValueError('Invalid episode source')
        identifier=uuid.uuid4().hex;folder=self.folder/identifier;folder.mkdir()
        manifest=dict(id=identifier,episode_id=identifier,format='explorer_typed_episode_v1',job_id=job_id,
                      goal=goal,task=task or goal,skill=skill,target=target,source=source,
                      timestamps={'started_at':time.time(),'ended_at':None},scene_version=scene_version,
                      policy_version=policy_version,reward_version=reward_version,state='recording',
                      started_at=time.time(),ended_at=None,outcome='unknown',failure_reason='',reward=None,
                      observations=[],actions=[],next_observations=[],proposed_actions=[],executed_actions=[],
                      base_commands=[],arm_commands=[],gripper_commands=[],interventions=[],transitions=[],
                      rgb_refs=[],depth_refs=[],video_path=None,artifacts=[],metadata=metadata or {})
        self.write(folder,manifest);return manifest

    def append(self, episode_id, kind, value):
        fields={'observation':'observations','action':'actions','next_observation':'next_observations',
                'proposed_action':'proposed_actions','executed_action':'executed_actions',
                'base_command':'base_commands','arm_command':'arm_commands','gripper_command':'gripper_commands',
                'intervention':'interventions','transition':'transitions','rgb':'rgb_refs','depth':'depth_refs'}
        if kind not in fields:raise ValueError('Invalid episode record kind')
        if not isinstance(value,dict):raise ValueError('Episode records must be objects')
        manifest=self.read(episode_id)
        if manifest.get('state')!='recording':raise ValueError('Episode is not recording')
        record=dict(value)
        if type(record.get('at')) not in (int,float):record['at']=time.time()
        json.dumps(record,allow_nan=False)
        manifest[fields[kind]].append(record)
        self.write(self.path(episode_id),manifest)
        return record

    def intervention(self,episode_id,started_at,ended_at,proposed_action,executed_action,metadata=None):
        if not (isinstance(started_at,(int,float)) and isinstance(ended_at,(int,float)) and ended_at>=started_at):
            raise ValueError('Invalid intervention interval')
        item=dict(at=ended_at,started_at=started_at,ended_at=ended_at,
                  proposed_action=proposed_action,executed_action=executed_action,
                  invalidated_autonomous_action=True,fresh_observation_required=True,
                  metadata=metadata or {})
        return self.append(episode_id,'intervention',item)

    def artifact(self, episode_id, path, kind, timestamps=None):
        folder=self.path(episode_id);source=Path(path).resolve()
        if not source.is_file():raise ValueError('Episode artifact not found')
        if self.root.resolve() not in source.parents:raise ValueError('Artifacts must remain under Explorer root')
        digest=hashlib.sha256(source.read_bytes()).hexdigest();manifest=self.read(episode_id)
        relative=str(source.relative_to(self.root));entry=dict(kind=kind,path=relative,bytes=source.stat().st_size,
                                                               sha256=digest,timestamps=timestamps or {})
        if not any(item['path']==relative for item in manifest['artifacts']):manifest['artifacts'].append(entry)
        self.write(folder,manifest);return entry

    def finish(self, episode_id, outcome, evidence, failure_reason='', reward=None):
        if outcome not in OUTCOMES:
            raise ValueError('Invalid episode outcome')
        if reward is not None and (type(reward) not in (int,float) or not -1<=reward<=1):
            raise ValueError('Reward must be finite and bounded')
        folder=self.path(episode_id);manifest=self.read(episode_id)
        if manifest['state']!='recording':raise ValueError('Episode already finalized')
        ended=time.time();manifest.update(state='complete',ended_at=ended,outcome=outcome,evidence=evidence,
            failure_reason=failure_reason,reward=reward)
        manifest['timestamps']['ended_at']=ended
        self.write(folder,manifest);return manifest

    def import_legacy(self,path,source,task='',skill=''):
        """Index an existing recorder episode without copying or relabelling it."""
        if source not in SOURCES:raise ValueError('Invalid episode source')
        path=Path(path).resolve();legacy=json.loads(path.read_text())
        identity=hashlib.sha256(('legacy:'+str(path)).encode()).hexdigest()[:32]
        folder=self.folder/identity
        if folder.exists():return self.read(identity)
        folder.mkdir();outcome=legacy.get('outcome','unknown')
        if outcome not in OUTCOMES:outcome='unknown'
        steps=legacy.get('steps',[]);manifest=dict(id=identity,episode_id=identity,
            format='explorer_typed_episode_v1',job_id=legacy.get('workflow_id',''),goal=task or legacy.get('name',''),
            task=task or legacy.get('name',''),skill=skill,target=legacy.get('object_label',''),source=source,
            timestamps={'started_at':legacy.get('started',legacy.get('created')),
                        'ended_at':legacy.get('ended')},scene_version=legacy.get('scene_version','legacy-unknown'),
            policy_version='human',reward_version='operator_and_verifier_v1',state='complete',
            started_at=legacy.get('started',legacy.get('created')),ended_at=legacy.get('ended'),outcome=outcome,
            failure_reason=legacy.get('reason',''),reward=1 if outcome=='success' else 0 if outcome=='failure' else None,
            observations=[],actions=[item.get('action',{}) for item in steps],next_observations=[],
            proposed_actions=[],executed_actions=[item.get('action',{}) for item in steps],
            base_commands=[],arm_commands=[item.get('action',{}).get('arm_command') for item in steps if item.get('action',{}).get('arm_command')],
            gripper_commands=[item.get('action',{}).get('gripper_command') for item in steps if item.get('action',{}).get('gripper_command') is not None],
            interventions=[],transitions=[],rgb_refs=[],depth_refs=[],video_path=legacy.get('video_path'),
            artifacts=[{'kind':'legacy_manifest','path':str(path.relative_to(self.root)),
                        'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'bytes':path.stat().st_size,'timestamps':{}}],
            evidence={'legacy_source':legacy.get('source'),'label_source':legacy.get('label_source')},
            metadata={'legacy_episode_id':legacy.get('id'),'joint_state_source':legacy.get('joint_state_source')})
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
