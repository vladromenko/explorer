"""Event segmentation and reusable skill proposals from Explorer episodes."""
import hashlib
import json
from pathlib import Path
import time
from autonomy_contracts import SkillSpec, Truth


EVENTS=('base_stop','approach_start','gripper_close','lift_verified','transport_start','place_verified')


def segment(events):
    ordered=sorted(events,key=lambda row:float(row['at']))
    boundaries=[row for row in ordered if row.get('event') in EVENTS]
    result=[]
    for index,item in enumerate(boundaries):
        end=boundaries[index+1]['at'] if index+1<len(boundaries) else ordered[-1]['at'] if ordered else item['at']
        rows=[row for row in ordered if item['at']<=row['at']<=end]
        result.append(dict(kind=item['event'],start_at=item['at'],end_at=end,events=rows,
                           completion_verified=any(row.get('outcome')=='success' for row in rows),
                           frame=item.get('frame'),goal=item.get('goal'),representation_version='event_segment_v1'))
    return result


class SkillDiscovery:
    def __init__(self, root, registry):
        self.root=Path(root);self.registry=registry;self.folder=self.root/'data/skills/proposals';self.folder.mkdir(parents=True,exist_ok=True)

    def propose(self, episodes):
        groups={}
        for episode in episodes:
            for part in segment(episode.get('events',[])):
                key=(part['kind'],str(part.get('goal')),str(part.get('frame')))
                groups.setdefault(key,[]).append(dict(episode_id=episode['id'],segment=part))
        proposals=[]
        for key,examples in groups.items():
            if len({item['episode_id'] for item in examples})>=3 and sum(item['segment']['completion_verified'] for item in examples)>=2:
                digest=hashlib.sha256(repr(key).encode()).hexdigest()[:12]
                item=dict(id='discovered_'+digest,kind=key[0],goal=key[1],frame=key[2],
                          examples=[example['episode_id'] for example in examples],state='proposed',
                          requires_operator_review=True,created_at=time.time(),representation_version='event_segment_v1')
                path=self.folder/(item['id']+'.json');temporary=path.with_suffix('.tmp')
                temporary.write_text(json.dumps(item,ensure_ascii=False,indent=2));temporary.replace(path);proposals.append(item)
        return proposals

    def accept(self, identifier, executor, verifier, preconditions, effects):
        path=self.folder/(identifier+'.json')
        try:proposal=json.loads(path.read_text())
        except (OSError,ValueError):raise ValueError('Skill proposal not found')
        if proposal.get('state')!='proposed':raise ValueError('Skill proposal already reviewed')
        skill=SkillSpec(identifier,'1',preconditions,effects,2.5,executor,verifier,source='discovered').validate()
        self.registry.register(skill);proposal.update(state='accepted',accepted_at=time.time(),skill=identifier)
        temporary=path.with_suffix('.tmp');temporary.write_text(json.dumps(proposal,ensure_ascii=False,indent=2));temporary.replace(path)
        return proposal


def replay_anchors(episodes, new_task, validation_fraction=.2):
    """Episode-level split plus task-balanced training anchors; validation stays out."""
    validation=[];training=[]
    for episode in episodes:
        bucket=int(hashlib.sha256(episode['id'].encode()).hexdigest(),16)%1000/1000
        target=validation if bucket<validation_fraction else training
        target.append(episode)
    by_task={}
    for episode in training:by_task.setdefault(episode.get('task','unknown'),[]).append(episode)
    count=max(1,max((len(rows) for rows in by_task.values()),default=1));anchors=[]
    for task,rows in by_task.items():
        ranked=sorted(rows,key=lambda row:(row.get('task')!=new_task,row.get('outcome')!='success',row['id']))
        anchors.extend((ranked*(count//len(ranked)+1))[:count])
    return dict(training_episode_ids=[row['id'] for row in training],validation_episode_ids=[row['id'] for row in validation],
                replay_anchor_episode_ids=[row['id'] for row in anchors],split='global_episode_sha256_v1',
                representation_version='event_segment_v1')
