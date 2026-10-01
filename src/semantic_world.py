"""Persistent semantic/episodic world memory with conservative 3-D fusion.

Every fresh perception frame becomes an episode tied to the robot map pose.
Object identities and moved/missing states are changed only when the detector
provides a validated map-frame point.  Until then queries return the viewpoint
from which an object was seen, rather than inventing an object coordinate.
"""
import json
import math
from pathlib import Path
import sqlite3
import threading
import time
import uuid


def finite(values):
    return all(type(v) in (int,float) and math.isfinite(v) for v in values)


class SemanticWorld:
    def __init__(self, root):
        self.path=Path(root)/'data/semantic-world.sqlite3'
        self.path.parent.mkdir(parents=True,exist_ok=True)
        self.lock=threading.RLock();self.last_frame=None
        with self.db() as db:
            db.executescript('''
            CREATE TABLE IF NOT EXISTS episodes(
              id TEXT PRIMARY KEY, seen REAL, map_epoch TEXT, pose TEXT,
              signature TEXT, objects TEXT, source TEXT);
            CREATE TABLE IF NOT EXISTS entities(
              id TEXT PRIMARY KEY, label TEXT, state TEXT, first_seen REAL,
              last_seen REAL, map_epoch TEXT, position TEXT, confidence REAL,
              observations INTEGER, missing_count INTEGER);
            CREATE TABLE IF NOT EXISTS entity_events(
              id INTEGER PRIMARY KEY, at REAL, entity_id TEXT, operation TEXT,
              position TEXT, evidence TEXT);
            CREATE TABLE IF NOT EXISTS places(
              id TEXT PRIMARY KEY, map_epoch TEXT, x REAL, y REAL, yaw REAL,
              signature TEXT, first_seen REAL, last_seen REAL, visits INTEGER,
              validated INTEGER DEFAULT 1);
            CREATE TABLE IF NOT EXISTS actions(
              id INTEGER PRIMARY KEY, at REAL, task TEXT, skill TEXT, state TEXT,
              before_episode TEXT, after_episode TEXT, result TEXT);
            CREATE TABLE IF NOT EXISTS relations(
              id INTEGER PRIMARY KEY, at REAL, subject TEXT, predicate TEXT,
              object TEXT, confidence REAL, evidence TEXT);
            CREATE TABLE IF NOT EXISTS memory_meta(key TEXT PRIMARY KEY, value TEXT);
            ''')
            db.execute("INSERT OR IGNORE INTO memory_meta VALUES('scene_generation','0')")
            columns={row[1] for row in db.execute('PRAGMA table_info(places)')}
            if 'validated' not in columns:
                db.execute('ALTER TABLE places ADD COLUMN validated INTEGER DEFAULT 0')
            row=db.execute('SELECT max(seen) FROM episodes').fetchone()
            self.last_frame=row[0] if row and row[0] is not None else None

    def db(self):
        db=sqlite3.connect(self.path,timeout=3)
        db.row_factory=sqlite3.Row
        return db

    @staticmethod
    def signature(objects):
        counts={}
        for obj in objects:
            label=str(obj.get('label','')).strip().casefold()
            if label and float(obj.get('confidence',0))>=.6:counts[label]=counts.get(label,0)+1
        return counts

    @staticmethod
    def similarity(a,b):
        keys=set(a)|set(b)
        if not keys:return 0.
        dot=sum(a.get(k,0)*b.get(k,0) for k in keys)
        na=math.sqrt(sum(v*v for v in a.values()));nb=math.sqrt(sum(v*v for v in b.values()))
        return dot/(na*nb) if na and nb else 0.

    def _place(self,db,epoch,pose,signature,now):
        if not pose or not finite([pose.get('x'),pose.get('y'),pose.get('yaw')]):return None
        if pose.get('provisional') is True or pose.get('validated') is False:return None
        rows=db.execute('SELECT * FROM places WHERE map_epoch=? AND validated=1',(epoch,)).fetchall()
        choices=[]
        for row in rows:
            distance=math.hypot(pose['x']-row['x'],pose['y']-row['y'])
            score=self.similarity(signature,json.loads(row['signature']))
            if distance<=.25 or distance<=.75 and score>=.65:choices.append((score-.25*distance,row))
        if choices:
            row=max(choices,key=lambda item:item[0])[1]
            visits=row['visits']+1
            weight=min(visits,20)
            merged=dict(json.loads(row['signature']))
            for key,value in signature.items():merged[key]=max(value,merged.get(key,0))
            db.execute('UPDATE places SET x=?,y=?,yaw=?,signature=?,last_seen=?,visits=? WHERE id=?',
                ((row['x']*(weight-1)+pose['x'])/weight,(row['y']*(weight-1)+pose['y'])/weight,
                 (row['yaw']*(weight-1)+pose['yaw'])/weight,json.dumps(merged),now,visits,row['id']))
            return row['id']
        if not signature:return None
        ident=uuid.uuid4().hex
        db.execute('INSERT INTO places VALUES(?,?,?,?,?,?,?,?,?,?)',
            (ident,epoch,pose['x'],pose['y'],pose['yaw'],json.dumps(signature),now,now,1,1))
        return ident

    def _validated_point(self,obj):
        value=obj.get('map_position')
        if obj.get('map_position_validated') is not True or not isinstance(value,dict):return None
        xyz=[value.get(k) for k in ('x','y','z')]
        return xyz if finite(xyz) and max(abs(v) for v in xyz)<100 else None

    def _entity(self,db,obj,epoch,now):
        point=self._validated_point(obj)
        if point is None:return None
        label=str(obj.get('label','')).casefold();confidence=float(obj.get('confidence',0))
        rows=db.execute("SELECT * FROM entities WHERE label=? AND map_epoch=? AND state!='removed'",(label,epoch)).fetchall()
        near=[]
        for row in rows:
            previous=json.loads(row['position'])
            distance=math.dist(point,[previous[k] for k in ('x','y','z')])
            if distance<=.45:near.append((distance,row))
        operation='appeared'
        if near:
            distance,row=min(near,key=lambda item:item[0]);ident=row['id']
            operation='moved' if distance>.08 else 'observed'
            observations=row['observations']+1
            db.execute('UPDATE entities SET state=?,last_seen=?,position=?,confidence=?,observations=?,missing_count=0 WHERE id=?',
                ('active',now,json.dumps(dict(zip(('x','y','z'),point))),confidence,observations,ident))
        else:
            ident=uuid.uuid4().hex
            db.execute('INSERT INTO entities VALUES(?,?,?,?,?,?,?,?,?,?)',
                (ident,label,'active',now,now,epoch,json.dumps(dict(zip(('x','y','z'),point))),confidence,1,0))
        db.execute('INSERT INTO entity_events(at,entity_id,operation,position,evidence) VALUES(?,?,?,?,?)',
            (now,ident,operation,json.dumps(point),json.dumps({'confidence':confidence,'validated_map_point':True})))
        return ident

    def ingest(self,perception,pose=None,map_epoch='unknown',source='live_perception'):
        stamp=perception.get('image_stamp',perception.get('at'))
        if not type(stamp) in (int,float) or not math.isfinite(stamp):raise ValueError('Invalid perception timestamp')
        objects=perception.get('objects',[])
        if not isinstance(objects,list):raise ValueError('Invalid objects')
        with self.lock:
            if self.last_frame is not None and stamp<=self.last_frame:return dict(added=False,reason='old_frame')
            self.last_frame=stamp
            signature=self.signature(objects);now=time.time();ident=uuid.uuid4().hex
            safe_pose=pose if pose and finite([pose.get('x'),pose.get('y'),pose.get('yaw')]) else None
            content=[dict(label=o.get('label'),confidence=o.get('confidence'),bbox=o.get('bbox'),
                          camera_position=o.get('position'),map_position=o.get('map_position'),
                          map_position_validated=o.get('map_position_validated') is True) for o in objects]
            with self.db() as db:
                previous=db.execute('SELECT seen,pose,signature FROM episodes ORDER BY seen DESC LIMIT 1').fetchone()
                has_validated_point=any(self._validated_point(obj) is not None for obj in objects)
                if (previous and not has_validated_point and perception.get('visibility_volume_validated') is not True
                        and stamp-previous['seen']<30 and signature==json.loads(previous['signature'])):
                    old_pose=json.loads(previous['pose'])
                    pose_still=(safe_pose is None and old_pose is None or safe_pose is not None and old_pose is not None and
                                math.hypot(safe_pose['x']-old_pose['x'],safe_pose['y']-old_pose['y'])<.15)
                    if pose_still:return dict(added=False,reason='unchanged_view')
                place=self._place(db,map_epoch,safe_pose,signature,now)
                db.execute('INSERT INTO episodes VALUES(?,?,?,?,?,?,?)',
                    (ident,stamp,map_epoch,json.dumps(safe_pose),json.dumps(signature),json.dumps(content),source))
                entities=[self._entity(db,o,map_epoch,now) for o in objects]
                observed={v for v in entities if v is not None}
                operations={row[0] for row in db.execute('SELECT operation FROM entity_events WHERE at=?',(now,))}
                changed=bool(operations & {'appeared','moved'})
                observable=perception.get('observable_entity_ids',[])
                if perception.get('visibility_volume_validated') is True and isinstance(observable,list):
                    for entity_id in observable:
                        if entity_id in observed:pass
                        elif isinstance(entity_id,str):
                            row=db.execute("SELECT state,missing_count FROM entities WHERE id=? AND map_epoch=?",(entity_id,map_epoch)).fetchone()
                            if row and row['state']!='removed':
                                count=row['missing_count']+1;state='missing' if count>=3 else row['state']
                                db.execute('UPDATE entities SET state=?,missing_count=? WHERE id=?',(state,count,entity_id))
                                db.execute('INSERT INTO entity_events(at,entity_id,operation,position,evidence) VALUES(?,?,?,?,?)',
                                    (now,entity_id,'missing' if count>=3 else 'not_observed',None,
                                     json.dumps({'validated_visibility_volume':True,'consecutive_misses':count})))
                                changed=changed or count>=3
                if changed:
                    generation=int(db.execute("SELECT value FROM memory_meta WHERE key='scene_generation'").fetchone()[0])+1
                    db.execute("UPDATE memory_meta SET value=? WHERE key='scene_generation'",(str(generation),))
            return dict(added=True,episode=ident,place=place,objects=len(objects),
                        fused_entities=sum(v is not None for v in entities),
                        viewpoint_known=safe_pose is not None)

    def query(self,label,limit=20):
        query=label.strip().casefold()
        with self.db() as db:
            entities=[dict(r) for r in db.execute('SELECT * FROM entities WHERE label LIKE ? ORDER BY last_seen DESC LIMIT ?',('%'+query+'%',limit))]
            episodes=[]
            for row in db.execute('SELECT * FROM episodes ORDER BY seen DESC LIMIT 500'):
                objects=json.loads(row['objects'])
                matches=[o for o in objects if query in str(o.get('label','')).casefold()]
                if matches:
                    episodes.append(dict(id=row['id'],seen=row['seen'],map_epoch=row['map_epoch'],
                                         observer_pose=json.loads(row['pose']),objects=matches))
                    if len(episodes)>=limit:break
        for entity in entities:entity['position']=json.loads(entity['position'])
        return dict(query=label,entities=entities,episodes=episodes,
                    last_seen=episodes[0] if episodes else None,
                    coordinate_contract='entity.position is map-frame only; episode.observer_pose is where the robot saw it')

    def record_action(self,task,skill,state,result=None,before_episode=None,after_episode=None):
        if state not in ('started','succeeded','failed','intervention','cancelled','unknown'):
            raise ValueError('Invalid action state')
        with self.db() as db:
            now=time.time()
            cursor=db.execute('INSERT INTO actions(at,task,skill,state,before_episode,after_episode,result) VALUES(?,?,?,?,?,?,?)',
                       (now,str(task)[:80],str(skill)[:80],state,before_episode,after_episode,
                        json.dumps(result or {},ensure_ascii=False)))
            db.execute('INSERT INTO relations(at,subject,predicate,object,confidence,evidence) VALUES(?,?,?,?,?,?)',
                       (now,'task:'+str(task)[:80],'executed_skill','skill:'+str(skill)[:80],1.,
                        json.dumps({'action_id':cursor.lastrowid,'state':state},ensure_ascii=False)))
            if state in ('succeeded','failed','unknown'):
                db.execute('INSERT INTO relations(at,subject,predicate,object,confidence,evidence) VALUES(?,?,?,?,?,?)',
                           (now,'skill:'+str(skill)[:80],'had_outcome',state,1.,json.dumps(result or {},ensure_ascii=False)))

    def scene_version(self,map_epoch='unknown'):
        with self.db() as db:generation=db.execute("SELECT value FROM memory_meta WHERE key='scene_generation'").fetchone()[0]
        return str(map_epoch)+':'+str(generation)

    def related(self,subject,limit=50):
        with self.db() as db:
            return [dict(row,evidence=json.loads(row['evidence'])) for row in db.execute(
                'SELECT * FROM relations WHERE subject=? OR object=? ORDER BY id DESC LIMIT ?',
                (subject,subject,limit))]

    def status(self):
        with self.db() as db:
            counts={name:db.execute('SELECT count(*) FROM '+name).fetchone()[0]
                    for name in ('episodes','entities','actions','relations')}
            counts['places']=db.execute('SELECT count(*) FROM places WHERE validated=1').fetchone()[0]
            provisional_places=db.execute('SELECT count(*) FROM places WHERE validated=0').fetchone()[0]
            validated=counts['entities']
            recent=[dict(r) for r in db.execute('SELECT at,task,skill,state FROM actions ORDER BY id DESC LIMIT 10')]
            generation=db.execute("SELECT value FROM memory_meta WHERE key='scene_generation'").fetchone()[0]
        return dict(**counts,provisional_places=provisional_places,validated_3d_entities=validated,recent_actions=recent,
                    scene_generation=int(generation),
                    dynamic_3d_enabled=True,unvalidated_camera_points_never_fused=True)


def active_view(perception,world_status,flags):
    """Choose a bounded next observation; the caller owns any physical action."""
    objects=perception.get('objects',[]) if isinstance(perception,dict) else []
    uncertain=[o for o in objects if o.get('position') is None or float(o.get('confidence',0))<.65]
    missing=[]
    if flags.get('camera_tf_validated') is not True:missing.append('camera_tf_validated')
    if flags.get('arm_commissioned') is not True:missing.append('arm_commissioned')
    if uncertain:
        target=min(uncertain,key=lambda o:float(o.get('confidence',0)))
        return dict(kind='inspect_target',target=target.get('label'),proposal='small_camera_viewpoint_change',
                    reason='depth_or_detector_uncertainty',physical_allowed=not missing,blocked_by=missing)
    return dict(kind='explore_frontier',proposal='highest_information_reachable_frontier',
                reason='target_not_visible',physical_allowed=False,
                blocked_by=[k for k in ('base_commissioned','mcu_watchdog_verified','lidar_tf_validated','localization_verified') if flags.get(k) is not True])


def guarded_closure(observations,soft=True):
    """Visual contact policy; returns one small gripper decision, never a force claim."""
    if not isinstance(observations,list) or len(observations)<2:return dict(action='observe',reason='need_two_fresh_frames')
    a,b=observations[-2:]
    required=('at','association_fraction','object_scale','object_tcp_distance_m')
    if any(not finite([row.get(k) for k in required]) for row in (a,b)):
        return dict(action='stop',reason='visual_contact_inputs_unavailable',contact='unknown')
    if not 0<b['at']-a['at']<.5:return dict(action='stop',reason='visual_feedback_stale',contact='unknown')
    slip=b['object_tcp_distance_m']-a['object_tcp_distance_m']>.008
    deformation=(b['object_scale']-a['object_scale'])/max(a['object_scale'],1e-6)
    contact=b['object_tcp_distance_m']<.025 and b['association_fraction']>=.7
    if b['association_fraction']<.7:return dict(action='stop',reason='object_tracking_lost',contact='unknown')
    if soft and deformation>.08:return dict(action='hold',reason='soft_object_deformation_limit',contact=True)
    if slip:return dict(action='tighten',step_deg=1,reason='visual_slip',contact=contact)
    if contact:return dict(action='hold',reason='minimum_sufficient_closure',contact=True)
    return dict(action='close',step_deg=1 if soft else 2,reason='no_contact_yet',contact=False)
