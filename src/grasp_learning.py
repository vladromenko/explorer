"""Robot-side replay memory and a small neural grasp ranker.

Learning changes candidate preference, never joint limits or execution permission.
Unknown outcomes and assistant-operated calibration are excluded from training.
"""
import hashlib
import json
from pathlib import Path
import sqlite3
import time
import uuid
import cv2
import numpy as np

FEATURE_VERSION = 'rgb8x8_xyz_aperture_roll_v1'
FEATURES = 197


def features(image, bbox, xyz, aperture_m, roll_deg):
    if image is None or image.ndim != 3:raise ValueError('RGB observation required')
    a = np.asarray([*bbox, *xyz, aperture_m, roll_deg], dtype=float)
    if a.shape != (9,) or not np.all(np.isfinite(a)):raise ValueError('Finite grasp features required')
    x1, y1, x2, y2 = map(int, bbox)
    h, w = image.shape[:2]
    if not (0 <= x1 < x2 <= w and 0 <= y1 < y2 <= h):raise ValueError('Invalid crop')
    if not 0 <= aperture_m <= .07 or max(abs(v) for v in xyz) > 1:raise ValueError('Invalid metric features')
    rgb = cv2.resize(image[y1:y2, x1:x2, ::-1], (8, 8)).astype(float).reshape(-1)/127.5-1
    return np.r_[rgb, np.asarray(xyz)*2, aperture_m/.07, roll_deg/180].tolist()


class GraspMemory:
    def __init__(self, directory):
        self.root = Path(directory); self.root.mkdir(parents=True, exist_ok=True)
        with self.db() as db:
            db.execute('CREATE TABLE IF NOT EXISTS attempts(id TEXT PRIMARY KEY, created REAL, scene TEXT, '
                       'source TEXT, features TEXT, proposal TEXT, outcome TEXT, evidence TEXT, feature_version TEXT)')

    def db(self):return sqlite3.connect(self.root/'grasp.sqlite3', timeout=3)

    def begin(self, scene, source, vector, proposal, image):
        x = np.asarray(vector, dtype=float)
        if x.shape != (FEATURES,) or not np.all(np.isfinite(x)):raise ValueError('Invalid feature vector')
        if source not in ('robot', 'observed_calibration'):raise ValueError('Invalid attempt source')
        if not isinstance(scene, str) or not scene or len(scene)>120:raise ValueError('Scene identifier required')
        identifier = uuid.uuid4().hex
        if not cv2.imwrite(str(self.root/(identifier+'-before.jpg')), image):raise ValueError('Image storage failed')
        with self.db() as db:
            db.execute('INSERT INTO attempts VALUES(?,?,?,?,?,?,?,?,?)',
                       (identifier, time.time(), scene, source, json.dumps(vector), json.dumps(proposal),
                        'unknown', '{}', FEATURE_VERSION))
        return identifier

    def finish(self, identifier, outcome, evidence, image):
        if outcome not in ('success', 'failure', 'unknown'):raise ValueError('Invalid outcome')
        # Only an independently validated verifier may provide training labels.
        # A language-model assertion or completed servo packet cannot label success.
        with self.db() as db:
            row = db.execute('SELECT outcome,source FROM attempts WHERE id=?', (identifier,)).fetchone()
            if row is None or row[0] != 'unknown':raise ValueError('Unknown or already labelled attempt')
            verified = (evidence.get('verifier') == 'rgbd_lift_v1' and
                        evidence.get('calibration_validated') is True and
                        evidence.get('object_identity_verified') is True)
            observed = row[1] == 'observed_calibration' and evidence.get('verifier') == 'external_camera_observer'
            if outcome != 'unknown' and not (verified or observed):
                raise ValueError('Outcome requires independent visual verification')
            if not cv2.imwrite(str(self.root/(identifier+'-after.jpg')), image):raise ValueError('Image storage failed')
            db.execute('UPDATE attempts SET outcome=?,evidence=? WHERE id=?',
                       (outcome, json.dumps(evidence), identifier))

    def status(self):
        with self.db() as db:
            counts = dict(db.execute('SELECT outcome,COUNT(*) FROM attempts GROUP BY outcome'))
            eligible = db.execute("SELECT COUNT(*) FROM attempts WHERE source='robot' AND outcome IN ('success','failure') AND feature_version=?", (FEATURE_VERSION,)).fetchone()[0]
        model = self.root/'model.json'
        metadata = json.loads(model.read_text()) if model.exists() else None
        return dict(attempts=counts, training_eligible=eligible, model=metadata,
                    learning_location='Jetson', execution_permission=False,
                    reason=None if metadata else 'Insufficient verified autonomous episodes; no trained ranker yet')

    def dataset(self):
        with self.db() as db:
            rows = db.execute("SELECT scene,features,outcome FROM attempts WHERE source='robot' AND outcome IN ('success','failure') AND feature_version=? ORDER BY created", (FEATURE_VERSION,)).fetchall()
        if len(rows)<50:raise ValueError('At least 50 verified robot attempts required')
        scenes = sorted(set(r[0] for r in rows))
        if len(scenes)<6:raise ValueError('At least six independently reset scenes required')
        # Scene-level split prevents nearly identical frames from leaking into validation.
        held = set(sorted(scenes, key=lambda s:hashlib.sha256(s.encode()).hexdigest())[::4])
        train = [r for r in rows if r[0] not in held]
        valid = [r for r in rows if r[0] in held]
        def arrays(items):
            x=np.asarray([json.loads(r[1]) for r in items]); y=np.asarray([r[2]=='success' for r in items],dtype=float)
            if x.ndim!=2 or x.shape[1]!=FEATURES or not np.all(np.isfinite(x)):raise ValueError('Corrupt replay data')
            if min(np.count_nonzero(y),np.count_nonzero(1-y))<3:raise ValueError('Both outcomes required in each scene split')
            return x,y
        return arrays(train), arrays(valid), scenes

    @staticmethod
    def forward(x, weights):
        h=np.tanh(x@weights['w1']+weights['b1'])
        p=1/(1+np.exp(-np.clip(h@weights['w2']+weights['b2'],-30,30)))
        return h,p

    def train(self):
        (x,y),(vx,vy),scenes=self.dataset()
        rng=np.random.default_rng(2026)
        weights=dict(w1=rng.normal(0,.05,(FEATURES,16)), b1=np.zeros(16),
                     w2=rng.normal(0,.05,16), b2=np.zeros(1))
        def loss(w):
            p=self.forward(vx,w)[1]
            return float(-np.mean(vy*np.log(p+1e-9)+(1-vy)*np.log(1-p+1e-9)))
        baseline=-np.mean(vy*np.log(np.mean(y)+1e-9)+(1-vy)*np.log(1-np.mean(y)+1e-9))
        best=None;best_loss=float(baseline)
        for _ in range(250):
            h,p=self.forward(x,weights); delta=(p-y)/len(y)
            hidden=delta[:,None]*weights['w2']*(1-h*h)
            gradients=dict(w1=x.T@hidden+.001*weights['w1'], b1=hidden.sum(axis=0),
                           w2=h.T@delta+.001*weights['w2'], b2=np.array([delta.sum()]))
            for key in weights:weights[key]-=.03*np.clip(gradients[key],-1,1)
            score=loss(weights)
            if score<best_loss:
                best_loss=score;best={k:v.copy() for k,v in weights.items()}
        if best is None or best_loss>baseline-.01:raise ValueError('Candidate does not improve held-out scene loss')
        # Re-evaluate the deployed model on the same validation scenes before promotion.
        current=self.root/'model.json'
        if current.exists():
            old=json.loads(current.read_text())
            with np.load(self.root/old['weights'],allow_pickle=False) as archive:
                if best_loss>=loss(dict(archive)):raise ValueError('Candidate regresses current model')
        version=uuid.uuid4().hex
        filename='ranker-'+version+'.npz'; np.savez(self.root/filename,**best)
        metadata=dict(version=version, weights=filename, feature_version=FEATURE_VERSION,
                      trained_at=time.time(), train_count=len(y), validation_count=len(vy),
                      eligible_at_training=len(y)+len(vy),
                      scenes=len(scenes), validation_loss=best_loss, baseline_loss=float(baseline),
                      architecture='197-16-1 tanh MLP', prior_version=json.loads(current.read_text()).get('version') if current.exists() else None)
        (self.root/('ranker-'+version+'.json')).write_text(json.dumps(metadata))
        temporary=self.root/'model.tmp';temporary.write_text(json.dumps(metadata));temporary.replace(current)
        return metadata

    def rank(self, candidates):
        path=self.root/'model.json'
        if not path.exists():return dict(ranked=False, reason='No validated learned model', candidates=candidates)
        metadata=json.loads(path.read_text())
        x=np.asarray([c['features'] for c in candidates],dtype=float)
        if x.shape!=(len(candidates),FEATURES) or not np.all(np.isfinite(x)):raise ValueError('Invalid candidate features')
        with np.load(self.root/metadata['weights'],allow_pickle=False) as weights:p=self.forward(x,weights)[1]
        ranked=[dict(c, learned_score=float(score)) for c,score in zip(candidates,p)]
        return dict(ranked=True, model_version=metadata['version'],
                    candidates=sorted(ranked,key=lambda c:c['learned_score'],reverse=True), execution_permission=False)
