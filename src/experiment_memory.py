"""Incremental, conservative object/episode memory with immutable evidence history."""
import json
import math
from pathlib import Path
import sqlite3
import time
import uuid


class ExperienceMemory:
    def __init__(self, root):
        self.path = Path(root)/'data/experiments.sqlite3'
        self.path.parent.mkdir(exist_ok=True)
        with self.db() as db:
            db.executescript('''
            CREATE TABLE IF NOT EXISTS objects(id TEXT PRIMARY KEY, label TEXT, active INTEGER,
              created REAL, confirmed REAL, position TEXT, provenance TEXT);
            CREATE TABLE IF NOT EXISTS events(id INTEGER PRIMARY KEY, at REAL, object_id TEXT,
              operation TEXT, evidence TEXT, source TEXT);
            CREATE TABLE IF NOT EXISTS views(id TEXT PRIMARY KEY, at REAL, epoch TEXT, fingerprint TEXT,
              content TEXT, image TEXT);
            CREATE TABLE IF NOT EXISTS labels(id INTEGER PRIMARY KEY, at REAL, task TEXT,
              condition TEXT, outcome TEXT, reason TEXT, episode TEXT);
            ''')

    def db(self):
        db = sqlite3.connect(self.path, timeout=2)
        db.row_factory = sqlite3.Row
        return db

    def update(self, operation, label='', object_id=None, evidence=None, source='operator'):
        if operation not in ('add', 'update', 'remove', 'no-change'):
            raise ValueError('Unknown memory operation')
        if source not in ('operator', 'detector'):
            raise ValueError('Unknown evidence source')
        evidence = evidence or {}
        if operation == 'remove' and (source != 'operator' or not evidence.get('reason')):
            raise ValueError('Удаление требует явного исправления владельца с причиной')
        if operation == 'add' and not 1 <= len(label) <= 80:
            raise ValueError('Нужно название предмета')
        now = time.time()
        with self.db() as db:
            if operation == 'add':
                object_id = uuid.uuid4().hex
                db.execute('INSERT INTO objects VALUES(?,?,?,?,?,?,?)', (object_id, label, 1, now,
                    now, json.dumps(evidence.get('position')), source))
            else:
                old = db.execute('SELECT * FROM objects WHERE id=?', (object_id,)).fetchone()
                if old is None:
                    raise ValueError('Предмет не найден')
                if operation == 'update':
                    db.execute('UPDATE objects SET label=?,confirmed=?,position=?,provenance=? WHERE id=?',
                        (label or old['label'], now, json.dumps(evidence['position']) if 'position' in evidence else old['position'], source, object_id))
                if operation == 'remove':
                    db.execute('UPDATE objects SET active=0,confirmed=?,provenance=? WHERE id=?',
                               (now, source, object_id))
            db.execute('INSERT INTO events(at,object_id,operation,evidence,source) VALUES(?,?,?,?,?)',
                       (now, object_id, operation, json.dumps(evidence, ensure_ascii=False), source))
        return dict(object_id=object_id, operation=operation, history_preserved=True)

    def objects(self, query=''):
        with self.db() as db:
            rows = db.execute('SELECT * FROM objects ORDER BY confirmed DESC LIMIT 200').fetchall()
            return [dict(row, position=json.loads(row['position'])) for row in rows
                    if query.casefold() in row['label'].casefold()]

    def history(self, object_id):
        with self.db() as db:
            return [dict(r, evidence=json.loads(r['evidence'])) for r in db.execute(
                'SELECT * FROM events WHERE object_id=? ORDER BY id', (object_id,))]

    def remember_view(self, perception, epoch, image=None):
        """No label-only identity merging; a view can contain several identical objects."""
        stamp = perception.get('image_stamp', perception.get('at', 0))
        if not isinstance(stamp, (int, float)) or not math.isfinite(stamp):
            raise ValueError('Invalid frame timestamp')
        objects = perception.get('objects', [])
        fingerprint = json.dumps([(o.get('label'), [round(float(x)/25) for x in o.get('bbox', [])])
                                  for o in objects], sort_keys=True)
        with self.db() as db:
            previous = db.execute('SELECT * FROM views ORDER BY at DESC LIMIT 1').fetchone()
            if previous and (stamp <= previous['at'] or
                (fingerprint == previous['fingerprint'] and epoch == previous['epoch'] and stamp-previous['at'] < 30)):
                return dict(id=previous['id'], added=False, reason='Повтор того же наблюдения')
            if db.execute('SELECT count(*) FROM views').fetchone()[0] >= 2000:
                raise ValueError('Лимит 2000 ракурсов: экспортируйте память; исходные данные не удалены')
            identifier = uuid.uuid4().hex
            content = dict(perception=perception, object_identity_verified=False,
                           map_object_positions_verified=False, unobserved_directions='unknown')
            db.execute('INSERT INTO views VALUES(?,?,?,?,?,?)', (identifier, stamp, epoch, fingerprint,
                                                               json.dumps(content), image))
        return dict(id=identifier, added=True, objects_seen=len(objects), identity_assignment='requires_confirmation')

    def retrieve(self, query, limit=6):
        with self.db() as db:
            rows = [dict(r) for r in db.execute('SELECT * FROM views ORDER BY at DESC LIMIT 200')]
        words = set(query.casefold().split())
        for r in rows:
            r['content'] = json.loads(r['content'])
            labels = ' '.join(o.get('label', '') for o in r['content']['perception'].get('objects', []))
            r['score'] = sum(w in labels.casefold() for w in words)
        relevant = sorted(rows, key=lambda r: (r['score'], r['at']), reverse=True)[:limit]
        return dict(retrieval=relevant, recent_baseline=rows[:limit], identical_input_view_ids=[r['id'] for r in rows],
                    objects=self.objects(query), method='label_relevance_and_recency_baseline')

    def label(self, task, condition, outcome, reason, episode=None):
        if outcome not in ('success', 'failure', 'intervention', 'unknown'):
            raise ValueError('Некорректный исход')
        if not task or not reason:
            raise ValueError('Нужны задача и причина оценки')
        with self.db() as db:
            db.execute('INSERT INTO labels(at,task,condition,outcome,reason,episode) VALUES(?,?,?,?,?,?)',
                       (time.time(), task[:80], condition[:120], outcome, reason[:500], episode))
        return self.curriculum()

    def curriculum(self):
        with self.db() as db:
            rows = [dict(r) for r in db.execute('''SELECT task,condition,count(*) AS samples,
                sum(outcome='success') AS successes,sum(outcome='failure') AS failures,
                sum(outcome='intervention') AS interventions FROM labels GROUP BY task,condition''')]
        for row in rows:
            row['priority'] = (row['failures']+2*row['interventions']+1)/(row['samples']+2)
            row['request'] = 'Покажите 3 успешных исправления: '+row['task']+' · '+row['condition']
            row['auto_training'] = False
        return dict(queue=sorted(rows, key=lambda r: r['priority'], reverse=True),
                    algorithm='failure_and_intervention_curriculum_baseline', weights_modified=False)
