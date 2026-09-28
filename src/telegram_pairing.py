"""Explicit local-owner pairing; never make the first messenger an administrator."""
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import secrets
import time


class Pairing:
    def __init__(self, root):
        self.root=Path(root)
        self.path=self.root/'data/telegram-pairing.json'
        self.path.parent.mkdir(exist_ok=True)

    @contextmanager
    def locked(self):
        with (self.root/'data/telegram-pairing.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            yield

    def load(self):
        try:return json.loads(self.path.read_text())
        except (OSError,ValueError):return {}

    def write(self, value):
        temporary=self.path.with_suffix('.tmp')
        fd=os.open(temporary,os.O_WRONLY|os.O_CREAT|os.O_TRUNC,0o600)
        with os.fdopen(fd,'w') as f:json.dump(value,f)
        temporary.replace(self.path)

    def begin(self):
        with self.locked():
            config=self.root/'config/telegram.json'
            if config.exists() and json.loads(config.read_text()).get('allowed_user_ids'):
                raise ValueError('Владелец уже привязан; смена владельца требует отдельного сброса через SSH')
            code=secrets.token_urlsafe(18)
            self.write(dict(hash=hashlib.sha256(code.encode()).hexdigest(),expires=time.time()+300,pending=None))
        return dict(command='/pair '+code,expires_in_s=300,requires_local_confirmation=True)

    def claim(self, code, user_id, chat_id, username):
        if type(user_id) is not int or type(chat_id) is not int or user_id<=0 or chat_id<=0:
            return False
        with self.locked():
            state=self.load()
            match=secrets.compare_digest(state.get('hash',''),hashlib.sha256(code.encode()).hexdigest())
            if time.time()>=state.get('expires',0) or not match or state.get('pending'):
                return False
            state.update(hash='',pending=dict(user_id=user_id,chat_id=chat_id,username=str(username)[:80]))
            self.write(state)
            return True

    def status(self):
        state=self.load()
        active=time.time()<state.get('expires',0)
        return dict(active=active,pending=state.get('pending') if active else None,
                    expires=state.get('expires'),code=None)

    def confirm(self, user_id):
        with self.locked():
            state=self.load();pending=state.get('pending')
            if time.time()>=state.get('expires',0) or not pending or pending['user_id']!=user_id:
                raise ValueError('Нет свежей заявки этого пользователя')
            path=self.root/'config/telegram.json'
            config=json.loads(path.read_text())
            if config.get('allowed_user_ids'):
                raise ValueError('Владелец уже установлен')
            config.update(allowed_user_ids=[pending['user_id']],allowed_chat_ids=[pending['chat_id']],enabled=True)
            fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_TRUNC,0o600)
            with os.fdopen(fd,'w') as f:json.dump(config,f)
            self.write(dict(expires=0,pending=None,confirmed_at=time.time()))
            return dict(paired=True,user_id=user_id)
