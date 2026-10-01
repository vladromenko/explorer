"""Typed, serializable contracts shared by autonomous Explorer components."""
from dataclasses import asdict, dataclass, field
from enum import Enum
import json
import math
import time
import uuid


class Truth(str, Enum):
    TRUE = 'true'
    FALSE = 'false'
    UNKNOWN = 'unknown'


def finite_vector(values, size=None):
    if not isinstance(values, (list, tuple)) or (size is not None and len(values) != size):
        return False
    return all(type(value) in (int, float) and math.isfinite(value) for value in values)


@dataclass(frozen=True)
class Observation:
    id: str
    at: float
    scene_version: str
    source: str
    frame: str
    values: dict
    uncertainty: dict = field(default_factory=dict)
    evidence: tuple = ()

    def validate(self):
        if not self.id or not math.isfinite(self.at) or not self.scene_version or not self.source or not self.frame:
            raise ValueError('Incomplete observation contract')
        json.dumps(asdict(self), allow_nan=False)
        return self


@dataclass(frozen=True)
class WorldSnapshot:
    id: str
    at: float
    scene_version: str
    map_epoch: str
    predicates: dict
    observations: tuple

    def validate(self):
        if not self.id or not self.scene_version or not self.map_epoch or not math.isfinite(self.at):
            raise ValueError('Incomplete world snapshot')
        allowed={value.value for value in Truth}
        if any(value not in allowed for value in self.predicates.values()):
            raise ValueError('Predicates must be true, false or unknown')
        json.dumps(asdict(self), allow_nan=False)
        return self


@dataclass(frozen=True)
class GoalSpec:
    id: str
    instruction: str
    target_predicates: dict
    object_query: str = ''
    destination: dict = field(default_factory=dict)
    constraints: dict = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)

    def validate(self):
        if not self.id or not self.instruction or not self.target_predicates:
            raise ValueError('Goal needs an instruction and observable target predicates')
        if any(value is not True for value in self.target_predicates.values()):
            raise ValueError('Goal predicates describe required true facts')
        json.dumps(asdict(self), allow_nan=False)
        return self


@dataclass(frozen=True)
class SkillSpec:
    id: str
    version: str
    preconditions: dict
    effects: dict
    cost: float
    executor: str
    verifier: str
    permissions: tuple = ()
    resolves: tuple = ()
    recovery_for: tuple = ()
    source: str = 'builtin'

    def validate(self):
        if not self.id or not self.version or not self.executor or not self.verifier or self.cost <= 0:
            raise ValueError('Invalid skill specification')
        allowed={value.value for value in Truth}
        if any(value not in allowed for value in self.preconditions.values()):
            raise ValueError('Invalid skill precondition')
        json.dumps(asdict(self), allow_nan=False)
        return self


@dataclass(frozen=True)
class ExecutionContext:
    job_id: str
    episode_id: str
    goal_id: str
    scene_version: str
    skill_id: str
    skill_version: str
    attempt: int
    deadline: float
    policy_version: str
    reward_version: str

    def validate(self):
        if not all((self.job_id, self.episode_id, self.goal_id, self.scene_version,
                    self.skill_id, self.skill_version, self.policy_version, self.reward_version)):
            raise ValueError('Incomplete execution context')
        if self.attempt < 1 or not math.isfinite(self.deadline):
            raise ValueError('Invalid execution budget')
        return self


@dataclass(frozen=True)
class Outcome:
    state: str
    reason: str
    verifier: str
    verifier_version: str
    observation_ids: tuple
    metrics: dict = field(default_factory=dict)
    failure_kind: str = ''

    def validate(self):
        if self.state not in ('success', 'failure', 'unknown', 'cancelled', 'infrastructure_error'):
            raise ValueError('Invalid outcome state')
        if not self.reason or not self.verifier or not self.verifier_version:
            raise ValueError('Outcome requires versioned evidence provenance')
        json.dumps(asdict(self), allow_nan=False)
        return self


@dataclass(frozen=True)
class PolicyVersion:
    id: str
    backend: str
    feature_version: str
    state: str
    checkpoint: str
    trained_episode_ids: tuple
    validation_episode_ids: tuple
    metrics: dict
    created_at: float = field(default_factory=time.time)

    def validate(self):
        allowed=('collecting','training','candidate','evaluating','accepted','rejected','needs_data','rolled_back')
        if self.state not in allowed or not self.id or not self.backend or not self.feature_version:
            raise ValueError('Invalid policy version')
        if set(self.trained_episode_ids) & set(self.validation_episode_ids):
            raise ValueError('Training and validation episodes overlap')
        json.dumps(asdict(self), allow_nan=False)
        return self


@dataclass(frozen=True)
class HelpRequest:
    id: str
    job_id: str
    question: str
    kind: str
    choices: tuple
    blocks_skill: str
    created_at: float = field(default_factory=time.time)
    expires_at: float = 0.

    def validate(self):
        if not self.id or not self.job_id or not self.question or not self.kind or not self.blocks_skill:
            raise ValueError('Incomplete help request')
        if self.expires_at and self.expires_at <= self.created_at:
            raise ValueError('Help request expiry is invalid')
        return self


def new_id():
    return uuid.uuid4().hex


def record(value):
    if hasattr(value, 'validate'):
        value.validate()
    return asdict(value)
