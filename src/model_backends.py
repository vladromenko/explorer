"""Explorer adapters for optional learned policy backends."""
import json
from pathlib import Path
import time
import numpy as np


ACTION_NAMES=('base','shoulder','elbow','wrist_pitch','wrist_roll','gripper','vx','vy','wz')


class ExplorerPolicyAdapter:
    VERSION='explorer_command_9dof_v1'
    def observation(self, image, command_state, instruction, at=None):
        array=np.asarray(image)
        state=np.asarray(command_state,dtype=np.float32)
        if array.ndim!=3 or array.shape[2]!=3:raise ValueError('RGB image required')
        if state.shape!=(9,) or not np.isfinite(state).all():raise ValueError('Nine finite Explorer command-state values required')
        if not isinstance(instruction,str) or not instruction.strip():raise ValueError('Task instruction required')
        return dict(image=array,state=state,instruction=instruction.strip(),at=time.time() if at is None else float(at),
                    state_source='command_estimate_not_proprioception',adapter_version=self.VERSION)

    def action(self, proposed, start, maximum_joint_step_deg=2.,maximum_velocity_step=.05):
        action=np.asarray(proposed,dtype=float);origin=np.asarray(start,dtype=float)
        if action.shape!=(9,) or origin.shape!=(9,) or not np.isfinite(np.r_[action,origin]).all():raise ValueError('Invalid Explorer action')
        limit=np.asarray([maximum_joint_step_deg]*6+[maximum_velocity_step]*3)
        bounded=origin+np.clip(action-origin,-limit,limit)
        return dict(proposed=action.tolist(),executed_request=bounded.tolist(),clipped=bool(np.any(np.abs(action-bounded)>1e-9)),
                    adapter_version=self.VERSION)


class SmolVLAAdapter(ExplorerPolicyAdapter):
    def __init__(self, root):self.root=Path(root)

    def status(self):
        checkpoint=self.root/'models/smolvla-explorer';environment=self.root/'.venv-learning'
        installed=False
        if environment.exists():
            installed=any((path/'site-packages/lerobot').exists() for path in (environment/'lib').glob('python*'))
        blockers=[]
        if not installed:blockers.append('LeRobot runtime missing')
        if not checkpoint.is_dir():blockers.append('Compatible Explorer SmolVLA checkpoint missing')
        blockers.append('Jetson 8 GB memory and latency acceptance not completed') if not blockers else None
        return dict(adapter=True,adapter_version=self.VERSION,runtime_installed=installed,
                    checkpoint=str(checkpoint) if checkpoint.is_dir() else None,ready=not blockers,blocked_by=blockers,
                    state_source='command_estimate_not_proprioception')

    def load(self):
        status=self.status()
        if not status['ready']:raise ValueError('; '.join(status['blocked_by']))
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
        return SmolVLAPolicy.from_pretrained(status['checkpoint'],local_files_only=True)


class ActionChunk:
    def __init__(self, observations_at, actions, policy_version, scene_version, maximum_age_s=.35):
        self.observed=float(observations_at);self.actions=[list(row) for row in actions]
        self.policy_version=policy_version;self.scene_version=scene_version;self.maximum_age=maximum_age_s;self.index=0
        if not self.actions or any(len(row)!=9 or not np.isfinite(row).all() for row in np.asarray(self.actions,dtype=float)):
            raise ValueError('Invalid policy action chunk')

    def next(self, now, scene_version, policy_version):
        if now-self.observed>self.maximum_age:raise ValueError('Action chunk observation is stale')
        if scene_version!=self.scene_version:raise ValueError('Scene changed; discard old action chunk')
        if policy_version!=self.policy_version:raise ValueError('Policy changed inside action chunk')
        if self.index>=len(self.actions):return None
        result=self.actions[self.index];self.index+=1;return result

    def cancel(self):self.index=len(self.actions)
