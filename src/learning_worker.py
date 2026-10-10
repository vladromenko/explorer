"""Dispatch persisted local training when budget permits; no actuator access."""
import time
from pathlib import Path
from grasp_learning import GraspMemory
from teaching import Demonstrations
from lerobot_bridge import LearningJobs,write_json
ROOT=Path('/home/vlad/Explorer')
def run():
    memory=GraspMemory(ROOT/'data/grasp-memory')
    demos=Demonstrations(ROOT/'data/demonstrations')
    jobs=LearningJobs(ROOT)
    while True:
        try:jobs.dispatch()
        except (OSError,ValueError,KeyError):pass
        record=memory.status()
        record.update(at=time.time(),framework='LeRobot',policy_type='ACT',
            demonstrations=demos.status(),jobs=jobs.status(),model=None,execution_permission=False,
            reason="Operator demonstrations feed official LeRobot ACT; no policy has actuator access",
            training_deferred="Queued demonstrations wait for idle and valid power; execution requires a separate observed trial")
        write_json(ROOT/'data/learning-status.json',record)
        time.sleep(10)
if __name__=='__main__':run()
