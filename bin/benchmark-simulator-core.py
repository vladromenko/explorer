#!/usr/bin/env python3
"""Finite component measurements. Does not train or qualify a policy."""
import argparse
import json
from pathlib import Path
import sys
import time
import xml.etree.ElementTree as ET

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    import psutil
    def reserve():
        m=psutil.virtual_memory()
        if m.available/m.total<.25:raise MemoryError('Component benchmark requires 25% available memory; live learners untouched')
    if args.output.exists():raise ValueError('Retain prior benchmark')
    reserve()
    import numpy as np
    import mujoco
    from simulator.environment import ExplorerEnv
    from simulator.explorer_model import wheel_targets
    from simulator.learning import write_json
    value=dict(at=time.time(),kind='component_benchmark_not_policy_acceptance',hardware_actuation=False,
        simulation_qualified=False,robot_verified=False,parallel_learner_may_affect_timings=True)
    env=None
    try:
        started=time.perf_counter();env=ExplorerEnv(render=False)
        value['initialization_s']=time.perf_counter()-started
        for _ in range(250):mujoco.mj_step(env.model,env.data)
        started=time.perf_counter()
        for _ in range(2000):mujoco.mj_step(env.model,env.data)
        elapsed=time.perf_counter()-started
        value['physics']=dict(steps=2000,wall_s=elapsed,simulated_s=2000*env.model.opt.timestep,
            substeps_per_second=2000/elapsed,finite=bool(np.isfinite(env.data.qpos).all()),
            warning_counts=env.data.warning.number.tolist(),flex_count=env.model.nflex,backend='MuJoCo CPU')
        rendered=[]
        for _ in range(12):
            reserve();mujoco.mj_step(env.model,env.data);started=time.perf_counter()
            image=env.render_rgb('wrist');rendered.append(time.perf_counter()-started)
        value['render']=dict(shape=list(image.shape),cold_s=rendered[0],
            warm_p50_s=float(np.percentile(rendered[1:],50)),warm_p95_s=float(np.percentile(rendered[1:],95)),
            backend='MuJoCo offscreen OpenGL',includes_image_transfer=True)
        value['compiled_inertias']=dict(dynamic_bodies=int((env.model.body_mass>0).sum()),
            positive=bool(np.all(env.model.body_inertia[env.model.body_mass>0]>0)))
        env.close();env=None;reserve()
        env=ExplorerEnv(render=False,scene='empty');initial_q=env.data.qpos.copy();initial_ctrl=env.data.ctrl.copy()
        rows=[]
        for axis in range(3):
            for sign in [-1,1]:
                mujoco.mj_resetData(env.model,env.data);env.data.qpos[:]=initial_q;env.data.ctrl[:]=initial_ctrl
                for _ in range(250):mujoco.mj_step(env.model,env.data)
                before=np.asarray(env.base_pose());velocity=np.zeros(3);velocity[axis]=sign*(.2 if axis==2 else .12)
                env.data.ctrl[env.wheel_ids]=wheel_targets(*velocity)
                for _ in range(600):mujoco.mj_step(env.model,env.data)
                after=np.asarray(env.base_pose());delta=after-before
                rows.append(dict(axis=axis,sign=sign,body_velocity=velocity.tolist(),duration_s=1.2,
                    measured_pose_delta=delta.tolist(),correct_sign=bool(delta[axis]*sign>0),
                    finite=bool(np.isfinite(env.data.qpos).all()),warnings=env.data.warning.number.tolist()))
        value['mecanum']=rows
        bad=[]
        for body in ET.parse(ROOT/'config/explorer.urdf').getroot().findall('link'):
            inertia=body.find('inertial/inertia')
            if inertia is not None:
                g=lambda k:float(inertia.get(k,'0'))
                matrix=np.array([[g('ixx'),g('ixy'),g('ixz')],[g('ixy'),g('iyy'),g('iyz')],[g('ixz'),g('iyz'),g('izz')]])
                eig=np.linalg.eigvalsh(matrix)
                if min(eig)<=0 or eig[-1]>sum(eig[:2])+1e-12:bad.append(dict(link=body.get('name'),eigenvalues=eig.tolist()))
        value['source_inertia_violations']=bad
        write_json(args.output,value);print(json.dumps(value))
    except Exception as error:
        value['error']=type(error).__name__+': '+str(error)
        write_json(args.output,value)
        raise
    finally:
        if env is not None:env.close()


if __name__=='__main__':main()
