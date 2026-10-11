"""Interactive native 3D model inspection; simulation only."""
import argparse
import time
import mujoco
import mujoco.viewer
from .environment import ExplorerEnv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-seconds", type=float, default=0)
    args = parser.parse_args()
    env = ExplorerEnv(render=False)
    try:
        with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
            viewer.cam.distance = 1.25
            viewer.cam.lookat[:] = [.15, 0., .2]
            viewer.cam.azimuth = 135
            viewer.cam.elevation = -25
            viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_FLEXFACE] = True
            viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_FLEXEDGE] = False
            opened_at = time.monotonic()
            # Built-in MuJoCo panels expose actuator and contact diagnostics.
            while viewer.is_running() and (not args.smoke_seconds or time.monotonic()-opened_at < args.smoke_seconds):
                start = time.monotonic()
                with viewer.lock():
                    mujoco.mj_step(env.model, env.data, nstep=10)
                viewer.sync()
                time.sleep(max(0., .02-(time.monotonic()-start)))
    finally:
        env.close()


if __name__ == "__main__":
    main()
