# Explorer room exploration, 2026-10-08

The robot ran with the installed factory-compatible `robotio` firmware. No STM32 write was performed. All observations below use the onboard arm camera, both lidars, live ROS 2 state, and saved task records on the Jetson. Joint positions remain command estimates.

## Physical observations

| Task ID | Observed result |
|---|---|
| `8b62363e8e3445279005fab88eb44e8f` | Three camera views completed. A collision-checked forward departure moved the chassis about 0.31 m from the bed. A later nearby rotation target was blocked, so the task was cancelled and STOP confirmed. |
| `0137cdd7171b4c2c80e52e9dad4f529b` | A second forward departure moved about 0.27 m. Navigation continued to approximately `(1.31, 0.07)` m, then a target rotation was blocked by furniture. The task was cancelled and STOP confirmed. |
| `4460298119a947a1a877bdba54ad5e3c` | The chassis travelled from approximately `(1.31, 0.07)` to `(1.54, -0.18)` m, then returned to `(1.32, 0.06)` m. The frontier was **not** counted because the measured settled heading missed the target; the map `room_route_20261008` was saved. |
| `1a95d1efd92d4e1a8cce144990152983` | A goal toward `(1.21, 0.88)` m stalled at a real obstacle. The chassis returned to within 0.07 m of its start; `room_verified_20261008` was saved. The failed attempt is recorded with 0.154 m actual motion. |
| `d71eb27295ba4115ab51935f248593cf` | Two Nav2 goals reported success, but fresh settled pose checks rejected both. The outbound attempts moved about 0.64 m and 0.40 m. Direct return stalled by furniture around `(1.06, 0.58)` m. STOP and zero measured velocity were confirmed; `room_recovery_partial_20261008` was saved as a partial map. |
| `e39e24a54237407d94cc6b744d53a4c3` | After matching Nav2's velocity model to the actual mapping limits, a short correction toward `(0.95, 0.39)` m succeeded: settled pose about `(0.957, 0.472)` m, yaw 0.59 rad, inside the configured position and heading tolerances. |
| `bb7e346b5f274b42a3ea567b07ac74e6`, `904607c2cd574022b9f449a48db2cdf7` | Return attempts with pre-alignment and explicit holonomic travel were blocked by nearby physical obstacles. Both ended with STOP and zero measured velocity. No reachable return is claimed. |
| `47f76031934b4a53a98517c841a2e0f7` | After the nearby basket/rack/chair clearance was changed, the onboard-camera-observed holonomic return from `(0.957, 0.472)` to target `(1.25, 0.05)` m succeeded. Fresh settled pose was `(1.245, 0.032)` m, yaw `0.496` rad, and the task independently reported the goal orientation verified. The core then reported MANUAL, STOP latched, commanded and measured velocity `[0,0,0]`, and battery 11.65 V. This confirms a short route back in the current map, not full-room exploration or repeat localization. |

Image evidence stays on the Jetson in `data/surveys/` and `data/release-checkpoints/room-resume-20261008/`; photographs are excluded from Git. Saved maps and task JSON remain under `data/maps/` and `data/navigation-tasks/`.

## Diagnosed software issues and fixes

- Cold MoveIt initialization starved the web process. `ArmPlannerClient` now runs the existing planner in a cancellable worker process; measured API response time stayed in milliseconds while the first planner launch took about 30 seconds.
- The mapping task selected AUTO after STOP but began camera panning before establishing a new measured stationary hold. Preparation now waits for the controller's real hold confirmation.
- Arm views were named left/right opposite to their measured optical axes. The view targets and axis check now match the physical camera images.
- A roughly 15 rad/s single-frame IMU spike reached the heading filter while the chassis was still. Impossible planar frames are rejected; the navigation supervisor also requires a fresh valid planar IMU. After a clean EKF/SLAM restart, stationary map yaw remained stable over the observed ten-second interval.
- The extra lidar margin blocked forward escape even when scan points stayed behind the hard arm footprint. The guard now allows motion away from an existing margin point, while any hard collision still blocks.
- Frontier planning at a near-obstacle start had no valid seed. A short candidate must lead through known-free cells to a footprint-clear point; Nav2's local collision check and the independent live lidar guard still decide whether it can execute.
- The planner could convert a distant grid route into a nearly coincident world goal and demand a turn beside furniture. Candidates now need at least 0.30 m Euclidean separation, and the target heading follows the accepted departure direction after pre-alignment.
- Nav2 DWB assumed 0.15 m/s and 0.4 rad/s, while the mapping controller actually allowed 0.10 m/s and 0.25 rad/s. Its model now matches the executed limits. Manual drive limits were not changed.
- A failed frontier now preserves its reason and actual displacement. A failed direct return can retry a bounded set of measured outbound waypoints; if the return still fails, the stopped task saves a clearly labelled partial map. These recovery branches passed isolated ROS-interface tests. A separately commanded physical return succeeded after the nearby obstacles were moved; the automatic multistage recovery branch itself has not been physically demonstrated.

Verification: **516 passed, 2 skipped** on the Jetson; the Mac web checks passed separately. The full room, repeat localization from a saved map, measured arm joints, and pick-and-deliver remain unaccepted.
