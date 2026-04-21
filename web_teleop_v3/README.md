# Web Teleop V3 (Phase 1: XYZ + Gripper)

This path now runs Cartesian end-effector teleoperation:

`camera/sim -> hand/target observation -> EE target xyz -> IK -> motors_deg -> safety -> twin/bridge`

Orientation control is intentionally **not** included in this phase.

## Architecture Summary

Direct joint mapping was previously in `arm_follow_mapper.py` (`extract_arm_features -> ArmFollowMapper.map`).

Current active modules:

- `hand_pose_tracker.py`: palm-centered camera-space observations (`raw_control_point_camera_xyz`, filtered control point, confidence, pinch)
- `ee_target_mapper.py`: camera-to-robot mapping (frame transform, scale, deadzone, clamps, confidence gating)
- `ik_solver.py`: DLS IK + shared FK model + joint-limit/singularity diagnostics
- `simulation_modes.py`: simulation-only scripted target and joint-axis diagnostic modes
- `target_visualizer.py`: per-frame debug text lines
- `app.py`: pipeline orchestration, websocket/api, safety, trajectory, ROS bridge

## Debug / Observability

`debug_mode` can be toggled from UI (`Debug On/Off`) or API (`POST /api/debug`).

In the restored Clutch shell, EE/IK controls and workflow/debug tools are available in the optional
`EE / IK Tools` drawer section (collapsed by default) so the default product view stays clean.

When enabled, every frame exposes:

- raw hand target in camera coordinates
- mapped EE target in robot coordinates
- IK output joint angles
- FK EE position from solved joints
- error norm between target EE and FK EE

Main websocket fields:

- `ee_target_xyz`
- `ee_fk_xyz`
- `ee_error_norm`
- `ik_ok`
- `joint_limit_hit`
- `singularity_warning`
- `ik_output_joints_deg`

## Session Logging for Real-World Tuning

Structured per-session logging is enabled by default and writes one JSONL file per run under:

- `logs/session_<run_id>.jsonl` (repo root by default)

Config:

- `session_logging.enabled`: set to `false` to disable logging
- `session_logging.logs_dir`: output directory (relative paths are resolved from repo root)
- `session_logging.file_name_template`: default `session_{run_id}.jsonl`

Per-frame records include:

- `timestamp`, `input_mode`, `hand_detected`, `tracking_confidence`
- `raw_hand_target_xyz`, `mapped_target_xyz`, `clamped_target_xyz`
- `ee_fk_xyz`, `ee_error_norm`, `ik_ok`, `ik_fail_reason`
- `joint_limit_hit`, `singularity_warning`
- `target_reachable`, `target_clamped`, `clamp_delta_xyz`, `workspace_violation_axes`
- `safety_suppressed`, `motion_armed`, `hold_to_enable_active`
- `motors_deg`

Event records include:

- calibration workflow step transitions
- neutral capture and robot-neutral capture
- config correction saves/applies
- motion arming/disarming and motion setting changes
- simulation mode startup / change requests

Practical post-session tuning flow:

1. Filter event rows (`record_type=event`) to find calibration and arming/disarming times.
2. Around those windows, inspect frame rows (`record_type=frame`) and trend:
   - `raw_hand_target_xyz -> mapped_target_xyz -> clamped_target_xyz`
   - `clamped_target_xyz` vs `ee_fk_xyz` (`ee_error_norm`)
3. If `workspace_violation_axes`/`target_clamped` spikes, widen or recenter workspace mapping.
4. If `ee_error_norm` stays high with little clamping, tune IK model (`link_vectors_m`, joint axis signs/limits).
5. If arming is unstable, inspect `motion_armed`, `hold_to_enable_active`, and `safety_suppressed` transitions.

## Calibration

Guided calibration is available through UI buttons and `POST /api/calibration/workflow`.

Workflow steps:

1. Move robot to neutral/home pose.
2. Capture current FK EE as robot neutral reference.
3. Capture hand neutral pose.
4. Run `+X` hand motion test.
5. Run `+Y` hand motion test.
6. Run `+Z` hand motion test.

The workflow updates axis sign/scale/offset correction candidates and stores them into calibration fields.

`POST /api/calibration/save`:

- saves runtime calibration JSON (`calibration_file`)
- also writes calibration values into active config (`runtime_settings["config_path"]`)

Important file split:

- `calibration_file` is for teleop neutral/workspace calibration (`center_offsets`, trims).
- `lerobot_calibration_file` is the hardware servo calibration source (e.g. `Group_Follower.json`) and should be treated as read-only from this app.

## Simulation Modes

Configured in `config.web_demo.json` under `simulation.mode`:

- `camera`: normal webcam teleop
- `scripted_target`: bypass webcam and drive EE target from scripted 3D trajectory
- `joint_axis_diagnostic`: same scripted target plus one-joint-at-a-time perturbation for sign/axis validation

Default is `camera` for normal live teleoperation.

## Joint Sign / Axis Validation Helpers

- joint axis names and signs are configurable in `ik_solver.joint_axes` and `ik_solver.joint_axis_signs`
- no code edit required for axis sign flips
- `GET /api/diagnostics/joint_axes` returns per-joint expected EE displacement hints for positive/negative perturbations

## How to Validate Correctness

1. **Simulation test**
   - set `simulation.mode` to `scripted_target`
   - run app and verify twin follows smooth periodic EE target
   - check `ee_error_norm`, `joint_limit_hit`, and `singularity_warning` stay acceptable
2. **Neutral calibration**
   - switch to `camera` mode
   - hold palm at desired neutral pose
   - trigger `Calibrate Neutral Pose`
   - save calibration (`Save Calibration JSON`) to persist both runtime and config calibration
3. **Frame alignment tuning**
   - tune `ee_target_mapper.camera_to_robot_rotation`, scale, and workspace bounds
   - use debug panel to compare raw camera target vs mapped EE target
4. **Joint direction validation**
   - set `simulation.mode` to `joint_axis_diagnostic`
   - review `/api/diagnostics/joint_axes` output and debug panel deltas
   - flip signs in `ik_solver.joint_axis_signs` if direction is inverted
5. **Safe robot-first procedure**
   - keep `freeze` ready and begin with tiny motion amplitudes
   - verify low `ee_error_norm` in simulation before connecting hardware bridge
   - enable bridge only after camera mapping and sign checks are stable

## How to Validate on the Real Robot Safely

1. **Simulation-first**
   - run with `simulation.mode=scripted_target`
   - confirm `raw_target_xyz`, `clamped_target_xyz`, `ee_fk_xyz`, and `target-FK error` are stable
2. **Run guided calibration workflow**
   - execute steps 1..6 in UI (`Guided Calibration Workflow`)
   - save corrections (`Save Corrections` then `Save Calibration JSON`)
3. **Joint-axis sign validation**
   - run `simulation.mode=joint_axis_diagnostic`
   - use `/api/diagnostics/joint_axes` and debug panel to verify signs
   - update `ik_solver.joint_axis_signs` only when needed
4. **Slow-mode first live motion**
   - keep `hardware_validation.enable_motion_default=false`
   - arm motion explicitly (`Motion Armed`) and hold `Hold-to-Enable`
   - start with `one_axis_mode=true` if needed
5. **Mismatch diagnosis**
   - **Frame mismatch:** target moves in wrong global direction while joint signs look consistent -> tune `camera_to_robot_rotation`
   - **Sign mismatch:** one or more joints move opposite expected response -> tune `joint_axis_signs` and rerun axis tests
   - **Geometry mismatch:** persistent FK/target error even with good signs and frame alignment -> tune `link_vectors_m` / robot kinematic model

## Run

```powershell
python -m pip install fastapi "uvicorn[standard]" opencv-python mediapipe numpy
python web_teleop_v3/app.py --camera 0 --config web_teleop_v3/config.web_demo.json --arm-side right --host 127.0.0.1 --port 8010
```

Open `http://127.0.0.1:8010`.

## Hardware-Specific TODOs

1. Confirm `ik_solver.link_vectors_m` against measured SO101 geometry.
2. Confirm `ik_solver.joint_axes` + `joint_axis_signs` against real motor conventions.
3. Tune `ee_target_mapper.camera_to_robot_rotation` for final camera mount.
4. Tune `ee_target_mapper.gripper_motor.open_deg/closed_deg` on hardware.
