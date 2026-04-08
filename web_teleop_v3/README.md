# Web Teleop V3 (Dual-Arm Camera + 3D Twins)

This version is focused on full-arm teleoperation:

- dual-arm follow: right human arm -> right robot, left human arm -> left robot
- full-arm-follow mapping (shoulder/elbow/wrist/gripper)
- required arm overlay: shoulder-elbow-wrist lines + key POI dots
- smoother/deadzoned motor output
- camera feed and 3D arm shown together on one web page
- hand landmark lines visible on the camera feed
- guide panel showing what movement controls each motor
- single-arm mode with runtime arm-side selection
- per-limb trim sliders for manual calibration
- uses local SO101 STL assets (downloaded from SO-ARM100 simulation files)
- safety supervisor (freeze / estop / home + confidence gating)
- gesture macros are available in code but disabled by default in config for stability
- trajectory record/replay (JSON files in `web_teleop_v3/trajectories`)
- runtime dashboard: FPS, processing latency, pose ratio, quality score, jitter
- camera quality checks: low-light + blur warnings
- calibration wizard steps in UI
- reset endpoint/button for both robot twins
- dual-arm collision guard (soft block when wrists move into collision)
- depth fusion pipeline for reach perception:
  - MediaPipe wrist z
  - shoulder->wrist image-scale change
  - palm-size change
- One Euro depth-only filtering
- per-arm depth calibration (neutral / near / far)
- Group_Follower calibration file support for SO101 min/max ranges

## Run

From project root:

```powershell
python -m pip install fastapi "uvicorn[standard]" opencv-python mediapipe numpy
```

Then:

```powershell
python web_teleop_v3/app.py --camera 0 --config web_teleop_v3/config.web_demo.json --arm-side right --host 127.0.0.1 --port 8010
```

Open:

```text
http://127.0.0.1:8010
```

## Self Test

Run a local API smoke test (starts/stops server automatically):

```powershell
python web_teleop_v3/self_test.py --camera 0 --port 8011 --config web_teleop_v3/config.web_demo.json
```

## Controls

- `Tracked Arm` selector: choose left or right arm only.
- `Calibrate Neutral Pose`: capture current pose as center.
- `motor_X trim` sliders: manually fine tune each limb until robot line-up matches your arm.
- `Save Calibration JSON`: persist selected arm + trim sliders + neutral offsets.
- Keep your whole arm visible for best stability.
- `Freeze` button or fist gesture: hold current safe pose.
- `E-Stop` button or pinch gesture: emergency stop latch.
- `Home` button or open-palm gesture: move to home pose safely.
- `Reset Robots`: reset both arms to safe home.
- `Depth Neutral`: capture neutral reach baseline (selected arm).
- `Depth Near`: move selected hand close to camera and capture.
- `Depth Far`: move selected hand farther from camera and capture.
- `Depth Reset`: clear depth calibration for selected arm.
- `Rec Start` / `Rec Stop`: record trajectories.
- `Play` / `Stop Play`: replay selected trajectory.

## How To Operate (Stable)

1. Place camera so both shoulders/elbows/wrists/hands are visible.
2. Start with small, slow arm motions first; avoid sudden fast swings.
3. Right arm drives right robot; left arm drives left robot.
4. Hold neutral reach and click `Depth Neutral`.
5. Move hand close to camera and click `Depth Near`.
6. Move hand far from camera and click `Depth Far`.
7. Hold neutral pose for 2 seconds, then click `Calibrate Neutral Pose`.
8. Adjust each `motor_X trim` slider until robot limb lines match your arm direction.
9. Click `Save Calibration JSON` to persist your setup (includes depth calibration).
10. Re-open app; verify saved trims/neutral offsets are auto-loaded.

If motion is still shaky, tune `config.web_demo.json`:

- Lower `smoothing_alpha` (more damping)
- Lower `feature_smoothing_alpha` (more damping)
- Increase `deadzone_deg` (ignore tiny motor changes)
- Increase `max_step_deg` only if movement feels too slow
- Increase `pose_visibility_threshold` to reject low-confidence pose frames

## Troubleshooting

- If the robot is not moving:
1. Check top badges: `pose:true` should appear while your shoulder-elbow-wrist are visible.
2. Make sure selected arm (`left/right`) matches your actual arm.
3. Re-run `Calibrate Neutral Pose`.
4. Lower `pose_visibility_threshold` in config (for weak lighting/camera).

## New API Endpoints (Phase Start)

- `GET /api/health`
- `POST /api/reset`
- `GET/POST /api/depth_calibration`
- `GET/POST /api/safety`
- `GET /api/trajectory`
- `POST /api/trajectory/start`
- `POST /api/trajectory/stop`
- `POST /api/trajectory/play`
- `POST /api/trajectory/stop_playback`
- `GET/POST /api/wizard`
- `POST /api/voice` (starter hook)
- `POST /api/ros2` (starter hook)
- `POST /api/validation/run` (sim-vs-real report starter)

## Notes

- This is simulation-oriented visualization output.
- Physical-arm control files remain untouched in the main project path.
- Saved calibration file path is configured by `calibration_file` in `config.web_demo.json`.
- Saved calibration is auto-loaded on next startup.
