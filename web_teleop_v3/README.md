# Web Teleop V3 (Camera + 3D Twin In One Page)

This version is focused on cleaner control:

- full-arm-follow style mapping (shoulder/elbow/wrist)
- smoother/deadzoned motor output
- camera feed and 3D arm shown together on one web page
- guide panel showing what movement controls each motor
- single-arm mode with runtime arm-side selection
- per-limb trim sliders for manual calibration
- raised robot model with fixed white support pole + human-arm ghost line
- uses local SO101 STL assets (downloaded from SO-ARM100 simulation files)

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

## Controls

- `Tracked Arm` selector: choose left or right arm only.
- `Calibrate Neutral Pose`: capture current pose as center.
- `motor_X trim` sliders: manually fine tune each limb until robot line-up matches your arm.
- `Save Calibration JSON`: persist selected arm + trim sliders + neutral offsets.
- Keep your whole arm visible for best stability.

## How To Operate (Stable)

1. Place camera so your selected arm (shoulder, elbow, wrist) is always visible.
2. Start with small, slow arm motions first; avoid sudden fast swings.
3. Choose `Tracked Arm` correctly (`left` or `right`).
4. Hold neutral pose for 2 seconds, then click `Calibrate Neutral Pose`.
5. Adjust each `motor_X trim` slider until robot limb lines match your arm direction.
6. Click `Save Calibration JSON` to persist your setup.
7. Re-open app; verify saved trims/neutral offsets are auto-loaded.

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

## Notes

- This is simulation-oriented visualization output.
- Physical-arm control files remain untouched in the main project path.
- Saved calibration file path is configured by `calibration_file` in `config.web_demo.json`.
- Saved calibration is auto-loaded on next startup.
