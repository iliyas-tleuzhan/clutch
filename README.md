# Clutch: Laptop Teleop Vertical Slice (SO101)
<p align="center">
  <img src="media\videos\demo_clutch.gif" width="900">
</p>

Primary path is now **Web Teleop V3** with end-effector control.

Runtime flow:

`camera hand tracking -> EE target (XYZ + grip + confidence) -> IK -> motors_deg -> twin/bridge`

## Primary Entrypoint

```powershell
python web_teleop_v3/app.py --camera 0 --config web_teleop_v3/config.web_demo.json --arm-side right --host 127.0.0.1 --port 8010
```

Open:

```text
http://127.0.0.1:8010
```

## What Web V3 Includes

- Restored polished Clutch shell (camera/twin/control layout)
- End-effector representation and retargeting layer
- Damped least-squares IK with FK consistency checks
- Reachability/clamp diagnostics and safety gating
- Guided calibration workflow
- Simulation test modes (`camera`, `scripted_target`, `joint_axis_diagnostic`)
- Structured session logging (`logs/session_<run_id>.jsonl`)

Detailed docs:

- `web_teleop_v3/README.md`

## Install

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -U pip
pip install -r requirements.txt
```

## Other Scripts (Legacy / Utilities)

- `hand_to_so101_positions.py`: legacy direct hand->joint mapping path.
- `so101_digital_twin_vpython.py`: standalone VPython twin utility.
- `so101_robot_bridge.py`: standalone robot bridge utility.
- `hand_tracking_yolo.py`: optional detector utility.

For current development and validation, use **Web V3** first.
