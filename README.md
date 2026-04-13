# Clutch: Laptop Teleop Vertical Slice (SO101)

Laptop-first teleoperation demo pipeline:

1. Webcam hand/arm tracking
2. Hand/arm features -> 6 motor targets
3. Real-time 3D digital twin
4. Optional physical robot bridge path from laptop

## Run Web V3 First (Recommended)

`Web V3` is the primary demo entrypoint.

```powershell
python web_teleop_v3/app.py --camera 0 --config web_teleop_v3/config.web_demo.json --arm-side right --host 127.0.0.1 --port 8010
```

Open:

```text
http://127.0.0.1:8010
```

## Demo Videos

Videos in this repo:

- [Web V3 Demo](media/videos/web_v3_demo.mov)
- [4Joystick + End Effector Control Demo](media/videos/joystick_end_effector_demo.mov)

## Install

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -U pip
pip install -r requirements.txt
```

## Local scripts

- `hand_to_so101_positions.py`: Webcam hand tracking and motor target generation (`motor_1..motor_6` + `motors_deg`).
- `so101_digital_twin_vpython.py`: Stage-1 3D digital twin viewer driven by 6 motor targets.
- `so101_robot_bridge.py`: Laptop-side UDP motor receiver with safety clamps and optional hardware driver adapter hook.
- `hand_tracking_yolo.py`: Optional YOLO hand detector/tracker utility.
- `so101_config.example.json`: Calibration-ready config shape.

## Config

Edit this file for limits/home/inversion/gain/offset:

```text
so101_config.example.json
```

## Hand output format

`hand_to_so101_positions.py` prints JSON lines:

```json
{
  "timestamp": 1710000000.0,
  "motors_deg": [10.2, -14.7, 32.1, 4.9, -11.3, 22.0],
  "motor_1": 10.2,
  "motor_2": -14.7,
  "motor_3": 32.1,
  "motor_4": 4.9,
  "motor_5": -11.3,
  "motor_6": 22.0,
  "hand_detected": true
}
```

## Digital twin without webcam

```powershell
python so101_digital_twin_vpython.py --source synthetic --config .\so101_config.example.json
```

## Physical SO101 path (adapter integration point)

`so101_robot_bridge.py` is safe by default (`--enable-robot` not set).

To enable physical control later, provide your adapter class:

```powershell
python so101_robot_bridge.py --enable-robot --driver-module your_so101_driver_module --driver-class YourSO101Driver --config .\so101_config.example.json
```

Your driver class should implement:

- `__init__(config_path: str)`
- `send_targets(motors_deg: list[float])`
- `close()`

## Simulation-only version (new folder)

If you want a clean mode with no physical-arm operation path, use:

```powershell
python sim_only_v2/sim_only_hand_and_twin.py --camera 0 --config sim_only_v2/config.sim_only.json
```

Docs:

- `sim_only_v2/README.md`

## Full Legacy Demo (Lower Priority)

If you want the original 3-process flow, run this below the Web V3 workflow.

Open 3 terminals from the project root.

Terminal A (3D viewer):

```powershell
python so101_digital_twin_vpython.py --source udp --udp-bind 127.0.0.1 --udp-port 5005 --config .\so101_config.example.json
```

Terminal B (hand -> motors stream):

```powershell
python hand_to_so101_positions.py --camera 0 --udp 127.0.0.1:5005 --config .\so101_config.example.json
```

Terminal C (robot bridge dry-run):

```powershell
python so101_robot_bridge.py --udp-bind 127.0.0.1 --udp-port 5005 --config .\so101_config.example.json
```
