# Sim-Only V2 (No Physical Arm)

This version runs only:

1. hand tracking (webcam)
2. motor target generation (`motor_1..motor_6`)
3. real-time 3D digital twin visualization

No physical robot commands are sent in this flow.

## Run

From project root:

```powershell
python sim_only_v2/sim_only_hand_and_twin.py --camera 0 --config sim_only_v2/config.sim_only.json
```

## Useful options

```powershell
# If hand is lost, keep last pose
python sim_only_v2/sim_only_hand_and_twin.py --lost-mode hold

# If hand is lost, return to home pose
python sim_only_v2/sim_only_hand_and_twin.py --lost-mode home

# Disable OpenCV preview (keep VPython twin + console JSON)
python sim_only_v2/sim_only_hand_and_twin.py --no-preview
```

Press `q` or `Esc` in the OpenCV window to stop.
