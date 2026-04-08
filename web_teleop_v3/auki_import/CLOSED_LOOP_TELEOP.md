# Closed-loop teleop (follow-on)

The browser stack now includes landmark filtering (One Euro), optional IK to the SO-101 URDF, joint velocity/acceleration limits, a grasp FSM, calibration in `localStorage`, and JSONL metrics (`teleop_metrics.jsonl`). For production-grade pick reliability, plan additional layers:

1. **Wrist or scene camera + fiducials** — AprilTag on the gripper or object enables visual servoing for the last centimeters when hand landmarks are occluded or unreliable.
2. **Motor current / effort** — LeRobot and similar stacks often expose per-joint effort; use a threshold to detect “gripped” and transition the FSM from `GRASP` to `LIFT` without relying on pinch alone.
3. **Depth sensing** — A RealSense-class camera supports segmentation and grasp planning; this is a different product surface than pure hand mimicry but removes ambiguity from 2D hand pose.

Integrate these in order of cost: effort signals (if available on your bus) are usually cheapest; fiducial visual servo is moderate; full depth + planning is highest effort.
