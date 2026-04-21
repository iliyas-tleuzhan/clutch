from typing import Dict, List, Optional

import numpy as np


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, float(v)))


def _parse_vec3(values, default: List[float]) -> np.ndarray:
    if isinstance(values, list) and len(values) == 3:
        try:
            return np.array([float(values[0]), float(values[1]), float(values[2])], dtype=float)
        except Exception:
            pass
    return np.array(default, dtype=float)


class EETargetMapper:
    """
    Maps hand tracker output in camera coordinates into robot EE targets.

    Phase 1 scope:
    - Position XYZ control only
    - Separate scalar gripper command from pinch ratio
    - Confidence gating + workspace clamping + smoothing
    """

    def __init__(self, config: dict):
        mapper_cfg = config.get("ee_target_mapper", {})
        grip_cfg = mapper_cfg.get("grip_from_pinch", {})

        self.workspace_center = _parse_vec3(
            mapper_cfg.get("workspace_center_m"),
            [0.22, 0.0, 0.14],
        )
        self.workspace_half_extents = _parse_vec3(
            mapper_cfg.get("workspace_half_extents_m"),
            [0.16, 0.20, 0.16],
        )
        self.position_scale = _parse_vec3(
            mapper_cfg.get("position_scale_m_per_camera_unit"),
            [0.55, 0.55, 0.75],
        )

        # Camera delta (x,y,z) -> robot delta (x,y,z)
        # Default mapping:
        # robot_x <- -camera_z
        # robot_y <-  camera_x
        # robot_z <- -camera_y
        default_rot = [
            [0.0, 0.0, -1.0],
            [1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
        ]
        rot_cfg = mapper_cfg.get("camera_to_robot_rotation")
        if isinstance(rot_cfg, list) and len(rot_cfg) == 3 and all(isinstance(r, list) and len(r) == 3 for r in rot_cfg):
            try:
                self.camera_to_robot = np.array(rot_cfg, dtype=float)
            except Exception:
                self.camera_to_robot = np.array(default_rot, dtype=float)
        else:
            self.camera_to_robot = np.array(default_rot, dtype=float)

        self.mirror_y_for_left_arm = bool(mapper_cfg.get("mirror_y_for_left_arm", True))
        self.position_alpha = _clamp(mapper_cfg.get("position_alpha", 0.28), 0.01, 1.0)
        self.deadzone_m = max(0.0, float(mapper_cfg.get("position_deadzone_m", 0.004)))
        self.min_confidence = _clamp(mapper_cfg.get("min_tracking_confidence", 0.25), 0.0, 1.0)
        self.target_min_reachable_margin_m = max(
            0.0, float(mapper_cfg.get("target_min_reachable_margin_m", 0.002))
        )

        self.axis_sign_correction = _parse_vec3(
            mapper_cfg.get("axis_sign_correction"),
            [1.0, 1.0, 1.0],
        )
        self.axis_sign_correction = np.where(self.axis_sign_correction >= 0.0, 1.0, -1.0)
        self.axis_scale_correction = _parse_vec3(
            mapper_cfg.get("axis_scale_correction"),
            [1.0, 1.0, 1.0],
        )
        self.axis_scale_correction = np.clip(self.axis_scale_correction, 0.05, 20.0)
        self.axis_offset_correction = _parse_vec3(
            mapper_cfg.get("axis_offset_correction_m"),
            [0.0, 0.0, 0.0],
        )

        self.pinch_closed_ratio = float(grip_cfg.get("pinch_closed_ratio", 0.30))
        self.pinch_open_ratio = float(grip_cfg.get("pinch_open_ratio", 1.00))
        self.default_grip_closedness = _clamp(grip_cfg.get("default_closedness", 0.1), 0.0, 1.0)
        self.grip_alpha = _clamp(grip_cfg.get("grip_alpha", 0.35), 0.01, 1.0)

        self.neutral_control: Optional[np.ndarray] = None
        self.prev_position: Optional[np.ndarray] = None
        self.prev_grip = self.default_grip_closedness

    def set_calibration_offsets(self, center_offsets: Dict[str, float]):
        try:
            cx = float(center_offsets["control_x"])
            cy = float(center_offsets["control_y"])
            cz = float(center_offsets["control_z"])
        except Exception:
            return
        self.neutral_control = np.array([cx, cy, cz], dtype=float)
        try:
            wx = float(center_offsets["workspace_center_x"])
            wy = float(center_offsets["workspace_center_y"])
            wz = float(center_offsets["workspace_center_z"])
            self.workspace_center = np.array([wx, wy, wz], dtype=float)
        except Exception:
            pass
        try:
            self.axis_sign_correction = np.where(
                np.array(
                    [
                        float(center_offsets["axis_sign_x"]),
                        float(center_offsets["axis_sign_y"]),
                        float(center_offsets["axis_sign_z"]),
                    ],
                    dtype=float,
                )
                >= 0.0,
                1.0,
                -1.0,
            )
        except Exception:
            pass
        try:
            self.axis_scale_correction = np.clip(
                np.array(
                    [
                        float(center_offsets["axis_scale_x"]),
                        float(center_offsets["axis_scale_y"]),
                        float(center_offsets["axis_scale_z"]),
                    ],
                    dtype=float,
                ),
                0.05,
                20.0,
            )
        except Exception:
            pass
        try:
            self.axis_offset_correction = np.array(
                [
                    float(center_offsets["axis_offset_x_m"]),
                    float(center_offsets["axis_offset_y_m"]),
                    float(center_offsets["axis_offset_z_m"]),
                ],
                dtype=float,
            )
        except Exception:
            pass

    def export_calibration_offsets(self) -> Dict[str, float]:
        if self.neutral_control is None:
            return {}
        return {
            "control_x": float(self.neutral_control[0]),
            "control_y": float(self.neutral_control[1]),
            "control_z": float(self.neutral_control[2]),
            "workspace_center_x": float(self.workspace_center[0]),
            "workspace_center_y": float(self.workspace_center[1]),
            "workspace_center_z": float(self.workspace_center[2]),
            "axis_sign_x": float(self.axis_sign_correction[0]),
            "axis_sign_y": float(self.axis_sign_correction[1]),
            "axis_sign_z": float(self.axis_sign_correction[2]),
            "axis_scale_x": float(self.axis_scale_correction[0]),
            "axis_scale_y": float(self.axis_scale_correction[1]),
            "axis_scale_z": float(self.axis_scale_correction[2]),
            "axis_offset_x_m": float(self.axis_offset_correction[0]),
            "axis_offset_y_m": float(self.axis_offset_correction[1]),
            "axis_offset_z_m": float(self.axis_offset_correction[2]),
        }

    def export_runtime_calibration(self) -> Dict:
        return {
            "neutral_control_camera_xyz": (
                [float(v) for v in self.neutral_control.tolist()]
                if self.neutral_control is not None
                else None
            ),
            "workspace_center_m": [float(self.workspace_center[0]), float(self.workspace_center[1]), float(self.workspace_center[2])],
            "axis_sign_correction": [float(v) for v in self.axis_sign_correction.tolist()],
            "axis_scale_correction": [float(v) for v in self.axis_scale_correction.tolist()],
            "axis_offset_correction_m": [float(v) for v in self.axis_offset_correction.tolist()],
        }

    def calibrate(self, observation: Dict, neutral_ee_reference_xyz: Optional[List[float]] = None):
        cp = observation.get("control_point_camera_xyz")
        if not (isinstance(cp, list) and len(cp) == 3):
            return
        self.neutral_control = np.array([float(cp[0]), float(cp[1]), float(cp[2])], dtype=float)
        if isinstance(neutral_ee_reference_xyz, list) and len(neutral_ee_reference_xyz) == 3:
            self.workspace_center = np.array(
                [
                    float(neutral_ee_reference_xyz[0]),
                    float(neutral_ee_reference_xyz[1]),
                    float(neutral_ee_reference_xyz[2]),
                ],
                dtype=float,
            )

    def set_axis_corrections(
        self,
        sign_xyz: Optional[List[float]] = None,
        scale_xyz: Optional[List[float]] = None,
        offset_xyz_m: Optional[List[float]] = None,
    ):
        if isinstance(sign_xyz, list) and len(sign_xyz) == 3:
            self.axis_sign_correction = np.where(
                np.array([float(sign_xyz[0]), float(sign_xyz[1]), float(sign_xyz[2])], dtype=float)
                >= 0.0,
                1.0,
                -1.0,
            )
        if isinstance(scale_xyz, list) and len(scale_xyz) == 3:
            self.axis_scale_correction = np.clip(
                np.array([float(scale_xyz[0]), float(scale_xyz[1]), float(scale_xyz[2])], dtype=float),
                0.05,
                20.0,
            )
        if isinstance(offset_xyz_m, list) and len(offset_xyz_m) == 3:
            self.axis_offset_correction = np.array(
                [float(offset_xyz_m[0]), float(offset_xyz_m[1]), float(offset_xyz_m[2])],
                dtype=float,
            )

    def project_robot_target(self, raw_target_xyz: List[float]) -> Dict:
        raw = np.array([float(raw_target_xyz[0]), float(raw_target_xyz[1]), float(raw_target_xyz[2])], dtype=float)
        delta_from_center = raw - self.workspace_center
        corrected = (
            self.workspace_center
            + self.axis_sign_correction * self.axis_scale_correction * delta_from_center
            + self.axis_offset_correction
        )
        lo = self.workspace_center - self.workspace_half_extents
        hi = self.workspace_center + self.workspace_half_extents
        clamped = np.clip(corrected, lo, hi)
        clamp_delta = corrected - clamped
        target_clamped = bool(np.linalg.norm(clamp_delta) > 1e-9)

        axes = []
        if corrected[0] < lo[0] or corrected[0] > hi[0]:
            axes.append("x")
        if corrected[1] < lo[1] or corrected[1] > hi[1]:
            axes.append("y")
        if corrected[2] < lo[2] or corrected[2] > hi[2]:
            axes.append("z")

        reachable_margin = min(float(clamped[0] - lo[0]), float(hi[0] - clamped[0]), float(clamped[1] - lo[1]), float(hi[1] - clamped[1]), float(clamped[2] - lo[2]), float(hi[2] - clamped[2]))
        target_reachable = bool((not target_clamped) and (reachable_margin >= self.target_min_reachable_margin_m))
        return {
            "raw_target_xyz": [float(raw[0]), float(raw[1]), float(raw[2])],
            "corrected_target_xyz": [float(corrected[0]), float(corrected[1]), float(corrected[2])],
            "clamped_target_xyz": [float(clamped[0]), float(clamped[1]), float(clamped[2])],
            "target_reachable": target_reachable,
            "target_clamped": target_clamped,
            "clamp_delta_xyz": [float(clamp_delta[0]), float(clamp_delta[1]), float(clamp_delta[2])],
            "workspace_violation_axes": axes,
        }

    def _map_grip(self, pinch_ratio: Optional[float]) -> float:
        if pinch_ratio is None:
            return self.prev_grip

        # 0.0=open, 1.0=closed
        denom = max(1e-6, self.pinch_open_ratio - self.pinch_closed_ratio)
        open_norm = _clamp((float(pinch_ratio) - self.pinch_closed_ratio) / denom, 0.0, 1.0)
        closedness = 1.0 - open_norm
        grip = self.prev_grip + (closedness - self.prev_grip) * self.grip_alpha
        self.prev_grip = _clamp(grip, 0.0, 1.0)
        return self.prev_grip

    def map(self, observation: Dict, arm_side: str) -> Dict:
        cp = observation.get("control_point_camera_xyz")
        raw_cp = observation.get("raw_control_point_camera_xyz")
        conf = float(observation.get("tracking_confidence", 0.0))
        hand_ok = bool(observation.get("hand_detected", False))
        pose_ok = bool(observation.get("pose_detected", False))

        if isinstance(cp, list) and len(cp) == 3:
            cp_vec = np.array([float(cp[0]), float(cp[1]), float(cp[2])], dtype=float)
            if self.neutral_control is None:
                self.neutral_control = cp_vec.copy()
        else:
            cp_vec = None

        confidence_gated = False
        workspace_clamped = False
        target_valid = cp_vec is not None and pose_ok

        raw_target_robot = None
        projection = None
        if not target_valid:
            if self.prev_position is None:
                self.prev_position = self.workspace_center.copy()
            position = self.prev_position.copy()
            confidence_gated = True
        else:
            delta_camera = cp_vec - self.neutral_control
            delta_scaled = delta_camera * self.position_scale
            delta_robot = self.camera_to_robot @ delta_scaled

            if self.mirror_y_for_left_arm and arm_side.lower() == "left":
                delta_robot[1] *= -1.0

            raw_target_robot = self.workspace_center + delta_robot
            projection = self.project_robot_target(raw_target_robot.tolist())
            clipped = np.array(projection["clamped_target_xyz"], dtype=float)
            workspace_clamped = bool(projection["target_clamped"])

            if self.prev_position is None:
                self.prev_position = clipped.copy()
            else:
                if conf < self.min_confidence:
                    clipped = self.prev_position.copy()
                    confidence_gated = True
                elif np.linalg.norm(clipped - self.prev_position) < self.deadzone_m:
                    clipped = self.prev_position.copy()
                clipped = self.prev_position + (clipped - self.prev_position) * self.position_alpha
                self.prev_position = clipped
            position = self.prev_position.copy()

        grip = self._map_grip(observation.get("pinch_ratio") if hand_ok else None)

        return {
            "position_xyz": [float(position[0]), float(position[1]), float(position[2])],
            "orientation_rpy": [0.0, 0.0, 0.0],
            "grip": float(grip),
            "valid": bool(target_valid),
            "confidence_gated": bool(confidence_gated),
            "workspace_clamped": bool(workspace_clamped),
            "tracking_confidence": float(conf),
            "raw_target_xyz": (
                projection["raw_target_xyz"]
                if projection is not None
                else [float(position[0]), float(position[1]), float(position[2])]
            ),
            "clamped_target_xyz": (
                projection["clamped_target_xyz"]
                if projection is not None
                else [float(position[0]), float(position[1]), float(position[2])]
            ),
            "target_reachable": (
                bool(projection["target_reachable"])
                if projection is not None
                else False
            ),
            "target_clamped": (
                bool(projection["target_clamped"])
                if projection is not None
                else False
            ),
            "clamp_delta_xyz": (
                projection["clamp_delta_xyz"]
                if projection is not None
                else [0.0, 0.0, 0.0]
            ),
            "workspace_violation_axes": (
                projection["workspace_violation_axes"]
                if projection is not None
                else []
            ),
            "raw_control_camera_xyz": (
                [float(raw_cp[0]), float(raw_cp[1]), float(raw_cp[2])]
                if isinstance(raw_cp, list) and len(raw_cp) == 3
                else None
            ),
            "mapped_control_camera_xyz": (
                [float(cp[0]), float(cp[1]), float(cp[2])]
                if isinstance(cp, list) and len(cp) == 3
                else None
            ),
            "neutral_control_camera_xyz": (
                [float(v) for v in self.neutral_control.tolist()]
                if self.neutral_control is not None
                else None
            ),
        }
