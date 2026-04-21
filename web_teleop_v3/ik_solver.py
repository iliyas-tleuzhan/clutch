import math
from typing import Dict, List, Tuple

import numpy as np


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, float(v)))


def _axis_from_name(name: str) -> np.ndarray:
    n = str(name).strip().lower()
    if n == "x":
        return np.array([1.0, 0.0, 0.0], dtype=float)
    if n == "y":
        return np.array([0.0, 1.0, 0.0], dtype=float)
    return np.array([0.0, 0.0, 1.0], dtype=float)


def _rotation_matrix(axis_local: np.ndarray, angle_rad: float) -> np.ndarray:
    ux, uy, uz = axis_local.tolist()
    c = math.cos(angle_rad)
    s = math.sin(angle_rad)
    one_c = 1.0 - c
    return np.array(
        [
            [c + ux * ux * one_c, ux * uy * one_c - uz * s, ux * uz * one_c + uy * s],
            [uy * ux * one_c + uz * s, c + uy * uy * one_c, uy * uz * one_c - ux * s],
            [uz * ux * one_c - uy * s, uz * uy * one_c + ux * s, c + uz * uz * one_c],
        ],
        dtype=float,
    )


class DampedLeastSquaresIKSolver:
    """
    Jacobian-based DLS IK for Phase 1 EE position (XYZ) control.
    Includes:
    - Forward kinematics using identical assumptions as IK
    - Joint-limit diagnostics
    - Singularity diagnostics (sigma_min / condition estimate)
    - Axis-sign validation helper
    """

    def __init__(self, config: dict):
        ik_cfg = config.get("ik_solver", {})
        motors = config.get("motors", [])

        self.n_motors = len(motors) if motors else 6
        self.n_pos_joints = int(ik_cfg.get("position_joint_count", 5))
        self.n_pos_joints = max(1, min(self.n_pos_joints, self.n_motors))

        default_axes = ["y", "z", "z", "x", "z"]
        axis_names = ik_cfg.get("joint_axes", default_axes)
        if not isinstance(axis_names, list) or len(axis_names) < self.n_pos_joints:
            axis_names = default_axes

        default_signs = [1.0] * self.n_pos_joints
        axis_signs = ik_cfg.get("joint_axis_signs", default_signs)
        if not isinstance(axis_signs, list) or len(axis_signs) < self.n_pos_joints:
            axis_signs = default_signs

        self.axes_local: List[np.ndarray] = []
        for i in range(self.n_pos_joints):
            axis = _axis_from_name(axis_names[i]) * (1.0 if float(axis_signs[i]) >= 0.0 else -1.0)
            self.axes_local.append(axis)

        default_links = [
            [0.10, 0.0, 0.0],
            [0.18, 0.0, 0.0],
            [0.19, 0.0, 0.0],
            [0.10, 0.0, 0.0],
            [0.08, 0.0, 0.0],
        ]
        links_cfg = ik_cfg.get("link_vectors_m", default_links)
        if not isinstance(links_cfg, list) or len(links_cfg) < self.n_pos_joints:
            links_cfg = default_links
        self.link_vectors = []
        for i in range(self.n_pos_joints):
            v = links_cfg[i]
            if not isinstance(v, list) or len(v) != 3:
                v = default_links[i]
            self.link_vectors.append(np.array([float(v[0]), float(v[1]), float(v[2])], dtype=float))

        base_offset = ik_cfg.get("base_offset_m", [0.0, 0.0, 0.0])
        if not isinstance(base_offset, list) or len(base_offset) != 3:
            base_offset = [0.0, 0.0, 0.0]
        self.base_offset = np.array(
            [float(base_offset[0]), float(base_offset[1]), float(base_offset[2])],
            dtype=float,
        )

        lo_deg: List[float] = []
        hi_deg: List[float] = []
        home_deg: List[float] = []
        for i in range(self.n_motors):
            m = motors[i] if i < len(motors) else {}
            lo_deg.append(float(m.get("min_deg", -180.0)))
            hi_deg.append(float(m.get("max_deg", 180.0)))
            home_deg.append(float(m.get("home_deg", 0.0)))
        self.home_deg = home_deg[:]
        self.lo_rad = np.radians(np.array(lo_deg[: self.n_pos_joints], dtype=float))
        self.hi_rad = np.radians(np.array(hi_deg[: self.n_pos_joints], dtype=float))

        self.max_iterations = int(ik_cfg.get("max_iterations", 12))
        self.convergence_eps_m = max(1e-5, float(ik_cfg.get("convergence_eps_m", 0.004)))
        self.max_error_m = max(self.convergence_eps_m, float(ik_cfg.get("max_error_m", 0.08)))
        self.damping = max(1e-4, float(ik_cfg.get("damping", 0.08)))
        self.low_confidence_damping_scale = max(
            1.0, float(ik_cfg.get("low_confidence_damping_scale", 2.5))
        )
        self.low_confidence_threshold = _clamp(ik_cfg.get("low_confidence_threshold", 0.35), 0.0, 1.0)
        self.regularization = max(0.0, float(ik_cfg.get("regularization_to_previous", 0.08)))
        self.posture_regularization = max(0.0, float(ik_cfg.get("posture_regularization", 0.02)))
        self.joint_limit_margin_rad = math.radians(
            max(0.0, float(ik_cfg.get("joint_limit_margin_deg", 1.5)))
        )
        self.singularity_sigma_min_threshold = max(
            1e-9, float(ik_cfg.get("singularity_sigma_min_threshold", 0.01))
        )
        self.singularity_condition_threshold = max(
            1.0, float(ik_cfg.get("singularity_condition_threshold", 250.0))
        )

        max_step_deg = ik_cfg.get("max_step_deg_per_update", [7.0] * self.n_pos_joints)
        if not isinstance(max_step_deg, list) or len(max_step_deg) < self.n_pos_joints:
            max_step_deg = [7.0] * self.n_pos_joints
        self.max_step_rad = np.radians(np.array([float(max_step_deg[i]) for i in range(self.n_pos_joints)]))

        default_seed = [0.0, -35.0, 70.0, 0.0, 35.0]
        seed_cfg = ik_cfg.get("initial_guess_deg", default_seed)
        if not isinstance(seed_cfg, list) or len(seed_cfg) < self.n_pos_joints:
            seed_cfg = default_seed
        seed_deg = np.array([float(seed_cfg[i]) for i in range(self.n_pos_joints)], dtype=float)
        seed_rad = np.radians(seed_deg)
        self.seed_q = np.clip(seed_rad, self.lo_rad, self.hi_rad)

        self.prev_q = self.seed_q.copy()
        self.prev_full = home_deg[:]
        for i in range(self.n_pos_joints):
            self.prev_full[i] = float(np.degrees(self.prev_q[i]))

    def _q_from_motors_deg(self, motors_deg: List[float]) -> np.ndarray:
        q = self.prev_q.copy()
        if isinstance(motors_deg, list):
            for i in range(min(self.n_pos_joints, len(motors_deg))):
                try:
                    q[i] = math.radians(float(motors_deg[i]))
                except Exception:
                    continue
        return np.clip(q, self.lo_rad, self.hi_rad)

    def _motors_from_q(self, q_rad: np.ndarray) -> List[float]:
        motors = self.home_deg[:]
        solved_deg = np.degrees(q_rad)
        for i in range(self.n_pos_joints):
            motors[i] = float(solved_deg[i])
        return motors

    def _fk_and_jacobian(self, q_rad: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        rot_world = np.eye(3, dtype=float)
        pos_world = self.base_offset.copy()
        joint_origins: List[np.ndarray] = []
        joint_axes_world: List[np.ndarray] = []

        for i in range(self.n_pos_joints):
            axis_local = self.axes_local[i]
            axis_world = rot_world @ axis_local
            joint_origins.append(pos_world.copy())
            joint_axes_world.append(axis_world)

            rot_world = rot_world @ _rotation_matrix(axis_local, q_rad[i])
            pos_world = pos_world + rot_world @ self.link_vectors[i]

        ee_pos = pos_world
        jac = np.zeros((3, self.n_pos_joints), dtype=float)
        for i in range(self.n_pos_joints):
            jac[:, i] = np.cross(joint_axes_world[i], ee_pos - joint_origins[i])
        return ee_pos, jac

    def forward_kinematics(self, motors_deg: List[float]) -> List[float]:
        q = self._q_from_motors_deg(motors_deg)
        ee, _ = self._fk_and_jacobian(q)
        return [float(ee[0]), float(ee[1]), float(ee[2])]

    def current_fk_xyz(self) -> List[float]:
        ee, _ = self._fk_and_jacobian(self.prev_q.copy())
        return [float(ee[0]), float(ee[1]), float(ee[2])]

    def axis_validation_report(self, step_deg: float = 4.0) -> Dict:
        step = math.radians(abs(float(step_deg)))
        base_q = self.seed_q.copy()
        base_ee, _ = self._fk_and_jacobian(base_q)
        entries = []
        axis_labels = ["x", "y", "z"]
        for i in range(self.n_pos_joints):
            q_plus = base_q.copy()
            q_minus = base_q.copy()
            q_plus[i] = _clamp(q_plus[i] + step, self.lo_rad[i], self.hi_rad[i])
            q_minus[i] = _clamp(q_minus[i] - step, self.lo_rad[i], self.hi_rad[i])
            ee_plus, _ = self._fk_and_jacobian(q_plus)
            ee_minus, _ = self._fk_and_jacobian(q_minus)
            d_plus = ee_plus - base_ee
            d_minus = ee_minus - base_ee

            dominant_idx = int(np.argmax(np.abs(d_plus)))
            dominant_axis = axis_labels[dominant_idx]
            dominant_sign = "+" if d_plus[dominant_idx] >= 0.0 else "-"
            entries.append(
                {
                    "joint_index": i,
                    "delta_plus_xyz_m": [float(d_plus[0]), float(d_plus[1]), float(d_plus[2])],
                    "delta_minus_xyz_m": [float(d_minus[0]), float(d_minus[1]), float(d_minus[2])],
                    "dominant_motion_hint": f"+joint_{i+1} -> {dominant_sign}{dominant_axis}",
                }
            )
        return {
            "step_deg": float(step_deg),
            "base_ee_xyz_m": [float(base_ee[0]), float(base_ee[1]), float(base_ee[2])],
            "entries": entries,
        }

    def solve(self, target_position_xyz: List[float], tracking_confidence: float) -> Dict:
        if not isinstance(target_position_xyz, list) or len(target_position_xyz) != 3:
            ee_fk = self.current_fk_xyz()
            return {
                "motors_deg": self.prev_full[:],
                "ik_ok": False,
                "ik_error_m": None,
                "ik_iterations": 0,
                "ee_fk_xyz": ee_fk,
                "joint_limit_hit": False,
                "singularity_warning": False,
                "ik_fail_reason": "invalid_target",
                "sigma_min": None,
                "condition_estimate": None,
            }

        q = self.prev_q.copy()
        q_ref = self.prev_q.copy()
        target = np.array(
            [float(target_position_xyz[0]), float(target_position_xyz[1]), float(target_position_xyz[2])],
            dtype=float,
        )
        conf = _clamp(tracking_confidence, 0.0, 1.0)
        damping = self.damping
        if conf < self.low_confidence_threshold:
            damping *= self.low_confidence_damping_scale

        err_norm = float("inf")
        used_iterations = 0
        for it in range(self.max_iterations):
            ee, jac = self._fk_and_jacobian(q)
            err = target - ee
            err_norm = float(np.linalg.norm(err))
            used_iterations = it + 1
            if err_norm <= self.convergence_eps_m:
                break

            jj_t = jac @ jac.T
            lhs = jj_t + (damping ** 2) * np.eye(3, dtype=float)
            try:
                y = np.linalg.solve(lhs, err)
            except np.linalg.LinAlgError:
                y = np.linalg.pinv(lhs) @ err

            dq = jac.T @ y
            dq = dq + self.regularization * (q_ref - q)
            dq = dq + self.posture_regularization * (self.seed_q - q)
            dq = np.clip(dq, -self.max_step_rad, self.max_step_rad)
            q = q + dq
            q = np.clip(q, self.lo_rad, self.hi_rad)

        ee_fk, jac_fk = self._fk_and_jacobian(q)
        self.prev_q = q
        motors = self._motors_from_q(q)
        self.prev_full = motors[:]

        try:
            sigma = np.linalg.svd(jac_fk, compute_uv=False)
            sigma_min = float(np.min(sigma))
            sigma_max = float(np.max(sigma))
            condition = float(sigma_max / max(1e-12, sigma_min))
        except np.linalg.LinAlgError:
            sigma_min = 0.0
            condition = float("inf")

        lower_hit = np.any((q - self.lo_rad) <= self.joint_limit_margin_rad)
        upper_hit = np.any((self.hi_rad - q) <= self.joint_limit_margin_rad)
        joint_limit_hit = bool(lower_hit or upper_hit)
        singularity_warning = bool(
            sigma_min <= self.singularity_sigma_min_threshold
            or condition >= self.singularity_condition_threshold
        )

        ee_error_norm = float(np.linalg.norm(target - ee_fk))
        ik_ok = bool(ee_error_norm <= self.max_error_m)
        if ik_ok:
            fail_reason = ""
        elif singularity_warning:
            fail_reason = "singularity"
        elif joint_limit_hit:
            fail_reason = "joint_limit"
        else:
            fail_reason = "max_error_exceeded"
        return {
            "motors_deg": motors,
            "ik_ok": ik_ok,
            "ik_error_m": float(ee_error_norm),
            "ik_iterations": int(used_iterations),
            "ee_fk_xyz": [float(ee_fk[0]), float(ee_fk[1]), float(ee_fk[2])],
            "joint_limit_hit": joint_limit_hit,
            "singularity_warning": singularity_warning,
            "ik_fail_reason": fail_reason,
            "sigma_min": sigma_min,
            "condition_estimate": condition,
        }
