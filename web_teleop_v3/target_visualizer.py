from typing import Dict, List


def _fmt_vec3(label: str, values) -> str:
    if isinstance(values, list) and len(values) == 3:
        return f"{label}: {values[0]:+0.3f} {values[1]:+0.3f} {values[2]:+0.3f}"
    return f"{label}: n/a"


def build_debug_lines(observation: Dict, ee_target: Dict, ik_result: Dict) -> List[str]:
    lines: List[str] = []

    if observation:
        lines.append(_fmt_vec3("raw_cam", observation.get("raw_control_point_camera_xyz")))
        lines.append(_fmt_vec3("mapped_cam", observation.get("control_point_camera_xyz")))

    if ee_target:
        lines.append(_fmt_vec3("ee_target", ee_target.get("position_xyz")))
        lines.append(_fmt_vec3("raw_target", ee_target.get("raw_target_xyz")))
        lines.append(_fmt_vec3("clamped_target", ee_target.get("clamped_target_xyz")))
        lines.append(f"track_conf: {float(ee_target.get('tracking_confidence', 0.0)):.2f}")
        lines.append(f"workspace_clamped: {bool(ee_target.get('workspace_clamped', False))}")
        lines.append(f"confidence_gated: {bool(ee_target.get('confidence_gated', False))}")
        lines.append(f"target_reachable: {bool(ee_target.get('target_reachable', False))}")
        lines.append(f"target_clamped: {bool(ee_target.get('target_clamped', False))}")
        lines.append(f"workspace_violation_axes: {ee_target.get('workspace_violation_axes', [])}")
        grip = ee_target.get("grip")
        if grip is not None:
            lines.append(f"grip_closedness: {float(grip):.2f}")

    if ik_result:
        joints = ik_result.get("motors_deg")
        if isinstance(joints, list) and len(joints) >= 6:
            lines.append(
                "ik_joints_deg: "
                + " ".join([f"{float(joints[i]):+05.1f}" for i in range(6)])
            )
        lines.append(_fmt_vec3("ee_fk", ik_result.get("ee_fk_xyz")))
        err = ik_result.get("ik_error_m")
        lines.append(f"ee_error_norm: {float(err):.4f}" if err is not None else "ee_error_norm: n/a")
        lines.append(f"ik_ok: {bool(ik_result.get('ik_ok', False))}")
        lines.append(f"ik_fail_reason: {ik_result.get('ik_fail_reason', '')}")
        lines.append(f"joint_limit_hit: {bool(ik_result.get('joint_limit_hit', False))}")
        lines.append(f"singularity_warning: {bool(ik_result.get('singularity_warning', False))}")
        lines.append(f"ik_iter: {int(ik_result.get('ik_iterations', 0))}")

    return lines
