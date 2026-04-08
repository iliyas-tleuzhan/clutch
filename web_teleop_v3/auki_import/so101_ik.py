"""
SO-101 inverse kinematics via ikpy + bundled URDF.

Joint order in URDF chain: shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll.
Demo / LeRobot motor order (syncUrdfMotors): same but wrist_roll and wrist_flex are swapped
(indices 3 and 4).
"""
from __future__ import annotations

import math
import warnings
from functools import lru_cache
from pathlib import Path

import numpy as np

BASE = Path(__file__).parent
URDF_PATH = BASE / "assets" / "so101_new_calib.urdf"


@lru_cache(maxsize=1)
def _chain():
    from ikpy.chain import Chain

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return Chain.from_urdf_file(str(URDF_PATH), base_elements=["base_link"], symbolic=False)


def _demo_deg_from_q(q: np.ndarray) -> tuple[float, float, float, float, float]:
    """Extract URDF joint angles (rad) from ikpy configuration vector; map to demo motor order (deg)."""
    # ikpy: q[0] base, q[1..5] = pan, lift, elbow, wrist_flex, wrist_roll (see chain link names)
    pan, lift, elbow, wf, wr = float(q[1]), float(q[2]), float(q[3]), float(q[4]), float(q[5])
    r2d = 180.0 / math.pi
    # Demo index 3 = wrist_roll, index 4 = wrist_flex (matches demo.html syncUrdfMotors)
    return (
        pan * r2d,
        lift * r2d,
        elbow * r2d,
        wr * r2d,
        wf * r2d,
    )


def solve_ee_position(
    target_xyz_m: list[float] | tuple[float, float, float],
    initial_motor_deg: list[float] | None = None,
) -> tuple[list[float], bool]:
    """
    Returns (6 motor degrees [pan..gripper], success). Gripper angle is copied from initial or 5.0.
    """
    chain = _chain()
    target = np.array([float(target_xyz_m[0]), float(target_xyz_m[1]), float(target_xyz_m[2])], dtype=np.float64)

    n = len(chain.links)
    q_init = np.zeros(n, dtype=np.float64)
    if initial_motor_deg is not None and len(initial_motor_deg) >= 5:
        m = initial_motor_deg
        d2r = math.pi / 180.0
        # URDF order: pan, lift, elbow, wrist_flex, wrist_roll — demo swaps last two indices.
        q_init[1] = m[0] * d2r
        q_init[2] = m[1] * d2r
        q_init[3] = m[2] * d2r
        q_init[4] = m[4] * d2r
        q_init[5] = m[3] * d2r

    try:
        q = chain.inverse_kinematics(target, initial_position=q_init)
    except Exception:
        return ([0.0] * 6, False)

    d0, d1, d2, d3, d4 = _demo_deg_from_q(q)
    grip = float(initial_motor_deg[5]) if initial_motor_deg and len(initial_motor_deg) > 5 else 5.0
    return ([d0, d1, d2, d3, d4, grip], True)
