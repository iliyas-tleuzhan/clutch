"""Regression checks for teleop IK + safety (no server)."""
import json
import sys
from pathlib import Path

import pytest

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))


def test_safety_rate_limit_monotonic():
    from safety import SafetySupervisor

    cfg = json.loads((BASE / "config.json").read_text())
    s = SafetySupervisor(cfg)
    t = 0.0
    m0 = [0.0, 0.0, 0.0, 0.0, 0.0, 5.0]
    o0 = s.filter_hand_command(m0, t)
    assert o0 == m0
    t += 0.1
    jump = [80.0, 0.0, 0.0, 0.0, 0.0, 5.0]
    o1 = s.filter_hand_command(jump, t)
    assert abs(o1[0] - m0[0]) < abs(jump[0] - m0[0])


def test_ik_solve_finite():
    from so101_ik import solve_ee_position

    sol, ok = solve_ee_position([0.18, 0.02, 0.22], [0, 0, 0, 0, 0, 5])
    assert ok
    assert len(sol) == 6
    assert all(abs(x) < 400 for x in sol)
