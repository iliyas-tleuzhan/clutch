import json
import time
from pathlib import Path


def generate_sim_real_report(sim_path: str, real_path: str, output_path: str):
    """
    Starter hook for item #9 (sim-to-real validation).
    Computes simple frame count and average joint delta.
    """
    with open(sim_path, "r", encoding="utf-8") as f:
        sim = json.load(f)
    with open(real_path, "r", encoding="utf-8") as f:
        real = json.load(f)

    sim_frames = sim.get("frames", [])
    real_frames = real.get("frames", [])
    n = min(len(sim_frames), len(real_frames))
    avg_abs = [0.0] * 6
    if n > 0:
        for i in range(n):
            sa = sim_frames[i]["motors_deg"]
            ra = real_frames[i]["motors_deg"]
            for j in range(6):
                avg_abs[j] += abs(float(sa[j]) - float(ra[j]))
        avg_abs = [v / n for v in avg_abs]

    report = {
        "created_at": time.time(),
        "sim_path": sim_path,
        "real_path": real_path,
        "aligned_frames": n,
        "avg_abs_joint_error_deg": avg_abs,
    }
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    return report
