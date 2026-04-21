# Clutch Teleoperation Safety Notes (Demo-Level)

This project now applies software safety layers inspired by industrial robot safety guidance:

- Risk assessment first: robot behavior must be validated in the real cell/workspace before use.
- Motion limits: joint position, speed, and acceleration limits should be enforced continuously.
- Safe stopping behavior: operator-accessible stop/freeze paths should immediately halt unsafe motion.
- Boundary/interaction protection: reduce speed or block motion near hazards/collisions.

## Sources consulted

- OSHA Technical Manual - Industrial Robot Systems and Safety  
  <https://www.osha.gov/otm/section-4-safety-hazards/chapter-4>
- Universal Robots Safety Functions Table (examples of joint limits, speed limits, stop time/distance concepts)  
  <https://www.universal-robots.com/manuals/EN/HTML/SW10_12/Content/prod-usr-man/complianceUR5e/safetyFunctionsAndinterfaces/safety_functions_table1.htm>

## What is implemented in this codebase

- Per-joint hard limits from config/calibration.
- Per-joint step, velocity, and acceleration limiting in `safety_supervisor.py`.
- Soft-limit braking near joint boundaries to avoid slamming endpoints.
- Freeze/E-stop and home recovery controls.
- Dual-arm anti-collision guard using segment-to-segment proximity checks.
- Collision response that blocks or freezes motions that increase collision risk.

## Important

These controls reduce demo risk but do **not** replace a formal task-based risk assessment,
mechanical end-stops, hardware E-stop chain, and integration validation for real hardware use.
