/**
 * Browser MediaPipe → SO-101 motors.
 * TeleopMapper matches upstream clutch hand_to_so101_positions.py (main branch):
 * https://github.com/iliyas-tleuzhan/clutch/blob/main/hand_to_so101_positions.py
 */
const MP_VER = "0.10.14";
const WASM_BASE = `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${MP_VER}/wasm`;
const MODEL_URL =
  "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task";

const HAND_EDGES = [
  [0, 1], [1, 2], [2, 3], [3, 4],
  [0, 5], [5, 6], [6, 7], [7, 8],
  [0, 9], [9, 10], [10, 11], [11, 12],
  [0, 13], [13, 14], [14, 15], [15, 16],
  [0, 17], [17, 18], [18, 19], [19, 20],
  [5, 9], [9, 13], [13, 17],
];

const DEFAULT_LIMITS_DEG = [
  [-160, 160], [-90, 90], [-120, 120], [-180, 180], [-90, 90], [0, 90],
];

/** Same default as upstream Python (0.25). */
const DEFAULT_SMOOTH_ALPHA = 0.25;

function clamp(v, lo, hi) {
  return Math.max(lo, Math.min(hi, v));
}

function lerp(a, b, t) {
  return a + (b - a) * t;
}

function mapUnitToRange(u, lo, hi) {
  return lerp(lo, hi, clamp(u, 0, 1));
}

function normalize01(value, lo, hi) {
  if (hi - lo < 1e-8) return 0.5;
  return clamp((value - lo) / (hi - lo), 0, 1);
}

function vecSub(a, b) {
  return [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
}

function norm3(v) {
  return Math.hypot(v[0], v[1], v[2]);
}

function angleBetween(v1, v2) {
  const n1 = norm3(v1);
  const n2 = norm3(v2);
  if (n1 < 1e-8 || n2 < 1e-8) return 0;
  let c = (v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2]) / (n1 * n2);
  c = clamp(c, -1, 1);
  return (Math.acos(c) * 180) / Math.PI;
}

/** Angle ABC in degrees (vertex B). */
function jointAngle(a, b, c) {
  return angleBetween(vecSub(a, b), vecSub(c, b));
}

class TeleopMapper {
  constructor(limitsDeg, smoothAlpha = DEFAULT_SMOOTH_ALPHA) {
    this.limitsDeg = limitsDeg;
    this.smoothAlpha = smoothAlpha;
    this.prev = null;
  }

  _flexFromFinger(lm, mcp, pip, dip) {
    const a = lm[mcp];
    const b = lm[pip];
    const c = lm[dip];
    const ang = jointAngle(a, b, c);
    return normalize01(180 - ang, 0, 100);
  }

  computeUnit(lm) {
    const wrist = lm[0];
    const indexMcp = lm[5];
    const middleMcp = lm[9];
    const pinkyMcp = lm[17];

    const palmCenter = [
      (wrist[0] + indexMcp[0] + middleMcp[0] + pinkyMcp[0]) / 4,
      (wrist[1] + indexMcp[1] + middleMcp[1] + pinkyMcp[1]) / 4,
      (wrist[2] + indexMcp[2] + middleMcp[2] + pinkyMcp[2]) / 4,
    ];

    const m1 = clamp(palmCenter[0], 0, 1);
    const m2 = clamp(1 - palmCenter[1], 0, 1);

    const zClose = -middleMcp[2];
    const m3 = normalize01(zClose, -0.1, 0.2);

    const palmVecX = pinkyMcp[0] - indexMcp[0];
    const palmVecY = pinkyMcp[1] - indexMcp[1];
    const roll = (Math.atan2(palmVecY, palmVecX) * 180) / Math.PI;
    const m4 = normalize01(roll, -90, 90);

    const handVecX = middleMcp[0] - wrist[0];
    const handVecY = middleMcp[1] - wrist[1];
    const pitch = (Math.atan2(-handVecY, handVecX) * 180) / Math.PI;
    const m5 = normalize01(pitch, -120, 60);

    const iFlex = this._flexFromFinger(lm, 5, 6, 7);
    const mFlex = this._flexFromFinger(lm, 9, 10, 11);
    const rFlex = this._flexFromFinger(lm, 13, 14, 15);
    const pFlex = this._flexFromFinger(lm, 17, 18, 19);
    const m6 = clamp((iFlex + mFlex + rFlex + pFlex) / 4, 0, 1);

    return [m1, m2, m3, m4, m5, m6];
  }

  unitToDegrees(u) {
    return u.map((value, idx) => {
      const [lo, hi] = this.limitsDeg[idx];
      return mapUnitToRange(value, lo, hi);
    });
  }

  smooth(values) {
    if (this.prev === null) {
      this.prev = values.slice();
      return values;
    }
    const smoothed = this.prev.map((p, i) => lerp(p, values[i], this.smoothAlpha));
    this.prev = smoothed;
    return smoothed;
  }

  mapLandmarks(lm) {
    const unit = this.computeUnit(lm);
    const deg = this.unitToDegrees(unit);
    return this.smooth(deg);
  }
}

function getLimitsFromPage() {
  const L = window.__demoMotorLimits;
  if (Array.isArray(L) && L.length === 6) return L.map((p) => [Number(p[0]), Number(p[1])]);
  return DEFAULT_LIMITS_DEG.map((p) => [...p]);
}

function landmarksNormArray(landmarks) {
  return landmarks.map((p) => [p.x, p.y, p.z ?? 0]);
}

function drawLandmarks(landmarks, w, h) {
  if (!ctx || !landmarks || !landmarks.length) return;
  ctx.clearRect(0, 0, w, h);
  ctx.strokeStyle = "rgba(109, 250, 170, 0.85)";
  ctx.lineWidth = 2;
  for (const [a, b] of HAND_EDGES) {
    const pa = landmarks[a];
    const pb = landmarks[b];
    if (!pa || !pb) continue;
    ctx.beginPath();
    ctx.moveTo(pa.x * w, pa.y * h);
    ctx.lineTo(pb.x * w, pb.y * h);
    ctx.stroke();
  }
  ctx.fillStyle = "rgba(255,255,255,0.9)";
  for (const p of landmarks) {
    ctx.beginPath();
    ctx.arc(p.x * w, p.y * h, 3, 0, Math.PI * 2);
    ctx.fill();
  }
}

function sendHand(deg, handDetected) {
  const fn = window.__demoSendMsg;
  if (typeof fn !== "function") return;
  fn({
    type: "hand",
    motors_deg: deg,
    hand_detected: handDetected,
  });
}

let landmarker = null;
let videoEl = null;
let canvasEl = null;
let ctx = null;
let stream = null;
let rafId = null;
let mapper = null;
let lastSendMs = 0;
let frameCount = 0;
let fpsLast = performance.now();
let fpsDisplay = 0;
const SEND_INTERVAL_MS = 33;

function resetTrackingState() {
  mapper = null;
}

function tick() {
  rafId = requestAnimationFrame(tick);
  if (!landmarker || !videoEl || videoEl.readyState < 2) return;

  const vw = videoEl.videoWidth;
  const vh = videoEl.videoHeight;
  if (canvasEl && (canvasEl.width !== vw || canvasEl.height !== vh)) {
    canvasEl.width = vw;
    canvasEl.height = vh;
  }

  const now = performance.now();
  frameCount += 1;
  if (now - fpsLast >= 1000) {
    fpsDisplay = frameCount;
    frameCount = 0;
    fpsLast = now;
    const fpsEl = document.getElementById("bh-fps");
    if (fpsEl) fpsEl.textContent = `${fpsDisplay} fps`;
  }

  let result;
  try {
    result = landmarker.detectForVideo(videoEl, now);
  } catch (e) {
    console.warn("[hand-browser] detectForVideo", e);
    return;
  }

  const marks = result?.landmarks?.[0];
  const w = canvasEl?.width || vw;
  const h = canvasEl?.height || vh;

  const det = document.getElementById("bh-detect");
  const rl = document.getElementById("bh-robot-line");

  if (marks && marks.length >= 21) {
    drawLandmarks(marks, w, h);
    if (!mapper) mapper = new TeleopMapper(getLimitsFromPage(), DEFAULT_SMOOTH_ALPHA);
    else mapper.limitsDeg = getLimitsFromPage();

    const lm = landmarksNormArray(marks);
    const deg = mapper.mapLandmarks(lm);

    if (det) det.textContent = "tracking";
    if (rl) {
      const hasMesh = window.__so101Robot && typeof window.syncUrdfMotors === "function";
      rl.textContent = hasMesh
        ? "Twin: upstream TeleopMapper (clutch main)"
        : "Mesh loading…";
    }

    const apply = window.__applyHandMotorsToRobot;
    if (typeof apply === "function") apply(deg);

    if (now - lastSendMs >= SEND_INTERVAL_MS) {
      lastSendMs = now;
      sendHand(deg, true);
    }
  } else {
    if (ctx && w && h) ctx.clearRect(0, 0, w, h);
    if (det) det.textContent = "no hand";
    if (rl && landmarker) rl.textContent = "Show hand — palm toward camera";
  }
}

export async function startBrowserHand() {
  videoEl = document.getElementById("browser-hand-video");
  canvasEl = document.getElementById("browser-hand-canvas");
  if (!videoEl || !canvasEl) {
    console.warn("[hand-browser] missing video/canvas");
    throw new Error("Missing video/canvas elements");
  }
  ctx = canvasEl.getContext("2d");

  const rl0 = document.getElementById("bh-robot-line");
  if (rl0) rl0.textContent = "Loading MediaPipe hand model…";

  if (typeof window.__demoSendMsg !== "function") {
    const ab = window.__demoAppendBoard;
    if (typeof ab === "function") ab(new Date().toISOString().slice(11, 23) + "  LOCAL    WebSocket not ready", "warn");
    if (rl0) rl0.textContent = "Need WebSocket — open this page from the demo server (not file://)";
    throw new Error("WebSocket not ready");
  }

  const { HandLandmarker, FilesetResolver } = await import(
    `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${MP_VER}/+esm`
  );

  const fileset = await FilesetResolver.forVisionTasks(WASM_BASE);
  const baseOpts = { modelAssetPath: MODEL_URL };
  /** Match upstream Python mediapipe Hands defaults (0.6 / 0.6). */
  const taskOpts = {
    baseOptions: baseOpts,
    runningMode: "VIDEO",
    numHands: 1,
    minHandDetectionConfidence: 0.6,
    minHandPresenceConfidence: 0.6,
    minTrackingConfidence: 0.6,
  };
  try {
    landmarker = await HandLandmarker.createFromOptions(fileset, {
      ...taskOpts,
      baseOptions: { ...baseOpts, delegate: "GPU" },
    });
  } catch (e) {
    console.warn("[hand-browser] GPU delegate failed, using CPU", e);
    landmarker = await HandLandmarker.createFromOptions(fileset, taskOpts);
  }

  if (!navigator.mediaDevices?.getUserMedia) {
    const msg = "Camera API unavailable (use HTTPS or localhost)";
    if (rl0) rl0.textContent = msg;
    throw new Error(msg);
  }

  stream = await navigator.mediaDevices.getUserMedia({
    video: {
      facingMode: "user",
      width: { ideal: 960 },
      height: { ideal: 540 },
      frameRate: { ideal: 30, max: 30 },
    },
    audio: false,
  });
  videoEl.srcObject = stream;
  await videoEl.play();

  resetTrackingState();
  mapper = new TeleopMapper(getLimitsFromPage(), DEFAULT_SMOOTH_ALPHA);
  lastSendMs = 0;

  const ab = window.__demoAppendBoard;
  if (typeof ab === "function") {
    ab(
      new Date().toTimeString().slice(0, 8) +
        "." +
        String(Date.now() % 1000).padStart(3, "0") +
        "  LOCAL    Browser hand: upstream clutch TeleopMapper (smooth_alpha=0.25)",
    );
  }

  cancelAnimationFrame(rafId);
  const chrome = window.__setHandTrackingActive;
  if (typeof chrome === "function") chrome(true);
  tick();
}

export function stopBrowserHand() {
  const chrome = window.__setHandTrackingActive;
  if (typeof chrome === "function") chrome(false);

  cancelAnimationFrame(rafId);
  rafId = null;
  landmarker = null;
  mapper = null;
  resetTrackingState();

  if (stream) {
    stream.getTracks().forEach((t) => t.stop());
    stream = null;
  }
  if (videoEl) {
    videoEl.srcObject = null;
  }
  if (ctx && canvasEl) {
    ctx.clearRect(0, 0, canvasEl.width, canvasEl.height);
  }

  const det = document.getElementById("bh-detect");
  if (det) det.textContent = "stopped";
  const fpsEl = document.getElementById("bh-fps");
  if (fpsEl) fpsEl.textContent = "—";
  const rl = document.getElementById("bh-robot-line");
  if (rl) rl.textContent = "URDF twin: idle — tap Camera";

  const ab = window.__demoAppendBoard;
  if (typeof ab === "function") {
    ab(
      new Date().toTimeString().slice(0, 8) +
        "." +
        String(Date.now() % 1000).padStart(3, "0") +
        "  LOCAL    Browser hand tracking stopped",
    );
  }
}

function wireHandPanel() {
  const toggle = document.getElementById("bh-toggle");
  if (!toggle) return;
  toggle.addEventListener("click", (e) => {
    e.stopPropagation();
    if (landmarker) {
      stopBrowserHand();
      return;
    }
    toggle.disabled = true;
    toggle.textContent = "Starting…";
    startBrowserHand().catch((err) => {
      console.error(err);
      const reset = window.__setHandTrackingActive;
      if (typeof reset === "function") reset(false);
      toggle.disabled = false;
      toggle.textContent = "Camera";
      const ab = window.__demoAppendBoard;
      if (typeof ab === "function") ab(String(err?.message || err), "warn");
      const rl = document.getElementById("bh-robot-line");
      if (rl) rl.textContent = "Error: " + (err?.message || err) + " — check console / use Chrome";
    });
  });
}

if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", wireHandPanel);
} else {
  wireHandPanel();
}
