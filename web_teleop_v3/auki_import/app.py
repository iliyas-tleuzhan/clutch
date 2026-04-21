import asyncio
import json
import sys
import time
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

BASE = Path(__file__).parent
CONFIG = json.loads((BASE / "config.json").read_text())

sys.path.insert(0, str(BASE))
from hand_bridge import start_hand_udp
from phone_mapper import PhoneMapper
from safety import SafetySupervisor, append_teleop_metric
from so101_ik import solve_ee_position
from voice_engine import VoiceEngine

HAND_UDP_HOST = "0.0.0.0"
HAND_UDP_PORT = int(__import__("os").environ.get("HAND_UDP_PORT", "5005"))

app = FastAPI(title="Clutch SO-101 demo server")
phone = PhoneMapper(CONFIG)
voice = VoiceEngine(CONFIG)
safety = SafetySupervisor(CONFIG)

motors: list[float] = [m["home"] for m in CONFIG["motors"]]
clients: set[WebSocket] = set()
last_imu_board_t = 0.0
imu_board_interval_s = 0.35
last_hand_ws_board_t = 0.0
hand_ws_board_interval_s = 0.35
hand_udp_transport: asyncio.DatagramTransport | None = None
last_hand_rx_mono: float = 0.0
hand_watchdog_task: asyncio.Task | None = None


def _ts() -> str:
    return datetime.now().strftime("%H:%M:%S")


async def broadcast(payload: dict) -> None:
    dead: set[WebSocket] = set()
    for ws in clients:
        try:
            await ws.send_json(payload)
        except Exception:
            dead.add(ws)
    clients.difference_update(dead)


async def board_line(text: str) -> None:
    await broadcast({"type": "board", "text": f"{_ts()}  {text}"})


async def push_motors(source: str, m: list[float], detail: str = "", log_board: bool = True) -> None:
    global motors
    motors = list(m)
    line = f"{source.upper():8}  " + " ".join(f"{x:6.1f}" for x in motors)
    if detail:
        line += f"  │  {detail}"
    if log_board:
        await board_line(line)
    await broadcast({"type": "motors", "motors": motors, "source": source})


def _try_ik_from_payload(payload: dict, seed: list[float]) -> tuple[list[float] | None, str]:
    if not payload.get("use_ik"):
        return None, ""
    em = payload.get("ee_m")
    if not isinstance(em, list) or len(em) != 3:
        return None, ""
    try:
        xyz = [float(em[0]), float(em[1]), float(em[2])]
    except (TypeError, ValueError):
        return None, ""
    sol, ok = solve_ee_position(xyz, seed)
    if not ok:
        return None, "ik_fail"
    grip = None
    md = payload.get("motors_deg")
    if isinstance(md, list) and len(md) >= 6:
        try:
            grip = float(md[5])
        except (TypeError, ValueError):
            pass
    if grip is None and payload.get("grip_deg") is not None:
        try:
            grip = float(payload["grip_deg"])
        except (TypeError, ValueError):
            grip = None
    if grip is None:
        grip = float(seed[5]) if len(seed) > 5 else 5.0
    sol[5] = grip
    return sol, "ik"


async def ingest_hand_payload(payload: dict, source_detail: str, log_board: bool = False) -> bool:
    """Resolve IK or joint targets, apply rate limits + clamp, broadcast. Returns False if nothing applied."""
    global motors, last_hand_rx_mono
    seed = list(motors)
    ik_sol, ik_tag = _try_ik_from_payload(payload, seed)
    raw = _extract_six_motors(payload)

    if ik_sol is not None:
        vals = ik_sol
        mode = "ik"
    elif raw is not None:
        vals = raw
        mode = "heuristic"
    else:
        return False

    last_hand_rx_mono = time.monotonic()
    filt = safety.filter_hand_command(vals)
    motors = filt
    detail = source_detail
    if ik_tag:
        detail += f"  {ik_tag}"
    detail += f"  mode={mode}"

    append_teleop_metric(
        BASE,
        {
            "t": time.time(),
            "mode": mode,
            "motors": motors,
            "ee_m": payload.get("ee_m") if mode == "ik" else None,
            "fsm": payload.get("teleop_fsm"),
            "jitter_rms": payload.get("teleop_jitter_rms"),
        },
    )
    await push_motors("hand", motors, detail, log_board=log_board)
    return True


def _extract_six_motors(payload: dict) -> list[float] | None:
    """
    iliyas-tleuzhan/clutch UDP JSON uses motors_deg (+ motor_1..6).
    Legacy / local scripts may send motors: [...].
    """
    md = payload.get("motors_deg")
    if isinstance(md, list) and len(md) == 6:
        try:
            return [float(x) for x in md]
        except (TypeError, ValueError):
            return None
    m = payload.get("motors")
    if isinstance(m, list) and len(m) == 6:
        try:
            return [float(x) for x in m]
        except (TypeError, ValueError):
            return None
    return None


async def handle_hand_payload(payload: dict, addr: tuple) -> None:
    if payload.get("_error") == "json":
        await board_line(f"HAND_UDP  BAD_JSON  {addr[0]}:{addr[1]}  {payload.get('raw', '')!r}")
        return

    parts = [f"UDP {addr[0]}:{addr[1]}"]
    if payload.get("source"):
        parts.append(f"src={payload['source']}")
    if "hand_detected" in payload:
        parts.append(f"hand_detected={payload['hand_detected']}")
    for key in ("op", "timestamp", "handedness", "confidence", "t"):
        if key in payload and payload[key] is not None:
            parts.append(f"{key}={payload[key]}")
    detail = "  ".join(parts)

    ok = await ingest_hand_payload(payload, detail, log_board=False)
    if not ok:
        await board_line(
            f"HAND_UDP  NO_VALID_HAND  from {addr[0]}:{addr[1]}  "
            f"(motors_deg[6] / motors[6], or use_ik + ee_m[3])"
        )


async def _hand_command_watchdog() -> None:
    """After hand_command_timeout with no packets, ease motors toward home."""
    global motors
    home = [m["home"] for m in CONFIG["motors"]]
    timeout = float((CONFIG.get("safety") or {}).get("hand_command_timeout_ms", 400)) / 1000.0
    while True:
        await asyncio.sleep(0.12)
        if last_hand_rx_mono <= 0:
            continue
        if time.monotonic() - last_hand_rx_mono < timeout:
            continue
        # stale: move slightly toward home until reset by new hand packet
        motors = [
            v + (h - v) * 0.08 for v, h in zip(motors, home)
        ]
        motors = safety.clamp(motors)
        await broadcast({"type": "motors", "motors": motors, "source": "system"})


@app.on_event("startup")
async def _startup() -> None:
    global hand_udp_transport, hand_watchdog_task
    hand_udp_transport = await start_hand_udp(
        HAND_UDP_HOST,
        HAND_UDP_PORT,
        handle_hand_payload,
    )
    hand_watchdog_task = asyncio.create_task(_hand_command_watchdog())
    if hand_udp_transport:
        await board_line(
            f"SYSTEM   Hand UDP :{HAND_UDP_PORT}  JSON: motors_deg[6] or use_ik+ee_m"
        )
    else:
        await board_line(f"SYSTEM   Hand UDP NOT bound (port {HAND_UDP_PORT} in use?) — use HTTP POST /api/hand")


@app.on_event("shutdown")
async def _shutdown() -> None:
    global hand_udp_transport, hand_watchdog_task
    if hand_watchdog_task:
        hand_watchdog_task.cancel()
        hand_watchdog_task = None
    if hand_udp_transport:
        hand_udp_transport.close()
        hand_udp_transport = None


@app.get("/")
async def landing() -> FileResponse:
    return FileResponse(BASE / "static" / "index.html")


@app.get("/demo")
async def demo() -> FileResponse:
    return FileResponse(BASE / "static" / "demo.html")


@app.get("/api/hand/status")
async def hand_status() -> dict:
    return {
        "hand_udp_port": HAND_UDP_PORT,
        "hand_udp_listening": hand_udp_transport is not None,
        "motors": motors,
    }


@app.post("/api/hand")
async def hand_http(payload: dict) -> dict:
    """Apply motors_deg / motors, or use_ik + ee_m (optional grip from motors_deg[5])."""
    meta = {k: payload[k] for k in ("op", "note", "handedness", "source", "hand_detected") if k in payload}
    detail = "HTTP /api/hand  " + "  ".join(f"{k}={v}" for k, v in meta.items())
    ok = await ingest_hand_payload(payload, detail, log_board=True)
    if not ok:
        return {"ok": False, "error": "need motors_deg or motors (6), or use_ik + ee_m[3]"}
    return {"ok": True, "motors": motors}


@app.post("/api/teleop/ik")
async def teleop_ik(payload: dict) -> dict:
    """Debug: IK-only solve; does not apply rate limits."""
    em = payload.get("ee_m") or payload.get("position_m")
    if not isinstance(em, list) or len(em) != 3:
        return {"ok": False, "error": "need ee_m: [x,y,z] meters"}
    seed = payload.get("motors_deg")
    if not isinstance(seed, list) or len(seed) < 5:
        seed = motors
    sol, ok = solve_ee_position([float(em[0]), float(em[1]), float(em[2])], seed)
    return {"ok": ok, "motors_deg": sol}


@app.post("/api/hand/test")
async def hand_test() -> dict:
    """One-shot fake pose so you can verify the letter board without a tracker."""
    safety.reset_hand_rate_state()
    test = [12.0, -8.0, 15.0, -20.0, 6.0, 55.0]
    await board_line("TEST     HTTP /api/hand/test  (built-in demo pose)")
    await push_motors("hand", safety.clamp(test), "op=test_pose")
    return {"ok": True, "motors": motors}


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket) -> None:
    global motors, last_imu_board_t, last_hand_ws_board_t, last_hand_rx_mono
    await ws.accept()
    clients.add(ws)
    await ws.send_json(
        {
            "type": "state",
            "motors": motors,
            "config": CONFIG,
            "hand_udp_port": HAND_UDP_PORT,
            "hand_udp_listening": hand_udp_transport is not None,
        }
    )
    await board_line(f"WS       client connected  ({len(clients)} open)")

    try:
        while True:
            data = await ws.receive_json()
            kind = data.get("type")

            if kind == "imu":
                last_hand_rx_mono = 0.0
                safety.reset_hand_rate_state()
                result = phone.compute(
                    data["alpha"],
                    data["beta"],
                    data["gamma"],
                    data.get("gripper_open", False),
                    data.get("clutch", False),
                )
                motors = safety.clamp(result)
                now = time.monotonic()
                if now - last_imu_board_t >= imu_board_interval_s:
                    last_imu_board_t = now
                    clutch = data.get("clutch", False)
                    go = data.get("gripper_open", False)
                    await board_line(
                        f"PHONE    imu α={data.get('alpha', 0):.0f} β={data.get('beta', 0):.0f} "
                        f"γ={data.get('gamma', 0):.0f}  clutch={clutch}  grip_open={go}"
                    )
                await broadcast({"type": "motors", "motors": motors, "source": "phone"})

            elif kind == "voice":
                last_hand_rx_mono = 0.0
                safety.reset_hand_rate_state()
                raw_text = (data.get("text") or "").strip()
                await board_line(f'VOICE    heard "{raw_text[:100]}"')
                result = voice.process(raw_text, motors)
                if result is not None:
                    motors = safety.clamp(result)
                    await board_line(
                        "VOICE    applied  "
                        + " ".join(f"{x:6.1f}" for x in motors)
                    )
                    await broadcast({"type": "motors", "motors": motors, "source": "voice"})
                else:
                    await board_line("VOICE    (no matching command)")

            elif kind == "hand":
                detail = data.get("detail") or "WebSocket type=hand"
                ok = await ingest_hand_payload(data, detail, log_board=False)
                if ok:
                    now = time.monotonic()
                    if now - last_hand_ws_board_t >= hand_ws_board_interval_s:
                        last_hand_ws_board_t = now
                        line = "HAND     " + " ".join(f"{x:6.1f}" for x in motors)
                        line += f"  │  {detail}"
                        await board_line(line)
                else:
                    await board_line("HAND_WS  need motors_deg[6] or use_ik+ee_m")

            elif kind == "home":
                last_hand_rx_mono = 0.0
                safety.reset_hand_rate_state()
                motors = [m["home"] for m in CONFIG["motors"]]
                await board_line("SYSTEM   GO HOME")
                await broadcast({"type": "motors", "motors": motors, "source": "system"})

            else:
                await board_line(f"WS       unknown type={kind!r}")

    except WebSocketDisconnect:
        clients.discard(ws)
        await board_line(f"WS       client disconnected  ({len(clients)} open)")


app.mount("/", StaticFiles(directory=str(BASE / "static")), name="static")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=3000, reload=True)
