"""
UDP receiver for external hand-tracking scripts (e.g. MediaPipe → SO-101 motor array).

Send UTF-8 JSON datagrams, one object per packet, e.g.:
  {"motors": [0, 10, -5, 0, 15, 45]}
Optional:
  {"motors": [...], "op": "track", "handedness": "right", "confidence": 0.92}
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Awaitable, Callable, Optional

log = logging.getLogger("hand_bridge")

ParseHandHandler = Callable[[dict, tuple], Awaitable[None]]


class HandUDPProtocol(asyncio.DatagramProtocol):
    def __init__(self, on_json: ParseHandHandler, loop: asyncio.AbstractEventLoop):
        self._on_json = on_json
        self._loop = loop

    def datagram_received(self, data: bytes, addr: tuple) -> None:
        text = data.decode("utf-8", errors="replace").strip()
        if not text:
            return
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as e:
            log.debug("bad json from %s: %s", addr, e)
            asyncio.run_coroutine_threadsafe(
                self._on_json({"_error": "json", "raw": text[:120]}, addr),
                self._loop,
            )
            return
        asyncio.run_coroutine_threadsafe(self._on_json(payload, addr), self._loop)


async def start_hand_udp(
    host: str,
    port: int,
    on_json: ParseHandHandler,
) -> Optional[asyncio.DatagramTransport]:
    loop = asyncio.get_running_loop()
    try:
        transport, _ = await loop.create_datagram_endpoint(
            lambda: HandUDPProtocol(on_json, loop),
            local_addr=(host, port),
        )
        log.info("Hand UDP listening on %s:%s", host, port)
        return transport
    except OSError as e:
        log.error("Hand UDP bind failed %s:%s — %s", host, port, e)
        return None
