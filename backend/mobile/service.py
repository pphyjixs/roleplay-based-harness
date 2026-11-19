"""AP1 pairing and outbound HTTPS rendezvous, with no public backend port."""
import asyncio
import base64
import gzip
import json
import logging
import ssl
import zlib
from urllib.parse import quote
from uuid import uuid4

import httpx

from backend.config import resolve_path
from .framing import ProtocolError
from .transport import PeerTransport

log = logging.getLogger("rpbh.mobile")


def pack(description):
    raw = json.dumps(description, separators=(",", ":")).encode()
    return "AP1." + base64.urlsafe_b64encode(gzip.compress(raw)).decode().rstrip("=")


def unpack(code):
    if not isinstance(code, str) or len(code) > 262144:
        raise ProtocolError("Invalid pairing code")
    code = "".join(code.split())
    if not code.startswith("AP1."):
        raise ProtocolError("Expected an AP1 pairing code")
    try:
        encoded = code[4:]
        raw = base64.b64decode(encoded + "=" * (-len(encoded) % 4), altchars=b"-_", validate=True)
        decoder = zlib.decompressobj(16 + zlib.MAX_WBITS)
        payload = decoder.decompress(raw, 262145)
        if len(payload) > 262144 or decoder.unconsumed_tail or not decoder.eof:
            raise ValueError()
        description = json.loads(payload)
        if (not isinstance(description, dict) or description.get("type") != "offer"
                or not isinstance(description.get("sdp"), str) or not description["sdp"].startswith("v=0")):
            raise ValueError()
        return {"type": "offer", "sdp": description["sdp"]}
    except (ValueError, TypeError, zlib.error, UnicodeError):
        raise ProtocolError("Invalid or oversized pairing code") from None


def read_json(path, default):
    file = resolve_path(path)
    return json.loads(file.read_text(encoding="utf-8-sig")) if file.is_file() else default


class MobileService:
    def __init__(self, app, settings):
        self.app, self.settings = app, settings
        self.enabled = bool(settings.get("enabled", False))
        self.peers = {}
        self.closing_peers = {}
        self.signal_task = None
        self.signal_online = False
        self.signal_error = None
        self.answer_lock = asyncio.Lock()

    def p2p_settings(self):
        return read_json(self.settings.get("p2p_file", "android/config/p2p.local.json"), {"ice_servers": [], "pairing_timeout_seconds": 120})

    async def start(self):
        if self.enabled:
            self.signal_task = asyncio.create_task(self.signaling(), name="RP mobile signaling")

    def status(self):
        return {"enabled": self.enabled, "protocol": 1, "signaling": {"online": self.signal_online, "error": self.signal_error},
                "peers": [{"id": key, "connection": item["pc"].connectionState, "ice": item["pc"].iceConnectionState}
                          for key, item in self.peers.items()]}

    async def answer(self, code):
        if not self.enabled:
            raise ProtocolError("Mobile transport is disabled")
        from aiortc import RTCConfiguration, RTCIceServer, RTCPeerConnection, RTCSessionDescription
        description = unpack(code)
        async with self.answer_lock:
            # The existing mobile app pairs one phone. Replace the prior transport
            # without touching persisted conversation IDs or acknowledged actions.
            await self.close_peers()
            config = self.p2p_settings()
            pc = RTCPeerConnection(RTCConfiguration(iceServers=[RTCIceServer(**item) for item in config.get("ice_servers", [])]))
            identity = uuid4().hex
            transport = PeerTransport(self.app, self.settings.get("transport", {}), lambda: self.drop(identity))
            self.peers[identity] = {"pc": pc, "transport": transport, "expiry": None}
            @pc.on("datachannel")
            def datachannel(channel):
                transport.attach(channel)
            @pc.on("connectionstatechange")
            async def changed():
                log.info("Mobile WebRTC %s", pc.connectionState)
                if pc.connectionState in {"failed", "closed"}:
                    await self.drop(identity)
            async def expire():
                await asyncio.sleep(config.get("pairing_timeout_seconds", 120))
                if pc.connectionState != "connected":
                    await self.drop(identity)
            self.peers[identity]["expiry"] = asyncio.create_task(expire())
            try:
                await pc.setRemoteDescription(RTCSessionDescription(**description))
                async with asyncio.timeout(self.settings.get("sdp_timeout_seconds", 30)):
                    await pc.setLocalDescription(await pc.createAnswer())
                return {"peer_id": identity, "code": pack({"type": pc.localDescription.type, "sdp": pc.localDescription.sdp})}
            except BaseException:
                await self.drop(identity)
                raise

    async def drop(self, identity):
        closing = self.closing_peers.get(identity)
        if closing is not None:
            if closing is not asyncio.current_task():
                await asyncio.shield(closing)
            return
        item = self.peers.pop(identity, None)
        if item:
            task = asyncio.create_task(self._drop(item))
            self.closing_peers[identity] = task
            task.add_done_callback(lambda completed: self.closing_peers.pop(identity, None))
            await asyncio.shield(task)

    async def _drop(self, item):
        current = asyncio.current_task()
        expiry = item["expiry"]
        if expiry and expiry is not current:
            expiry.cancel()
            await asyncio.gather(expiry, return_exceptions=True)
        await item["transport"].close()
        await item["pc"].close()

    async def close_peers(self):
        await asyncio.gather(*(self.drop(identity) for identity in set(self.peers) | set(self.closing_peers)))

    async def signaling(self):
        config = read_json(self.settings.get("connection_file", "android/config/connection.local.json"), {"enabled": False})
        if not config.get("enabled"):
            return
        try:
            address = config["signal_url"].rstrip("/")
            if not address.startswith("https://"):
                raise ValueError("Signaling requires HTTPS")
            ca_path = config.get("ca_file") or self.settings.get("ca_file", "android/config/connection-ca.pem")
            # The copied old ca_file was relative to its former project root.
            if not resolve_path(ca_path).is_file():
                ca_path = self.settings.get("ca_file", "android/config/connection-ca.pem")
            context = ssl.create_default_context(cafile=str(resolve_path(ca_path)))
            prefix = address + "/v1/rooms/" + quote(config["room"], safe="")
            headers = {"Authorization": "Bearer " + config["host_token"]}
            async with httpx.AsyncClient(verify=context, headers=headers, timeout=35, trust_env=False) as client:
                while True:
                    try:
                        response = await client.get(prefix + "/next")
                        response.raise_for_status()
                        self.signal_online, self.signal_error = True, None
                        if response.status_code == 204:
                            continue
                        request = response.json()
                        try:
                            answer = await self.answer(request["code"])
                            result = {"id": request["id"], "code": answer["code"]}
                        except Exception as error:
                            log.warning("Mobile pairing failed (%s)", type(error).__name__)
                            result = {"id": request["id"], "error": "RPBH pairing failed; regenerate connection request"}
                        reply = await client.post(prefix + "/answer", json=result)
                        if reply.status_code != 404:
                            reply.raise_for_status()
                    except (httpx.HTTPError, OSError, ValueError, KeyError) as error:
                        self.signal_online = False
                        self.signal_error = type(error).__name__
                        await asyncio.sleep(config.get("retry_seconds", 5))
        except asyncio.CancelledError:
            raise
        except Exception as error:
            self.signal_error = type(error).__name__
            log.warning("Mobile signaling unavailable (%s)", type(error).__name__)
        finally:
            self.signal_online = False

    async def close(self):
        if self.signal_task:
            self.signal_task.cancel()
            await asyncio.gather(self.signal_task, return_exceptions=True)
        await self.close_peers()
