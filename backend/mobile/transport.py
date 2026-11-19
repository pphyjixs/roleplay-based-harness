"""A restricted in-process HTTP / WebSocket transport over a WebRTC peer."""
import asyncio
import base64
import binascii
import json
import logging
import re
import time
from contextlib import suppress
from urllib.parse import unquote, urlsplit

import httpx

from .framing import Framing, ProtocolError

log = logging.getLogger("rpbh.mobile")
IDENTITY = re.compile(r"^[A-Za-z0-9_.:-]{1,160}$")
WS_PATH = re.compile(r"^/ws/(rp|asr)/[A-Za-z0-9_-]{1,160}$")


def safe_id(value):
    if not isinstance(value, str) or not IDENTITY.fullmatch(value):
        raise ProtocolError("Invalid request identity")
    return value


def safe_path(value, websocket=False):
    if not isinstance(value, str) or len(value) > 1024:
        raise ProtocolError("Invalid API path")
    parsed = urlsplit(value)
    decoded = unquote(value)
    if (parsed.scheme or parsed.netloc or parsed.query or parsed.fragment or "\\" in value
            or "\x00" in value or ".." in decoded.split("/") or decoded != value):
        raise ProtocolError("Only local canonical API paths are permitted")
    if websocket:
        allowed = WS_PATH.fullmatch(value)
    else:
        allowed = value == "/api/health" or (value.startswith("/api/rp/") and re.fullmatch(r"/[A-Za-z0-9_./:-]+", value))
    if not allowed:
        raise ProtocolError("API path is outside the mobile allowlist")
    return value


class SocketSession:
    def __init__(self, peer, identity, path):
        self.peer, self.identity, self.path = peer, identity, path
        self.incoming = asyncio.Queue(maxsize=peer.settings.get("ws_queue_messages", 128))
        self.queued_bytes = 0
        self.accepted = False
        self.closed = False
        self.task = None

    async def run(self):
        await self.incoming.put((0, {"type": "websocket.connect"}))
        scope = {"type": "websocket", "asgi": {"version": "3.0", "spec_version": "2.3"},
                 "scheme": "ws", "path": self.path, "raw_path": self.path.encode(), "query_string": b"",
                 "root_path": "", "headers": [(b"host", b"127.0.0.1"), (b"origin", b"http://127.0.0.1")],
                 "client": ("127.0.0.1", 0), "server": ("127.0.0.1", 8770), "subprotocols": [], "state": {}}
        async def receive():
            size, packet = await self.incoming.get()
            self.queued_bytes -= size
            self.peer.pending_ws_bytes -= size
            return packet
        async def send(packet):
            kind = packet["type"]
            if kind == "websocket.accept":
                self.accepted = True
                await self.peer.send({"type": "ws.opened", "id": self.identity})
            elif kind == "websocket.send":
                event = {"type": "ws.message", "id": self.identity}
                if packet.get("bytes") is not None:
                    event["data_base64"] = base64.b64encode(packet["bytes"]).decode()
                else:
                    event["text"] = packet.get("text", "")
                await self.peer.send(event)
            elif kind == "websocket.close":
                self.closed = True
                await self.peer.send({"type": "ws.closed", "id": self.identity, "code": packet.get("code", 1000),
                                      "reason": packet.get("reason", "")})
        try:
            await self.peer.app(scope, receive, send)
        except asyncio.CancelledError:
            raise
        except Exception as error:
            log.warning("Mobile WebSocket stopped (%s)", type(error).__name__)
        finally:
            self.peer.remember_closed_socket(self.identity)
            if not self.closed and not self.peer.closed:
                with suppress(Exception):
                    await self.peer.send({"type": "ws.closed", "id": self.identity, "code": 1000})
            self.peer.pending_ws_bytes -= self.queued_bytes
            self.queued_bytes = 0
            if self.peer.sockets.get(self.identity) is self:
                del self.peer.sockets[self.identity]

    def feed(self, packet):
        if not self.accepted:
            raise ProtocolError("WebSocket has not opened")
        text, encoded = packet.get("text"), packet.get("data_base64")
        if (text is None) == (encoded is None):
            raise ProtocolError("WebSocket frame needs exactly one payload")
        if text is not None:
            if not isinstance(text, str):
                raise ProtocolError("Invalid WebSocket text")
            size, message = len(text.encode()), {"type": "websocket.receive", "text": text}
        else:
            try:
                data = base64.b64decode(encoded, validate=True)
            except (ValueError, TypeError, binascii.Error):
                raise ProtocolError("Invalid WebSocket binary encoding") from None
            size, message = len(data), {"type": "websocket.receive", "bytes": data}
        if size > self.peer.settings.get("max_websocket_message_bytes", 9437184):
            raise ProtocolError("WebSocket frame exceeds application limit")
        if (self.incoming.full() or self.peer.pending_ws_bytes + size > self.peer.max_pending):
            raise ProtocolError("WebSocket input backpressure limit reached")
        self.queued_bytes += size
        self.peer.pending_ws_bytes += size
        self.incoming.put_nowait((size, message))

    async def close(self, code=1000):
        if self.task and not self.task.done():
            if not self.incoming.full():
                self.incoming.put_nowait((0, {"type": "websocket.disconnect", "code": code}))
            # The server receives disconnect first and can clean ASR/playback subscriptions.
            try:
                await asyncio.wait_for(asyncio.shield(self.task), 1)
            except asyncio.TimeoutError:
                self.task.cancel()
                await asyncio.gather(self.task, return_exceptions=True)


class PeerTransport:
    def __init__(self, app, settings, close_peer):
        self.app, self.settings, self.close_peer = app, settings, close_peer
        self.framing = Framing(settings)
        self.channels = {}
        self.sockets = {}
        self.closed_socket_ids = {}
        self.requests = {}
        self.pending_ws_bytes = 0
        # Includes HTTP responses and WebSocket events waiting behind the sender.
        self.pending_response_bytes = 0
        self.max_pending = settings.get("max_pending_bytes", 33554432)
        self.queued_bytes = 0
        self.queue = asyncio.Queue()
        self.closed = False
        self.close_task = None
        self.send_lock = asyncio.Lock()
        self.worker = asyncio.create_task(self.receive_loop())
        self.expiry = asyncio.create_task(self.expire_loop())
        self.http = httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1", timeout=60)

    def attach(self, channel):
        if channel.label not in {"rp-control", "rp-media"} or channel.label in self.channels:
            channel.close()
            return
        self.channels[channel.label] = channel
        @channel.on("message")
        def incoming(message):
            if self.closed:
                return
            size = len(message.encode()) if isinstance(message, str) else len(message)
            if not isinstance(message, str) or size > self.framing.chunk_bytes * 2 + 1024 or self.queued_bytes + size > self.max_pending:
                asyncio.create_task(self.close_peer())
                return
            self.queued_bytes += size
            self.queue.put_nowait((size, message))
        @channel.on("close")
        def closed():
            if not self.closed:
                asyncio.create_task(self.close_peer())

    async def expire_loop(self):
        while True:
            await asyncio.sleep(min(5, self.framing.ttl))
            self.framing.expire()
            self.expire_closed_sockets()

    def expire_closed_sockets(self):
        now = time.monotonic()
        for identity, deadline in tuple(self.closed_socket_ids.items()):
            if deadline <= now:
                del self.closed_socket_ids[identity]

    def remember_closed_socket(self, identity):
        self.expire_closed_sockets()
        self.closed_socket_ids.pop(identity, None)
        self.closed_socket_ids[identity] = time.monotonic() + self.settings.get("closed_socket_ttl_seconds", 120)
        while len(self.closed_socket_ids) > self.settings.get("closed_socket_ids", 128):
            del self.closed_socket_ids[next(iter(self.closed_socket_ids))]

    async def receive_loop(self):
        while True:
            size, message = await self.queue.get()
            self.queued_bytes -= size
            try:
                packet = self.framing.feed(message)
                if packet is not None:
                    await self.dispatch(packet)
            except (ProtocolError, ValueError, TypeError, KeyError):
                with suppress(Exception):
                    await self.send({"type": "rp.error", "message": "Invalid or over-limit mobile request"})
                asyncio.create_task(self.close_peer())
                return

    def reserve_response(self, size):
        if self.closed:
            raise ConnectionError("Mobile transport is closed")
        if self.pending_response_bytes + size > self.max_pending:
            raise BufferError("Mobile outbound response memory limit reached")
        self.pending_response_bytes += size
        return size

    async def send(self, packet, *, reserved_bytes=0):
        # HTTP reserves before allocating JSON/base64. Other small event producers
        # reserve immediately before their first await, so lock waiters stay bounded.
        data = json.dumps(packet, ensure_ascii=False, separators=(",", ":")).encode()
        if len(data) > self.framing.maximum:
            raise ProtocolError("Logical packet exceeds transport limit")
        if reserved_bytes:
            if len(data) > reserved_bytes:
                raise ProtocolError("Response exceeded its reserved memory budget")
            owned_reservation = 0
        else:
            owned_reservation = self.reserve_response(len(data))
        try:
            async with asyncio.timeout(self.settings.get("send_timeout_seconds", 30)):
                async with self.send_lock:
                    channel = self.channels.get("rp-control")
                    if self.closed or channel is None or channel.readyState != "open":
                        raise ConnectionError("Mobile channel is closed")
                    for frame in self.framing.iter_encoded(data):
                        while channel.bufferedAmount > self.settings.get("send_buffer_bytes", 1048576):
                            if self.closed or channel.readyState != "open":
                                raise ConnectionError("Mobile channel closed during send")
                            await asyncio.sleep(.01)
                        channel.send(frame)
        finally:
            self.pending_response_bytes -= owned_reservation

    async def dispatch(self, packet):
        kind = packet.get("type")
        identity = safe_id(packet.get("id"))
        if kind == "http.request":
            if identity in self.requests:
                raise ProtocolError("Request identity is already active")
            if len(self.requests) >= self.settings.get("max_http_requests", 16):
                await self.send({"type": "http.response", "id": identity, "status": 429, "body": {"detail": "Mobile request concurrency limit"}})
                return
            safe_path(packet.get("path"))
            if packet.get("method", "GET") not in {"GET", "POST", "PATCH", "DELETE"}:
                raise ProtocolError("Unsupported HTTP method")
            if packet.get("body") is not None and not isinstance(packet["body"], dict):
                raise ProtocolError("HTTP body must be an object")
            task = asyncio.create_task(self.request(packet))
            self.requests[identity] = task
            task.add_done_callback(lambda completed: self.requests.pop(identity, None))
        elif kind == "ws.open":
            path = safe_path(packet.get("path"), websocket=True)
            self.expire_closed_sockets()
            if identity in self.closed_socket_ids:
                await self.send({"type": "ws.closed", "id": identity, "code": 1000, "reason": "Socket was already closed"})
                return
            if identity in self.sockets or len(self.sockets) >= self.settings.get("max_websockets", 8):
                await self.send({"type": "ws.closed", "id": identity, "code": 1008, "reason": "WebSocket limit or duplicate identity"})
                return
            socket = SocketSession(self, identity, path)
            self.sockets[identity] = socket
            socket.task = asyncio.create_task(socket.run())
        elif kind == "ws.send":
            socket = self.sockets.get(identity)
            if socket is None:
                self.expire_closed_sockets()
                if identity in self.closed_socket_ids:
                    return  # A late media packet cannot tear down the RP channel.
                raise ProtocolError("Unknown WebSocket")
            socket.feed(packet)
        elif kind == "ws.close":
            code = packet.get("code", 1000)
            if type(code) is not int or not 1000 <= code <= 4999:
                raise ProtocolError("Invalid WebSocket close code")
            self.remember_closed_socket(identity)
            socket = self.sockets.get(identity)
            if socket:
                await socket.close(code)
            else:
                # control and media channels have independent ordering. Closing
                # while CONNECTING may reach us before that socket's open packet.
                await self.send({"type": "ws.closed", "id": identity, "code": code})
        else:
            raise ProtocolError("Unsupported mobile packet")

    async def request(self, packet):
        identity = packet["id"]
        reservation = 0
        response = result = None
        try:
            method = packet.get("method", "GET")
            if method not in {"GET", "POST", "PATCH", "DELETE"}:
                raise ProtocolError("Unsupported HTTP method")
            body = packet.get("body")
            if body is not None and not isinstance(body, dict):
                raise ProtocolError("HTTP body must be an object")
            # httpx's network timeouts do not constrain an in-process ASGI app.
            async with asyncio.timeout(self.settings.get("http_timeout_seconds", 60)):
                response = await self.http.request(method, packet["path"], json=body if body is not None else None)
            if len(response.content) > self.framing.maximum * 3 // 4 - 1024:
                result = {"type": "http.response", "id": identity, "status": 413, "body": {"detail": "Mobile response exceeds transport limit"}}
            else:
                # Reserve before retaining both decoded/base64 body and raw bytes.
                estimated = min(self.framing.maximum, (len(response.content) + 2) // 3 * 4 + 4096)
                reservation = self.reserve_response(estimated)
                result = {"type": "http.response", "id": identity, "status": response.status_code,
                          "headers": {key: value for key, value in response.headers.items() if key in {"content-type", "cache-control", "content-length"}}}
                if "application/json" in response.headers.get("content-type", ""):
                    result["body"] = response.json()
                else:
                    result["data_base64"] = base64.b64encode(response.content).decode()
            await self.send(result, reserved_bytes=reservation)
        except asyncio.CancelledError:
            raise
        except Exception as error:
            log.warning("Mobile HTTP stopped (%s)", type(error).__name__)
            # Drop large responses before waiting to send a small failure notice.
            response = result = None
            self.pending_response_bytes -= reservation
            reservation = 0
            status = 503 if isinstance(error, BufferError) else 504 if isinstance(error, TimeoutError) else 502
            with suppress(Exception):
                await self.send({"type": "http.response", "id": identity, "status": status, "body": {"detail": "Mobile API request failed or transport is busy"}})
        finally:
            self.pending_response_bytes -= reservation

    async def close(self):
        if self.close_task is None:
            self.close_task = asyncio.create_task(self._close())
        if self.close_task is not asyncio.current_task():
            await asyncio.shield(self.close_task)

    async def _close(self):
        if self.closed:
            return
        self.closed = True
        self.framing.clear()
        self.worker.cancel()
        self.expiry.cancel()
        requests = list(self.requests.values())
        for task in requests:
            task.cancel()
        await asyncio.gather(*(socket.close(1001) for socket in tuple(self.sockets.values())), return_exceptions=True)
        await asyncio.gather(self.worker, self.expiry, *requests, return_exceptions=True)
        self.requests.clear()
        self.closed_socket_ids.clear()
        self.channels.clear()
        while not self.queue.empty():
            self.queue.get_nowait()
        self.queued_bytes = 0
        await self.http.aclose()
