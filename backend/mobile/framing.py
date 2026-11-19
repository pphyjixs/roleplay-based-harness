"""Bounded UTF-8 fragmentation shared by native Android and aiortc."""
import base64
import binascii
import json
import math
import time
from uuid import uuid4


class ProtocolError(ValueError):
    pass


class Framing:
    def __init__(self, settings=None):
        settings = settings or {}
        self.chunk_bytes = settings.get("chunk_bytes", 12288)
        self.maximum = settings.get("max_message_bytes", 25165824)
        self.max_transfers = settings.get("max_transfers", 8)
        self.aggregate_max = settings.get("max_pending_bytes", 33554432)
        self.ttl = settings.get("fragment_timeout_seconds", 30)
        self.transfers = {}
        self.retained_bytes = 0

    def expire(self):
        now = time.monotonic()
        for key, item in tuple(self.transfers.items()):
            if item["deadline"] <= now:
                self.retained_bytes -= item["size"]
                del self.transfers[key]

    def clear(self):
        self.transfers.clear()
        self.retained_bytes = 0

    def encode(self, packet):
        data = json.dumps(packet, ensure_ascii=False, separators=(",", ":")).encode()
        return list(self.iter_encoded(data))

    def iter_encoded(self, data):
        """Generate one wire frame at a time; never duplicate a complete WAV train."""
        if len(data) > self.maximum:
            raise ProtocolError("Logical packet exceeds transport limit")
        if len(data) <= self.chunk_bytes:
            yield data.decode()
            return
        identity = uuid4().hex
        total = math.ceil(len(data) / self.chunk_bytes)
        for index, offset in enumerate(range(0, len(data), self.chunk_bytes)):
            yield json.dumps({"type": "rp.fragment", "transfer_id": identity, "index": index,
                              "total": total, "data": base64.b64encode(data[offset:offset+self.chunk_bytes]).decode()},
                             separators=(",", ":"))

    def feed(self, text):
        self.expire()
        if not isinstance(text, str) or len(text.encode()) > self.chunk_bytes * 2 + 1024:
            raise ProtocolError("Invalid transport frame")
        try:
            packet = json.loads(text)
        except (ValueError, RecursionError):
            raise ProtocolError("Invalid JSON frame") from None
        if not isinstance(packet, dict):
            raise ProtocolError("Expected an object")
        if packet.get("type") != "rp.fragment":
            if len(text.encode()) > self.chunk_bytes:
                raise ProtocolError("Unfragmented packet exceeds frame limit")
            return packet
        key, index, total, encoded = (packet.get(field) for field in ("transfer_id", "index", "total", "data"))
        if (not isinstance(key, str) or not 1 <= len(key) <= 80 or type(index) is not int
                or type(total) is not int or not 1 <= total <= math.ceil(self.maximum / self.chunk_bytes)
                or not 0 <= index < total or not isinstance(encoded, str)
                or len(encoded) > math.ceil(self.chunk_bytes / 3) * 4):
            raise ProtocolError("Invalid fragment metadata")
        try:
            part = base64.b64decode(encoded, validate=True)
        except (ValueError, binascii.Error):
            raise ProtocolError("Invalid fragment encoding") from None
        if not part or len(part) > self.chunk_bytes:
            raise ProtocolError("Invalid fragment size")
        if index == 0:
            if key in self.transfers or len(self.transfers) >= self.max_transfers:
                raise ProtocolError("Too many simultaneous fragment transfers")
            self.transfers[key] = {"parts": [], "total": total, "size": 0, "deadline": time.monotonic() + self.ttl}
        item = self.transfers.get(key)
        if item is None or total != item["total"] or index != len(item["parts"]):
            raise ProtocolError("Unexpected fragment sequence")
        item["size"] += len(part)
        self.retained_bytes += len(part)
        if item["size"] > self.maximum or self.retained_bytes > self.aggregate_max:
            self.clear()
            raise ProtocolError("Fragment memory limit exceeded")
        item["parts"].append(part)
        if index != total - 1:
            return None
        data = b"".join(item["parts"])
        self.retained_bytes -= item["size"]
        del self.transfers[key]
        try:
            result = json.loads(data.decode("utf-8"))
        except (ValueError, UnicodeError, RecursionError):
            raise ProtocolError("Invalid fragmented JSON") from None
        if not isinstance(result, dict) or result.get("type") == "rp.fragment":
            raise ProtocolError("Invalid nested logical packet")
        return result
