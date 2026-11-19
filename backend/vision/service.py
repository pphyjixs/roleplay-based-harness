"""Bounded in-memory images for one RP request; this module never writes files.

DeepSeek Responses vision uses input_image blocks:
https://api-docs.deepseek.com/zh-cn/guides/vision/
"""

from __future__ import annotations

import asyncio
import base64
import binascii
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from io import BytesIO
import hashlib
import re
import time
from uuid import uuid4
import warnings
import math


class ImageInputError(ValueError):
    """A safe, payload-free error suitable for the local API."""


@dataclass(frozen=True)
class PreparedImage:
    image_id: str
    captured_at: str
    source: str
    revision: int | None
    mime: str
    width: int
    height: int
    detail: str
    data: bytes = field(repr=False)
    digest: str = field(init=False)

    def __post_init__(self):
        object.__setattr__(self, "digest", hashlib.sha256(self.data).hexdigest())

    def metadata(self):
        return {"image_id": self.image_id, "captured_at": self.captured_at,
                "source": self.source, "revision": self.revision, "mime": self.mime,
                "width": self.width, "height": self.height, "size_bytes": len(self.data), "digest": self.digest}

    def model_block(self):
        return {"type": "input_image", "image_url": "data:" + self.mime + ";base64," + base64.b64encode(self.data).decode("ascii"),
                "detail": self.detail}


def _positive(settings, key, default):
    value = settings.get(key, default)
    if type(value) is not int or value <= 0:
        raise ImageInputError(f"Invalid image setting: {key}")
    return value


def normalize_image(packet: dict, settings: dict | None = None) -> PreparedImage:
    """Decode, validate, orient and encode an image within configured byte limits."""
    settings = settings or {}
    if not isinstance(packet, dict):
        raise ImageInputError("Image input must be an object")
    image_id = packet.get("image_id") or "img_" + uuid4().hex
    if not isinstance(image_id, str) or re.fullmatch(r"[A-Za-z0-9_-]{1,160}", image_id) is None:
        raise ImageInputError("Invalid image ID")
    source = packet.get("source", "upload")
    if source not in {"upload", "camera", "speech_end"}:
        raise ImageInputError("Invalid image source")
    revision = packet.get("revision")
    if revision is not None and (type(revision) is not int or revision < 0):
        raise ImageInputError("Invalid image revision")
    captured_at = packet.get("captured_at") or datetime.now(timezone.utc).isoformat()
    if not isinstance(captured_at, str) or len(captured_at) > 64:
        raise ImageInputError("Invalid image capture time")
    try:
        timestamp = datetime.fromisoformat(captured_at.replace("Z", "+00:00"))
        if timestamp.tzinfo is None:
            raise ValueError("Timezone is required")
    except ValueError:
        raise ImageInputError("Invalid image capture time") from None
    data_url = packet.get("data_url")
    if not isinstance(data_url, str):
        raise ImageInputError("Image data_url must be a string")
    maximum_input = _positive(settings, "max_input_bytes_per_image", 2 * 1024 * 1024)
    if len(data_url) > (maximum_input + 2) // 3 * 4 + 128:
        raise ImageInputError("Image input exceeds the byte limit")
    match = re.fullmatch(r"data:image/(?:jpeg|jpg|png|webp|gif);base64,([A-Za-z0-9+/]*={0,2})", data_url, re.ASCII)
    if match is None:
        raise ImageInputError("Only base64 JPEG, PNG, WebP or GIF images are accepted")
    try:
        raw = base64.b64decode(match.group(1), validate=True)
    except (ValueError, binascii.Error):
        raise ImageInputError("Invalid image encoding") from None
    if not raw or len(raw) > maximum_input:
        raise ImageInputError("Image input exceeds the byte limit")
    policy = settings.get("camera" if source != "upload" else "upload", {})
    longest = _positive(policy, "max_dimension", 960 if source != "upload" else 1600)
    maximum_output = _positive(policy, "max_bytes", 192 * 1024 if source != "upload" else 512 * 1024)
    quality = policy.get("quality", 75 if source != "upload" else 85)
    minimum_quality = policy.get("min_quality", 50)
    if type(quality) is not int or type(minimum_quality) is not int or not 1 <= minimum_quality <= quality <= 100:
        raise ImageInputError("Invalid image encoding quality")
    quality_step = _positive(policy, "quality_step", 10)
    resize_factor = policy.get("resize_factor", .8)
    if type(resize_factor) not in (float, int) or not math.isfinite(resize_factor) or not 0 < resize_factor < 1:
        raise ImageInputError("Invalid image resize factor")
    webp_method = policy.get("webp_method", 2)
    if type(webp_method) is not int or not 0 <= webp_method <= 6:
        raise ImageInputError("Invalid WebP encoding method")
    attempts = _positive(settings.get("processing", {}), "max_encode_attempts", 12)
    output_format = policy.get("format", "webp").lower().removeprefix("image/")
    if output_format not in {"webp", "jpeg", "png"}:
        raise ImageInputError("Invalid image output format")
    detail = policy.get("detail", "low" if source != "upload" else "original")
    if detail not in {"low", "high", "original", "auto"}:
        raise ImageInputError("Invalid image detail level")
    maximum_pixels = _positive(settings, "max_input_pixels", 16_000_000)
    # Pillow is imported lazily so text-only operation has no codec dependency.
    from PIL import Image, ImageOps, UnidentifiedImageError
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(BytesIO(raw)) as opened:
                if opened.format not in {"JPEG", "PNG", "WEBP", "GIF"}:
                    raise ImageInputError("Unsupported image file content")
                if opened.width * opened.height > maximum_pixels:
                    raise ImageInputError("Image exceeds the decoded pixel limit")
                # A video sample is a still image, never an animation payload.
                opened.seek(0)
                image = ImageOps.exif_transpose(opened).convert("RGBA")
                background = Image.new("RGB", image.size, "white")
                background.paste(image, mask=image.getchannel("A"))
                image.close()
                image = background
                image.thumbnail((longest, longest), Image.Resampling.LANCZOS)
                minimum_dimension = _positive(policy, "min_dimension", 320)
                minimum_dimension = min(minimum_dimension, max(image.size))
                for attempt in range(attempts):
                    output = BytesIO()
                    options = {}
                    if output_format == "webp":
                        options = {"quality": quality, "method": webp_method, "lossless": bool(policy.get("lossless", False))}
                    elif output_format == "jpeg":
                        options = {"quality": quality, "optimize": False}
                    image.save(output, format=output_format.upper(), **options)
                    encoded = output.getvalue()
                    if len(encoded) <= maximum_output:
                        width, height = image.size
                        image.close()
                        return PreparedImage(image_id, timestamp.astimezone(timezone.utc).isoformat(), source, revision,
                                             "image/" + output_format, width, height, detail, encoded)
                    if output_format != "png" and not policy.get("lossless", False) and quality > minimum_quality:
                        quality = max(minimum_quality, quality - quality_step)
                    elif max(image.size) > minimum_dimension:
                        next_dimension = max(minimum_dimension, int(max(image.size) * resize_factor))
                        image.thumbnail((next_dimension, next_dimension), Image.Resampling.LANCZOS)
                    else:
                        image.close()
                        raise ImageInputError("Image cannot fit the configured output byte limit")
                image.close()
                raise ImageInputError("Image encoding exceeded the configured attempt limit")
    except ImageInputError:
        raise
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError, Image.DecompressionBombError, Image.DecompressionBombWarning):
        raise ImageInputError("Invalid or oversized image file") from None


def image_blocks(images):
    """Build ephemeral model blocks; callers must not store these in traces."""
    return [image.model_block() for image in images]


class TransientImages:
    """A bounded cache with atomic multi-image ownership and deterministic cleanup.

    add/add_async normalizes a packet. take reserves all requested images for one
    model turn (no partial reservation), and release removes pixels immediately.
    Reserved entries still count towards limits and never get evicted underneath
    active inference. Clear/expiry only affect entries not reserved by a turn.
    """

    def __init__(self, settings=None, *, clock=time.monotonic):
        self.settings = settings or {}
        cache = self.settings.get("cache", {})
        self.max_bytes = _positive(cache, "max_bytes", 16 * 1024 * 1024)
        self.max_images = _positive(cache, "max_images", 32)
        self.ttl = _positive(cache, "ttl_seconds", 120)
        self.camera_frames = _positive(cache, "camera_frames", 3)
        self.max_per_turn = _positive(self.settings, "max_images_per_turn", 3)
        processing = self.settings.get("processing", {})
        self.max_processing_jobs = _positive(processing, "max_jobs", 2)
        self.processing_timeout = processing.get("timeout_seconds", 3)
        if (type(self.processing_timeout) not in (int, float) or not math.isfinite(self.processing_timeout)
                or self.processing_timeout <= 0):
            raise ImageInputError("Invalid image processing timeout")
        self.processing_slots = asyncio.Semaphore(self.max_processing_jobs)
        self.processing_jobs = set()
        self.clock = clock
        self.entries = OrderedDict()
        self.used_bytes = 0

    def add(self, conversation_id, packet):
        return self.add_prepared(conversation_id, normalize_image(packet, self.settings))

    async def add_async(self, conversation_id, packet):
        # Refuse excess work rather than queueing arbitrary base64 payloads in
        # memory. An abandoned to_thread call holds its slot until it truly ends.
        if self.processing_slots.locked():
            raise ImageInputError("Image processing is busy; please try again")
        await self.processing_slots.acquire()
        try:
            task = asyncio.create_task(asyncio.to_thread(normalize_image, packet, self.settings))
        except BaseException:
            self.processing_slots.release()
            raise
        self.processing_jobs.add(task)

        def completed(job):
            self.processing_jobs.discard(job)
            self.processing_slots.release()
            if not job.cancelled():
                job.exception()  # Retrieve failures even after the caller left.

        task.add_done_callback(completed)
        try:
            prepared = await asyncio.wait_for(asyncio.shield(task), self.processing_timeout)
        except TimeoutError:
            raise ImageInputError("Image processing timed out; please try a smaller image") from None
        return self.add_prepared(conversation_id, prepared)

    def add_prepared(self, conversation_id, image: PreparedImage):
        self.expire()
        existing = self.entries.get(image.image_id)
        if existing:
            if existing["conversation_id"] != conversation_id or existing["image"] != image:
                raise ImageInputError("Image ID was already used for different content")
            return image.metadata()
        size = len(image.data)
        if size > self.max_bytes:
            raise ImageInputError("Image exceeds the temporary cache limit")
        while self.used_bytes + size > self.max_bytes or len(self.entries) >= self.max_images:
            disposable = next((key for key, value in self.entries.items() if not value["reserved"]), None)
            if disposable is None:
                raise ImageInputError("The temporary image cache is busy")
            self._remove(disposable)
        self.entries[image.image_id] = {"image": image, "conversation_id": conversation_id,
                                       "expires": self.clock() + self.ttl, "reserved": False}
        self.used_bytes += size
        if image.source in {"camera", "speech_end"}:
            buffered = [key for key, value in self.entries.items()
                        if value["conversation_id"] == conversation_id and not value["reserved"]
                        and value["image"].source in {"camera", "speech_end"}]
            for key in buffered[:-self.camera_frames]:
                self._remove(key)
        return image.metadata()

    def take(self, image_ids, conversation_id, revision=None):
        self.expire()
        ids = list(image_ids)
        if len(ids) > self.max_per_turn or len(set(ids)) != len(ids):
            raise ImageInputError("Invalid image count for a turn")
        entries = []
        for image_id in ids:
            entry = self.entries.get(image_id)
            if entry is None or entry["conversation_id"] != conversation_id:
                raise ImageInputError("The temporary image is unavailable; please send it again")
            if entry["reserved"]:
                raise ImageInputError("The temporary image already belongs to an admitted turn")
            image = entry["image"]
            if revision is not None and image.revision is not None and image.revision != revision:
                raise ImageInputError("Image belongs to an outdated speech revision")
            entries.append(entry)
        for entry in entries:
            entry["reserved"] = True
        return [entry["image"] for entry in entries]

    def release(self, image_ids):
        for image_id in image_ids:
            self._remove(image_id)

    def clear(self, conversation_id=None, *, include_reserved=False):
        for image_id, entry in list(self.entries.items()):
            if (conversation_id is None or entry["conversation_id"] == conversation_id) and (include_reserved or not entry["reserved"]):
                self._remove(image_id)

    def expire(self):
        now = self.clock()
        for image_id, entry in list(self.entries.items()):
            if not entry["reserved"] and entry["expires"] <= now:
                self._remove(image_id)

    def _remove(self, image_id):
        entry = self.entries.pop(image_id, None)
        if entry:
            self.used_bytes -= len(entry["image"].data)

    def status(self):
        self.expire()
        return {"images": len(self.entries), "bytes": self.used_bytes,
                "reserved": sum(bool(entry["reserved"]) for entry in self.entries.values())}
