"""Server-side Tencent streaming ASR and orchestration voice input bridge.

Wire protocol follows Tencent's official realtime API:
https://cloud.tencent.com/document/product/1093/48982
Signed URLs and cloud credentials never cross the application websocket.
"""

import array
import asyncio
import base64
from contextlib import asynccontextmanager, suppress
import hashlib
import hmac
import json
import math
import secrets
import sys
import time
from urllib.parse import urlencode, urlsplit
from uuid import uuid4

from fastapi import WebSocketDisconnect

from backend.config import resolve_path
from backend.speech.turns import VoiceTurnController


def signed_url(credentials, engine, *, timestamp=None, nonce=None, voice_id=None,
               endpoint="wss://asr.cloud.tencent.com/asr/v2"):
    timestamp = int(time.time()) if timestamp is None else timestamp
    parameters = {
        "secretid": credentials["secret_id"], "timestamp": timestamp, "expired": timestamp + 300,
        "nonce": secrets.randbelow(2**31 - 1) + 1 if nonce is None else nonce,
        "engine_model_type": engine, "voice_id": voice_id or uuid4().hex,
        "voice_format": 1, "needvad": 1, "filter_empty_result": 1,
    }
    pairs = sorted(parameters.items())
    address = urlsplit(endpoint.rstrip("/") + "/" + str(credentials["app_id"]))
    if address.scheme != "wss" or address.username or address.password or address.query:
        raise ValueError("Invalid ASR endpoint")
    resource = address.netloc + address.path
    original = resource + "?" + "&".join(f"{key}={value}" for key, value in pairs)
    digest = hmac.new(credentials["secret_key"].encode(), original.encode(), hashlib.sha1).digest()
    signature = base64.b64encode(digest).decode()
    return address.scheme + "://" + resource + "?" + urlencode(pairs + [("signature", signature)])


def sentence_updates(message):
    """Normalize old result and V2 sentence_list formats without concatenating revisions."""
    updates = []
    result = message.get("result")
    if isinstance(result, dict) and type(result.get("index")) is int:
        updates.append((result["index"], result.get("voice_text_str", ""), result.get("slice_type") == 2))
    sentences = message.get("sentences", {})
    if isinstance(sentences, dict):
        entries = sentences.get("sentence_list", [sentences] if "sentence" in sentences else [])
        if isinstance(entries, list):
            for sentence in entries:
                if isinstance(sentence, dict) and type(sentence.get("sentence_id")) is int:
                    updates.append((sentence["sentence_id"], sentence.get("sentence", ""), sentence.get("sentence_type") == 1))
    return sorted(updates, key=lambda update: update[0])


class TencentASR:
    def __init__(self, settings, *, root=None):
        self.settings = settings or {}
        self.root = root

    def _credentials(self):
        path = self.settings.get("credentials_file")
        if not path:
            return {}
        try:
            target = resolve_path(path, self.root) if self.root is not None else resolve_path(path)
            return json.loads(target.read_text(encoding="utf-8-sig"))
        except (OSError, ValueError, TypeError):
            return {}

    def status(self):
        credentials = self._credentials()
        return {"configured": bool(self.settings.get("enabled", False) and all(
                    credentials.get(key) for key in ("app_id", "secret_id", "secret_key"))),
                "provider": "tencent", "sample_rate": 16000,
                "engine": self.settings.get("engine_model_type", "16k_zh_en_2.0"),
                "max_recording_seconds": self.settings.get("max_recording_seconds", 60)}

    @asynccontextmanager
    async def connection(self):
        if not self.status()["configured"]:
            raise RuntimeError("腾讯 ASR 未配置，请检查本地凭据文件。")
        try:
            from websockets.asyncio.client import connect
        except ImportError:
            raise RuntimeError("ASR 运行依赖未安装。") from None
        url = signed_url(self._credentials(), self.status()["engine"],
                         endpoint=self.settings.get("endpoint", "wss://asr.cloud.tencent.com/asr/v2"))
        try:
            async with connect(url, open_timeout=15, close_timeout=3, max_queue=8,
                               max_size=256 * 1024, proxy=None) as upstream:
                greeting = json.loads(await asyncio.wait_for(upstream.recv(), 15))
                self.check_response(greeting)
                yield upstream
        except asyncio.CancelledError:
            raise
        except RuntimeError:
            raise
        except Exception as error:
            # Connection exception strings may contain the complete signed URL.
            raise RuntimeError(f"腾讯 ASR 连接中断（{type(error).__name__}）。") from None

    @staticmethod
    def check_response(message):
        if not isinstance(message, dict):
            raise RuntimeError("腾讯 ASR 返回了无效消息。")
        if message.get("code", 0) != 0:
            # No provider-supplied message is reflected back to the UI/log.
            code = message.get("code")
            safe_code = str(code) if isinstance(code, int) else "unknown"
            raise RuntimeError(f"腾讯 ASR 识别失败（code={safe_code}）。")


def pcm_level(data):
    values = array.array("h")
    values.frombytes(data)
    if sys.byteorder != "little":
        values.byteswap()
    return math.sqrt(sum(sample * sample for sample in values) / len(values)) / 32768


async def serve_asr(websocket, settings, conversation_id, emit_to_conversation,
                    speculate, commit, interrupt, *, activity=None, vision=None):
    """Accept a socket, recognize PCM and submit confirmed turns through commit.

    First command is {type:'start',continuous:bool}; then binary PCM16/16k/mono,
    optional speech.start/silence VAD edges, finally end or cancel. Emits ready,
    transcript/utterance, speech.started and input.queued. Frontends must never
    submit transcript/utterance back as a second user turn.
    """
    await websocket.accept()
    send_lock = asyncio.Lock()
    controller = None
    connected = True
    vision = vision or {}
    frame = None
    end_images = []
    captures = {}
    speech_epoch = 0
    last_activity = 0

    def external_activity(*, force=False):
        nonlocal last_activity
        now = time.monotonic()
        if activity and (force or now - last_activity >= settings.get("activity_interval_ms", 1000) / 1000):
            activity()
            last_activity = now

    async def send(event):
        nonlocal connected
        if not connected:
            return
        try:
            async with send_lock:
                await websocket.send_json(event)
        except (WebSocketDisconnect, RuntimeError):
            connected = False

    async def emit(event):
        if event["type"] == "input.error":
            await send({"type": "error", "message": event.get("message", "语音输入未能提交，请重新录音。")})
            await emit_to_conversation({**event, "conversation_id": conversation_id})
            return
        await send(event)
        if event["type"] in {"speech.started", "input.queued"}:
            await emit_to_conversation({**event, "conversation_id": conversation_id})

    async def commit_turn(text, request_id, candidate):
        nonlocal frame, end_images
        packets = list(end_images)
        if frame is not None:
            packets.append(frame)
        if packets:
            unique = {packet.get("image_id") or str(index): packet for index, packet in enumerate(packets)}
            if len(unique) > vision.get("max_images_per_turn", 3):
                raise ValueError("语音附图总数超过本轮上限。")
            admission = await commit(text, request_id, candidate, images=list(unique.values()))
        else:
            admission = await commit(text, request_id, candidate)
        frame = None
        end_images = []
        captures.clear()
        return admission

    def accept_image(packet):
        if not vision.get("enabled", False):
            raise ValueError("图片输入未启用。")
        if not isinstance(packet, dict) or not isinstance(packet.get("data_url"), str):
            raise ValueError("图片分片无效。")
        if len(packet["data_url"]) > (vision["max_input_bytes_per_image"] + 2) // 3 * 4 + 128:
            raise ValueError("图片分片超过容量限制。")
        # Full decoding runs once, off the event loop, during durable admission.
        return {key: value for key, value in packet.items() if key in {"image_id", "data_url", "captured_at", "source"}}

    try:
        asr = TencentASR(settings)
        if not asr.status()["configured"]:
            raise RuntimeError("腾讯 ASR 未配置，请检查本地凭据文件。")
        start = await asyncio.wait_for(websocket.receive_json(), 15)
        if (not isinstance(start, dict) or start.get("type") != "start"
                or type(start.get("continuous", False)) is not bool):
            raise ValueError("请先发送有效的 start 语音输入指令。")
        continuous = start.get("continuous", False)
        video = start.get("video", False)
        if type(video) is not bool:
            raise ValueError("video 必须是布尔值。")
        use_speculation = continuous and (not video or settings.get("turn_detection", {}).get("video_speculation", False))
        controller = VoiceTurnController(emit=emit, commit=commit_turn, speculate=speculate if use_speculation else None,
                                         interrupt=interrupt, settings=settings.get("turn_detection"))
        async with asr.connection() as upstream:
            await send({"type": "ready", "sample_rate": 16000, "continuous": continuous})
            if not continuous:
                external_activity(force=True)
                await controller.speech_start()
            final = asyncio.Event()
            end_requested = False
            explicit_vad = False
            voiced = False
            threshold = settings.get("rms_threshold", 0.012)

            async def upload():
                nonlocal end_requested, explicit_vad, voiced, speech_epoch, frame, end_images
                while True:
                    packet = await asyncio.wait_for(websocket.receive(), settings.get("socket_receive_timeout_seconds", 15))
                    if packet["type"] == "websocket.disconnect":
                        return
                    data = packet.get("bytes")
                    if data is not None:
                        if not data or len(data) > settings.get("max_pcm_chunk_bytes", 12800) or len(data) % 2:
                            raise ValueError("PCM16 音频分片无效。")
                        if continuous and not explicit_vad:
                            speech = pcm_level(data) >= threshold
                            if speech and not voiced:
                                speech_epoch += 1
                                frame = None
                                captures.clear()
                                external_activity(force=True)
                                await controller.speech_start()
                            elif not speech and voiced:
                                await controller.silence()
                            voiced = speech
                        if controller.speaking:
                            external_activity()
                        await upstream.send(data)
                        continue
                    try:
                        if len(packet.get("text", "")) > settings.get("max_frame_packet_bytes", 3145728):
                            raise ValueError("语音输入图片消息过大。")
                        command = json.loads(packet.get("text", ""))
                    except (ValueError, TypeError):
                        raise ValueError("语音输入指令不是有效 JSON。") from None
                    if not isinstance(command, dict):
                        raise ValueError("语音输入指令无效。")
                    kind = command.get("type")
                    if kind == "speech.start":
                        explicit_vad = True
                        if not controller.speaking:
                            speech_epoch += 1
                            frame = None
                            captures.clear()
                        external_activity(force=True)
                        await controller.speech_start()
                    elif kind == "silence":
                        explicit_vad = True
                        capture_id = command.get("capture_id")
                        if isinstance(capture_id, str) and len(capture_id) <= 160:
                            captures.clear()
                            captures[capture_id] = speech_epoch
                        if command.get("image") is not None:
                            frame = accept_image(command["image"])
                        if continuous:
                            await controller.silence()
                    elif kind == "camera.frame":
                        capture_id = command.get("capture_id")
                        if (not controller.speaking and isinstance(capture_id, str) and capture_id in captures
                                and captures[capture_id] == speech_epoch):
                            frame = accept_image(command.get("image"))
                            external_activity()
                    elif kind == "end":
                        if command.get("image") is not None:
                            frame = accept_image(command["image"])
                        supplied = command.get("images", [])
                        if not isinstance(supplied, list) or len(supplied) > vision.get("max_images_per_turn", 3):
                            raise ValueError("语音附图数量无效。")
                        end_images = [accept_image(image) for image in supplied]
                        end_requested = True
                        await upstream.send(json.dumps({"type": "end"}))
                        await asyncio.wait_for(final.wait(), settings.get("finish_timeout_ms", 20000) / 1000)
                        return
                    elif kind == "cancel":
                        return
                    else:
                        raise ValueError("不支持的语音输入指令。")

            async def download():
                async for packet in upstream:
                    message = json.loads(packet)
                    asr.check_response(message)
                    for sentence_id, text, stable in sentence_updates(message):
                        await controller.update(sentence_id, text, stable)
                    if message.get("final") == 1:
                        await controller.finish()
                        final.set()
                        await send({"type": "finished" if end_requested else "restart", "reason": "stream_complete"})
                        return
                raise RuntimeError("腾讯 ASR 在返回最终结果前断开连接。")

            tasks = [asyncio.create_task(upload()), asyncio.create_task(download())]
            try:
                seconds = settings.get("max_session_seconds", 3600) if continuous else settings.get("max_recording_seconds", 60) + 30
                async with asyncio.timeout(seconds):
                    done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
                    for task in done:
                        task.result()
            finally:
                for task in tasks:
                    task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
    except WebSocketDisconnect:
        pass
    except TimeoutError:
        await send({"type": "error", "message": "语音识别超时，请重新录音。"})
    except ValueError as error:
        await send({"type": "error", "message": str(error)})
    except Exception:
        await send({"type": "error", "message": "语音识别暂不可用，请检查 ASR 配置和网络后重试。"})
    finally:
        if controller:
            await controller.close()
        with suppress(RuntimeError, WebSocketDisconnect):
            await websocket.close()
