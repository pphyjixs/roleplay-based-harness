"""Local API for persistent text RP sessions and simulated task callbacks."""

import asyncio
import hmac
import secrets
from contextlib import asynccontextmanager, suppress
from typing import Literal
from uuid import uuid4

from fastapi import FastAPI, Header, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field

from backend.config import PROJECT_ROOT, load_config, model_settings, resolve_path, presentation_manifest
from backend.roleplay.loop import ConversationEngine
from backend.roleplay.model import ResponsesModel
from backend.roleplay.tools import DecisionError
from backend.state.store import Store


class Input(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True, strict=True)


class ConversationInput(Input):
    conversation_id: str | None = Field(default=None, min_length=1, max_length=160)
    character_id: str | None = Field(default=None, min_length=1, max_length=100)
    user_id: str | None = Field(default=None, min_length=1, max_length=100)


class MessageInput(Input):
    conversation_id: str = Field(min_length=1, max_length=160)
    request_id: str = Field(min_length=1, max_length=160)
    text: str = ""
    source: Literal["text", "asr", "vision"] = "text"
    images: list[dict] = Field(default_factory=list)


class ActivityInput(Input):
    kind: Literal["typing", "speech", "camera", "image", "interaction"]


class PresentationInput(Input):
    presentation_id: str = Field(min_length=1, max_length=100)


class InterruptInput(Input):
    action_ids: list[str] | None = Field(default=None, max_length=256)


class PlaybackInput(Input):
    status: Literal["playing", "completed", "interrupted", "failed"]


class DeliveryInput(Input):
    model_config = ConfigDict(extra="forbid", strict=True, str_strip_whitespace=False)
    text: str | None = Field(default=None, max_length=16000)


class SocketDeliveryInput(DeliveryInput):
    action_id: str = Field(min_length=1, max_length=160)


class TaskEventInput(Input):
    task_id: str = Field(min_length=1, max_length=160)
    event_id: str = Field(min_length=1, max_length=160)
    kind: Literal["progress", "completed", "failed", "cancelled"]
    summary: str = Field(min_length=1, max_length=8000)
    result_ref: str | None = Field(default=None, max_length=2000)


def callback_token(config):
    path = resolve_path(config["server"]["callback_token_file"])
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        try:
            with path.open("x", encoding="utf-8") as target:
                target.write(secrets.token_urlsafe(32))
        except FileExistsError:
            pass
    token = path.read_text(encoding="utf-8").strip()
    if len(token) < 24:
        raise ValueError("The internal callback token must have at least 24 characters")
    return token


def create_app(config=None, *, engine=None):
    settings = config or load_config()

    @asynccontextmanager
    async def lifespan(app):
        owned_engine = engine is None
        active = engine
        if active is None:
            store = Store(resolve_path(settings["paths"]["data"]) / "state" / "rpbh.sqlite3")
            active = ConversationEngine(settings, store,
                ResponsesModel(model_settings(settings, "roleplay"), PROJECT_ROOT),
                ResponsesModel(model_settings(settings, "memory"), PROJECT_ROOT))
        app.state.engine = active
        app.state.callback_token = callback_token(settings)
        from backend.mobile.service import MobileService
        mobile_settings = settings.get("mobile", {})
        app.state.mobile = MobileService(app, {**mobile_settings, "transport": {
            **mobile_settings.get("transport", {}), "max_websocket_message_bytes": settings["server"].get("max_websocket_message_bytes", 9437184)}})
        try:
            await active.recover()
            await app.state.mobile.start()
            yield
        finally:
            await app.state.mobile.close()
            if owned_engine:
                await active.close()

    app = FastAPI(title="RPBH orchestration", lifespan=lifespan)

    from backend.api.limits import MessageBodyLimit
    app.add_middleware(MessageBodyLimit, maximum=settings.get("vision", {}).get("max_request_bytes", 9437184))

    def current():
        return app.state.engine

    def require_conversation(conversation_id):
        conversation = current().store.get_conversation(conversation_id)
        if conversation is None:
            raise HTTPException(404, "Unknown conversation")
        return conversation

    @app.get("/api/health")
    async def health():
        from backend.speech.asr import TencentASR
        return {"status": "ready", "model": settings["roleplay"]["model"], "api_format": "responses",
                "executor": settings["tasks"]["executor"], "character": settings["characters"]["default"],
                "asr": TencentASR(settings.get("speech", {}).get("asr", {})).status(),
                "tts": current().voice.tts.status() if current().voice else {"enabled": False},
                "mobile": app.state.mobile.status()}

    @app.get("/api/p2p/status")
    async def mobile_status():
        return app.state.mobile.status()

    @app.post("/api/p2p/answer")
    async def mobile_answer(body: dict):
        try:
            return await app.state.mobile.answer(body.get("code"))
        except (ValueError, KeyError):
            raise HTTPException(400, "Invalid mobile pairing request") from None

    @app.delete("/api/p2p/connection")
    async def mobile_disconnect():
        await app.state.mobile.close_peers()
        return {"disconnected": True}

    @app.get("/p2p", include_in_schema=False)
    @app.get("/p2p/", include_in_schema=False)
    async def mobile_pairing_page():
        page = resolve_path(settings["server"].get("frontend_directory", "frontend/dist")) / "p2p.html"
        if not page.is_file():
            raise HTTPException(404, "Mobile pairing page is unavailable")
        return FileResponse(page, media_type="text/html")

    @app.get("/api/rp/characters")
    async def characters():
        return {"default_character": settings["characters"]["default"], "characters": {
            character_id: presentation_manifest(settings, character_id)
            for character_id in settings["characters"]["profiles"]}}

    @app.get("/api/rp/config")
    async def public_config():
        asr = settings.get("speech", {}).get("asr", {})
        return {"vision": settings.get("vision", {}), "idle": settings.get("idle", {}),
                "context": settings.get("context", {}),
                "memory": {key: value for key, value in settings["memory"].items() if key not in {"provider", "model"}},
                "retention": settings.get("retention", {}),
                "asr": {**{key: asr[key] for key in ("enabled", "rms_threshold", "max_recording_seconds", "open_timeout_ms", "finish_timeout_ms", "flush_timeout_ms", "max_buffered_bytes", "restart_delay_ms", "max_restart_attempts") if key in asr},
                        "sample_rate": 16000}}

    @app.get("/api/rp/conversations")
    async def conversations():
        return current().store.list_conversations()

    @app.post("/api/rp/conversations")
    async def create_conversation(body: ConversationInput):
        try:
            return current().create_conversation(body.conversation_id or "conv_" + uuid4().hex,
                body.character_id or settings["characters"]["default"],
                body.user_id or settings["server"]["default_user_id"])
        except ValueError as error:
            raise HTTPException(409, str(error)) from error

    @app.get("/api/rp/conversations/{conversation_id}")
    async def conversation(conversation_id: str):
        return {**require_conversation(conversation_id),
                "messages": current().store.list_messages(conversation_id),
                "pending_actions": current().store.list_pending_actions(conversation_id),
                "tasks": current().store.list_tasks(conversation_id),
                "audio_jobs": current().store.list_audio_jobs(conversation_id),
                "turns": [{key: turn[key] for key in ("turn_id", "request_id", "source", "input_source", "status", "error", "seq")}
                          for turn in current().store.list_turns(conversation_id) if not turn.get("archived")],
                "memory": current().memory.context(conversation_id)}

    @app.post("/api/rp/conversations/{conversation_id}/archive", status_code=202)
    async def archive(conversation_id: str):
        require_conversation(conversation_id)
        return await current().archive_conversation(conversation_id)

    @app.post("/api/rp/conversations/{conversation_id}/activity")
    async def activity(conversation_id: str, body: ActivityInput):
        conversation = require_conversation(conversation_id)
        if conversation.get("status", "active") != "active":
            raise HTTPException(409, "Conversation is not active")
        current().external_input(conversation_id, body.kind)
        return {"accepted": True}

    @app.post("/api/rp/conversations/{conversation_id}/presentation")
    @app.patch("/api/rp/conversations/{conversation_id}/presentation")
    async def change_presentation(conversation_id: str, body: PresentationInput):
        require_conversation(conversation_id)
        try:
            return await current().set_presentation(conversation_id, body.presentation_id)
        except ValueError as error:
            raise HTTPException(422, str(error)) from error

    @app.post("/api/rp/conversations/{conversation_id}/interrupt")
    async def interrupt(conversation_id: str, body: InterruptInput):
        require_conversation(conversation_id)
        try:
            return await current().interrupt(conversation_id, body.action_ids)
        except KeyError as error:
            raise HTTPException(404, str(error)) from error

    @app.post("/api/rp/conversations/{conversation_id}/playback/{action_id}")
    async def playback(conversation_id: str, action_id: str, body: PlaybackInput):
        require_conversation(conversation_id)
        action = current().store.get_action(action_id)
        if action is None or action["conversation_id"] != conversation_id:
            raise HTTPException(404, "Unknown action in this conversation")
        try:
            job = await current().playback(conversation_id, action_id, body.status)
        except KeyError as error:
            raise HTTPException(404, str(error)) from error
        except ValueError as error:
            raise HTTPException(409, str(error)) from error
        return job

    @app.post("/api/rp/messages", status_code=202)
    async def enqueue_message(body: MessageInput):
        require_conversation(body.conversation_id)
        try:
            if not body.text and not body.images:
                raise ValueError("A message needs text or an image")
            return await current().enqueue(body.conversation_id, body.request_id, body.text,
                source="vision" if body.source == "vision" else "user", input_source=body.source, images=body.images)
        except ValueError as error:
            raise HTTPException(409, str(error)) from error

    @app.post("/api/rp/message")
    async def message(body: MessageInput):
        require_conversation(body.conversation_id)
        try:
            if not body.text and not body.images:
                raise ValueError("A message needs text or an image")
            return await current().reply(body.conversation_id, body.request_id, body.text,
                source="vision" if body.source == "vision" else "user", input_source=body.source, images=body.images)
        except DecisionError as error:
            raise HTTPException(502, "The model returned invalid tool decisions after repair.") from error
        except ValueError as error:
            raise HTTPException(409, str(error)) from error
        except Exception as error:
            raise HTTPException(502, "RP generation failed. See the turn.failed event or persisted turn error.") from error

    @app.post("/api/rp/conversations/{conversation_id}/deliveries/{action_id}")
    async def delivered(conversation_id: str, action_id: str, body: DeliveryInput):
        require_conversation(conversation_id)
        try:
            return await current().acknowledge(conversation_id, action_id, body.text)
        except KeyError as error:
            raise HTTPException(404, str(error)) from error
        except ValueError as error:
            raise HTTPException(409, str(error)) from error

    @app.get("/api/rp/tasks/{task_id}")
    async def task(task_id: str):
        result = current().store.get_task(task_id)
        if result is None:
            raise HTTPException(404, "Unknown task")
        return result

    @app.get("/api/rp/audio/{action_id}")
    async def audio_job(action_id: str):
        result = current().store.get_audio_job(action_id)
        if result is None:
            raise HTTPException(404, "Unknown audio generation job")
        return result

    @app.get("/api/rp/audio/{action_id}/file")
    async def audio_file(action_id: str):
        job = current().store.get_audio_job(action_id)
        if job is None:
            raise HTTPException(404, "Unknown audio job")
        if job.get("file_state") == "deleted" or not job.get("wav_path"):
            raise HTTPException(410 if job.get("file_state") == "deleted" else 409, "Audio is no longer retained" if job.get("file_state") == "deleted" else "Audio is not ready")
        if job["status"] != "completed":
            raise HTTPException(409, "Audio generation is not complete")
        expected_root = resolve_path(settings["paths"]["data"]) / "audio"
        path = resolve_path(job["wav_path"])
        if not path.is_relative_to(expected_root) or not path.is_file():
            raise HTTPException(404, "Audio file is unavailable")
        return FileResponse(path, media_type="audio/wav", headers={"Cache-Control": "no-store"})

    @app.websocket("/ws/asr/{conversation_id}")
    async def asr_socket(websocket: WebSocket, conversation_id: str):
        if current().store.get_conversation(conversation_id) is None:
            await websocket.close(code=1008)
            return
        from backend.speech.asr import serve_asr
        speech_settings = settings.get("speech", {})
        asr_settings = {**speech_settings.get("asr", {}),
                        "turn_detection": speech_settings.get("asr", {}).get("turn_detection", speech_settings.get("turn_detection", {}))}
        await serve_asr(websocket, asr_settings, conversation_id,
            current().events.emit,
            lambda text: current().prepare_speculation(conversation_id, text),
            lambda text, request_id, prepared, images=None: current().enqueue(conversation_id, request_id, text, input_source="asr", prepared=prepared, images=images),
            lambda: current().interrupt(conversation_id),
            activity=lambda: current().external_input(conversation_id, "speech"),
            vision=settings.get("vision", {}))

    @app.post("/api/internal/task-events")
    async def task_event(body: TaskEventInput, authorization: str | None = Header(default=None)):
        expected = "Bearer " + app.state.callback_token
        if authorization is None or not hmac.compare_digest(authorization, expected):
            raise HTTPException(401, "Invalid task callback credentials")
        try:
            return await current().task_event(**body.model_dump())
        except KeyError as error:
            raise HTTPException(404, str(error)) from error
        except ValueError as error:
            raise HTTPException(409, str(error)) from error

    @app.websocket("/ws/rp/{conversation_id}")
    async def socket(websocket: WebSocket, conversation_id: str):
        if current().store.get_conversation(conversation_id) is None:
            await websocket.close(code=1008)
            return
        await websocket.accept()
        current().idle.arm(conversation_id)
        queue = current().events.subscribe(conversation_id)

        async def send_events():
            while True:
                await websocket.send_json(await queue.get())

        async def submit(body):
            try:
                result = await current().reply(conversation_id, body.request_id, body.text,
                    source="vision" if body.source == "vision" else "user", input_source=body.source, images=body.images)
                await current().events.emit({"type": "request.result", "conversation_id": conversation_id,
                                            "request_id": body.request_id, "result": result})
            except Exception:
                await current().events.emit({"type": "request.failed", "conversation_id": conversation_id, "request_id": body.request_id})

        async def receive_inputs():
            while True:
                try:
                    packet = await websocket.receive_json()
                    if not isinstance(packet, dict):
                        raise ValueError("Expected an object")
                    if packet.get("type") == "message":
                        body = MessageInput.model_validate({"conversation_id": conversation_id,
                            "request_id": packet.get("request_id"), "text": packet.get("text", ""), "source": packet.get("source", "text"), "images": packet.get("images", [])})
                        if not body.text and not body.images:
                            raise ValueError("A message needs text or an image")
                        current().spawn(submit(body))
                    elif packet.get("type") == "activity":
                        body = ActivityInput.model_validate({"kind": packet.get("kind")})
                        current().external_input(conversation_id, body.kind)
                    elif packet.get("type") == "delivery.ack":
                        receipt = SocketDeliveryInput.model_validate({"action_id": packet.get("action_id"), "text": packet.get("text")})
                        action = await current().acknowledge(conversation_id, receipt.action_id, receipt.text)
                        await queue.put({"type": "delivery.recorded", "action": action})
                    else:
                        raise ValueError("Unsupported event type")
                except (ValueError, KeyError) as error:
                    await queue.put({"type": "input.error", "message": str(error)})

        # Replaying a large backlog can fill the bounded event queue. Supervise
        # both socket directions while replay runs so a disconnected sender
        # cannot leave replay blocked forever on queue.put().
        await queue.put({"type": "session.ready", "conversation_id": conversation_id})
        sender = asyncio.create_task(send_events())
        replay = asyncio.create_task(current().delivery.replay(conversation_id, lambda event: put_event(queue, event)))
        receiver = asyncio.create_task(receive_inputs())
        tasks = {sender, replay, receiver}
        active = set(tasks)
        try:
            while active:
                done, active = await asyncio.wait(active, return_when=asyncio.FIRST_COMPLETED)
                for task in done:
                    task.result()
                if sender in done or receiver in done:
                    break
        except (WebSocketDisconnect, RuntimeError, asyncio.CancelledError):
            pass
        finally:
            current().events.unsubscribe(conversation_id, queue)
            for task in tasks:
                task.cancel()
            with suppress(asyncio.CancelledError):
                await asyncio.gather(*tasks, return_exceptions=True)

    # Only the frontend build is public. Configuration, persona and data are never mounted.
    frontend_directory = resolve_path(settings["server"].get("frontend_directory", "frontend/dist"))
    if frontend_directory.is_dir():
        app.mount("/", StaticFiles(directory=frontend_directory, html=True), name="frontend")
    return app


async def put_event(queue, event):
    await queue.put(event)
