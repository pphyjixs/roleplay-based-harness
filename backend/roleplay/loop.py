"""One ordered RP conversation loop, with asynchronous delivery and memory work."""

import asyncio
import json
import logging
import hashlib
import inspect
import re
from datetime import datetime, timezone

from backend.config import character_profile, resolve_path
from backend.performance.service import DeliveryService
from backend.performance.voice import VoiceService
from backend.performance.retention import AudioRetention
from backend.roleplay.context import bound_context
from backend.roleplay.idle import IdleScheduler
from backend.roleplay.prompts import SYSTEM_PROMPT
from backend.roleplay.tools import DecisionError, build_tools, parse_decision
from backend.state.memory import MemoryManager, token_upper_bound
from backend.vision import TransientImages, image_blocks
from backend.speech.client import GPTSoVITS
from backend.tasks.service import TaskService


log = logging.getLogger("rpbh")


class EventBus:
    def __init__(self):
        self.subscribers = {}

    def subscribe(self, conversation_id):
        queue = asyncio.Queue(maxsize=256)
        self.subscribers.setdefault(conversation_id, set()).add(queue)
        return queue

    def unsubscribe(self, conversation_id, queue):
        listeners = self.subscribers.get(conversation_id, set())
        listeners.discard(queue)
        if not listeners:
            self.subscribers.pop(conversation_id, None)

    async def emit(self, event):
        for queue in tuple(self.subscribers.get(event["conversation_id"], ())):
            if queue.full():
                while not queue.empty():
                    queue.get_nowait()
                queue.put_nowait({"type": "resync_required", "conversation_id": event["conversation_id"]})
            else:
                queue.put_nowait(event)


class ConversationEngine:
    def __init__(self, config, store, model, memory_model):
        self.config = config
        self.store = store
        self.model = model
        memory_settings = {**config["memory"], "log_retention": config.get("retention", {}).get("logs", {})}
        self.memory = MemoryManager(store, memory_model, memory_settings)
        self.events = EventBus()
        self.delivery = DeliveryService(store)
        self.voice = None
        speech = config.get("speech", {}).get("tts", {})
        if speech.get("enabled", False):
            data_root = resolve_path(config["paths"]["data"])
            tts = GPTSoVITS(speech, runtime_dir=data_root / "runtime" / "tts")
            self.voice = VoiceService(store, tts, speech, data_root, emit=self.events.emit)
        self.tasks = TaskService(store, self.events.emit)
        self.locks = {}
        self.workers = {}
        self.finished = {}
        self.prepared = {}
        self.interrupt_epochs = {}
        self.background = set()
        self.closing = False
        self.services = set()
        self.media = {}
        self.processing = {}
        self.archives = {}
        self.images = TransientImages(config.get("vision", {}))
        self.idle = IdleScheduler(self, config.get("idle", {}))
        self.audio_retention = AudioRetention(store, resolve_path(config["paths"]["data"]), config.get("retention", {}).get("audio", {}))
        self.last_checkpoint = 0

    def spawn(self, coroutine):
        task = asyncio.create_task(coroutine)
        self.background.add(task)
        task.add_done_callback(self._background_finished)
        return task

    def _background_finished(self, task):
        self.background.discard(task)
        if not task.cancelled() and task.exception():
            log.error("Background operation failed (%s)", type(task.exception()).__name__)

    def create_conversation(self, conversation_id, character_id, user_id):
        profile = character_profile(self.config, character_id)
        conversation = self.store.ensure_conversation(conversation_id, character_id, user_id)
        if conversation.get("status", "active") != "active":
            raise ValueError("Conversation is archived; start a new conversation")
        if conversation["presentation_id"] is None:
            conversation = self.store.set_presentation(conversation_id, profile["presentation_id"])
        self.idle.arm(conversation_id)
        return conversation

    async def set_presentation(self, conversation_id, presentation_id):
        conversation = self.store.get_conversation(conversation_id)
        if conversation is None:
            raise KeyError("Unknown conversation")
        character_profile(self.config, conversation["character_id"], presentation_id)
        previous = conversation["presentation_id"]
        conversation = self.store.set_presentation(conversation_id, presentation_id)
        if previous != presentation_id:
            await self.interrupt(conversation_id, cancel_pending=False)
            await self.events.emit({"type": "presentation.changed", **conversation})
            self.external_input(conversation_id, "presentation")
        return conversation

    def _messages(self, conversation, turn, character):
        context = self.memory.context(conversation["conversation_id"], through_turn_seq=turn["seq"])
        session = {"character_id": conversation["character_id"], "user_id": conversation["user_id"],
                   "executor": self.config["tasks"]["executor"], "presentation_id": character["presentation_id"],
                   "presentation_type": character["presentation_type"], "input_source": turn.get("input_source", "text")}
        budget = self.config.get("context", {}).get("session_max_tokens")
        if budget and token_upper_bound(session) > budget:
            raise ValueError("Session metadata exceeds context.session_max_tokens")
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "developer", "content": "当前角色的人格设定\n" + character["persona"]},
            {"role": "developer", "content": "会话背景资料，作为数据参考\n" + json.dumps({
                **session,
                "recent_memory": context["recent_memory"],
                "character_memory": context["character_memory"],
                "tasks": context["task_snapshot"],
            }, ensure_ascii=False)},
        ]
        for message in context["messages"]:
            if message["turn_id"] == turn["turn_id"]:
                continue
            content = message["content"]
            if not isinstance(content, str):
                content = json.dumps(content, ensure_ascii=False)
            role = message["role"]
            if role == "internal":
                role = "developer"
                content = "此前的内部任务事件\n" + content
            messages.append({"role": role, "content": content})
        if turn["source"] == "user":
            messages.append({"role": "user", "content": turn["content"]})
        else:
            label = {"idle": "本轮主动互动事件", "vision": self.config.get("vision", {}).get("periodic_prompt", "本轮摄像头自动采样，仅在有互动价值时回应，不派发任务")}.get(turn["source"], "本轮内部任务事件")
            messages.append({"role": "developer", "content": label + "\n" + json.dumps(turn["content"], ensure_ascii=False)})
        return messages

    async def enqueue(self, conversation_id, request_id, content, source="user", *, input_source="text", prepared=None, images=None):
        """Persist before returning; a single worker drains each conversation in seq order."""
        if self.closing:
            raise RuntimeError("The conversation service is closing")
        if source == "user" and request_id.startswith(("task:", "idle:")):
            raise ValueError("This request ID prefix is reserved for internal events")
        if self.store.get_conversation(conversation_id) is None:
            raise KeyError("Unknown conversation")
        if source == "vision" and any(turn["status"] in {"queued", "running"} for turn in self.store.list_turns(conversation_id)):
            self.external_input(conversation_id, "camera")
            return {"status": "skipped", "reason": "conversation_busy", "request_id": request_id}
        context_settings = self.config.get("context", {})
        if source == "user" and context_settings and token_upper_bound(content) > context_settings["current_input_max_tokens"]:
            raise ValueError("Input text exceeds context.current_input_max_tokens")
        if images and not self.config.get("vision", {}).get("enabled", False):
            raise ValueError("Image input is disabled")
        if source == "vision" and not images:
            raise ValueError("Camera observations need an image")
        input_fingerprint = None
        if images:
            if any(not isinstance(image, dict) or not isinstance(image.get("data_url"), str) for image in images):
                raise ValueError("Image packets need a data_url string")
            input_fingerprint = hashlib.sha256(json.dumps({"text": content, "source": source, "input_source": input_source,
                "images": [{"image_id": image.get("image_id"), "captured_at": image.get("captured_at"),
                            "digest": hashlib.sha256(image.get("data_url", "").encode()).hexdigest()} for image in images]},
                sort_keys=True, ensure_ascii=False).encode()).hexdigest()
        duplicate = self.store.get_turn_by_request(conversation_id, request_id)
        held = []
        metadata = []
        try:
            if images and duplicate is None:
                if len(images) > self.config["vision"]["max_images_per_turn"]:
                    raise ValueError("Too many images in one turn")
                for packet in images:
                    item = await self.images.add_async(conversation_id, packet)
                    metadata.append(item)
                held = self.images.take([item["image_id"] for item in metadata], conversation_id)
            durable_content = content
            if held:
                placeholders = [f"[图片 {index + 1}，拍摄时间 {image.captured_at}：等待本轮文字摘要；原图不留存]" for index, image in enumerate(held)]
                durable_content = str(content) + "\n" + "\n".join(placeholders)
            admitted, created = self.store.begin_turn(conversation_id, request_id, durable_content, source,
                input_source=input_source, queued=True, input_fingerprint=input_fingerprint,
                evidence_content=content if images else None, image_count=len(images or []))
            if created and held:
                self.media[admitted["turn_id"]] = {"images": held, "original": content}
            elif held:
                self.images.release([image.image_id for image in held])
        except BaseException:
            if held:
                self.images.release([image.image_id for image in held])
            elif images and duplicate is None:
                self.images.release([image["image_id"] for image in metadata])
            raise
        if prepared is not None and created:
            self.prepared[admitted["turn_id"]] = prepared
        elif isinstance(prepared, asyncio.Future):
            prepared.cancel()
        if created:
            if source in {"user", "vision"}:
                self.external_input(conversation_id, "camera" if source == "vision" else input_source)
            active = self.processing.get(conversation_id)
            if source == "user" and active and active["turn"]["source"] == "vision" and not active["task"].done():
                self.store.mark_turn_failed(active["turn"]["turn_id"], "Camera observation superseded by external user input")
                active["task"].cancel()
            user_message = next(message for message in self.store.list_messages(conversation_id) if message["turn_id"] == admitted["turn_id"])
            await self.events.emit({"type": "input.accepted", "conversation_id": conversation_id,
                "turn_id": admitted["turn_id"], "request_id": request_id, "source": input_source, "message": user_message})
        if admitted["status"] in {"queued", "running"}:
            self.finished.setdefault(admitted["turn_id"], asyncio.Event())
            self._start_worker(conversation_id)
        return {key: admitted[key] for key in ("turn_id", "request_id", "status", "input_source", "seq")}

    def _start_worker(self, conversation_id):
        if self.closing:
            return
        existing = self.workers.get(conversation_id)
        if existing is None or existing.done():
            self.workers[conversation_id] = self.spawn(self._drain(conversation_id))

    async def _drain(self, conversation_id):
        lock = self.locks.setdefault(conversation_id, asyncio.Lock())
        async with lock:
            while not self.closing:
                pending = [turn for turn in self.store.list_turns(conversation_id) if turn["status"] in {"queued", "running"}]
                if not pending:
                    return
                turn = pending[0]
                try:
                    operation = asyncio.create_task(self._process_turn(turn))
                    self.processing[conversation_id] = {"task": operation, "turn": turn}
                    await operation
                except asyncio.CancelledError:
                    if self.closing or asyncio.current_task().cancelling():
                        raise
                    self.store.mark_turn_failed(turn["turn_id"], "Camera observation superseded by external user input")
                    await self.events.emit({"type": "vision.skipped", "conversation_id": conversation_id, "turn_id": turn["turn_id"]})
                except Exception:
                    # A failed request is durable and does not block later inputs.
                    log.warning("RP turn failed (%s)", turn["turn_id"])
                finally:
                    self.processing.pop(conversation_id, None)
                    media = self.media.pop(turn["turn_id"], None)
                    if media:
                        self.images.release([image.image_id for image in media["images"]])
                        if self.store.get_turn(turn["turn_id"])["status"] != "completed":
                            self.store.replace_turn_content(turn["turn_id"], str(media["original"]) + "\n[本轮图片未完成识别，原图已清理；如需查看请重新发送。]")
                    event = self.finished.pop(turn["turn_id"], None)
                    if event:
                        event.set()
                    self.prepared.pop(turn["turn_id"], None)

    async def reply(self, conversation_id, request_id, content, source="user", *, input_source="text", images=None):
        admission = await self.enqueue(conversation_id, request_id, content, source, input_source=input_source, images=images)
        if admission["status"] == "skipped":
            return admission
        turn = self.store.get_turn(admission["turn_id"])
        if turn["status"] not in {"completed", "failed"}:
            event = self.finished.setdefault(turn["turn_id"], asyncio.Event())
            await event.wait()
            turn = self.store.get_turn(turn["turn_id"])
            if turn["status"] == "failed":
                error_type = DecisionError if (turn["error"] or "").startswith("DecisionError:") else RuntimeError
                raise error_type(turn["error"])
        if turn["status"] == "failed":
            return {"turn_id": turn["turn_id"], "status": "failed", "error": turn["error"]}
        return self._current_result(turn["result"])

    async def _decide(self, messages, tools, character, allow_tasks, images=None):
        request = list(messages)
        for attempt in range(self.config["roleplay"].get("max_repair_attempts", 1) + 1):
            request = bound_context(request, tools, self.config.get("context", {}),
                                    image_count=len(images or []), tail_count=1 if attempt == 0 else 2)
            response = await self.model.decide(request, tools)
            try:
                return parse_decision(response, allow_tasks=allow_tasks, character=character, images=images,
                    image_summary_max_tokens=self.config.get("vision", {}).get("summary_max_tokens", 800)), response
            except DecisionError as error:
                if attempt == self.config["roleplay"].get("max_repair_attempts", 1):
                    raise
                request.append({"role": "developer", "content": "上次响应未通过参数校验，请重新给出本轮完整决定。校验原因\n" + str(error)})

    @staticmethod
    def _fingerprint(messages, tools):
        return hashlib.sha256(json.dumps([messages, tools], ensure_ascii=False, sort_keys=True).encode()).hexdigest()

    async def prepare_speculation(self, conversation_id, text):
        """Build a discardable candidate; never persist, dispatch, synthesize or publish."""
        conversation = self.store.get_conversation(conversation_id)
        if conversation is None:
            raise KeyError("Unknown conversation")
        turns = self.store.list_turns(conversation_id)
        if any(turn["status"] in {"running", "queued"} for turn in turns):
            return None
        turn = {"source": "user", "input_source": "asr", "content": text, "turn_id": None, "seq": max([item["seq"] for item in turns], default=0) + 1}
        character = character_profile(self.config, conversation["character_id"], conversation["presentation_id"])
        messages = self._messages(conversation, turn, character)
        tools = build_tools(character)
        messages = bound_context(messages, tools, self.config.get("context", {}))
        fingerprint = self._fingerprint(messages, tools)
        epoch = self.interrupt_epochs.get(conversation_id, 0)
        decision, response = await self._decide(messages, tools, character, True)
        return {"fingerprint": fingerprint, "decision": decision, "response": response,
                "interrupt_epoch": epoch,
                "presentation_revision": conversation["presentation_revision"]}

    async def _process_turn(self, turn):
        conversation_id = turn["conversation_id"]
        self.store.mark_turn_running(turn["turn_id"])
        await self.events.emit({"type": "turn.started", "conversation_id": conversation_id, "turn_id": turn["turn_id"]})
        try:
            if turn.get("image_count", 0) and turn["turn_id"] not in self.media:
                original = turn.get("evidence_content") or ""
                self.store.replace_turn_content(turn["turn_id"], str(original) + "\n[图片临时内容已清理，无法继续识别，请重新发送。]")
                raise ValueError("ImageInputUnavailable: temporary images cannot be recovered after restart")
            while True:
                conversation = self.store.get_conversation(conversation_id)
                character = character_profile(self.config, conversation["character_id"], conversation["presentation_id"])
                messages = self._messages(conversation, turn, character)
                allow_tasks = turn["source"] == "user"
                media = self.media.get(turn["turn_id"])
                visible_images = media["images"] if media else []
                tools = build_tools(character, allow_tasks=allow_tasks, images=visible_images,
                    image_summary_max_tokens=self.config.get("vision", {}).get("summary_max_tokens", 800))
                messages = bound_context(messages, tools, self.config.get("context", {}), image_count=len(visible_images))
                if visible_images:
                    messages[-1]["content"] = [{"type": "input_text", "text": messages[-1]["content"]}, *image_blocks(visible_images)]
                prepared = self.prepared.pop(turn["turn_id"], None)
                if prepared and visible_images:
                    if isinstance(prepared, asyncio.Future):
                        prepared.cancel()
                    prepared = None
                if inspect.isawaitable(prepared):
                    try:
                        prepared = await prepared
                    except asyncio.CancelledError:
                        if asyncio.current_task().cancelling():
                            raise
                        prepared = None
                    except Exception:
                        prepared = None
                    # Receipts, memories or the selected rig can change while
                    # an admitted speculative request is still completing.
                    conversation = self.store.get_conversation(conversation_id)
                    character = character_profile(self.config, conversation["character_id"], conversation["presentation_id"])
                    messages = self._messages(conversation, turn, character)
                    tools = build_tools(character, allow_tasks=allow_tasks)
                    messages = bound_context(messages, tools, self.config.get("context", {}))
                if (prepared and prepared["fingerprint"] == self._fingerprint(messages, tools)
                        and prepared.get("interrupt_epoch", 0) == self.interrupt_epochs.get(conversation_id, 0)
                        and prepared["presentation_revision"] == conversation["presentation_revision"]):
                    decision, response = prepared["decision"], prepared["response"]
                else:
                    decision, response = await self._decide(messages, tools, character, allow_tasks, visible_images)
                if self.store.get_conversation(conversation_id)["presentation_revision"] == conversation["presentation_revision"]:
                    break
            summaries = decision.pop("image_summaries", [])
            if media:
                by_id = {summary["image_id"]: summary["summary"] for summary in summaries}
                observations = [f"[图片观察，{image.captured_at}] {by_id[image.image_id]}" for image in visible_images]
                self.store.replace_turn_content(turn["turn_id"], str(media["original"]) + "\n" + "\n".join(observations))
            if self.store.get_turn(turn["turn_id"])["rp_cancelled"]:
                decision = {**decision, "rp": {"execute": False}}
            if decision["rp"]["execute"]:
                decision["rp"] = {**decision["rp"], "presentation_id": character["presentation_id"],
                                  "presentation_revision": conversation["presentation_revision"]}
            # Responses are traces, never a place to retain echoed request media.
            trace = {key: response[key] for key in ("id", "status", "model", "usage", "output") if key in response}
            trace["output"] = [{key: item[key] for key in ("type", "name", "call_id", "arguments") if key in item}
                               for item in response.get("output", []) if item.get("type") == "function_call"]
            trace = json.loads(re.sub(r"data:image/[^;\s]+;base64,[A-Za-z0-9+/=]+", "[temporary image omitted]", json.dumps(trace, ensure_ascii=False)))
            result = self.store.finish_turn(turn["turn_id"], **decision, model_response=trace)
        except asyncio.CancelledError:
            # A running input is durable and can be resumed on startup.
            raise
        except Exception as error:
            if turn["source"] == "vision" and self.store.get_turn(turn["turn_id"])["status"] == "failed":
                return {"turn_id": turn["turn_id"], "status": "failed", "rp": None, "task": None}
            self.store.mark_turn_failed(turn["turn_id"], type(error).__name__ + ": " + str(error))
            await self.events.emit({"type": "turn.failed", "conversation_id": conversation_id, "turn_id": turn["turn_id"], "error": str(error)})
            raise
        if result["rp"]:
            await self.delivery.publish(result["rp"], self.events.emit)
            if self.voice:
                self.voice.enqueue(result["rp"], character.get("tts_voice"))
        if result["task"]:
            result["task"] = await self.tasks.dispatch(result["task"])
        await self.events.emit({"type": "turn.completed", "conversation_id": conversation_id, "turn_id": turn["turn_id"], "request_id": turn["request_id"]})
        self.memory.maintenance(conversation_id)
        self.schedule_memory(conversation_id)
        return self._current_result(result)

    async def interrupt(self, conversation_id, action_ids=None, *, cancel_pending=True):
        self.interrupt_epochs[conversation_id] = self.interrupt_epochs.get(conversation_id, 0) + 1
        cancelled_turns = self.store.cancel_pending_performances(conversation_id) if cancel_pending and action_ids is None else []
        actions = self.store.interrupt_actions(conversation_id, action_ids)
        ids = [action["action_id"] for action in actions]
        if self.voice:
            await self.voice.cancel(ids)
        for action_id in ids:
            self.audio_retention.release(action_id)
        await self.events.emit({"type": "rp.interrupted", "conversation_id": conversation_id, "action_ids": ids, "turn_ids": cancelled_turns})
        return {"action_ids": ids, "turn_ids": cancelled_turns}

    def _current_result(self, result):
        public = {key: result[key] for key in ("turn_id", "status", "rp", "task")}
        public["decisions"] = result.get("decisions", {})
        if public["rp"]:
            public["rp"] = self.store.get_action(public["rp"]["action_id"])
            if self.voice:
                public["rp"]["audio"] = self.store.get_audio_job(public["rp"]["action_id"])
        if public["task"]:
            public["task"] = self.store.get_task(public["task"]["task_id"])
        return public

    def schedule_memory(self, conversation_id):
        if self.config["memory"].get("enabled", True) and not self.closing:
            self.spawn(self._update_memory(conversation_id))

    async def _update_memory(self, conversation_id):
        try:
            await self.memory.update(conversation_id)
        except Exception as error:
            log.error("Memory update failed (%s)", type(error).__name__)
            await self.events.emit({"type": "memory.warning", "conversation_id": conversation_id, "message": "记忆更新暂未完成，已保留上次成功的记忆。"})

    async def acknowledge(self, conversation_id, action_id, text=None):
        action = self.store.get_action(action_id)
        if action is None or action["conversation_id"] != conversation_id:
            raise KeyError("Unknown action in this conversation")
        result = self.delivery.acknowledge(action_id, text)
        self.schedule_memory(conversation_id)
        return result

    async def task_event(self, task_id, event_id, kind, summary, result_ref=None):
        if event_id.startswith("mock-dispatch:"):
            raise ValueError("The mock-dispatch: event prefix is reserved for the executor")
        task, accepted = await self.tasks.handle_event(task_id, event_id, kind, summary, result_ref)
        if accepted:
            self.external_input(task["conversation_id"], "task")
            if self.store.get_conversation(task["conversation_id"]).get("status", "active") == "active":
                self.spawn(self._task_notice(task, event_id, kind, summary, result_ref))
        self.memory.maintenance(task["conversation_id"])
        return {"accepted": accepted, "task": task}

    async def _task_notice(self, task, event_id, kind, summary, result_ref=None):
        return await self.reply(task["conversation_id"], "task:" + task["task_id"] + ":" + event_id, {
            "task_id": task["task_id"], "kind": kind, "summary": summary, "result_ref": result_ref,
        }, source="task")

    def external_input(self, conversation_id, kind):
        return self.idle.touch(conversation_id, kind)

    async def playback(self, conversation_id, action_id, status):
        action = self.store.get_action(action_id)
        if action is None or action["conversation_id"] != conversation_id:
            raise KeyError("Unknown action in this conversation")
        job = self.store.record_playback(action_id, status)
        if status in {"interrupted", "failed"} and job["status"] in {"pending", "running"}:
            self.store.set_audio_job(action_id, "cancelled")
            if self.voice:
                await self.voice.cancel([action_id])
        job = self.audio_retention.release(action_id) or self.store.get_audio_job(action_id)
        await self.events.emit({"type": "audio.updated", "conversation_id": conversation_id, "action_id": action_id, "job": job})
        return job

    async def archive_conversation(self, conversation_id):
        conversation = self.store.get_conversation(conversation_id)
        if conversation is None:
            raise KeyError("Unknown conversation")
        if conversation.get("status") == "archived":
            return conversation
        existing = self.archives.get(conversation_id)
        if existing and not existing.done():
            return conversation
        self.store.set_conversation_status(conversation_id, "archiving")
        await self.interrupt(conversation_id)
        self.archives[conversation_id] = self.spawn(self._archive(conversation_id))
        return self.store.get_conversation(conversation_id)

    async def _archive(self, conversation_id):
        try:
            self._start_worker(conversation_id)
            worker = self.workers.get(conversation_id)
            if worker:
                await worker
            await self.memory.archive(conversation_id)
            self.idle.activity.pop(conversation_id, None)
            self.idle.notified.discard(conversation_id)
            await self.events.emit({"type": "conversation.archived", "conversation_id": conversation_id})
        except asyncio.CancelledError:
            raise
        except Exception as error:
            # The raw text remains intact; a user can retry the same archive.
            self.store.set_conversation_status(conversation_id, "active")
            await self.events.emit({"type": "archive.failed", "conversation_id": conversation_id,
                                   "message": "归档暂未完成，原始文字已保留，可稍后重试。"})
            log.warning("Conversation archive failed (%s)", type(error).__name__)
        finally:
            self.archives.pop(conversation_id, None)

    async def _idle_loop(self):
        settings = self.config.get("idle", {})
        if not settings.get("enabled", False):
            return
        while not self.closing:
            await asyncio.sleep(settings["check_interval_seconds"])
            try:
                await self.idle.tick()
            except Exception as error:
                log.warning("Idle timer failed (%s)", type(error).__name__)

    async def maintenance(self):
        self.images.expire()
        retention = self.config.get("retention")
        if not retention:
            return
        for conversation in self.store.list_conversations():
            conversation_id = conversation["conversation_id"]
            self.memory.maintenance(conversation_id)
            abandoned = [action["action_id"] for action in self.store.list_pending_actions(conversation_id)
                         if (datetime.now(timezone.utc) - datetime.fromisoformat(action["created_at"])).total_seconds() >= retention["audio"]["unplayed_max_age_seconds"]]
            if abandoned:
                await self.interrupt(conversation_id, abandoned)
        expired = self.audio_retention.candidates()
        for action_id in expired:
            job = self.store.get_audio_job(action_id)
            if job["playback_status"] not in {"completed", "interrupted"}:
                self.store.record_playback(action_id, "interrupted")
            if job["status"] in {"pending", "running"}:
                self.store.set_audio_job(action_id, "cancelled")
                if self.voice:
                    await self.voice.cancel([action_id])
            self.audio_retention.release(action_id, force=True)
        now = asyncio.get_running_loop().time()
        if now - self.last_checkpoint >= retention["sqlite"]["checkpoint_interval_seconds"]:
            self.store.checkpoint(retention["sqlite"]["incremental_vacuum_pages"])
            self.last_checkpoint = now

    async def _maintenance_loop(self):
        retention = self.config.get("retention")
        if not retention:
            return
        while not self.closing:
            try:
                await self.maintenance()
            except Exception as error:
                log.warning("Retention maintenance failed (%s)", type(error).__name__)
            await asyncio.sleep(retention["maintenance_interval_seconds"])

    async def recover(self):
        if self.voice:
            self.spawn(self.voice.tts.start())
            for action in self.store.actions_without_audio_jobs():
                conversation = self.store.get_conversation(action["conversation_id"])
                profile = character_profile(self.config, conversation["character_id"])
                self.voice.enqueue(action, profile.get("tts_voice"))
            self.voice.recover()
        for conversation in self.store.list_conversations():
            conversation_id = conversation["conversation_id"]
            if conversation.get("status") == "archiving":
                self.archives[conversation_id] = self.spawn(self._archive(conversation_id))
                continue
            if conversation.get("status", "active") != "active":
                continue
            if any(turn["status"] in {"queued", "running"} for turn in self.store.list_turns(conversation_id)):
                self._start_worker(conversation_id)
            for task in self.store.list_tasks(conversation_id):
                if task["status"] == "pending":
                    await self.tasks.dispatch(task)
            self.schedule_memory(conversation_id)
        for event in self.store.pending_task_notifications():
            if self.store.get_conversation(event["conversation_id"]).get("status", "active") == "active":
                self.spawn(self._task_notice(event, event["event_id"], event["kind"], event["summary"], event.get("result_ref")))
        if not self.services:
            for coroutine in (self._idle_loop(), self._maintenance_loop()):
                task = asyncio.create_task(coroutine)
                self.services.add(task)
                task.add_done_callback(self.services.discard)

    async def close(self):
        self.closing = True
        for task in tuple(self.services):
            task.cancel()
        if self.services:
            await asyncio.gather(*tuple(self.services), return_exceptions=True)
        for prepared in self.prepared.values():
            if isinstance(prepared, asyncio.Future):
                prepared.cancel()
        for task in tuple(self.background):
            task.cancel()
        if self.voice:
            await self.voice.close()
        if self.background:
            await asyncio.gather(*tuple(self.background), return_exceptions=True)
        for media in self.media.values():
            self.images.release([image.image_id for image in media["images"]])
        await self.model.close()
        if self.memory.model is not self.model:
            await self.memory.model.close()
        self.store.close()
