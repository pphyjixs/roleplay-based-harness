"""External-input timer; proactive turns never pretend to be user messages."""
import time
from uuid import uuid4


class IdleScheduler:
    def __init__(self, engine, settings, *, clock=time.monotonic):
        self.engine = engine
        self.settings = settings
        self.clock = clock
        self.activity = {}
        self.notified = set()

    def arm(self, conversation_id):
        if conversation_id not in self.activity:
            self.activity[conversation_id] = self.clock()
            self.engine.store.reset_idle_notification(conversation_id)

    def touch(self, conversation_id, kind):
        self.activity[conversation_id] = self.clock()
        self.notified.discard(conversation_id)
        return self.engine.store.touch_activity(conversation_id, kind)

    async def tick(self):
        if not self.settings.get("enabled", False):
            return
        for conversation_id, last in tuple(self.activity.items()):
            if conversation_id in self.notified:
                continue
            if self.clock() - last < self.settings["timeout_seconds"]:
                continue
            if self.settings["require_connected_client"] and not self.engine.events.subscribers.get(conversation_id):
                continue
            conversation = self.engine.store.get_conversation(conversation_id)
            if not conversation or conversation.get("status", "active") != "active":
                continue
            if any(turn["status"] in {"running", "queued"} for turn in self.engine.store.list_turns(conversation_id)):
                continue
            if any(job["playback_status"] == "playing" for job in self.engine.store.list_audio_jobs(conversation_id)):
                continue
            if not self.engine.store.claim_idle(conversation_id, conversation["last_external_input_at"]):
                self.notified.add(conversation_id)
                continue
            self.notified.add(conversation_id)
            await self.engine.enqueue(conversation_id, "idle:" + uuid4().hex,
                {"kind": "idle", "elapsed_seconds": round(self.clock() - last), "prompt": self.settings["prompt"]},
                source="idle", input_source="internal")
            if self.settings.get("repeat_without_input", False):
                self.notified.discard(conversation_id)
                self.activity[conversation_id] = self.clock()
                self.engine.store.reset_idle_notification(conversation_id)
