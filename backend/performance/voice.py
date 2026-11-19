"""Persist and synthesize accepted RP speech without claiming playback delivery."""

import asyncio
from contextlib import aclosing
from pathlib import Path
import re
import wave

from backend.speech.text import SentenceBuffer, SpeechText


class VoiceService:
    def __init__(self, store, tts, settings: dict, data_root: Path, *, emit=None):
        self.store = store
        self.tts = tts
        self.settings = settings
        self.audio_dir = Path(data_root).resolve() / "audio"
        self.queue = asyncio.Queue()
        self.queued = set()
        self.worker = None
        self.closing = False
        self.emit = emit
        self.active_id = None
        self.active_task = None

    async def _emit_job(self, action_id):
        if self.emit:
            job = self.store.get_audio_job(action_id)
            url = f"/api/rp/audio/{action_id}/file" if job["status"] == "completed" else None
            await self.emit({"type": "audio.updated", "conversation_id": job["conversation_id"],
                             "action_id": action_id, "job": {**job, "url": url}})

    async def cancel(self, action_ids):
        if self.active_id in action_ids and self.active_task and not self.active_task.done():
            self.active_task.cancel()
            await asyncio.gather(self.active_task, return_exceptions=True)
        for action_id in action_ids:
            if self.store.get_audio_job(action_id):
                await self._emit_job(action_id)

    def enqueue(self, action: dict, voice_id: str | None = None) -> dict:
        if self.closing:
            raise RuntimeError("Speech generation is closing")
        voice = voice_id or self.settings["default_voice"]
        job, _ = self.store.ensure_audio_job(action["action_id"], voice)
        if self.store.get_action(action["action_id"])["status"] == "interrupted":
            self.store.set_audio_job(action["action_id"], "cancelled")
            return self.store.set_audio_file_state(action["action_id"], "deleted")
        if job["status"] in {"pending", "running"} and action["action_id"] not in self.queued:
            self.queued.add(action["action_id"])
            self.queue.put_nowait(action["action_id"])
            if self.worker is None or self.worker.done():
                self.worker = asyncio.create_task(self._run(), name="RP speech generation")
        return job

    def recover(self):
        for job in self.store.pending_audio_jobs():
            self.enqueue(self.store.get_action(job["action_id"]), job["voice_id"])

    async def wait_idle(self):
        await self.queue.join()

    def _segments(self, text: str) -> list[str]:
        settings = self.settings.get("text", {})
        splitter = SentenceBuffer(settings.get("max_sentence_chars", 48), settings.get("first_segment_min_chars", 12))
        transformer = SpeechText(settings.get("spoken_replacements", {}))
        segments = []
        for sentence in splitter.feed(text, final=True):
            spoken = transformer.convert(sentence)
            if spoken:
                segments.append(spoken)
        return segments

    async def _run(self):
        while True:
            action_id = await self.queue.get()
            self.active_id = action_id
            try:
                self.active_task = asyncio.create_task(self._synthesize(action_id))
                await self.active_task
            except asyncio.CancelledError:
                if self.closing:
                    if self.store.get_audio_job(action_id)["status"] != "cancelled":
                        self.store.set_audio_job(action_id, "pending", error=None)
                    raise
                self.store.set_audio_job(action_id, "cancelled", error=None)
            except Exception as error:
                self.store.set_audio_job(action_id, "failed", error=type(error).__name__ + ": " + str(error))
            finally:
                await self._emit_job(action_id)
                self.active_id = None
                self.active_task = None
                self.queued.discard(action_id)
                self.queue.task_done()

    async def _synthesize(self, action_id: str):
        job = self.store.get_audio_job(action_id)
        if job["status"] not in {"pending", "running"}:
            return
        if re.fullmatch(r"rp_[0-9a-f]+", action_id) is None:
            raise ValueError("Invalid performance action ID")
        action = self.store.get_action(action_id)
        segments = self._segments(action.get("text", ""))
        if not segments:
            self.store.set_audio_job(action_id, "skipped", segments=[], error=None)
            return
        self.store.set_audio_job(action_id, "running", segments=segments, error=None)
        await self._emit_job(action_id)
        await self.tts.start()
        sample_rate = self.tts.sample_rate
        self.audio_dir.mkdir(parents=True, exist_ok=True)
        destination = self.audio_dir / (action_id + ".wav")
        temporary = self.audio_dir / (action_id + ".wav.part")
        pcm_bytes = 0
        try:
            with wave.open(str(temporary), "wb") as audio:
                audio.setnchannels(1)
                audio.setsampwidth(2)
                audio.setframerate(sample_rate)
                for segment in segments:
                    tail = b""
                    iterator = self.tts.stream(segment, self.settings.get("language", "zh"), job["voice_id"])
                    async with aclosing(iterator):
                        async for chunk in iterator:
                            data = tail + chunk
                            complete = len(data) // 2 * 2
                            if complete:
                                audio.writeframesraw(data[:complete])
                                pcm_bytes += complete
                            tail = data[complete:]
                    if tail:
                        raise ValueError("TTS returned an incomplete PCM16 sample")
            if not pcm_bytes:
                raise ValueError("TTS returned no PCM audio")
            temporary.replace(destination)
            self.store.set_audio_job(action_id, "completed", wav_path=str(destination), sample_rate=sample_rate,
                pcm_bytes=pcm_bytes, duration_seconds=pcm_bytes / (sample_rate * 2), segments=segments, error=None)
        finally:
            temporary.unlink(missing_ok=True)

    async def close(self):
        self.closing = True
        if self.worker is not None and not self.worker.done():
            self.worker.cancel()
            await asyncio.gather(self.worker, return_exceptions=True)
        while not self.queue.empty():
            action_id = self.queue.get_nowait()
            self.queued.discard(action_id)
            self.queue.task_done()
        await self.tts.close()
