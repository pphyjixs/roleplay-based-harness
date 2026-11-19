"""Revision-aware voice turns; speculative decisions never perform effects here."""

import asyncio
import time
from uuid import uuid4


class TranscriptLedger:
    """Replace each cloud sentence by ID, retaining only uncommitted text."""

    def __init__(self):
        self.sentences = {}
        self.consumed = {}
        self.stable = set()
        self.retired_through = -1

    def update(self, sentence_id, text, stable=False):
        if type(sentence_id) is not int or sentence_id < 0 or not isinstance(text, str):
            raise ValueError("Invalid ASR sentence")
        if sentence_id <= self.retired_through:
            return self.text
        if len(self.sentences) >= 512 and sentence_id not in self.sentences:
            # Cloud indices increase throughout a call; old committed entries can go.
            for index in sorted(self.sentences)[:256]:
                if index in self.consumed:
                    self.sentences.pop(index, None)
                    self.consumed.pop(index, None)
                    self.stable.discard(index)
                    self.retired_through = max(self.retired_through, index)
        if len(self.sentences) >= 512 and sentence_id not in self.sentences:
            raise ValueError("Too many uncommitted ASR segments")
        old = self.sentences.get(sentence_id)
        if sentence_id in self.stable and old is not None and old != text:
            return self.text
        self.sentences[sentence_id] = text[:16000]
        if stable:
            self.stable.add(sentence_id)
        return self.text

    @property
    def text(self):
        parts = []
        for index in sorted(self.sentences):
            text = self.sentences[index]
            prefix = self.consumed.get(index, "")
            # Late corrections must not repeat or rewrite already committed words.
            if prefix and not text.startswith(prefix):
                continue
            remainder = text[len(prefix):]
            # A late final full stop after the 800ms boundary isn't a new turn.
            if any(character.isalnum() for character in remainder):
                parts.append(remainder)
        return "".join(parts).strip()[:16000]

    @property
    def is_stable(self):
        active = [index for index, text in self.sentences.items()
                  if text != self.consumed.get(index, "")]
        return bool(active) and all(index in self.stable for index in active)

    def consume(self):
        self.consumed.update(self.sentences)


class VoiceTurnController:
    """Coordinate VAD, ASR revisions and pure speculative model work.

    Callback contract: speculate(text) -> opaque candidate; commit(text,
    request_id, candidate) -> durable admission. Only commit may have effects.
    A resumed voice or revised transcript invalidates earlier candidates even
    when an uncooperative model coroutine ignores cancellation.
    """

    def __init__(self, *, emit, commit, speculate=None, interrupt=None, settings=None):
        settings = settings or {}
        self.emit = emit
        self.commit = commit
        self.speculate = speculate
        self.interrupt = interrupt
        self.speculative_delay = settings.get("speculative_silence_ms", 300) / 1000
        self.commit_delay = settings.get("commit_silence_ms", 800) / 1000
        self.max_attempts = settings.get("max_speculative_attempts", 2)
        if not 0 <= self.speculative_delay < self.commit_delay or self.max_attempts < 0:
            raise ValueError("Invalid voice turn timing")
        self.transcript = TranscriptLedger()
        self.revision = 0
        self.request_id = "asr_" + uuid4().hex
        self.attempts = 0
        self.invalidated_attempts = 0
        self.speaking = False
        self.silent_since = None
        self.timer = None
        self.speculation = None
        self.speculation_revision = None
        self.speculation_text = None
        self.candidate = None
        self.candidate_revision = None
        self.candidate_text = None
        self.closed = False
        self.lock = asyncio.Lock()
        self.children = set()

    def _task(self, coroutine):
        task = asyncio.create_task(coroutine)
        self.children.add(task)
        task.add_done_callback(self.children.discard)
        return task

    def _invalidate(self):
        if self.speculation is not None or self.candidate is not None:
            self.invalidated_attempts += 1
        self.revision += 1
        if self.speculation is not None:
            self.speculation.cancel()
            self.speculation = None
        self.speculation_revision = self.speculation_text = None
        self.candidate = self.candidate_revision = self.candidate_text = None

    async def speech_start(self):
        async with self.lock:
            if self.closed or self.speaking:
                return
            self.speaking = True
            self.silent_since = None
            if self.timer:
                self.timer.cancel()
            self._invalidate()
            await self.emit({"type": "speech.started", "request_id": self.request_id,
                             "revision": self.revision})
            if self.interrupt:
                await self.interrupt()

    async def silence(self):
        async with self.lock:
            if self.closed or self.silent_since is not None:
                return
            self.speaking = False
            self.silent_since = time.monotonic()
            self._schedule()

    async def update(self, sentence_id, text, stable=False):
        async with self.lock:
            if self.closed:
                return
            before = self.transcript.text
            current = self.transcript.update(sentence_id, text, stable)
            if current != before:
                self._invalidate()
            if current:
                await self.emit({"type": "utterance" if stable else "transcript",
                                 "text": current, "final": bool(stable),
                                 "sentence_id": sentence_id, "revision": self.revision,
                                 "request_id": self.request_id})
            if not self.speaking:
                if self.silent_since is None:
                    self.silent_since = time.monotonic()
                self._schedule()

    def _schedule(self):
        if self.timer:
            self.timer.cancel()
        self.timer = self._task(self._silence_cycle())

    async def _silence_cycle(self):
        try:
            delay = max(0, self.speculative_delay - (time.monotonic() - self.silent_since))
            await asyncio.sleep(delay)
            async with self.lock:
                if self.closed or self.speaking:
                    return
                text = self.transcript.text
                if (text and self.speculate and self.attempts < self.max_attempts
                        and self.speculation is None and self.candidate is None):
                    self.attempts += 1
                    revision = self.revision
                    self.speculation = self._task(self._prepare(text, revision))
                    self.speculation_revision = revision
                    self.speculation_text = text
                remaining = max(0, self.commit_delay - (time.monotonic() - self.silent_since))
            await asyncio.sleep(remaining)
            async with self.lock:
                if self.closed or self.speaking:
                    return
                if self.invalidated_attempts >= self.max_attempts > 0 and not self.transcript.is_stable:
                    await self.emit({"type": "input.waiting_stable", "request_id": self.request_id})
                    return
                await self._commit()
        except asyncio.CancelledError:
            return
        except Exception:
            await self.emit({"type": "input.error", "message": "语音输入提交失败，请重新发送。"})

    async def _prepare(self, text, revision):
        owner = asyncio.current_task()
        try:
            candidate = await self.speculate(text)
            async with self.lock:
                if not self.closed and not self.speaking and revision == self.revision and text == self.transcript.text:
                    self.candidate = candidate
                    self.candidate_revision = revision
                    self.candidate_text = text
            return candidate
        except asyncio.CancelledError:
            raise
        except Exception:
            # Speculation is an optimization; the durable turn can recompute.
            pass
        finally:
            if self.speculation is owner:
                self.speculation = None

    async def _commit(self):
        text = self.transcript.text
        if not text:
            return
        candidate = self.candidate if self.candidate_revision == self.revision and self.candidate_text == text else None
        pending = self.speculation
        if (candidate is None and pending is not None and self.speculation_revision == self.revision
                and self.speculation_text == text):
            # At the 800ms boundary the durable queue takes ownership of ongoing
            # pure model work; it revalidates its context fingerprint before use.
            candidate = pending
        request_id = self.request_id
        admission = await self.commit(text, request_id, candidate)
        if candidate is pending and pending is not None:
            self.children.discard(pending)
            self.speculation = None
            self.speculation_revision = self.speculation_text = None
        self.transcript.consume()
        self._invalidate()
        self.attempts = self.invalidated_attempts = 0
        self.request_id = "asr_" + uuid4().hex
        self.silent_since = None
        event = {"type": "input.queued", "text": text, "request_id": request_id,
                 "admission": admission}
        if isinstance(admission, dict) and admission.get("turn_id"):
            event["turn_id"] = admission["turn_id"]
        await self.emit(event)

    async def finish(self):
        async with self.lock:
            if self.closed:
                return
            self.speaking = False
            if self.timer:
                self.timer.cancel()
            await self._commit()

    async def close(self):
        async with self.lock:
            self.closed = True
            self._invalidate()
            children = tuple(self.children)
            for task in children:
                task.cancel()
        if children:
            await asyncio.gather(*children, return_exceptions=True)
