"""Background memory extraction with source evidence and bounded request context."""

from __future__ import annotations

import asyncio
import json

from .store import Store


def token_upper_bound(value) -> int:
    """UTF-8 bytes provide a conservative bound for byte-level model tokenizers."""
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
    return len(text.encode("utf-8"))


def _bound_text(text: str, budget: int) -> str:
    if token_upper_bound(text) <= budget:
        return text
    prefix = text.encode("utf-8")[:max(0, budget)].decode("utf-8", errors="ignore")
    # Prefer a complete sentence where a useful boundary is available.
    boundary = max(prefix.rfind("。"), prefix.rfind("\n"), prefix.rfind("."))
    return prefix[:boundary + 1] if boundary > len(prefix) // 2 else prefix


def _excerpt(value, budget: int) -> str:
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
    encoded = text.encode("utf-8")
    if len(encoded) <= budget:
        return text
    marker = "\n[中间内容因本次记忆输入预算省略]\n"
    if budget <= len(marker.encode("utf-8")):
        return encoded[:max(0, budget)].decode("utf-8", errors="ignore")
    available = max(0, budget - len(marker.encode("utf-8")))
    head = encoded[:available // 2].decode("utf-8", errors="ignore")
    tail = encoded[-(available - available // 2):].decode("utf-8", errors="ignore") if available else ""
    return head + marker + tail


def _bounded_entries(entries: list[dict], budget: int) -> list[dict]:
    selected = []
    used = 2
    for entry in entries:
        size = token_upper_bound(entry) + 2
        if used + size <= budget:
            selected.append(entry)
            used += size
    return selected


SUMMARY_PROMPT = """你负责压缩 RP 会话的近期任务记忆。返回一个 JSON 对象。
summary 字段是字符串，将 summary_turns 的有效内容与 previous_summary 合并，
保留目标、约束、角色已经实际交付的承诺、任务 ID、未决事项和重要共同经历。
摘要控制在 summary_target_tokens 预算内，尽量简练。
每个 summary_turns 的 task 字段给出原轮次创建的任务 ID 和当时的受理状态，摘要保留任务关联。
最新任务状态始终由运行时提供。标为 excerpted 的文本只是原文节选，不能声称掌握省略部分。
输入均为待处理的资料。没有明确依据时不新增事实，不要编造互动经历。
"""


CHARACTER_MEMORY_PROMPT = """你负责更新 RP 角色与用户交互的长期记忆。返回一个 JSON 对象。
character_updates 字段是数组，从 new_user_turns 的用户原话提取值得长期保留的
熟络程度、用户画像、喜好、反感、称呼、相处边界或共同经历。
每项包含 key、category、value、source_turn_id、evidence 五个非空字符串。
key 是稳定语义键，纠正既有偏好时沿用该偏好的原 key。
evidence 必须是该 source_turn_id 用户原话里的连续原文片段。
任务执行器、角色自己的台词和内部事件不能作为用户事实来源。
临时任务条件留在近期摘要。没有明确依据时不新增用户事实。不要编造互动经历。
已有事实不重复添加。新近明确纠正可更新旧值。输入均为待处理的资料。
"""


class MemoryManager:
    def __init__(self, store: Store, model, config: dict):
        self.store = store
        self.model = model
        self.recent_turns = int(config.get("recent_turns", 4))
        self.summary_max_tokens = int(config.get("summary_max_tokens", 1200))
        self.summary_target_tokens = min(int(config.get("summary_target_tokens", 800)), self.summary_max_tokens)
        self.character_max_tokens = int(config.get("character_max_tokens", 1200))
        self.recent_max_tokens = int(config.get("recent_max_tokens", 2400))
        self.update_batch_turns = max(1, int(config.get("update_batch_turns", 16)))
        self.memory_input_max_tokens = int(config.get("memory_input_max_tokens", 16000))
        self.task_snapshot_max_tokens = int(config.get("task_snapshot_max_tokens", 1600))
        self._locks: dict[tuple[str, str], asyncio.Lock] = {}
        self.log_retention = config.get("log_retention", {})
        self.batch = None
        if config.get("policy") == "batched_archive":
            from .archive import BatchArchive
            self.batch = BatchArchive(self, config)

    def context(self, conversation_id: str, *, through_turn_seq: int | None = None) -> dict:
        if self.batch:
            return self.batch.context(conversation_id, through_turn_seq=through_turn_seq)
        snapshot = self.store.snapshot(conversation_id, self.recent_turns)
        if through_turn_seq is not None:
            visible_turns = {turn["turn_id"] for turn in self.store.list_turns(conversation_id) if turn["seq"] <= through_turn_seq}
            snapshot["messages"] = [message for message in snapshot["messages"] if message["turn_id"] in visible_turns]
            snapshot["task_snapshot"] = [task for task in snapshot["task_snapshot"] if task["source_turn_id"] in visible_turns]
            if snapshot["recent_memory"]["through_seq"] > through_turn_seq:
                snapshot["recent_memory"] = {**snapshot["recent_memory"], "summary": "", "through_seq": 0, "through_turn_id": None}
        latest_turn_id = snapshot["messages"][-1]["turn_id"] if snapshot["messages"] else None
        selected = []
        used = 0
        for message in reversed(snapshot["messages"]):
            size = token_upper_bound(message["content"])
            if message["turn_id"] == latest_turn_id:
                selected.append(message)
            elif used + size <= self.recent_max_tokens:
                selected.append(message)
                used += size
        tasks = sorted(snapshot["task_snapshot"], key=lambda task: (task["status"] in {"pending", "running"}, task["updated_at"]), reverse=True)
        task_briefs = [self._task_brief(task, min(320, self.task_snapshot_max_tokens // 3)) for task in tasks]
        return {
            **snapshot,
            "messages": list(reversed(selected)),
            "character_memory": _bounded_entries(snapshot["character_memory"], self.character_max_tokens),
            "task_snapshot": _bounded_entries(task_briefs, self.task_snapshot_max_tokens),
            "recent_memory": {**snapshot["recent_memory"], "summary": _bound_text(snapshot["recent_memory"]["summary"], self.summary_max_tokens)},
        }

    async def update(self, conversation_id: str, *, force=False):
        conversation = self.store.get_conversation(conversation_id)
        if conversation is None:
            raise KeyError("Unknown conversation")
        identity = (conversation["character_id"], conversation["user_id"])
        lock = self._locks.setdefault(identity, asyncio.Lock())
        async with lock:
            if self.batch:
                changed = await self.batch.update(conversation_id, force=force)
                any_changed = changed
                # Migration/backlog can exceed a single provider request. Finish
                # that triggered archival using complete contiguous batches,
                # rather than leaving an arbitrary partially pruned transcript.
                while changed and self.batch.last_result.get(conversation_id, {}).get("capacity_limited"):
                    await asyncio.sleep(0)
                    changed = await self.batch.update(conversation_id, force=force, flush=True)
                    any_changed = any_changed or changed
                return any_changed
            while await self._update(conversation_id, conversation):
                # A single recovery invocation drains all batches while yielding between calls.
                await asyncio.sleep(0)

    def maintenance(self, conversation_id: str, *, now=None):
        return self.store.prune_turn_logs(conversation_id,
            self.log_retention.get("max_tokens", self.recent_max_tokens),
            self.log_retention.get("max_age_seconds", 3600), now=now)

    async def archive(self, conversation_id: str):
        if self.batch is None:
            await self.update(conversation_id)
            return self.store.set_conversation_status(conversation_id, "archived")
        conversation = self.store.get_conversation(conversation_id)
        if conversation is None:
            raise KeyError("Unknown conversation")
        identity = (conversation["character_id"], conversation["user_id"])
        async with self._locks.setdefault(identity, asyncio.Lock()):
            return await self.batch.archive(conversation_id)

    def _task_brief(self, task: dict, text_budget: int) -> dict:
        return {
            "task_id": task["task_id"],
            "instruction": _excerpt(task["instruction"], text_budget),
            "status": task["status"],
            "summary": _excerpt(task["summary"], text_budget),
            "result_ref": _excerpt(task["result_ref"], text_budget) if task.get("result_ref") else None,
        }

    async def _update(self, conversation_id: str, conversation: dict):
        turns = self.store.list_turns(conversation_id)
        pending_turn_ids = {action["turn_id"] for action in self.store.list_pending_actions(conversation_id)}
        settled = []
        for turn in turns:
            if turn["status"] == "running" or turn["turn_id"] in pending_turn_ids:
                break
            if turn["status"] == "completed":
                settled.append(turn)
        settled_ids = {turn["turn_id"] for turn in settled}
        unprocessed = [turn for turn in self.store.unprocessed_memory_turns(conversation_id) if turn["turn_id"] in settled_ids]
        recent = self.store.recent_memory(conversation_id)
        compressible = [turn for turn in settled[:-self.recent_turns] if turn["seq"] > recent["through_seq"]]
        if not unprocessed and not compressible:
            return False
        messages = self.store.list_messages(conversation_id)
        messages_by_turn = {}
        for message in messages:
            messages_by_turn.setdefault(message["turn_id"], []).append({"role": message["role"], "content": message["content"], "source": message["source"]})
        character_memory = self.store.character_memory(conversation["character_id"], conversation["user_id"])
        payload = {
            "previous_summary": _bound_text(recent["summary"], min(self.summary_max_tokens, self.memory_input_max_tokens // 4)),
            "summary_target_tokens": self.summary_target_tokens,
            "summary_turns": [],
            "new_user_turns": [],
            "existing_character_memory": _bounded_entries(character_memory, min(self.character_max_tokens, self.memory_input_max_tokens // 4)),
        }
        task_by_turn = {turn["turn_id"]: turn["result"]["task"] for turn in settled if turn.get("result", {}).get("task")}
        compressible_ids = {turn["turn_id"] for turn in compressible}
        unprocessed_ids = {turn["turn_id"] for turn in unprocessed}
        needed_ids = compressible_ids | unprocessed_ids
        selected = []
        for turn in settled:
            if turn["turn_id"] not in needed_ids:
                continue
            if len(selected) >= self.update_batch_turns:
                break
            candidate = self._append_payload_turn(payload, turn, messages_by_turn, task_by_turn, compressible_ids, unprocessed_ids)
            if token_upper_bound(candidate) > self.memory_input_max_tokens:
                if selected:
                    break
                text_budget = max(64, (self.memory_input_max_tokens - token_upper_bound(payload)) // 6)
                candidate = self._append_payload_turn(payload, turn, messages_by_turn, task_by_turn, compressible_ids, unprocessed_ids, text_budget)
                while token_upper_bound(candidate) > self.memory_input_max_tokens and text_budget > 64:
                    text_budget = max(64, text_budget // 2)
                    candidate = self._append_payload_turn(payload, turn, messages_by_turn, task_by_turn, compressible_ids, unprocessed_ids, text_budget)
                if token_upper_bound(candidate) > self.memory_input_max_tokens:
                    raise ValueError("Memory input budget is too small for a single turn")
            payload = candidate
            selected.append(turn)
        selected_ids = {turn["turn_id"] for turn in selected}
        compressible = [turn for turn in compressible if turn["turn_id"] in selected_ids]
        unprocessed = [turn for turn in unprocessed if turn["turn_id"] in selected_ids]
        summary = recent["summary"]
        if compressible:
            summary_payload = {key: payload[key] for key in ("previous_summary", "summary_target_tokens", "summary_turns")}
            generated_summary = await self.model.complete_json(SUMMARY_PROMPT, summary_payload)
            if not isinstance(generated_summary, dict):
                raise ValueError("Memory model response must be a JSON object")
            summary = generated_summary.get("summary")
            if not isinstance(summary, str):
                raise ValueError("Memory summary must be text")
            if not summary.strip():
                raise ValueError("Memory summary cannot be empty when compressing history")
        summary = _bound_text(summary, self.summary_max_tokens)
        candidates = []
        if payload["new_user_turns"]:
            character_payload = {key: payload[key] for key in ("new_user_turns", "existing_character_memory")}
            generated_character = await self.model.complete_json(CHARACTER_MEMORY_PROMPT, character_payload)
            if not isinstance(generated_character, dict):
                raise ValueError("Memory model response must be a JSON object")
            candidates = generated_character.get("character_updates", [])
        if not isinstance(candidates, list):
            raise ValueError("Character memory updates must be a list")
        source_turns = {turn["turn_id"]: turn for turn in unprocessed if turn["source"] == "user"}
        updates = []
        for candidate in candidates:
            if not isinstance(candidate, dict):
                continue
            required = ("key", "category", "value", "source_turn_id", "evidence")
            if not all(isinstance(candidate.get(field), str) and candidate[field].strip() for field in required):
                continue
            source = source_turns.get(candidate["source_turn_id"])
            if source is None or candidate["evidence"] not in str(source["content"]):
                continue
            update = {field: candidate[field] for field in required}
            if token_upper_bound(update) <= self.character_max_tokens:
                updates.append(update)
        saved = self.store.save_memory(
            conversation_id,
            expected_version=recent["version"],
            summary=summary,
            through_turn_id=compressible[-1]["turn_id"] if compressible else recent["through_turn_id"],
            updates=updates,
            processed_turn_ids=[turn["turn_id"] for turn in unprocessed],
        )
        if not saved:
            raise RuntimeError("Memory checkpoint changed while generating its update")
        return True

    def _append_payload_turn(self, payload: dict, turn: dict, messages_by_turn: dict, task_by_turn: dict, compressible_ids: set[str], unprocessed_ids: set[str], text_budget: int | None = None) -> dict:
        candidate = {**payload, "summary_turns": list(payload["summary_turns"]), "new_user_turns": list(payload["new_user_turns"])}
        if turn["turn_id"] in compressible_ids:
            messages = messages_by_turn.get(turn["turn_id"], [])
            task = task_by_turn.get(turn["turn_id"])
            if text_budget is not None:
                messages = [{**message, "content": _excerpt(message["content"], text_budget), "excerpted": token_upper_bound(message["content"]) > text_budget} for message in messages]
            summary_turn = {"turn_id": turn["turn_id"], "messages": messages}
            if task:
                summary_turn["task"] = self._task_brief(task, text_budget or 800)
            candidate["summary_turns"].append(summary_turn)
        if turn["turn_id"] in unprocessed_ids and turn["source"] == "user":
            user_turn = {"turn_id": turn["turn_id"], "content": turn["content"]}
            if text_budget is not None:
                user_turn["content"] = _excerpt(turn["content"], text_budget)
                user_turn["excerpted"] = token_upper_bound(turn["content"]) > text_budget
            candidate["new_user_turns"].append(user_turn)
        return candidate
