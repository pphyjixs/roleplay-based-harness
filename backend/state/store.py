"""Durable RP conversations, delivery receipts, memory, and mock task records."""

from __future__ import annotations

import json
import hashlib
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json(value) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


class Store:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._db = sqlite3.connect(self.path, check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        self._db.execute("PRAGMA foreign_keys = ON")
        # Effective for new databases. Existing files keep their format and
        # reuse freed pages without an expensive migration VACUUM.
        self._db.execute("PRAGMA auto_vacuum = INCREMENTAL")
        self._db.execute("PRAGMA journal_mode = WAL")
        self._db.executescript("""
            CREATE TABLE IF NOT EXISTS conversations (
                conversation_id TEXT PRIMARY KEY,
                character_id TEXT NOT NULL,
                user_id TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS turns (
                seq INTEGER PRIMARY KEY AUTOINCREMENT,
                turn_id TEXT NOT NULL UNIQUE,
                conversation_id TEXT NOT NULL REFERENCES conversations(conversation_id),
                request_id TEXT NOT NULL,
                source TEXT NOT NULL,
                content TEXT NOT NULL,
                status TEXT NOT NULL,
                result TEXT,
                error TEXT,
                created_at TEXT NOT NULL,
                UNIQUE (conversation_id, request_id)
            );
            CREATE TABLE IF NOT EXISTS messages (
                seq INTEGER PRIMARY KEY AUTOINCREMENT,
                message_id TEXT NOT NULL UNIQUE,
                conversation_id TEXT NOT NULL REFERENCES conversations(conversation_id),
                turn_id TEXT NOT NULL REFERENCES turns(turn_id),
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                source TEXT NOT NULL,
                delivery_state TEXT NOT NULL,
                action_id TEXT UNIQUE,
                created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS actions (
                action_id TEXT PRIMARY KEY,
                conversation_id TEXT NOT NULL REFERENCES conversations(conversation_id),
                turn_id TEXT NOT NULL UNIQUE REFERENCES turns(turn_id),
                payload TEXT NOT NULL,
                status TEXT NOT NULL,
                delivered_text TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS tasks (
                task_id TEXT PRIMARY KEY,
                conversation_id TEXT NOT NULL REFERENCES conversations(conversation_id),
                source_turn_id TEXT NOT NULL UNIQUE REFERENCES turns(turn_id),
                instruction TEXT NOT NULL,
                context TEXT NOT NULL,
                constraints TEXT NOT NULL,
                expected_output TEXT,
                status TEXT NOT NULL,
                summary TEXT NOT NULL,
                result_ref TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS task_events (
                event_id TEXT PRIMARY KEY,
                task_id TEXT NOT NULL REFERENCES tasks(task_id),
                kind TEXT NOT NULL,
                summary TEXT NOT NULL,
                result_ref TEXT,
                accepted INTEGER NOT NULL,
                created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS recent_memory (
                conversation_id TEXT PRIMARY KEY REFERENCES conversations(conversation_id),
                summary TEXT NOT NULL DEFAULT '',
                through_seq INTEGER NOT NULL DEFAULT 0,
                through_turn_id TEXT,
                version INTEGER NOT NULL DEFAULT 0,
                updated_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS character_memory (
                character_id TEXT NOT NULL,
                user_id TEXT NOT NULL,
                memory_key TEXT NOT NULL,
                category TEXT NOT NULL,
                value TEXT NOT NULL,
                source_turn_id TEXT NOT NULL REFERENCES turns(turn_id),
                evidence TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                PRIMARY KEY (character_id, user_id, memory_key)
            );
            CREATE TABLE IF NOT EXISTS memory_processed_turns (
                turn_id TEXT PRIMARY KEY REFERENCES turns(turn_id)
            );
            CREATE TABLE IF NOT EXISTS audio_jobs (
                action_id TEXT PRIMARY KEY REFERENCES actions(action_id),
                voice_id TEXT NOT NULL,
                status TEXT NOT NULL,
                wav_path TEXT,
                sample_rate INTEGER,
                pcm_bytes INTEGER NOT NULL DEFAULT 0,
                duration_seconds REAL NOT NULL DEFAULT 0,
                segments TEXT NOT NULL DEFAULT '[]',
                error TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS turns_by_conversation ON turns(conversation_id, seq);
            CREATE INDEX IF NOT EXISTS messages_by_turn ON messages(turn_id, seq);
            CREATE INDEX IF NOT EXISTS tasks_by_conversation ON tasks(conversation_id);
        """)
        # Additive migrations retain existing conversations and receipts.
        migrations = {
            "conversations": {"presentation_id": "TEXT", "presentation_revision": "INTEGER NOT NULL DEFAULT 0", "status": "TEXT NOT NULL DEFAULT 'active'", "archived_at": "TEXT", "last_external_input_at": "TEXT", "idle_notified_at": "TEXT"},
            "turns": {"input_source": "TEXT NOT NULL DEFAULT 'text'", "rp_cancelled": "INTEGER NOT NULL DEFAULT 0", "archived": "INTEGER NOT NULL DEFAULT 0", "input_hash": "TEXT", "evidence_content": "TEXT", "image_count": "INTEGER NOT NULL DEFAULT 0"},
            "audio_jobs": {"playback_status": "TEXT NOT NULL DEFAULT 'pending'", "file_state": "TEXT NOT NULL DEFAULT 'present'", "file_deleted_at": "TEXT"},
            "task_events": {"event_hash": "TEXT"},
        }
        for table, columns in migrations.items():
            existing = {row[1] for row in self._db.execute(f"PRAGMA table_info({table})")}
            for name, definition in columns.items():
                if name not in existing:
                    self._db.execute(f"ALTER TABLE {table} ADD COLUMN {name} {definition}")
        self._db.commit()

    @staticmethod
    def content_hash(content, source="user", input_source="text"):
        encoded = json.dumps([content, source, input_source], ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def close(self):
        with self._lock:
            self._db.close()

    def ensure_conversation(self, conversation_id: str, character_id: str, user_id: str) -> dict:
        with self._lock, self._db:
            self._db.execute(
                "INSERT OR IGNORE INTO conversations (conversation_id, character_id, user_id, created_at) VALUES (?, ?, ?, ?)",
                (conversation_id, character_id, user_id, _now()),
            )
            conversation = self.get_conversation(conversation_id)
            if conversation["character_id"] != character_id or conversation["user_id"] != user_id:
                raise ValueError("Conversation already belongs to a different character or user")
            return conversation

    def get_conversation(self, conversation_id: str) -> dict | None:
        with self._lock:
            row = self._db.execute(
                "SELECT * FROM conversations WHERE conversation_id = ?", (conversation_id,)
            ).fetchone()
            return dict(row) if row else None

    def list_conversations(self) -> list[dict]:
        with self._lock:
            return [dict(row) for row in self._db.execute("SELECT * FROM conversations ORDER BY created_at").fetchall()]

    def touch_activity(self, conversation_id: str, kind=None, *, now=None) -> dict:
        with self._lock, self._db:
            if self.get_conversation(conversation_id) is None:
                raise KeyError("Unknown conversation")
            self._db.execute("UPDATE conversations SET last_external_input_at = ?, idle_notified_at = NULL WHERE conversation_id = ?", (now or _now(), conversation_id))
            return self.get_conversation(conversation_id)

    def claim_idle(self, conversation_id: str, expected_activity_at, now=None) -> bool:
        with self._lock, self._db:
            cursor = self._db.execute("UPDATE conversations SET idle_notified_at = ? WHERE conversation_id = ? AND status = 'active' AND last_external_input_at IS ? AND idle_notified_at IS NULL", (now or _now(), conversation_id, expected_activity_at))
            return cursor.rowcount == 1

    def reset_idle_notification(self, conversation_id: str):
        with self._lock, self._db:
            self._db.execute("UPDATE conversations SET idle_notified_at = NULL WHERE conversation_id = ?", (conversation_id,))

    def set_conversation_status(self, conversation_id: str, status: str) -> dict:
        if status not in {"active", "archiving", "archived"}:
            raise ValueError("Invalid conversation status")
        with self._lock, self._db:
            if self.get_conversation(conversation_id) is None:
                raise KeyError("Unknown conversation")
            self._db.execute("UPDATE conversations SET status = ?, archived_at = ? WHERE conversation_id = ?", (status, _now() if status == "archived" else None, conversation_id))
            return self.get_conversation(conversation_id)

    def replace_turn_content(self, turn_id: str, content: str) -> dict:
        if not isinstance(content, str):
            raise ValueError("Replacement transcript must be text")
        with self._lock, self._db:
            turn = self.get_turn(turn_id)
            if turn is None:
                raise KeyError("Unknown turn")
            if turn["archived"]:
                raise ValueError("Archived text is already represented by memory")
            original_hash = turn["input_hash"] or self.content_hash(turn["content"], turn["source"], turn["input_source"])
            self._db.execute("UPDATE turns SET content = ?, input_hash = ?, evidence_content = COALESCE(evidence_content, ?) WHERE turn_id = ?", (_json(content), original_hash, _json(turn["content"]), turn_id))
            self._db.execute("UPDATE messages SET content = ? WHERE turn_id = ? AND role IN ('user', 'internal')", (_json(content), turn_id))
            return self.get_turn(turn_id)

    def set_presentation(self, conversation_id: str, presentation_id: str) -> dict:
        with self._lock, self._db:
            current = self.get_conversation(conversation_id)
            if current is None:
                raise KeyError("Unknown conversation")
            if current["presentation_id"] != presentation_id:
                self._db.execute("UPDATE conversations SET presentation_id = ?, presentation_revision = presentation_revision + 1 WHERE conversation_id = ?", (presentation_id, conversation_id))
            return self.get_conversation(conversation_id)

    def begin_turn(self, conversation_id: str, request_id: str, content, source: str = "user", *, input_source="text", queued=False, input_fingerprint=None, evidence_content=None, image_count=0) -> tuple[dict, bool]:
        """Admit a confirmed input once; speculative candidates stay outside this store."""
        with self._lock, self._db:
            existing = self._db.execute(
                "SELECT turn_id, content, source, input_source, input_hash FROM turns WHERE conversation_id = ? AND request_id = ?",
                (conversation_id, request_id),
            ).fetchone()
            if existing:
                input_hash = existing["input_hash"] or self.content_hash(json.loads(existing["content"]), existing["source"], existing["input_source"])
                if input_hash != (input_fingerprint or self.content_hash(content, source, input_source)):
                    raise ValueError("Request ID was already used for a different input")
                return self.get_turn(existing["turn_id"]), False
            conversation = self.get_conversation(conversation_id)
            if conversation is None:
                raise KeyError("Unknown conversation")
            if conversation["status"] != "active":
                raise ValueError("Conversation is not active")
            turn_id = "turn_" + uuid4().hex
            created_at = _now()
            self._db.execute(
                "INSERT INTO turns (turn_id, conversation_id, request_id, source, content, status, created_at, input_source, input_hash, evidence_content, image_count) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (turn_id, conversation_id, request_id, source, _json(content), "queued" if queued else "running", created_at, input_source, input_fingerprint or self.content_hash(content, source, input_source), _json(evidence_content) if evidence_content is not None else None, image_count),
            )
            self._db.execute(
                "INSERT INTO messages (message_id, conversation_id, turn_id, role, content, source, delivery_state, created_at) VALUES (?, ?, ?, ?, ?, ?, 'delivered', ?)",
                ("msg_" + uuid4().hex, conversation_id, turn_id, "user" if source == "user" else "internal", _json(content), source, created_at),
            )
            return self.get_turn(turn_id), True

    def mark_turn_running(self, turn_id: str):
        with self._lock, self._db:
            self._db.execute("UPDATE turns SET status = 'running' WHERE turn_id = ? AND status = 'queued'", (turn_id,))

    def _turn(self, row) -> dict | None:
        if row is None:
            return None
        turn = dict(row)
        turn["content"] = json.loads(turn["content"])
        turn["result"] = json.loads(turn["result"]) if turn["result"] else None
        if turn.get("evidence_content") is not None:
            turn["evidence_content"] = json.loads(turn["evidence_content"])
        return turn

    def get_turn(self, turn_id: str) -> dict | None:
        with self._lock:
            return self._turn(self._db.execute("SELECT * FROM turns WHERE turn_id = ?", (turn_id,)).fetchone())

    def get_turn_by_request(self, conversation_id: str, request_id: str) -> dict | None:
        with self._lock:
            return self._turn(self._db.execute("SELECT * FROM turns WHERE conversation_id = ? AND request_id = ?", (conversation_id, request_id)).fetchone())

    def list_turns(self, conversation_id: str) -> list[dict]:
        with self._lock:
            rows = self._db.execute("SELECT * FROM turns WHERE conversation_id = ? ORDER BY seq", (conversation_id,)).fetchall()
            return [self._turn(row) for row in rows]

    def list_messages(self, conversation_id: str) -> list[dict]:
        with self._lock:
            rows = self._db.execute(
                "SELECT m.* FROM messages m JOIN turns t ON t.turn_id = m.turn_id WHERE m.conversation_id = ? ORDER BY t.seq, m.seq",
                (conversation_id,),
            ).fetchall()
            return [{**dict(row), "content": json.loads(row["content"])} for row in rows]

    def _action(self, row) -> dict | None:
        if row is None:
            return None
        action = dict(row)
        payload = json.loads(action.pop("payload"))
        return {**payload, **action}

    def get_action(self, action_id: str) -> dict | None:
        with self._lock:
            return self._action(self._db.execute("SELECT * FROM actions WHERE action_id = ?", (action_id,)).fetchone())

    def list_pending_actions(self, conversation_id: str) -> list[dict]:
        with self._lock:
            rows = self._db.execute(
                "SELECT * FROM actions WHERE conversation_id = ? AND status IN ('pending', 'partial') ORDER BY created_at",
                (conversation_id,),
            ).fetchall()
            return [self._action(row) for row in rows]

    def finish_turn(self, turn_id: str, rp: dict, task: dict, *, model_response=None) -> dict:
        """Commit both decisions and their durable work in one transaction."""
        with self._lock, self._db:
            turn = self.get_turn(turn_id)
            if turn is None:
                raise KeyError("Unknown turn")
            if turn["result"] is not None:
                return turn["result"]
            if turn["status"] != "running":
                raise ValueError("Turn is no longer running")
            conversation_id = turn["conversation_id"]
            now = _now()
            action = None
            task_record = None
            if rp["execute"]:
                action_id = "rp_" + uuid4().hex
                self._db.execute(
                    "INSERT INTO actions (action_id, conversation_id, turn_id, payload, status, created_at) VALUES (?, ?, ?, ?, 'pending', ?)",
                    (action_id, conversation_id, turn_id, _json(rp), now),
                )
                action = self.get_action(action_id)
            if task["execute"]:
                task_id = "task_" + uuid4().hex
                self._db.execute(
                    "INSERT INTO tasks VALUES (?, ?, ?, ?, ?, ?, ?, 'pending', ?, NULL, ?, ?)",
                    (task_id, conversation_id, turn_id, task["instruction"], _json(task.get("context", {})), _json(task.get("constraints", [])), task.get("expected_output"), task["instruction"], now, now),
                )
                task_record = self.get_task(task_id)
            result = {"turn_id": turn_id, "status": "completed", "rp": action, "task": task_record}
            self._db.execute(
                "UPDATE turns SET status = 'completed', result = ? WHERE turn_id = ?",
                (_json({**result, "decisions": {"rp_perform": rp, "task_perform": task}, "model_response": model_response}), turn_id),
            )
            return self.get_turn(turn_id)["result"]

    def mark_turn_failed(self, turn_id: str, error: str):
        with self._lock, self._db:
            self._db.execute("UPDATE turns SET status = 'failed', error = ? WHERE turn_id = ? AND status IN ('running', 'queued')", (error, turn_id))

    def record_delivery(self, action_id: str, text: str | None = None) -> dict:
        """Acknowledge a monotonic delivered prefix, keeping a single history message."""
        with self._lock, self._db:
            action = self.get_action(action_id)
            if action is None:
                raise KeyError("Unknown performance action")
            if self.get_turn(action["turn_id"])["archived"]:
                return action
            expected = action.get("text", "")
            delivered = expected if text is None else text
            previous = action["delivered_text"]
            if not isinstance(delivered, str) or not expected.startswith(delivered):
                raise ValueError("Delivered text must be a prefix of the accepted reply")
            if previous.startswith(delivered):
                delivered = previous
            elif not delivered.startswith(previous):
                raise ValueError("Delivery receipt conflicts with the earlier prefix")
            status = "interrupted" if action["status"] == "interrupted" else "delivered" if delivered == expected else "partial"
            self._db.execute("UPDATE actions SET status = ?, delivered_text = ? WHERE action_id = ?", (status, delivered, action_id))
            if delivered:
                self._db.execute(
                    "INSERT INTO messages (message_id, conversation_id, turn_id, role, content, source, delivery_state, action_id, created_at) VALUES (?, ?, ?, 'assistant', ?, 'rp_perform', 'delivered', ?, ?) ON CONFLICT(action_id) DO UPDATE SET content = excluded.content",
                    ("msg_" + uuid4().hex, action["conversation_id"], action["turn_id"], _json(delivered), action_id, _now()),
                )
            return self.get_action(action_id)

    def ensure_audio_job(self, action_id: str, voice_id: str) -> tuple[dict, bool]:
        """Register synthesis once for an accepted performance action."""
        with self._lock, self._db:
            if self.get_action(action_id) is None:
                raise KeyError("Unknown performance action")
            existing = self.get_audio_job(action_id)
            if existing is not None:
                if existing["voice_id"] != voice_id:
                    raise ValueError("This audio job already belongs to a different voice")
                return existing, False
            now = _now()
            self._db.execute(
                "INSERT INTO audio_jobs (action_id, voice_id, status, created_at, updated_at) VALUES (?, ?, 'pending', ?, ?)",
                (action_id, voice_id, now, now),
            )
            return self.get_audio_job(action_id), True

    def _audio_job(self, row) -> dict | None:
        if row is None:
            return None
        job = dict(row)
        job["segments"] = json.loads(job["segments"])
        return job

    def get_audio_job(self, action_id: str) -> dict | None:
        with self._lock:
            row = self._db.execute(
                "SELECT j.*, a.conversation_id, a.turn_id FROM audio_jobs j JOIN actions a ON a.action_id = j.action_id WHERE j.action_id = ?",
                (action_id,),
            ).fetchone()
            return self._audio_job(row)

    def list_audio_jobs(self, conversation_id: str) -> list[dict]:
        with self._lock:
            rows = self._db.execute(
                "SELECT j.*, a.conversation_id, a.turn_id FROM audio_jobs j JOIN actions a ON a.action_id = j.action_id WHERE a.conversation_id = ? ORDER BY j.created_at, j.action_id",
                (conversation_id,),
            ).fetchall()
            return [self._audio_job(row) for row in rows]

    def pending_audio_jobs(self) -> list[dict]:
        with self._lock:
            rows = self._db.execute(
                "SELECT j.*, a.conversation_id, a.turn_id FROM audio_jobs j JOIN actions a ON a.action_id = j.action_id WHERE j.status IN ('pending', 'running') ORDER BY j.created_at, j.action_id"
            ).fetchall()
            return [self._audio_job(row) for row in rows]

    def actions_without_audio_jobs(self) -> list[dict]:
        """Find accepted performances whose speech registration was interrupted."""
        with self._lock:
            rows = self._db.execute(
                "SELECT a.* FROM actions a JOIN turns t ON t.turn_id = a.turn_id LEFT JOIN audio_jobs j ON j.action_id = a.action_id WHERE j.action_id IS NULL AND a.status NOT IN ('interrupted', 'archived') AND t.archived = 0 ORDER BY a.created_at, a.action_id"
            ).fetchall()
            return [self._action(row) for row in rows]

    def set_audio_job(self, action_id: str, status: str, **details) -> dict:
        statuses = {"pending", "running", "completed", "failed", "skipped", "cancelled"}
        fields = {"wav_path", "sample_rate", "pcm_bytes", "duration_seconds", "segments", "error"}
        if status not in statuses or set(details) - fields:
            raise ValueError("Invalid audio job update")
        with self._lock, self._db:
            if self.get_audio_job(action_id) is None:
                raise KeyError("Unknown audio job")
            values = {"status": status, "updated_at": _now(), **details}
            if "segments" in values:
                values["segments"] = _json(values["segments"])
            assignments = ", ".join(f"{field} = ?" for field in values)
            self._db.execute(
                f"UPDATE audio_jobs SET {assignments} WHERE action_id = ?",
                (*values.values(), action_id),
            )
            return self.get_audio_job(action_id)

    def set_audio_file_state(self, action_id: str, state: str) -> dict:
        if state not in {"present", "deleted"}:
            raise ValueError("Invalid audio file state")
        with self._lock, self._db:
            if self.get_audio_job(action_id) is None:
                raise KeyError("Unknown audio job")
            if state == "deleted":
                self._db.execute("UPDATE audio_jobs SET file_state = 'deleted', wav_path = NULL, file_deleted_at = ?, updated_at = ? WHERE action_id = ?", (_now(), _now(), action_id))
            else:
                self._db.execute("UPDATE audio_jobs SET file_state = 'present', file_deleted_at = NULL WHERE action_id = ?", (action_id,))
            return self.get_audio_job(action_id)

    def record_playback(self, action_id: str, status: str) -> dict:
        if status not in {"playing", "completed", "interrupted", "failed"}:
            raise ValueError("Invalid playback status")
        with self._lock, self._db:
            job = self.get_audio_job(action_id)
            if job is None:
                raise KeyError("Unknown audio job")
            previous = job["playback_status"]
            if previous in {"completed", "interrupted"}:
                return job
            if status in {"playing", "completed"} and job["status"] != "completed":
                raise ValueError("Audio is not ready for playback")
            self._db.execute("UPDATE audio_jobs SET playback_status = ?, updated_at = ? WHERE action_id = ?", (status, _now(), action_id))
            return self.get_audio_job(action_id)

    def cancel_pending_performances(self, conversation_id: str) -> list[str]:
        """Cancel speech from admitted inputs; task intent still belongs to the user."""
        with self._lock, self._db:
            rows = self._db.execute("SELECT turn_id FROM turns WHERE conversation_id = ? AND status IN ('queued', 'running') AND rp_cancelled = 0", (conversation_id,)).fetchall()
            self._db.execute("UPDATE turns SET rp_cancelled = 1 WHERE conversation_id = ? AND status IN ('queued', 'running')", (conversation_id,))
            return [row[0] for row in rows]

    def interrupt_actions(self, conversation_id: str, action_ids=None) -> list[dict]:
        """Retire existing performances without erasing already displayed history."""
        with self._lock, self._db:
            if action_ids is None:
                rows = self._db.execute("SELECT action_id FROM actions WHERE conversation_id = ?", (conversation_id,)).fetchall()
                action_ids = [row[0] for row in rows]
            actions = []
            for action_id in action_ids:
                action = self.get_action(action_id)
                if action is None or action["conversation_id"] != conversation_id:
                    raise KeyError("Unknown action in this conversation")
                changed = False
                if action["status"] in {"pending", "partial"}:
                    self._db.execute("UPDATE actions SET status = 'interrupted' WHERE action_id = ?", (action_id,))
                    changed = True
                job = self.get_audio_job(action_id)
                if job is None and action["status"] == "delivered":
                    # A display receipt can beat synthesis registration; do not
                    # recreate that retired speech after a crash in this gap.
                    self._db.execute("UPDATE actions SET status = 'interrupted' WHERE action_id = ?", (action_id,))
                    changed = True
                if job and job["playback_status"] not in {"completed", "interrupted"}:
                    synthesis_status = "cancelled" if job["status"] in {"pending", "running"} else job["status"]
                    self._db.execute("UPDATE audio_jobs SET status = ?, playback_status = 'interrupted', updated_at = ? WHERE action_id = ?", (synthesis_status, _now(), action_id))
                    changed = True
                if changed:
                    actions.append(self.get_action(action_id))
            return actions

    def _task(self, row) -> dict | None:
        if row is None:
            return None
        task = dict(row)
        task["context"] = json.loads(task["context"])
        task["constraints"] = json.loads(task["constraints"])
        return task

    def get_task(self, task_id: str) -> dict | None:
        with self._lock:
            return self._task(self._db.execute("SELECT * FROM tasks WHERE task_id = ?", (task_id,)).fetchone())

    def list_tasks(self, conversation_id: str) -> list[dict]:
        with self._lock:
            return [self._task(row) for row in self._db.execute("SELECT * FROM tasks WHERE conversation_id = ? ORDER BY created_at", (conversation_id,)).fetchall()]

    def add_task_event(self, task_id: str, event_id: str, kind: str, summary: str, result_ref: str | None = None) -> tuple[dict, bool]:
        states = {"progress": "running", "completed": "completed", "failed": "failed", "cancelled": "cancelled"}
        if kind not in states:
            raise ValueError("Unknown task event kind")
        with self._lock, self._db:
            task = self.get_task(task_id)
            if task is None:
                raise KeyError("Unknown task")
            existing = self._db.execute("SELECT * FROM task_events WHERE event_id = ?", (event_id,)).fetchone()
            if existing:
                event_hash = self.content_hash([task_id, kind, summary, result_ref], "event", "task")
                previous_hash = existing["event_hash"] or self.content_hash([existing["task_id"], existing["kind"], existing["summary"], existing["result_ref"]], "event", "task")
                if previous_hash != event_hash:
                    raise ValueError("Event ID was already used for a different event")
                return task, False
            accepted = task["status"] not in {"completed", "failed", "cancelled"}
            now = _now()
            event_hash = self.content_hash([task_id, kind, summary, result_ref], "event", "task")
            self._db.execute("INSERT INTO task_events (event_id, task_id, kind, summary, result_ref, accepted, created_at, event_hash) VALUES (?, ?, ?, ?, ?, ?, ?, ?)", (event_id, task_id, kind, summary, result_ref, int(accepted), now, event_hash))
            if accepted:
                self._db.execute("UPDATE tasks SET status = ?, summary = ?, result_ref = COALESCE(?, result_ref), updated_at = ? WHERE task_id = ?", (states[kind], summary, result_ref, now, task_id))
            return self.get_task(task_id), accepted

    def pending_task_notifications(self) -> list[dict]:
        with self._lock:
            rows = self._db.execute(
                "SELECT e.task_id, e.event_id, e.kind, e.summary, e.result_ref, t.conversation_id FROM task_events e JOIN tasks t ON t.task_id = e.task_id WHERE e.accepted = 1 AND e.event_id NOT LIKE 'mock-dispatch:%' AND NOT EXISTS (SELECT 1 FROM turns r WHERE r.conversation_id = t.conversation_id AND r.request_id = 'task:' || e.task_id || ':' || e.event_id) ORDER BY e.created_at"
            ).fetchall()
            return [dict(row) for row in rows]

    def recent_memory(self, conversation_id: str) -> dict:
        with self._lock:
            row = self._db.execute("SELECT * FROM recent_memory WHERE conversation_id = ?", (conversation_id,)).fetchone()
            return dict(row) if row else {"conversation_id": conversation_id, "summary": "", "through_seq": 0, "through_turn_id": None, "version": 0}

    def character_memory(self, character_id: str, user_id: str) -> list[dict]:
        with self._lock:
            return [dict(row) for row in self._db.execute("SELECT memory_key AS key, category, value, source_turn_id, evidence, updated_at FROM character_memory WHERE character_id = ? AND user_id = ? ORDER BY updated_at DESC, memory_key", (character_id, user_id)).fetchall()]

    def unprocessed_memory_turns(self, conversation_id: str) -> list[dict]:
        with self._lock:
            rows = self._db.execute("SELECT t.* FROM turns t LEFT JOIN memory_processed_turns p ON p.turn_id = t.turn_id WHERE t.conversation_id = ? AND t.status = 'completed' AND p.turn_id IS NULL ORDER BY t.seq", (conversation_id,)).fetchall()
            return [self._turn(row) for row in rows]

    def save_memory(self, conversation_id: str, *, expected_version: int, summary: str, through_turn_id: str | None, updates: list[dict], processed_turn_ids: list[str], archive_turn_ids=None, close_conversation=False, character_budget=None, expected_transcript_hash=None, transcript_turn_ids=None) -> bool:
        """Apply a background summary only when its input checkpoint is still current."""
        with self._lock, self._db:
            conversation = self.get_conversation(conversation_id)
            if conversation is None:
                raise KeyError("Unknown conversation")
            current = self.recent_memory(conversation_id)
            if current["version"] != expected_version:
                return False
            if expected_transcript_hash is not None and self.transcript_fingerprint(transcript_turn_ids or []) != expected_transcript_hash:
                return False
            through_seq = current["through_seq"]
            if through_turn_id is not None:
                turn = self.get_turn(through_turn_id)
                if turn is None or turn["conversation_id"] != conversation_id or turn["seq"] < through_seq:
                    raise ValueError("Invalid memory coverage checkpoint")
                through_seq = turn["seq"]
            now = _now()
            self._db.execute("INSERT INTO recent_memory VALUES (?, ?, ?, ?, ?, ?) ON CONFLICT(conversation_id) DO UPDATE SET summary = excluded.summary, through_seq = excluded.through_seq, through_turn_id = excluded.through_turn_id, version = excluded.version, updated_at = excluded.updated_at", (conversation_id, summary, through_seq, through_turn_id or current["through_turn_id"], expected_version + 1, now))
            for update in updates:
                source = self.get_turn(update["source_turn_id"])
                if source is None or source["conversation_id"] != conversation_id or source["source"] != "user" or source["status"] != "completed":
                    raise ValueError("Character memory requires a completed user turn as evidence")
                if not update["evidence"] or update["evidence"] not in str(source.get("evidence_content") if source.get("evidence_content") is not None else source["content"]):
                    raise ValueError("Character memory evidence must quote the source user input")
                self._db.execute("INSERT INTO character_memory VALUES (?, ?, ?, ?, ?, ?, ?, ?) ON CONFLICT(character_id, user_id, memory_key) DO UPDATE SET category = excluded.category, value = excluded.value, source_turn_id = excluded.source_turn_id, evidence = excluded.evidence, updated_at = excluded.updated_at WHERE (SELECT seq FROM turns WHERE turn_id = excluded.source_turn_id) >= (SELECT seq FROM turns WHERE turn_id = character_memory.source_turn_id)", (conversation["character_id"], conversation["user_id"], update["key"], update["category"], update["value"], update["source_turn_id"], update["evidence"], now))
            for turn_id in processed_turn_ids:
                self._db.execute("INSERT OR IGNORE INTO memory_processed_turns VALUES (?)", (turn_id,))
            for turn_id in archive_turn_ids or []:
                turn = self.get_turn(turn_id)
                if turn is None or turn["conversation_id"] != conversation_id or turn["status"] not in {"completed", "failed"}:
                    raise ValueError("Cannot archive an unfinished or foreign turn")
                pending = self._db.execute("SELECT 1 FROM actions WHERE turn_id = ? AND status IN ('pending', 'partial')", (turn_id,)).fetchone()
                if pending:
                    raise ValueError("Cannot archive an unacknowledged performance")
                self._db.execute("DELETE FROM messages WHERE turn_id = ?", (turn_id,))
                result = self._compact_result(turn["result"], turn_id, turn["status"])
                original_hash = turn["input_hash"] or self.content_hash(turn["content"], turn["source"], turn["input_source"])
                self._db.execute("UPDATE turns SET archived = 1, content = '\"\"', evidence_content = NULL, input_hash = ?, result = ?, error = NULL WHERE turn_id = ?", (original_hash, _json(result), turn_id))
                action = self._db.execute("SELECT * FROM actions WHERE turn_id = ?", (turn_id,)).fetchone()
                if action:
                    payload = json.loads(action["payload"])
                    retained = {key: payload[key] for key in ("execute", "presentation_id", "presentation_revision") if key in payload}
                    self._db.execute("UPDATE actions SET payload = ?, status = 'archived', delivered_text = '' WHERE action_id = ?", (_json(retained), action["action_id"]))
                    self._db.execute("UPDATE audio_jobs SET segments = '[]', error = NULL WHERE action_id = ?", (action["action_id"],))
                self._db.execute("UPDATE tasks SET instruction = '', context = '[]', constraints = '[]', expected_output = NULL WHERE source_turn_id = ? AND status IN ('completed', 'failed', 'cancelled')", (turn_id,))
            if character_budget is not None:
                used = 2
                for fact in self.character_memory(conversation["character_id"], conversation["user_id"]):
                    size = len(_json(fact).encode("utf-8")) + 1
                    if used + size > character_budget:
                        self._db.execute("DELETE FROM character_memory WHERE character_id = ? AND user_id = ? AND memory_key = ?", (conversation["character_id"], conversation["user_id"], fact["key"]))
                    else:
                        used += size
            if close_conversation:
                self._db.execute("UPDATE conversations SET status = 'archived', archived_at = ? WHERE conversation_id = ?", (now, conversation_id))
            return True

    def transcript_fingerprint(self, turn_ids):
        with self._lock:
            records = []
            for turn_id in sorted(set(turn_ids)):
                turn = self._db.execute("SELECT turn_id, content, status, archived FROM turns WHERE turn_id = ?", (turn_id,)).fetchone()
                messages = self._db.execute("SELECT role, content, delivery_state, action_id FROM messages WHERE turn_id = ? ORDER BY seq", (turn_id,)).fetchall()
                records.append([dict(turn) if turn else None, [dict(message) for message in messages]])
            return hashlib.sha256(_json(records).encode("utf-8")).hexdigest()

    @staticmethod
    def _compact_result(result, turn_id, status):
        if result is None:
            return {"turn_id": turn_id, "status": status, "rp": None, "task": None,
                    "decisions": {"rp_perform": {"execute": False}, "task_perform": {"execute": False}}}
        rp = {"action_id": result["rp"]["action_id"]} if result.get("rp") else None
        task = {"task_id": result["task"]["task_id"]} if result.get("task") else None
        return {"turn_id": turn_id, "status": status, "rp": rp, "task": task,
                "decisions": {"rp_perform": {"execute": rp is not None}, "task_perform": {"execute": task is not None}}}

    def prune_turn_logs(self, conversation_id: str, max_tokens: int, max_age_seconds: float, *, now=None) -> int:
        """Remove raw model/tool traces at either the age or aggregate-size boundary."""
        current = datetime.fromisoformat(now) if isinstance(now, str) else now or datetime.now(timezone.utc)
        used = 0
        removed = 0
        with self._lock, self._db:
            items = []
            for turn in self.list_turns(conversation_id):
                result = turn["result"]
                compact = self._compact_result(result, turn["turn_id"], turn["status"])
                trace = {"model_response": result.get("model_response"), "decisions": result.get("decisions")} if result and result != compact else {}
                if turn["error"]:
                    trace["error"] = turn["error"]
                size = len(_json(trace).encode("utf-8")) if trace else 0
                if size and turn["status"] not in {"queued", "running"}:
                    items.append((turn["created_at"], size, "turn", turn))
            events = self._db.execute("SELECT e.* FROM task_events e JOIN tasks t ON t.task_id = e.task_id JOIN conversations c ON c.conversation_id = t.conversation_id WHERE t.conversation_id = ? AND (c.status = 'archived' OR e.accepted = 0 OR e.event_id LIKE 'mock-dispatch:%' OR EXISTS (SELECT 1 FROM turns r WHERE r.conversation_id = t.conversation_id AND r.request_id = 'task:' || e.task_id || ':' || e.event_id))", (conversation_id,)).fetchall()
            for row in events:
                event = dict(row)
                if event["summary"] or event["result_ref"]:
                    items.append((event["created_at"], len(_json([event["summary"], event["result_ref"]]).encode("utf-8")), "event", event))
            for created_at, size, kind, item in sorted(items, key=lambda item: item[0], reverse=True):
                age = (current - datetime.fromisoformat(created_at)).total_seconds()
                if age <= max_age_seconds and used + size <= max_tokens:
                    used += size
                    continue
                if kind == "turn":
                    compact = self._compact_result(item["result"], item["turn_id"], item["status"])
                    self._db.execute("UPDATE turns SET result = ?, error = NULL WHERE turn_id = ?", (_json(compact), item["turn_id"]))
                else:
                    event_hash = item["event_hash"] or self.content_hash([item["task_id"], item["kind"], item["summary"], item["result_ref"]], "event", "task")
                    self._db.execute("UPDATE task_events SET summary = '', result_ref = NULL, event_hash = ? WHERE event_id = ?", (event_hash, item["event_id"]))
                removed += 1
        return removed

    def checkpoint(self, incremental_vacuum_pages=0):
        with self._lock:
            self._db.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            if incremental_vacuum_pages > 0:
                self._db.execute(f"PRAGMA incremental_vacuum({int(incremental_vacuum_pages)})")

    def snapshot(self, conversation_id: str, recent_turns: int = 4) -> dict:
        with self._lock:
            conversation = self.get_conversation(conversation_id)
            if conversation is None:
                raise KeyError("Unknown conversation")
            memory = self.recent_memory(conversation_id)
            turns = self.list_turns(conversation_id)
            retained_ids = {turn["turn_id"] for turn in turns[-recent_turns:]}
            uncovered_ids = {turn["turn_id"] for turn in turns if turn["seq"] > memory["through_seq"]}
            included_ids = retained_ids | uncovered_ids
            messages = [message for message in self.list_messages(conversation_id) if message["turn_id"] in included_ids and message["delivery_state"] == "delivered"]
            return {
                "conversation": conversation,
                "messages": messages,
                "task_snapshot": self.list_tasks(conversation_id),
                "recent_memory": memory,
                "character_memory": self.character_memory(conversation["character_id"], conversation["user_id"]),
            }
