"""Durable task notifications and the first-phase simulated executor."""

from collections.abc import Awaitable, Callable


class TaskService:
    def __init__(self, store, emit: Callable[[dict], Awaitable[None]]):
        self.store = store
        self.emit = emit

    async def dispatch(self, task: dict) -> dict:
        """Accept mock work once and wait for explicit simulated task events."""
        task_id = task["task_id"]
        event_id = f"mock-dispatch:{task_id}"
        summary = "模拟执行器已受理任务，等待测试回调。"
        current, accepted = self.store.add_task_event(
            task_id, event_id, "progress", summary
        )
        if accepted:
            await self.emit(
                {
                    "type": "task.created",
                    "conversation_id": current["conversation_id"],
                    "task_id": task_id,
                    "executor": "mock",
                    "task": current,
                }
            )
            await self.emit(
                {
                    "type": "task.progress",
                    "conversation_id": current["conversation_id"],
                    "task_id": task_id,
                    "event_id": event_id,
                    "summary": summary,
                    "status": current["status"],
                    "executor": "mock",
                }
            )
        return current

    async def handle_event(
        self,
        task_id: str,
        event_id: str,
        kind: str,
        summary: str,
        result_ref: str | None = None,
    ) -> tuple[dict, bool]:
        """Store one callback and publish its accepted state transition."""
        task, accepted = self.store.add_task_event(
            task_id, event_id, kind, summary, result_ref
        )
        if accepted:
            await self.emit(
                {
                    "type": f"task.{kind}",
                    "conversation_id": task["conversation_id"],
                    "task_id": task_id,
                    "event_id": event_id,
                    "summary": summary,
                    "status": task["status"],
                    "result_ref": result_ref,
                }
            )
        return task, accepted
