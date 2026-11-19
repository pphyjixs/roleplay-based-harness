"""Text delivery and explicit client receipts for RP actions."""

from collections.abc import Awaitable, Callable


class DeliveryService:
    def __init__(self, store):
        self.store = store

    async def publish(
        self, action: dict, emit: Callable[[dict], Awaitable[None]]
    ) -> dict:
        """Offer an action to clients. Its receipt commits delivered dialogue."""
        await emit(
            {
                "type": "rp.message",
                "conversation_id": action["conversation_id"],
                "turn_id": action["turn_id"],
                "action_id": action["action_id"],
                "text": action.get("text", ""),
                "emotion": action.get("emotion"),
                "motion_ids": action.get("motion_ids", []),
                "presentation_id": action.get("presentation_id"),
                "presentation_revision": action.get("presentation_revision"),
            }
        )
        return action

    def acknowledge(self, action_id: str, text: str | None = None) -> dict:
        """Persist a full or partial delivered-text receipt idempotently."""
        return self.store.record_delivery(action_id, text)

    async def replay(
        self, conversation_id: str, emit: Callable[[dict], Awaitable[None]]
    ) -> list[dict]:
        """Redeliver pending actions after a client connects or reconnects."""
        actions = self.store.list_pending_actions(conversation_id)
        for action in actions:
            await self.publish(action, emit)
        return actions
