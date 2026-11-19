"""Hard request budgets; UTF-8 bytes conservatively bound text token counts."""
import copy
import json

from backend.state.memory import token_upper_bound


def _cost(value):
    def text_only(item):
        if isinstance(item, dict):
            if item.get("type") == "input_image":
                return {"type": "input_image"}
            return {key: text_only(value) for key, value in item.items()}
        if isinstance(item, list):
            return [text_only(value) for value in item]
        return item
    return token_upper_bound(json.dumps(text_only(value), ensure_ascii=False, separators=(",", ":")))


def bound_context(messages, tools, settings, *, image_count=0, tail_count=1):
    if not settings:
        return messages
    result = copy.deepcopy(messages)
    for index, key in ((0, "system_max_tokens"), (1, "persona_max_tokens")):
        if token_upper_bound(result[index]["content"]) > settings[key]:
            raise ValueError(f"Configured {key} cannot hold the current prompt")
    if _cost(tools) > settings["tools_max_tokens"]:
        raise ValueError("Current character tool definitions exceed context.tools_max_tokens")
    reserve = image_count * settings["image_token_reserve_per_image"]
    limit = settings["max_input_tokens"] - reserve

    def total():
        return _cost(result) + _cost(tools)

    # Only the first three framework/persona/background items and the current
    # input are mandatory. Drop oldest history as whole user-led groups.
    while total() > limit and len(result) > 3 + tail_count:
        del result[3]
        while len(result) > 3 + tail_count and result[3]["role"] == "assistant":
            del result[3]
    if total() > limit:
        background = result[2]["content"]
        prefix, encoded = background.split("\n", 1)
        context = json.loads(encoded)
        # Existing per-category budgets were already applied by MemoryManager.
        # Under unusually large persona/tools/input, trim optional background.
        for key in ("tasks", "character_memory", "recent_memory"):
            if total() <= limit:
                break
            context[key] = [] if key != "recent_memory" else {"summary": ""}
            result[2]["content"] = prefix + "\n" + json.dumps(context, ensure_ascii=False, separators=(",", ":"))
    if total() > limit:
        raise ValueError("The current input and fixed character context exceed the configured request budget")
    return result
