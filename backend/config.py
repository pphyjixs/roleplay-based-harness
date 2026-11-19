"""Load project configuration without copying credentials into configuration."""

import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def resolve_path(value, root=PROJECT_ROOT):
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def load_config(path=None):
    config_path = Path(path) if path else PROJECT_ROOT / "config.json"
    if path is None and not config_path.is_file():
        raise FileNotFoundError("Copy config.example.json to config.json before starting RPBH.")
    config = json.loads(config_path.read_text(encoding="utf-8-sig"))
    for section in ("paths", "server", "llm", "roleplay", "characters", "memory", "tasks"):
        if section not in config:
            raise ValueError(f"Missing configuration section: {section}")
    if config["tasks"]["executor"] != "mock":
        raise ValueError("This backend currently supports tasks.executor=mock")
    for role in ("roleplay", "memory"):
        settings = config[role]
        if settings["provider"] not in config["llm"]["providers"]:
            raise ValueError(f"Unknown LLM provider for {role}")
        if settings.get("thinking", False):
            raise ValueError(f"{role} requires non-thinking mode")
    memory = config["memory"]
    for key in ("recent_turns", "recent_max_tokens", "summary_max_tokens", "summary_target_tokens", "character_max_tokens", "update_batch_turns", "memory_input_max_tokens", "task_snapshot_max_tokens"):
        if type(memory[key]) is not int or memory[key] <= 0:
            raise ValueError(f"memory.{key} must be a positive integer")
    repairs = config["roleplay"]["max_repair_attempts"]
    if type(repairs) is not int or repairs not in (0, 1):
        raise ValueError("roleplay.max_repair_attempts must be 0 or 1")
    if memory["summary_target_tokens"] > memory["summary_max_tokens"]:
        raise ValueError("Summary target must fit within the summary budget")
    default = config["characters"]["default"]
    if default not in config["characters"]["profiles"]:
        raise ValueError("Default character profile does not exist")
    _validate_runtime_policy(config)
    return config


def _validate_runtime_policy(config):
    def positive(section, keys):
        for key in keys:
            if type(section.get(key)) not in {int, float} or section[key] <= 0:
                raise ValueError(f"Policy parameter {key} must be positive")

    mobile = config.get("mobile")
    if mobile:
        positive(mobile, ("sdp_timeout_seconds",))
        transport = mobile["transport"]
        positive(transport, ("chunk_bytes", "max_message_bytes", "max_pending_bytes", "max_transfers", "fragment_timeout_seconds", "send_buffer_bytes", "send_timeout_seconds", "max_http_requests", "max_websockets", "ws_queue_messages"))
        if not 1024 <= transport["chunk_bytes"] <= 16384:
            raise ValueError("mobile.transport.chunk_bytes must fit a native SCTP frame")
        if transport["max_message_bytes"] > transport["max_pending_bytes"]:
            raise ValueError("Mobile logical packet must fit the pending memory budget")
        if type(transport.get("http_timeout_seconds", 60)) not in {int, float} or transport.get("http_timeout_seconds", 60) <= 0:
            raise ValueError("mobile.transport.http_timeout_seconds must be positive")
        for key, default in (("closed_socket_ids", 128), ("closed_socket_ttl_seconds", 120)):
            if type(transport.get(key, default)) not in {int, float} or transport.get(key, default) <= 0:
                raise ValueError(f"mobile.transport.{key} must be positive")

    context = config.get("context")
    if context:
        positive(context, ("max_input_tokens", "system_max_tokens", "persona_max_tokens", "tools_max_tokens", "session_max_tokens", "current_input_max_tokens", "image_token_reserve_per_image"))
    memory = config["memory"]
    if memory.get("policy") == "batched_archive":
        positive(memory, ("recent_max_age_seconds", "archive_trigger_multiplier", "archive_input_max_tokens", "archive_prompt_reserve_tokens", "archive_summary_max_tokens", "archive_summary_target_tokens"))
        required = memory["recent_max_tokens"] * memory["archive_trigger_multiplier"] + memory["archive_summary_max_tokens"] + memory["character_max_tokens"] + memory["archive_prompt_reserve_tokens"]
        if required > memory["archive_input_max_tokens"]:
            raise ValueError("Archive trigger and memory reserves exceed archive_input_max_tokens; reduce the multiplier or raise the verified input capacity")
        if memory["archive_summary_target_tokens"] > memory["archive_summary_max_tokens"]:
            raise ValueError("Archive summary target must fit its memory budget")
    retention = config.get("retention")
    if retention:
        positive(retention, ("maintenance_interval_seconds", "archive_poll_interval_ms", "archive_wait_timeout_ms"))
        if retention["archive_wait_timeout_ms"] < retention["archive_poll_interval_ms"]:
            raise ValueError("Archive wait timeout must exceed its polling interval")
        positive(retention["logs"], ("max_tokens", "max_age_seconds"))
        positive(retention["audio"], ("unplayed_max_age_seconds", "max_cached_bytes", "playback_ack_grace_seconds"))
        positive(retention["sqlite"], ("checkpoint_interval_seconds", "incremental_vacuum_pages"))
    idle = config.get("idle")
    if idle:
        positive(idle, ("timeout_seconds", "check_interval_seconds", "activity_throttle_ms"))
        if not isinstance(idle.get("prompt"), str) or not idle["prompt"].strip():
            raise ValueError("idle.prompt must contain the proactive instruction")
    vision = config.get("vision")
    if vision:
        positive(vision, ("max_images_per_turn", "max_input_bytes_per_image", "max_input_pixels", "max_request_bytes", "encoding_timeout_ms", "summary_max_tokens"))
        positive(vision["processing"], ("max_jobs", "timeout_seconds", "max_encode_attempts"))
        positive(vision["cache"], ("max_bytes", "max_images", "ttl_seconds", "camera_frames"))
        for name in ("camera", "upload"):
            policy = vision[name]
            positive(policy, ("max_dimension", "quality", "min_quality", "max_bytes", "min_dimension", "quality_step", "resize_factor"))
            if not 1 <= policy["min_quality"] <= policy["quality"] <= 100 or not 0 < policy["resize_factor"] < 1:
                raise ValueError("Invalid image quality or resize factor")
        positive(vision["camera"], ("interval_ms", "analysis_interval_ms", "batch_size", "max_buffered_frames"))
        if vision["camera"]["batch_size"] > min(vision["max_images_per_turn"], vision["camera"]["max_buffered_frames"]):
            raise ValueError("Camera batch exceeds the per-turn or frame cache capacity")


def model_settings(config, role):
    settings = config[role]
    return {**config["llm"]["providers"][settings["provider"]], **settings}


def presentation_manifest(config, character_id):
    profile = config["characters"]["profiles"].get(character_id)
    if profile is None:
        raise ValueError(f"Unknown character: {character_id}")
    filename = profile.get("presentation_file")
    if not filename:
        return {"schema_version": 1, "character_id": character_id, "default_presentation": "live2d", "presentations": {
            "live2d": {"id": "live2d", "type": "live2d", "name": character_id,
                       "emotions": profile.get("emotions", []), "motions": profile.get("motions", [])}}}
    manifest = json.loads(resolve_path(filename).read_text(encoding="utf-8-sig"))
    if manifest.get("character_id") != character_id or manifest.get("default_presentation") not in manifest.get("presentations", {}):
        raise ValueError("Invalid character presentation manifest")
    return manifest


def character_profile(config, character_id, presentation_id=None):
    profiles = config["characters"]["profiles"]
    if character_id not in profiles:
        raise ValueError(f"Unknown character: {character_id}")
    profile = dict(profiles[character_id])
    profile["character_id"] = character_id
    profile["persona"] = resolve_path(profile["persona_file"]).read_text(encoding="utf-8-sig")
    manifest = presentation_manifest(config, character_id)
    presentation_id = presentation_id or manifest["default_presentation"]
    if presentation_id not in manifest["presentations"]:
        raise ValueError(f"Unknown presentation: {presentation_id}")
    presentation = manifest["presentations"][presentation_id]
    profile["presentation_id"] = presentation_id
    profile["presentation_type"] = presentation["type"]
    # Inference receives semantic IDs and descriptions, never mesh/rig parameters.
    for key in ("emotions", "motions"):
        profile[key] = [{name: item[name] for name in ("id", "label", "description") if name in item}
                        if isinstance(item, dict) else item for item in presentation.get(key, [])]
    return profile
