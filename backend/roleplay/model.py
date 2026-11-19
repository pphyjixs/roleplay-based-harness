"""Small async client for DeepSeek's stateless Responses API."""

import json
import os
from pathlib import Path
import re

import httpx


class ModelError(RuntimeError):
    """Provider or transport failure with a credential-safe message."""


class ResponsesModel:
    def __init__(self, settings, project_root, *, client=None):
        self.settings = settings
        self.project_root = Path(project_root)
        self.client = client or httpx.AsyncClient(timeout=settings.get("timeout_seconds", 90))
        self._owns_client = client is None
        self._key = None

    def _api_key(self):
        if self._key:
            return self._key
        env_name = self.settings.get("api_key_env")
        value = os.environ.get(env_name, "") if env_name else ""
        if not value:
            path = Path(self.settings.get("api_key_file", "dsapi.md")).expanduser()
            if not path.is_absolute():
                path = self.project_root / path
            try:
                value = path.read_text(encoding="utf-8-sig").strip()
            except OSError as error:
                raise ModelError("The configured API key file cannot be read") from error
        matches = re.findall(r"(?<![A-Za-z0-9_-])sk-[A-Za-z0-9_-]+", value)
        if len(matches) != 1:
            raise ModelError("The API credential source must contain exactly one sk- key")
        self._key = matches[0]
        return self._key

    async def _request(self, payload):
        base_url = self.settings.get("base_url", "https://api.deepseek.com").rstrip("/")
        try:
            response = await self.client.post(
                f"{base_url}/responses",
                headers={"Authorization": f"Bearer {self._api_key()}"},
                json=payload,
            )
        except httpx.TimeoutException as error:
            raise ModelError("DeepSeek Responses request timed out") from error
        except httpx.HTTPError as error:
            raise ModelError("DeepSeek Responses transport failed") from error
        if response.is_error:
            raise ModelError(f"DeepSeek Responses returned HTTP {response.status_code}")
        try:
            result = response.json()
        except ValueError as error:
            raise ModelError("DeepSeek Responses returned invalid JSON") from error
        if not isinstance(result, dict) or result.get("status") != "completed":
            status = result.get("status", "unknown") if isinstance(result, dict) else "invalid"
            raise ModelError(f"DeepSeek Responses did not complete ({status})")
        return result

    def _base_payload(self, input_messages, max_output_tokens):
        return {
            "model": self.settings.get("model", "deepseek-flash"),
            "input": input_messages,
            "reasoning": {"effort": "none"},
            "max_output_tokens": max_output_tokens,
            "stream": False,
        }

    async def decide(self, input_messages, tools):
        payload = self._base_payload(input_messages, self.settings.get("max_output_tokens", 1400))
        payload.update(tools=tools, tool_choice="required")
        return await self._request(payload)

    async def complete_json(self, system, payload):
        request = self._base_payload([
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ], self.settings.get("max_output_tokens", 2400))
        request["text"] = {"format": {"type": "json_object"}}
        response = await self._request(request)
        text = "".join(
            part.get("text", "")
            for item in response.get("output", [])
            if item.get("type") == "message"
            for part in item.get("content", [])
            if part.get("type") == "output_text"
        )
        try:
            result = json.loads(text)
        except ValueError as error:
            raise ModelError("The memory response is not a JSON object") from error
        if not isinstance(result, dict):
            raise ModelError("The memory response is not a JSON object")
        return result

    async def close(self):
        if self._owns_client:
            await self.client.aclose()
