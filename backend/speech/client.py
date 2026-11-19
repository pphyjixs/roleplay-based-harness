"""Resident GPT-SoVITS client using the migrated Apeiria inference settings."""

import asyncio
import os
from pathlib import Path
import subprocess
from urllib.parse import urlsplit

import httpx

from backend.config import PROJECT_ROOT


def resolve_voice(config, voice_id, project_root):
    voices = config["voices"]
    if voice_id not in voices:
        raise ValueError(f"Unknown TTS voice: {voice_id}")
    voice = dict(voices[voice_id])
    for field in ("gpt_weights", "sovits_weights", "reference_audio"):
        path = Path(voice[field]).expanduser()
        voice[field] = str((project_root / path).resolve()) if not path.is_absolute() else str(path.resolve())
    return voice


def synthesis_request(text, language, voice, config):
    """Preserve ai-Apeiria/assistant/tts.py inference and warm-up parameters."""
    return dict(
        text=text, text_lang={"zh": "all_zh", "ja": "all_ja", "en": "en"}[language],
        ref_audio_path=voice["reference_audio"], prompt_text=voice["reference_text"],
        prompt_lang=voice.get("reference_language", "all_ja"), media_type="raw", gpt_streaming=False,
        streaming_mode=config.get("streaming_mode", 1), text_split_method="cut0",
        batch_size=1, parallel_infer=False, split_bucket=False,
        use_cuda_graph=False, use_flash_attention=False,
        min_chunk_length=config.get("min_chunk_length", 24),
        overlap_length=config.get("overlap_length", 4),
        fragment_interval=config.get("fragment_interval", 0.08),
        top_k=config.get("top_k", 15), top_p=config.get("top_p", 1.0),
        temperature=config.get("temperature", 0.9),
        repetition_penalty=config.get("repetition_penalty", 1.35),
        seed=config.get("seed", 1234), speed_factor=config.get("speed", 1.0),
    )


class GPTSoVITS:
    def __init__(self, config, project_root=None, *, runtime_dir=None, config_path=None, client=None):
        self.config = config
        self.project_root = Path(project_root or PROJECT_ROOT).resolve()
        self.runtime_dir = Path(runtime_dir or self.project_root / "data/runtime/tts").resolve()
        self.config_path = Path(config_path or self.project_root / "config.json").resolve()
        self.url = config["url"].rstrip("/")
        self.default_voice = config["default_voice"]
        self.voice = resolve_voice(config, self.default_voice, self.project_root)
        self.sample_rate = self.voice.get("sample_rate", 48000)
        self.client = client or httpx.AsyncClient(timeout=httpx.Timeout(180, connect=3), trust_env=False)
        self._owns_client = client is None
        self.process = None
        self.log = None
        self.ready = asyncio.Event()
        self.error = None
        self._starting = False
        self._closed = False
        self._start_lock = asyncio.Lock()
        self._synthesis_lock = asyncio.Lock()

    def status(self):
        return {
            "enabled": self.config.get("enabled", True),
            "ready": self.ready.is_set() and self.error is None and not self._closed,
            "loading": self._starting,
            "error": self.error,
            "name": self.voice["name"],
            "voice_id": self.default_voice,
            "sample_rate": self.sample_rate,
        }

    def check_model(self, health):
        for field in ("gpt_weights", "sovits_weights", "reference_audio"):
            if field not in health or Path(health[field]).resolve() != Path(self.voice[field]).resolve():
                raise RuntimeError("TTS service model or reference audio differs from the configured voice")
        if health.get("voice_id", self.default_voice) != self.default_voice:
            raise RuntimeError("TTS service voice differs from the configured voice")
        sample_rate = health.get("sample_rate", self.sample_rate)
        if type(sample_rate) is not int or sample_rate <= 0:
            raise RuntimeError("TTS service returned an invalid sample rate")
        self.sample_rate = sample_rate

    async def _health(self):
        response = await self.client.get(self.url + "/health")
        response.raise_for_status()
        health = response.json()
        if not isinstance(health, dict):
            raise RuntimeError("TTS health response must be an object")
        if health.get("ready"):
            self.check_model(health)
            return True
        return False

    def _spawn(self):
        address = urlsplit(self.url)
        if address.hostname not in {"127.0.0.1", "localhost", "::1"}:
            raise ValueError("TTS auto-start requires a loopback service URL")
        if address.port is None:
            raise ValueError("TTS auto-start requires an explicit port")
        executable = Path(self.config["runtime"]["python"])
        if not executable.is_absolute():
            executable = self.project_root / executable
        if not executable.is_file():
            raise FileNotFoundError("The configured TTS Python runtime does not exist")
        self.runtime_dir.mkdir(parents=True, exist_ok=True)
        self.log = (self.runtime_dir / "service.log").open("w", encoding="utf-8")
        command = [str(executable), "-B", "-m", "backend.speech.server",
                   "--config", str(self.config_path), "--port", str(address.port),
                   "--runtime-dir", str(self.runtime_dir)]
        if not self.config.get("warmup", True):
            command.append("--no-warmup")
        self.process = subprocess.Popen(
            command, cwd=self.project_root, stdout=self.log, stderr=subprocess.STDOUT,
            env={**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8", "PYTHONDONTWRITEBYTECODE": "1"},
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )

    async def start(self):
        async with self._start_lock:
            if self._closed:
                raise RuntimeError("The TTS client is closed")
            if not self.config.get("enabled", True):
                raise RuntimeError("TTS is disabled in configuration")
            if self.ready.is_set():
                if self.error:
                    raise RuntimeError(self.error)
                return
            self._starting = True
            try:
                try:
                    if await self._health():
                        return
                except httpx.ConnectError:
                    if not self.config.get("auto_start", True):
                        raise RuntimeError("TTS service is not running")
                    self._spawn()
                async with asyncio.timeout(self.config.get("startup_timeout_seconds", 180)):
                    while True:
                        if self.process and self.process.poll() is not None:
                            raise RuntimeError(f"TTS model loading failed. See {self.runtime_dir / 'service.log'}")
                        try:
                            if await self._health():
                                return
                        except httpx.ConnectError:
                            pass
                        await asyncio.sleep(1)
            except BaseException as error:
                self.error = "TTS startup cancelled" if isinstance(error, asyncio.CancelledError) else str(error)
                await self._stop_process()
                raise
            finally:
                self._starting = False
                self.ready.set()

    async def stream(self, text, language=None, voice_id=None):
        selected_voice = voice_id or self.default_voice
        if selected_voice != self.default_voice:
            raise ValueError("The resident TTS service supports its configured default voice")
        selected_language = language or self.config.get("language", "zh")
        if selected_language not in {"zh", "ja", "en"}:
            raise ValueError("Unsupported TTS language")
        await self.start()
        request = synthesis_request(text, selected_language, self.voice, self.config)
        async with self._synthesis_lock:
            async with self.client.stream("POST", self.url + "/tts", json=request) as response:
                if response.is_error:
                    raise RuntimeError(f"TTS synthesis returned HTTP {response.status_code}")
                async for chunk in response.aiter_bytes(4096):
                    yield chunk

    async def _stop_process(self):
        if self.process and self.process.poll() is None:
            self.process.terminate()
            try:
                await asyncio.to_thread(self.process.wait, 15)
            except subprocess.TimeoutExpired:
                self.process.kill()
                await asyncio.to_thread(self.process.wait, 5)

    async def close(self):
        if self._closed:
            return
        self._closed = True
        await self._stop_process()
        if self._owns_client:
            await self.client.aclose()
        if self.log:
            self.log.close()
