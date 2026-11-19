"""Resident GPT-SoVITS API using the project's copied runtime and voice assets.

Adapted from ai-Apeiria/tools/run_apeiria_api.py.
"""

import argparse
from contextlib import redirect_stderr, redirect_stdout
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

from backend.config import PROJECT_ROOT, resolve_path


def private_inference(run):
    def stream(*args, **kwargs):
        with open(os.devnull, "w", encoding="utf-8") as discard:
            iterator = run(*args, **kwargs)
            try:
                while True:
                    with redirect_stdout(discard), redirect_stderr(discard):
                        try:
                            packet = next(iterator)
                        except StopIteration:
                            return
                    yield packet
            finally:
                with redirect_stdout(discard), redirect_stderr(discard):
                    iterator.close()
    return stream


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "config.json")
    parser.add_argument("--port", type=int, default=9881)
    parser.add_argument("--runtime-dir", type=Path)
    parser.add_argument("--no-warmup", action="store_true")
    options = parser.parse_args()
    config = json.loads(options.config.read_text(encoding="utf-8-sig"))
    settings = config["speech"]["tts"]
    runtime = settings["runtime"]
    runtime_dir = options.runtime_dir or resolve_path(config["paths"]["data"]) / "runtime" / "tts"
    runtime_dir = runtime_dir.resolve()
    runtime_dir.mkdir(parents=True, exist_ok=True)
    voice_id = settings["default_voice"]
    voice = dict(settings["voices"][voice_id])
    for key in ("gpt_weights", "sovits_weights", "reference_audio"):
        voice[key] = str(resolve_path(voice[key]))
    app_dir = resolve_path(runtime["directory"])
    os.chdir(app_dir)
    sys.path[:0] = [str(app_dir), str(app_dir / "GPT_SoVITS")]
    os.environ.update({
        "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8", "PYTHONDONTWRITEBYTECODE": "1",
        "language": "zh_CN", "version": voice["version"], "OMP_NUM_THREADS": "4",
        "MKL_NUM_THREADS": "4", "CUDAGraph": "1", "TOKENIZERS_PARALLELISM": "false",
        "NO_PROXY": "localhost,127.0.0.1,::1", "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
    })
    os.environ["PATH"] = str(app_dir) + os.pathsep + os.environ.get("PATH", "")
    os.environ["NLTK_DATA"] = str(app_dir / "runtime" / "nltk_data")
    for variable, directory in {
        "NUMBA_CACHE_DIR": "numba", "MPLCONFIGDIR": "matplotlib",
        "TORCHINDUCTOR_CACHE_DIR": "torchinductor", "TRITON_CACHE_DIR": "triton",
        "HF_HOME": "huggingface", "XDG_CACHE_HOME": "xdg", "TEMP": "tmp", "TMP": "tmp",
    }.items():
        cache = runtime_dir / "cache" / directory
        cache.mkdir(parents=True, exist_ok=True)
        os.environ[variable] = str(cache)

    import yaml
    model_config = {"custom": {
        "device": runtime["device"], "is_half": runtime["is_half"], "version": voice["version"],
        "t2s_weights_path": voice["gpt_weights"], "vits_weights_path": voice["sovits_weights"],
        "bert_base_path": str(resolve_path(runtime["bert_path"])),
        "cnhuhbert_base_path": str(resolve_path(runtime["hubert_path"])),
    }}
    inference_config = runtime_dir / "api-config.yaml"
    inference_config.write_text(yaml.safe_dump(model_config, allow_unicode=True), encoding="utf-8")
    sys.argv = [str(app_dir / "api_v2.py"), "-c", str(inference_config), "-p", str(options.port), "-a", "127.0.0.1"]
    started = time.perf_counter()
    spec = importlib.util.spec_from_file_location("rpbh_gpt_sovits_api", app_dir / "api_v2.py")
    api = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(api)
    api.tts_pipeline.run = private_inference(api.tts_pipeline.run)
    import torch
    torch.set_num_threads(4)
    warmup_seconds = 0.0
    if settings.get("warmup", True) and not options.no_warmup:
        from backend.speech.client import synthesis_request
        warmup_start = time.perf_counter()
        request = synthesis_request("声音准备好了，今天也很高兴见到你。", "zh", voice, settings)
        mode = request["streaming_mode"]
        request.update(streaming_mode=mode in (2, 3), fixed_length_chunk=mode == 3, return_fragment=mode == 1)
        if not sum(len(audio) for _, audio in api.tts_pipeline.run(request)):
            raise RuntimeError("TTS warmup produced no audio")
        warmup_seconds = time.perf_counter() - warmup_start
    sample_rate = (api.tts_pipeline.vocoder_configs["sr"] if api.tts_pipeline.configs.use_vocoder
                   else api.tts_pipeline.configs.sampling_rate)
    profile = {
        "ready": True, "voice_id": voice_id, "model_name": voice["name"], "pid": os.getpid(),
        "gpt_weights": voice["gpt_weights"], "sovits_weights": voice["sovits_weights"],
        "reference_audio": voice["reference_audio"], "sample_rate": sample_rate,
        "streaming_mode": settings["streaming_mode"], "warmup_seconds": round(warmup_seconds, 3),
        "startup_seconds": round(time.perf_counter() - started, 3),
    }
    (runtime_dir / "startup-profile.json").write_text(json.dumps(profile, ensure_ascii=False, indent=2), encoding="utf-8")

    @api.APP.get("/health")
    def health():
        return profile

    print(f"READY TTS http://127.0.0.1:{options.port}", flush=True)
    # Match the previous resident service's handling of inference diagnostics.
    with open(os.devnull, "w", encoding="utf-8") as discard:
        with redirect_stdout(discard), redirect_stderr(discard):
            api.uvicorn.run(api.APP, host="127.0.0.1", port=options.port, workers=1, access_log=False)


if __name__ == "__main__":
    main()
