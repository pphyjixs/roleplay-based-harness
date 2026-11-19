"""Delete disposable RP audio; never touch model or TTS-owned cache directories."""
from datetime import datetime, timezone
from pathlib import Path


class AudioRetention:
    def __init__(self, store, data_root, settings):
        self.store = store
        self.directory = (Path(data_root).resolve() / "audio").resolve()
        self.settings = settings

    def release(self, action_id, *, force=False):
        job = self.store.get_audio_job(action_id)
        if not job or job.get("file_state") == "deleted":
            return job
        policies = {"completed": "delete_after_playback", "interrupted": "delete_on_interrupted", "failed": "delete_on_failed"}
        if not force and not self.settings.get(policies.get(job.get("playback_status")), False):
            return job
        filename = job.get("wav_path")
        if filename:
            target = Path(filename).resolve()
            if target.parent != self.directory or target.name != action_id + ".wav":
                raise ValueError("Audio retention refused a path outside its own audio directory")
            try:
                target.unlink(missing_ok=True)
            except PermissionError:
                # An in-flight Windows file response may still own the handle;
                # the next maintenance pass retries after the terminal receipt.
                return job
        return self.store.set_audio_file_state(action_id, "deleted")

    def candidates(self, now=None):
        now = now or datetime.now(timezone.utc)
        jobs = []
        for conversation in self.store.list_conversations():
            jobs.extend(self.store.list_audio_jobs(conversation["conversation_id"]))
        jobs.sort(key=lambda job: (job["created_at"], job["action_id"]))
        total = sum(job.get("pcm_bytes", 0) for job in jobs if job.get("file_state") != "deleted")
        expired = []
        for job in jobs:
            if job.get("file_state") == "deleted":
                continue
            self.release(job["action_id"])
            current = self.store.get_audio_job(job["action_id"])
            if current.get("file_state") == "deleted":
                total -= job.get("pcm_bytes", 0)
                continue
            age = (now - datetime.fromisoformat(job["created_at"])).total_seconds()
            playback_age = (now - datetime.fromisoformat(job["updated_at"])).total_seconds()
            stale_playback = job["playback_status"] == "playing" and playback_age >= job["duration_seconds"] + self.settings["playback_ack_grace_seconds"]
            if stale_playback or age >= self.settings["unplayed_max_age_seconds"] or total > self.settings["max_cached_bytes"]:
                expired.append(job["action_id"])
                total -= job.get("pcm_bytes", 0)
        return expired
