import redis
import json
from typing import Optional, Dict, Any


class RedisProgressTracker:
    def __init__(self, host="localhost", port=6379, db=0):
        self.redis_client = redis.Redis(
            host=host,
            port=port,
            db=db,
            decode_responses=True,
            socket_timeout=5,
            retry_on_timeout=True,
        )
        self.stages = {
            "initializing": (0, 5),
            "source_separation_start": (5, 6),
            "source_separation_loading": (6, 16),
            "source_separation_processing": (16, 18),
            "source_separation_enhancing": (18, 20),
            "transcription": (20, 50),
            "alignment": (50, 70),
            "diarization": (70, 90),
            "finalizing": (90, 100),
        }

    def _get_progress_key(self, job_id: str) -> str:
        return f"progress:{job_id}"

    def _get_status_key(self, job_id: str) -> str:
        return f"status:{job_id}"

    def initialize_job(self, job_id: str) -> None:
        """Initialize a new job's progress tracking"""
        progress_data = {"stage": "initializing", "progress": 0, "status": "processing"}
        self.redis_client.setex(
            self._get_progress_key(job_id),
            3600,  # 1 hour expiration
            json.dumps(progress_data),
        )

    def update_progress(self, job_id: str, stage: str, sub_progress: float = 0) -> None:
        """Update job progress"""
        try:
            if stage not in self.stages:
                return

            stage_start, stage_end = self.stages[stage]
            stage_range = stage_end - stage_start
            overall_progress = stage_start + (stage_range * (sub_progress / 100))

            progress_data = {
                "stage": stage,
                "progress": round(overall_progress, 2),
                "status": "processing",
            }

            self.redis_client.setex(
                self._get_progress_key(job_id),
                3600,  # 1 hour expiration
                json.dumps(progress_data),
            )
        except Exception as e:
            print(f"Failed to update progress: {str(e)}")

    def get_progress(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get current progress for a job"""
        try:
            progress_data = self.redis_client.get(self._get_progress_key(job_id))
            if progress_data:
                return json.loads(progress_data)
            return None
        except Exception as e:
            print(f"Failed to get progress: {str(e)}")
            return None

    def complete_job(self, job_id: str, output_file: str) -> None:
        """Mark job as completed"""
        self.redis_client.setex(
            self._get_progress_key(job_id),
            3600,
            json.dumps(
                {"status": "completed", "output_file": output_file, "progress": 100}
            ),
        )

    def fail_job(self, job_id: str, error: str) -> None:
        """Mark job as failed"""
        self.redis_client.setex(
            self._get_progress_key(job_id),
            3600,
            json.dumps({"status": "failed", "error": error, "progress": 0}),
        )

    def cancel_job(self, job_id: str) -> None:
        """Mark job as canceled"""
        self.redis_client.setex(
            self._get_progress_key(job_id),
            3600,
            json.dumps({"status": "canceled", "progress": 0}),
        )
