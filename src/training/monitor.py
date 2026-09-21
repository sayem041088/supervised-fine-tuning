"""
Monitoring module for Vertex AI Gemini fine-tuning jobs.
"""

import argparse
import time
from typing import Any, Optional

from src.utils.gcp import GCPManager
from src.utils.logger import get_logger

logger = get_logger("src.training.monitor")


class TuningMonitor:
    """Monitors asynchronous Vertex AI SFT jobs and reports final endpoints."""

    def __init__(self, gcp_manager: Optional[GCPManager] = None):
        self.gcp = gcp_manager or GCPManager()
        self.client = self.gcp.get_genai_client()

    def monitor(
        self, job_name: str, poll_interval_seconds: int = 60, timeout_minutes: int = 180
    ) -> Any:
        """
        Poll tuning job until terminal state.

        Args:
            job_name: Full resource name or ID of tuning job.
            poll_interval_seconds: Seconds between polling requests.
            timeout_minutes: Maximum duration to wait before timing out.

        Returns:
            Completed TuningJob object.
        """
        logger.info("Starting monitoring for job: %s", job_name)
        start_time = time.time()

        while True:
            elapsed_minutes = (time.time() - start_time) / 60
            if elapsed_minutes > timeout_minutes:
                logger.error("Monitoring timed out after %d minutes.", timeout_minutes)
                raise TimeoutError(f"Job {job_name} timed out after {timeout_minutes} minutes.")

            try:
                job = self.client.tunings.get(name=job_name)
                state = getattr(job.state, "name", str(job.state))
                logger.info("Elapsed: %.1f mins | Current State: %s", elapsed_minutes, state)

                if state in ("JOB_STATE_SUCCEEDED", "SUCCEEDED", "COMPLETED"):
                    logger.info("✅ Fine-tuning succeeded!")
                    if hasattr(job, "tuned_model") and job.tuned_model:
                        endpoint = getattr(job.tuned_model, "endpoint", "N/A")
                        model_id = getattr(job.tuned_model, "model", "N/A")
                        logger.info("Tuned Model ID: %s", model_id)
                        logger.info("Tuned Model Endpoint: %s", endpoint)
                    return job

                elif state in ("JOB_STATE_FAILED", "FAILED", "JOB_STATE_CANCELLED", "CANCELLED"):
                    error_msg = getattr(job, "error", "Unknown error")
                    logger.error("❌ Job terminated with state %s: %s", state, error_msg)
                    raise RuntimeError(f"Tuning job failed: {error_msg}")

            except Exception as exc:
                if "failed" in str(exc).lower():
                    raise
                logger.warning(
                    "Transient error fetching job status: %s. Retrying in %ds...",
                    exc,
                    poll_interval_seconds,
                )

            time.sleep(poll_interval_seconds)


def main() -> None:
    parser = argparse.ArgumentParser(description="Monitor a Vertex AI Gemini fine-tuning job.")
    parser.add_argument("--job-name", "-j", required=True, help="Tuning job resource name or ID.")
    parser.add_argument("--interval", type=int, default=60, help="Polling interval in seconds.")
    parser.add_argument("--timeout", type=int, default=180, help="Timeout in minutes.")
    args = parser.parse_args()

    gcp = GCPManager()
    monitor = TuningMonitor(gcp_manager=gcp)
    monitor.monitor(
        job_name=args.job_name, poll_interval_seconds=args.interval, timeout_minutes=args.timeout
    )


if __name__ == "__main__":
    main()
