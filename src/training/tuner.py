"""
Vertex AI Supervised Fine-Tuning launcher for Gemini models.
"""

import argparse
import sys
from typing import Any, Dict, Optional

from src.config import PipelineConfig
from src.utils.gcp import GCPManager
from src.utils.logger import get_logger

logger = get_logger("src.training.tuner")


class GeminiTuner:
    """Orchestrates Gemini SFT jobs on Vertex AI."""

    def __init__(self, gcp_manager: Optional[GCPManager] = None):
        self.gcp = gcp_manager or GCPManager()
        self.client = self.gcp.get_genai_client()

    def launch(
        self,
        base_model: str,
        train_gcs_uri: str,
        val_gcs_uri: Optional[str] = None,
        display_name: str = "gemini-flash-wikilingua-summarizer",
        epochs: Optional[int] = None,
        learning_rate_multiplier: Optional[float] = None,
        adapter_size: Optional[int] = None,
    ) -> Any:
        """
        Launch fine-tuning job on Vertex AI via google-genai SDK.

        Args:
            base_model: Base foundation model identifier (e.g. 'gemini-2.5-flash').
            train_gcs_uri: GCS path to training JSONL.
            val_gcs_uri: Optional GCS path to validation JSONL.
            display_name: Target display name for the tuned model.
            epochs: Optional custom epoch count.
            learning_rate_multiplier: Optional learning rate multiplier.
            adapter_size: Optional LoRA adapter rank.

        Returns:
            TuningJob object from GenAI client.
        """
        logger.info("Launching SFT job on base model: %s", base_model)
        logger.info("Training dataset URI: %s", train_gcs_uri)
        if val_gcs_uri:
            logger.info("Validation dataset URI: %s", val_gcs_uri)

        training_dataset = {"gcs_uri": train_gcs_uri}
        validation_dataset = {"gcs_uri": val_gcs_uri} if val_gcs_uri else None

        tuning_config: Dict[str, Any] = {
            "tuned_model_display_name": display_name,
        }
        if epochs is not None:
            tuning_config["epoch_count"] = epochs
        if learning_rate_multiplier is not None:
            tuning_config["learning_rate_multiplier"] = learning_rate_multiplier
        if adapter_size is not None:
            tuning_config["adapter_size"] = adapter_size

        try:
            job = self.client.tunings.tune(
                base_model=base_model,
                training_dataset=training_dataset,
                validation_dataset=validation_dataset,
                config=tuning_config,
            )
            logger.info("Tuning job initiated successfully!")
            logger.info("Job Name: %s", job.name)
            logger.info("Job State: %s", getattr(job.state, "name", str(job.state)))
            return job
        except Exception as exc:
            logger.error("Failed to launch fine-tuning job: %s", exc)
            raise


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch Gemini SFT job on Vertex AI.")
    parser.add_argument(
        "--config", "-c", default="configs/tuning_config.yaml", help="Path to tuning config YAML."
    )
    parser.add_argument("--train-uri", required=False, help="Override training GCS URI.")
    parser.add_argument("--val-uri", required=False, help="Override validation GCS URI.")
    parser.add_argument("--display-name", required=False, help="Override model display name.")
    args = parser.parse_args()

    cfg = PipelineConfig.from_yaml(args.config)
    gcp = GCPManager(
        project_id=cfg.gcp.project_id,
        region=cfg.gcp.region,
        bucket_name=cfg.gcp.bucket_name,
    )

    if not gcp.verify_environment():
        logger.error("GCP environment verification failed. Exiting.")
        sys.exit(1)

    train_uri = (
        args.train_uri
        or f"{cfg.gcp.bucket_uri}/{cfg.data.gcs_dataset_prefix}/train/train_gemini.jsonl"
    )
    val_uri = (
        args.val_uri or f"{cfg.gcp.bucket_uri}/{cfg.data.gcs_dataset_prefix}/val/val_gemini.jsonl"
    )
    display_name = args.display_name or cfg.tuning.tuned_model_display_name

    tuner = GeminiTuner(gcp_manager=gcp)
    tuner.launch(
        base_model=cfg.tuning.base_model,
        train_gcs_uri=train_uri,
        val_gcs_uri=val_uri,
        display_name=display_name,
        epochs=cfg.tuning.epochs,
        learning_rate_multiplier=cfg.tuning.learning_rate_multiplier,
        adapter_size=cfg.tuning.adapter_size,
    )


if __name__ == "__main__":
    main()
