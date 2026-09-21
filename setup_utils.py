#!/usr/bin/env python3
"""
Legacy LabUtils module maintained for backward compatibility.
For new projects and CLI pipelines, use the modular packages in `src/`.
"""

import json
import os
import warnings
from typing import Any, Dict, Optional
import pandas as pd
from google.cloud import storage

from src.utils.gcp import GCPManager
from src.evaluation.metrics import RougeEvaluator


class LabUtils:
    """Utility class providing backward-compatible helper functions."""

    @staticmethod
    def verify_environment() -> bool:
        """Verify that the environment is set up correctly."""
        gcp = GCPManager()
        return gcp.verify_environment()

    @staticmethod
    def check_bucket_exists(bucket_name: str) -> bool:
        """Check if a GCS bucket exists."""
        gcp = GCPManager(bucket_name=bucket_name)
        return gcp.check_bucket_exists(bucket_name)

    @staticmethod
    def format_jsonl_sample(sample: Dict[str, Any]) -> str:
        """Pretty print a JSONL sample for inspection."""
        return json.dumps(sample, indent=2, ensure_ascii=False)

    @staticmethod
    def calculate_rouge_improvement(
        baseline_scores: pd.DataFrame,
        tuned_scores: pd.DataFrame,
        metric: str = "rougeL_precision",
    ) -> Dict[str, float]:
        """Calculate improvement metrics between baseline and tuned models."""
        baseline_mean = baseline_scores[metric].mean()
        tuned_mean = tuned_scores[metric].mean()

        improvement = ((tuned_mean - baseline_mean) / baseline_mean) * 100 if baseline_mean else 0.0

        return {
            "baseline_mean": baseline_mean,
            "tuned_mean": tuned_mean,
            "improvement_percent": improvement,
            "improvement_absolute": tuned_mean - baseline_mean,
        }

    @staticmethod
    def create_model_card_template() -> Dict[str, Any]:
        """Create a template for model documentation."""
        return {
            "model_name": "",
            "base_model": "gemini-2.5-flash",
            "task": "Article Summarization",
            "dataset": {
                "name": "WikiLingua",
                "language": "English",
                "train_samples": 0,
                "val_samples": 0,
                "test_samples": 0,
            },
            "training_config": {
                "epochs": "default",
                "learning_rate": "default",
                "batch_size": "default",
            },
            "performance": {
                "baseline_rouge_l": 0.0,
                "tuned_rouge_l": 0.0,
                "improvement": 0.0,
            },
            "training_duration": "",
            "notes": "",
        }


if __name__ == "__main__":
    print("Lab utilities loaded successfully (legacy bridge active).")
    utils = LabUtils()
    utils.verify_environment()
