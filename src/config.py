"""
Configuration management for Gemini Fine-Tuning Pipeline.
Loads settings from YAML files and environment variables.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


@dataclass
class GCPConfig:
    project_id: str = field(default_factory=lambda: os.getenv("PROJECT_ID", "your-gcp-project-id"))
    region: str = field(default_factory=lambda: os.getenv("REGION", "us-central1"))
    bucket_name: str = field(
        default_factory=lambda: os.getenv("BUCKET_NAME", "fine-tunning-bucket-summarization")
    )

    @property
    def bucket_uri(self) -> str:
        return f"gs://{self.bucket_name}"


@dataclass
class DataConfig:
    raw_train_path: str = "data/raw/sft_train_samples.jsonl"
    raw_val_path: str = "data/raw/sft_val_samples.jsonl"
    raw_test_path: str = "data/raw/sft_test_samples.csv"
    processed_train_path: str = "data/processed/train_gemini.jsonl"
    processed_val_path: str = "data/processed/val_gemini.jsonl"
    gcs_dataset_prefix: str = "datasets"


@dataclass
class TuningConfig:
    base_model: str = "gemini-2.5-flash"
    tuned_model_display_name: str = "gemini-flash-wikilingua-summarizer"
    epochs: Optional[int] = None
    learning_rate_multiplier: Optional[float] = None
    adapter_size: Optional[int] = None


@dataclass
class EvalConfig:
    temperature: float = 0.1
    max_output_tokens: int = 194
    top_p: float = 0.8
    metrics: List[str] = field(default_factory=lambda: ["rouge1", "rouge2", "rougeL"])


@dataclass
class PipelineConfig:
    gcp: GCPConfig = field(default_factory=GCPConfig)
    data: DataConfig = field(default_factory=DataConfig)
    tuning: TuningConfig = field(default_factory=TuningConfig)
    evaluation: EvalConfig = field(default_factory=EvalConfig)

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "PipelineConfig":
        """Load configuration from a YAML file with environment fallback."""
        config_path = Path(yaml_path)
        data: Dict[str, Any] = {}
        if config_path.is_file():
            with open(config_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}

        gcp_data = data.get("gcp", {})
        data_data = data.get("data", {})
        tuning_data = data.get("tuning", {})
        eval_data = data.get("evaluation", {})

        return cls(
            gcp=GCPConfig(**gcp_data) if gcp_data else GCPConfig(),
            data=DataConfig(**data_data) if data_data else DataConfig(),
            tuning=TuningConfig(**tuning_data) if tuning_data else TuningConfig(),
            evaluation=EvalConfig(**eval_data) if eval_data else EvalConfig(),
        )
