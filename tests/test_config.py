"""Unit tests for configuration loading."""

from src.config import GCPConfig, PipelineConfig


def test_gcp_config_defaults():
    cfg = GCPConfig(project_id="test-proj", region="us-central1", bucket_name="test-bucket")
    assert cfg.project_id == "test-proj"
    assert cfg.region == "us-central1"
    assert cfg.bucket_uri == "gs://test-bucket"


def test_pipeline_config_from_yaml(tmp_path):
    yaml_content = """
gcp:
  project_id: custom-project
  region: us-east4
  bucket_name: custom-bucket
tuning:
  base_model: gemini-2.5-flash
  epochs: 4
"""
    cfg_file = tmp_path / "test_config.yaml"
    cfg_file.write_text(yaml_content)

    config = PipelineConfig.from_yaml(str(cfg_file))
    assert config.gcp.project_id == "custom-project"
    assert config.gcp.region == "us-east4"
    assert config.tuning.epochs == 4
    assert config.tuning.base_model == "gemini-2.5-flash"
