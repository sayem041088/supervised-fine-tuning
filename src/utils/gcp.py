"""
Google Cloud Platform (GCP) utilities for Vertex AI and Cloud Storage.
"""

import os
from typing import Optional

import vertexai
from google import genai
from google.cloud import storage

from src.utils.logger import get_logger

logger = get_logger("src.utils.gcp")


class GCPManager:
    """Manages GCP authentication, storage operations, and Vertex AI init."""

    def __init__(
        self,
        project_id: Optional[str] = None,
        region: Optional[str] = None,
        bucket_name: Optional[str] = None,
    ):
        self.project_id = project_id or os.getenv("PROJECT_ID")
        self.region = region or os.getenv("REGION", "us-central1")
        self.bucket_name = bucket_name or os.getenv("BUCKET_NAME")
        self._storage_client: Optional[storage.Client] = None
        self._genai_client: Optional[genai.Client] = None

    def verify_environment(self) -> bool:
        """Verify that necessary GCP environment variables are defined."""
        missing = []
        if not self.project_id or self.project_id == "your-gcp-project-id":
            missing.append("PROJECT_ID")
        if not self.region:
            missing.append("REGION")

        if missing:
            logger.warning(
                "Missing or default GCP environment variables: %s. "
                "Ensure PROJECT_ID and REGION are configured in your environment or .env",
                ", ".join(missing),
            )
            return False

        logger.info(
            "GCP Environment verified: project=%s, region=%s",
            self.project_id,
            self.region,
        )
        return True

    def init_vertexai(self) -> None:
        """Initialize Vertex AI SDK and Google GenAI client."""
        self.verify_environment()
        logger.info("Initializing Vertex AI for project %s in %s...", self.project_id, self.region)
        vertexai.init(project=self.project_id, location=self.region)

    def get_genai_client(self) -> genai.Client:
        """Get or initialize the Google GenAI client backed by Vertex AI."""
        if self._genai_client is None:
            self._genai_client = genai.Client(
                vertexai=True,
                project=self.project_id,
                location=self.region,
            )
        return self._genai_client

    @property
    def storage_client(self) -> storage.Client:
        """Lazy load Cloud Storage client."""
        if self._storage_client is None:
            self._storage_client = storage.Client(project=self.project_id)
        return self._storage_client

    def check_bucket_exists(self, bucket_name: Optional[str] = None) -> bool:
        """Check if target GCS bucket exists and is accessible."""
        target_bucket = bucket_name or self.bucket_name
        if not target_bucket:
            logger.error("Bucket name is not specified.")
            return False

        try:
            self.storage_client.get_bucket(target_bucket)
            logger.info("GCS Bucket '%s' exists and is accessible.", target_bucket)
            return True
        except Exception as exc:
            logger.error("Error accessing GCS bucket '%s': %s", target_bucket, exc)
            return False

    def upload_file(
        self, local_path: str, gcs_blob_name: str, bucket_name: Optional[str] = None
    ) -> str:
        """
        Upload local file to GCS.

        Args:
            local_path: Path to local file.
            gcs_blob_name: Destination path within bucket.
            bucket_name: Optional bucket name override.

        Returns:
            Destination GCS URI (gs://bucket/blob).
        """
        target_bucket = bucket_name or self.bucket_name
        if not target_bucket:
            raise ValueError("Target bucket must be provided.")

        bucket = self.storage_client.bucket(target_bucket)
        blob = bucket.blob(gcs_blob_name)

        logger.info("Uploading %s to gs://%s/%s...", local_path, target_bucket, gcs_blob_name)
        blob.upload_from_filename(local_path)
        uri = f"gs://{target_bucket}/{gcs_blob_name}"
        logger.info("Uploaded successfully: %s", uri)
        return uri
