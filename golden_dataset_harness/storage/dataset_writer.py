"""Dataset writer — writes annotations to JSONL, PostgreSQL, and MinIO.

Provides a unified write interface that always produces a local JSONL file
and optionally persists to PostgreSQL and MinIO when configured.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

from golden_dataset_harness.schemas.annotation import AnnotationRecord
from golden_dataset_harness.storage.minio_client import ObjectStorageClient
from golden_dataset_harness.storage.postgres import DatabaseManager

logger = logging.getLogger(__name__)


class DatasetWriter:
    """Unified writer that outputs to JSONL and optionally to DB/object store."""

    def __init__(
        self,
        output_dir: str,
        db: DatabaseManager | None = None,
        storage: ObjectStorageClient | None = None,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.jsonl_path = self.output_dir / "dataset.jsonl"
        self.db = db
        self.storage = storage

    async def write(
        self,
        record: AnnotationRecord,
        image_bytes: bytes | None = None,
    ) -> None:
        """Write a single annotation record.

        1. Append to local JSONL (always)
        2. Upload image to MinIO (if storage configured and bytes provided)
        3. Save to PostgreSQL (if db configured)
        """
        # 1. JSONL — always
        with open(self.jsonl_path, "a") as f:
            f.write(record.model_dump_json() + "\n")

        # 2. MinIO
        if image_bytes and self.storage:
            try:
                await self.storage.upload_image(record.image_id, image_bytes)
                logger.debug("Uploaded image %s to object storage", record.image_id)
            except Exception:
                logger.exception("Failed to upload image %s", record.image_id)

        # 3. PostgreSQL
        if self.db:
            try:
                ann_id = await self.db.save_annotation(record)
                logger.debug("Saved annotation %s (id=%s)", record.image_id, ann_id)
            except Exception:
                logger.exception("Failed to save annotation %s to DB", record.image_id)

    async def export_jsonl(
        self,
        output_path: str,
        status_filter: str | None = None,
    ) -> int:
        """Export annotations from the database to a JSONL file.

        Args:
            output_path: Destination file path.
            status_filter: If set, only export records with this review status.

        Returns:
            Number of records exported.
        """
        if not self.db:
            logger.warning("No database configured — cannot export")
            return 0

        if status_filter:
            records = await self.db.query_by_status(status_filter)
        else:
            records = await self.db.get_all_annotations()

        with open(output_path, "w") as f:
            for rec in records:
                f.write(rec.model_dump_json() + "\n")

        logger.info("Exported %d records to %s", len(records), output_path)
        return len(records)
