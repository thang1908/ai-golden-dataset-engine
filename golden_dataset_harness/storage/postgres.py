"""PostgreSQL storage layer using SQLAlchemy 2.0 async.

Provides ORM models for images, annotations, and evaluations, plus a
``DatabaseManager`` with complete CRUD operations.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import Column, DateTime, Float, ForeignKey, Integer, String, Text, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase

from golden_dataset_harness.schemas.annotation import AnnotationRecord, PersonAttributes


# ---------------------------------------------------------------------------
# ORM Models
# ---------------------------------------------------------------------------

class Base(DeclarativeBase):
    pass


class ImageRecord(Base):
    """Stores metadata about ingested images."""

    __tablename__ = "images"

    image_id = Column(String, primary_key=True)
    image_path = Column(String, nullable=False)
    width = Column(Integer, nullable=True)
    height = Column(Integer, nullable=True)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))


class AnnotationORM(Base):
    """Stores the golden annotation produced by the pipeline."""

    __tablename__ = "annotations"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    image_id = Column(String, ForeignKey("images.image_id"), nullable=False, index=True)
    caption = Column(Text, nullable=False)
    attributes_json = Column(Text, nullable=False)  # JSON-serialized PersonAttributes
    confidence = Column(Float, nullable=False)
    consensus_score = Column(Float, nullable=False)
    grounding_score = Column(Float, nullable=False)
    judge_score = Column(Float, nullable=False)
    model_version = Column(String, nullable=False)
    prompt_version = Column(String, default="v1")
    review_status = Column(String, nullable=False, index=True)
    issues_json = Column(Text, default="[]")  # JSON-serialized list[str]
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))


class EvaluationORM(Base):
    """Stores human review decisions."""

    __tablename__ = "evaluations"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    annotation_id = Column(String, ForeignKey("annotations.id"), nullable=False)
    judge_score = Column(Float, nullable=True)
    grounding_score = Column(Float, nullable=True)
    review_status = Column(String, nullable=False)
    reviewed_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    reviewer_notes = Column(Text, default="")


# ---------------------------------------------------------------------------
# Database Manager
# ---------------------------------------------------------------------------

class DatabaseManager:
    """Async database manager for the golden dataset harness."""

    def __init__(self, db_url: str) -> None:
        self.engine = create_async_engine(db_url, echo=False)
        self.session_factory = async_sessionmaker(
            bind=self.engine,
            class_=AsyncSession,
            expire_on_commit=False,
        )

    async def init_db(self) -> None:
        """Create all tables if they don't exist."""
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    async def close(self) -> None:
        """Dispose of the engine connection pool."""
        await self.engine.dispose()

    # --- Image CRUD ---

    async def save_image(self, image_id: str, image_path: str, width: int = 0, height: int = 0) -> None:
        async with self.session_factory() as session:
            existing = await session.get(ImageRecord, image_id)
            if not existing:
                session.add(ImageRecord(
                    image_id=image_id,
                    image_path=image_path,
                    width=width,
                    height=height,
                ))
                await session.commit()

    # --- Annotation CRUD ---

    async def save_annotation(self, record: AnnotationRecord) -> str:
        """Persist an AnnotationRecord. Returns the generated annotation ID."""
        async with self.session_factory() as session:
            # Ensure image record exists
            existing_img = await session.get(ImageRecord, record.image_id)
            if not existing_img:
                session.add(ImageRecord(
                    image_id=record.image_id,
                    image_path=record.image_path,
                ))

            ann_id = str(uuid.uuid4())
            orm = AnnotationORM(
                id=ann_id,
                image_id=record.image_id,
                caption=record.caption,
                attributes_json=record.attributes.model_dump_json(),
                confidence=record.confidence,
                consensus_score=record.consensus_score,
                grounding_score=record.grounding_score,
                judge_score=record.judge_score,
                model_version=record.model_version,
                prompt_version=record.prompt_version,
                review_status=record.review_status.value if hasattr(record.review_status, 'value') else record.review_status,
                issues_json=json.dumps(record.issues),
                created_at=record.created_at,
            )
            session.add(orm)
            await session.commit()
            return ann_id

    async def get_annotation(self, image_id: str) -> AnnotationRecord | None:
        """Retrieve the latest annotation for an image."""
        async with self.session_factory() as session:
            stmt = (
                select(AnnotationORM)
                .where(AnnotationORM.image_id == image_id)
                .order_by(AnnotationORM.created_at.desc())
                .limit(1)
            )
            result = await session.execute(stmt)
            row = result.scalar_one_or_none()
            if not row:
                return None
            return self._orm_to_record(row)

    async def query_by_status(self, status: str) -> list[AnnotationRecord]:
        """Query annotations by review status."""
        async with self.session_factory() as session:
            stmt = select(AnnotationORM).where(AnnotationORM.review_status == status)
            result = await session.execute(stmt)
            rows = result.scalars().all()
            return [self._orm_to_record(r) for r in rows]

    async def get_all_annotations(self) -> list[AnnotationRecord]:
        """Retrieve all annotations."""
        async with self.session_factory() as session:
            stmt = select(AnnotationORM).order_by(AnnotationORM.created_at.desc())
            result = await session.execute(stmt)
            rows = result.scalars().all()
            return [self._orm_to_record(r) for r in rows]

    async def update_review_status(
        self, image_id: str, status: str, notes: str = ""
    ) -> bool:
        """Update the review status and optionally save reviewer notes."""
        async with self.session_factory() as session:
            stmt = (
                select(AnnotationORM)
                .where(AnnotationORM.image_id == image_id)
                .order_by(AnnotationORM.created_at.desc())
                .limit(1)
            )
            result = await session.execute(stmt)
            ann = result.scalar_one_or_none()
            if not ann:
                return False

            ann.review_status = status

            # Save evaluation record
            evaluation = EvaluationORM(
                annotation_id=ann.id,
                judge_score=ann.judge_score,
                grounding_score=ann.grounding_score,
                review_status=status,
                reviewer_notes=notes,
            )
            session.add(evaluation)
            await session.commit()
            return True

    # --- Helpers ---

    @staticmethod
    def _orm_to_record(row: AnnotationORM) -> AnnotationRecord:
        """Convert an ORM row back to a Pydantic AnnotationRecord."""
        return AnnotationRecord(
            image_id=row.image_id,
            image_path="",  # Not stored in annotation table
            caption=row.caption,
            attributes=PersonAttributes.model_validate_json(row.attributes_json),
            confidence=row.confidence,
            consensus_score=row.consensus_score,
            grounding_score=row.grounding_score,
            judge_score=row.judge_score,
            model_version=row.model_version,
            prompt_version=row.prompt_version,
            review_status=row.review_status,
            issues=json.loads(row.issues_json) if row.issues_json else [],
            created_at=row.created_at,
        )
