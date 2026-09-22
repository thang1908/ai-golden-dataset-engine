"""FastAPI application — HTTP interface for the annotation pipeline.

Provides endpoints for single/batch annotation, querying, human review,
Label Studio export, and pipeline statistics.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import HTMLResponse, Response
from pydantic import BaseModel

from golden_dataset_harness.api.label_studio import format_for_label_studio, label_studio_config
from golden_dataset_harness.schemas.annotation import AnnotationRecord, PersonAttributes, ReviewStatus
from golden_dataset_harness.schemas.taxonomy import TAXONOMY, attribute_cells
from golden_dataset_harness.workflow.graph import build_annotation_graph, load_settings

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# App state (initialized at startup)
# ---------------------------------------------------------------------------

class AppState:
    """Mutable application state shared across requests."""

    def __init__(self) -> None:
        self.graph: Any = None
        self.settings: dict[str, Any] = {}
        self.records: dict[str, AnnotationRecord] = {}  # in-memory store for MVP


_state = AppState()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize the pipeline graph on startup."""
    _state.settings = load_settings()
    _state.graph = build_annotation_graph(_state.settings)
    logger.info("Pipeline graph initialized")
    yield
    logger.info("Shutting down")


app = FastAPI(
    title="Golden Dataset Harness API",
    description="AI-powered annotation pipeline for person images",
    version="0.1.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class ReviewRequest(BaseModel):
    status: Literal["human_approved", "human_rejected"]
    corrected_caption: str | None = None
    corrected_attributes: PersonAttributes | None = None
    notes: str = ""


class StatsResponse(BaseModel):
    total_processed: int
    auto_accepted: int
    pending_review: int
    acceptance_rate: float
    mean_confidence: float


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    """Serve the visual interactive web dashboard."""
    template_path = Path(__file__).parent / "templates" / "index.html"
    if template_path.exists():
        return HTMLResponse(content=template_path.read_text(encoding="utf-8"))
    return HTMLResponse(content="<h1>AI Golden Dataset Harness API is running. Go to <a href='/docs'>/docs</a></h1>")


@app.get("/taxonomy")
async def get_taxonomy():
    """Expose the exact attribute contract used by extraction and review."""
    return {"schema_version": "siglip-v1", "num_attributes": len(TAXONOMY),
            "total_classes": sum(len(v["classes"]) for v in TAXONOMY.values()),
            "attributes": TAXONOMY}


@app.get("/export/siglip")
async def export_siglip():
    """Export accepted records as SigLIP-compatible table rows."""
    return {"schema_version": "siglip-v1", "rows": [
        {"image_id": record.image_id, "caption": record.caption,
         **attribute_cells(record.attributes.model_dump())}
        for record in _state.records.values()
        if record.review_status in {ReviewStatus.AUTO_ACCEPTED, ReviewStatus.HUMAN_APPROVED}
    ]}


@app.post("/annotate", response_model=AnnotationRecord)
async def annotate_single(file: UploadFile = File(...)):
    """Upload a single image and run the full annotation pipeline."""
    image_bytes = await file.read()
    image_id = file.filename or str(uuid.uuid4())

    try:
        result = await _state.graph.ainvoke(
            {"image_bytes": image_bytes, "image_id": image_id, "image_path": file.filename or ""},
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pipeline error: {e}")

    annotation = result.get("annotation")
    if not annotation:
        raise HTTPException(status_code=500, detail="Pipeline produced no annotation")

    _state.records[annotation.image_id] = annotation
    return annotation


@app.post("/annotate/batch", response_model=list[AnnotationRecord])
async def annotate_batch(files: list[UploadFile] = File(...)):
    """Upload multiple images and process them through the pipeline."""
    results: list[AnnotationRecord] = []

    async def process_one(f: UploadFile) -> AnnotationRecord | None:
        image_bytes = await f.read()
        image_id = f.filename or str(uuid.uuid4())
        try:
            result = await _state.graph.ainvoke(
                {"image_bytes": image_bytes, "image_id": image_id, "image_path": f.filename or ""},
            )
            return result.get("annotation")
        except Exception:
            logger.exception("Batch item failed: %s", f.filename)
            return None

    tasks = [process_one(f) for f in files]
    raw_results = await asyncio.gather(*tasks)

    for annotation in raw_results:
        if annotation:
            _state.records[annotation.image_id] = annotation
            results.append(annotation)

    return results


@app.get("/annotations/{image_id}", response_model=AnnotationRecord)
async def get_annotation(image_id: str):
    """Retrieve an annotation by image ID."""
    record = _state.records.get(image_id)
    if not record:
        raise HTTPException(status_code=404, detail=f"Annotation not found: {image_id}")
    return record


@app.get("/annotations", response_model=list[AnnotationRecord])
async def list_annotations(status: str | None = Query(None)):
    """List annotations, optionally filtered by review status."""
    records = list(_state.records.values())
    if status:
        records = [r for r in records if r.review_status == status]
    return records


@app.get("/export/label-studio/config")
async def export_label_studio_config():
    return Response(content=label_studio_config(), media_type="application/xml")


@app.get("/export/label-studio")
async def export_label_studio(
    image_base_url: str = Query("http://localhost:9000/golden-dataset-images"),
):
    """Export pending-review annotations in Label Studio import format."""
    pending = [
        r for r in _state.records.values()
        if r.review_status in ("pending_review", "PENDING_REVIEW")
    ]
    if not pending:
        return {"tasks": [], "message": "No pending reviews"}
    return {"tasks": format_for_label_studio(pending, image_base_url)}


@app.post("/review/{image_id}")
async def submit_review(image_id: str, review: ReviewRequest):
    """Submit a human review decision for an annotation."""
    record = _state.records.get(image_id)
    if not record:
        raise HTTPException(status_code=404, detail=f"Annotation not found: {image_id}")

    # Apply only explicitly supplied fields, retaining untouched labels.
    if review.corrected_attributes is not None:
        merged = record.attributes.model_dump()
        merged.update(review.corrected_attributes.model_dump(exclude_unset=True))
        record.attributes = PersonAttributes.model_validate(merged)
    if review.corrected_caption is not None:
        record.caption = review.corrected_caption
    record.review_status = ReviewStatus(review.status)
    return {"image_id": image_id, "new_status": review.status, "notes": review.notes}


@app.get("/stats", response_model=StatsResponse)
async def get_stats():
    """Return pipeline statistics."""
    records = list(_state.records.values())
    if not records:
        return StatsResponse(
            total_processed=0, auto_accepted=0, pending_review=0,
            acceptance_rate=0.0, mean_confidence=0.0,
        )

    auto = [r for r in records if r.review_status in ("auto_accepted", "AUTO_ACCEPTED")]
    pending = [r for r in records if r.review_status in ("pending_review", "PENDING_REVIEW")]

    return StatsResponse(
        total_processed=len(records),
        auto_accepted=len(auto),
        pending_review=len(pending),
        acceptance_rate=len(auto) / len(records) if records else 0.0,
        mean_confidence=sum(r.confidence for r in records) / len(records),
    )
