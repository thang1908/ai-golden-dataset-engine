"""Prediction envelopes; no synthetic quality scores or review decisions."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from golden_dataset_harness.schemas.taxonomy import AttributeValue, validate_attributes

Method = Literal["c1_vllm_medium", "c2_siglip"]
Probability = Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]


class InferenceResult(BaseModel):
    model_config = ConfigDict(extra="forbid")
    attributes: dict[str, AttributeValue]
    scores: dict[str, dict[str, Probability]] | None = None

    @field_validator("attributes", mode="before")
    @classmethod
    def check_attributes(cls, value):
        return validate_attributes(value, require_all=True)


class PredictionError(BaseModel):
    code: str
    message: str


class PredictionRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")
    schema_version: Literal["baseline-v1"] = "baseline-v1"
    attribute_schema_version: Literal["siglip-v1"] = "siglip-v1"
    run_id: str
    image_id: str
    image_path: str
    image_sha256: str | None
    method: Method
    model_id: str
    status: Literal["success", "error"]
    attributes: dict[str, AttributeValue] | None = None
    scores: dict[str, dict[str, Probability]] | None = None
    elapsed_ms: float = Field(ge=0)
    model_calls: int = Field(ge=0)
    http_attempts: int = Field(ge=0)
    error: PredictionError | None = None

    @model_validator(mode="after")
    def check_status(self):
        if self.status == "success":
            if self.attributes is None or self.error is not None:
                raise ValueError("Success requires attributes and no error")
            validate_attributes(self.attributes, require_all=True)
        elif self.attributes is not None or self.scores is not None or self.error is None:
            raise ValueError("Error requires an error object and no prediction")
        return self
