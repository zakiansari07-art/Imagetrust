from typing import Literal

from pydantic import BaseModel, Field


Verdict = Literal[
    "likely_real",
    "likely_ai_generated",
    "uncertain",
]

SourceStatus = Literal[
    "identified",
    "unknown",
    "not_applicable",
]


class ModelPrediction(BaseModel):
    label: str
    confidence: float = Field(ge=0, le=1)
    probabilities: dict[str, float]
    model_name: str


class ForensicsReport(BaseModel):
    file_name: str
    file_format: str | None
    width: int
    height: int
    original_color_mode: str
    has_exif_metadata: bool
    sharpness_score: float
    average_brightness: float


class AnalysisResult(BaseModel):
    verdict: Verdict
    confidence: float | None

    source_status: SourceStatus
    likely_generator: str | None
    generator_confidence: float | None

    explanation: str
    llm_report: str | None = None

    real_vs_ai: ModelPrediction
    generator_attribution: ModelPrediction
    forensics: ForensicsReport