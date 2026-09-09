from typing import Literal

from pydantic import BaseModel, Field


# ============================================================
# EXISTING TYPES
# ============================================================

Verdict = Literal[
    "likely_real",
    "likely_ai_generated",
]

ConfidenceLevel = Literal[
    "Very_High",
    "High",
    "Medium",
    "Low",
    "not_applicable"
]



# ============================================================
# MODEL PREDICTION
# ============================================================

class ModelPrediction(BaseModel):
    label: str | None
    confidence: float = Field(ge=0, le=1)
    probabilities: dict[str, float]
    model_name: str


# ============================================================
# METADATA / PROVENANCE
# ============================================================

class CameraMetadata(BaseModel):
    make: str | None = None
    model: str | None = None
    software: str | None = None
    date_time: str | None = None


class ExifReport(BaseModel):
    present: bool
    camera: CameraMetadata
    fields: dict[str, str] = {}


class XmpReport(BaseModel):
    present: bool
    fields: dict[str, str] = {}


class C2PASignature(BaseModel):
    issuer: str | None = None
    time: str | None = None
    algorithm: str | None = None


class C2PAAction(BaseModel):
    action: str | None = None
    when: str | None = None
    software_agent: str | None = None
    parameters: dict | None = None


class C2PAIngredient(BaseModel):
    title: str | None = None
    format: str | None = None
    relationship: str | None = None
    instance_id: str | None = None


class C2PAManifest(BaseModel):
    manifest_id: str | None = None
    title: str | None = None
    format: str | None = None
    claim_generator: str | None = None

    signature: C2PASignature

    actions: list[C2PAAction] = []
    ingredients: list[C2PAIngredient] = []


class C2PAReport(BaseModel):
    available: bool

    # None means the check failed and we cannot determine
    # whether a manifest exists.
    present: bool | None = None

    # None means validation could not be determined.
    verified: bool | str | None = None

    validation_results: str | None = None

    active_manifest: str | None = None

    manifests: list[C2PAManifest] = []

    active_manifest_summary: C2PAManifest | None = None

    analysis_error: bool = False
    error: str | None = None


class ProvenanceReport(BaseModel):
    exif: ExifReport
    xmp: XmpReport
    c2pa: C2PAReport


# ============================================================
# COLOR
# ============================================================

class ColorReport(BaseModel):
    mean_rgb: list[float] = Field(
        min_length=3,
        max_length=3
    )

    std_rgb: list[float] = Field(
        min_length=3,
        max_length=3
    )

    mean_saturation: float
    mean_brightness: float


# ============================================================
# QUALITY
# ============================================================

class QualityReport(BaseModel):
    sharpness_score: float
    blur_score: float
    contrast: float
    entropy: float


# ============================================================
# NOISE
# ============================================================

class NoiseReport(BaseModel):
    noise_mean: float
    noise_std: float
    noise_median: float
    noise_mad: float
    noise_spatial_std: float


# ============================================================
# EDGES
# ============================================================

class EdgeReport(BaseModel):
    edge_density: float
    edge_mean: float


# ============================================================
# FREQUENCY
# ============================================================

class FrequencyReport(BaseModel):
    low_frequency_energy: float
    high_frequency_energy: float
    high_frequency_ratio: float


# ============================================================
# DCT
# ============================================================

class DCTReport(BaseModel):
    mean_low_frequency_dct_energy: float
    mean_high_frequency_dct_energy: float
    high_frequency_dct_ratio: float


# ============================================================
# COMPRESSION
# ============================================================

class CompressionReport(BaseModel):
    jpeg_detected: bool

    estimated_jpeg_quality: int | None = Field(
        default=None,
        ge=1,
        le=100
    )

    block_artifact_score: float

    dct: DCTReport


# ============================================================
# ELA
# ============================================================

class ELAReport(BaseModel):
    applicable: bool

    mean_error: float | None = None
    std_error: float | None = None
    max_error: float | None = None

    error: str | None = None


# ============================================================
# RESAMPLING
# ============================================================

class ResamplingReport(BaseModel):
    resampling_score: float | None = None
    interpolation_error: float | None = None


# ============================================================
# DOUBLE COMPRESSION
# ============================================================

class DoubleCompressionReport(BaseModel):
    applicable: bool

    double_compression_indicator: bool | None = None

    score: float | None = None


# ============================================================
# RESIZING
# ============================================================

class ResizingReport(BaseModel):
    common_dimension_detected: bool
    reconstruction_error: float
# ============================================================
# FileReport
# ============================================================


class FileReport(BaseModel):
    file_name: str
    file_size_bytes: int
    file_format: str | None
# ============================================================
# ImageReport
# ============================================================

class ImageReport(BaseModel):
    width: int
    height: int
    channels: int
    aspect_ratio: float




# ============================================================
# COMPLETE FORENSICS REPORT
# ============================================================

class ForensicsReport(BaseModel):
    file: FileReport
    image: ImageReport

    metadata: ProvenanceReport

    color: ColorReport
    quality: QualityReport
    noise: NoiseReport
    edges: EdgeReport
    frequency: FrequencyReport
    compression: CompressionReport
    ela: ELAReport
    resampling: ResamplingReport
    double_compression: DoubleCompressionReport
    resizing: ResizingReport

# ============================================================
# COMPLETE ANALYSIS RESULT
# ============================================================

class AnalysisResult(BaseModel):

    verdict: Verdict

    confidence: float | None = Field(
        default=None,
        ge=0,
        le=1
    )

    confidence_level: ConfidenceLevel

    likely_generator: str | None = None

    generator_confidence: float | None = Field(
        default=None,
        ge=0,
        le=1
    )

    explanation: str

    llm_report: str | None = None

    real_vs_ai: ModelPrediction

    generator_attribution: ModelPrediction

    forensics: ForensicsReport