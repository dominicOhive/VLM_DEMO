from pydantic import BaseModel, Field
from typing import Any, List, Optional, Literal


class BBox(BaseModel):
    x: float
    y: float
    width: float
    height: float


class Detection(BaseModel):
    label: str
    description: str
    bbox: Optional[BBox] = None
    confidence: float = Field(ge=0.0, le=1.0)
    evidence: Optional[str] = None
    condition_id: Optional[str] = None
    mask: Optional[dict[str, Any]] = None
    segmentation_mode: Optional[str] = None
    segmentation_status: Optional[str] = None
    sam3_prompt: Optional[str] = None


class AssessmentCheck(BaseModel):
    name: str
    status: Literal["pass", "fail", "uncertain", "not_applicable"] = "uncertain"
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    evidence: Optional[str] = None
    bbox: Optional[BBox] = None


class VisionResult(BaseModel):
    result_type: Literal["assessment", "detection", "mixed"] = "mixed"
    status: Literal["pass", "fail", "uncertain", "not_applicable"] = "uncertain"
    summary: str = ""
    answer: Optional[str] = None
    vision_plan: Optional[dict[str, Any]] = None
    preprocessing: Optional[dict[str, Any]] = None
    counts: Optional[dict[str, int]] = None
    checks: List[AssessmentCheck] = Field(default_factory=list)
    detections: List[Detection] = Field(default_factory=list)


class StepCondition(BaseModel):
    id: Optional[str] = None
    name: str
    type: Optional[
        Literal["visual", "spatial", "temporal", "attribute", "reference"]
    ] = "visual"
    value: Optional[str] = None
    reference_image_url: Optional[str] = None


class DetectRequest(BaseModel):
    sequence_id: Optional[str] = None
    step_id: Optional[str] = None
    capture_id: Optional[str] = None
    condition_name: str
    condition_type: Optional[
        Literal["visual", "spatial", "temporal", "attribute", "reference"]
    ] = "visual"
    prompt: Optional[str] = None
    step_name: Optional[str] = None
    step_description: Optional[str] = None
    conditions: List[StepCondition] = Field(default_factory=list)
    image_base64: str


class DetectResponse(BaseModel):
    model: str
    sequence_id: Optional[str] = None
    step_id: Optional[str] = None
    capture_id: Optional[str] = None
    image_width: int
    image_height: int
    result_type: Literal["assessment", "detection", "mixed"] = "mixed"
    status: Literal["pass", "fail", "uncertain", "not_applicable"] = "uncertain"
    summary: str = ""
    checks: List[AssessmentCheck] = Field(default_factory=list)
    detections: List[Detection]
    raw_text: Optional[str] = None


class HealthResponse(BaseModel):
    status: str
    model: str
    queue: Optional[dict[str, Any]] = None
