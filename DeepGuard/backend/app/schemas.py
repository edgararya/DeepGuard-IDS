from pydantic import BaseModel
from typing import Optional, List


class HealthResponse(BaseModel):
    status: str
    models_loaded: bool
    version: str


class ErrorResponse(BaseModel):
    error_code: str
    message: str
    detail: str


class PerModelScores(BaseModel):
    mesonet_1: Optional[float] = None
    mesonet_2: Optional[float] = None
    mesonet_3: Optional[float] = None
    xception: Optional[float] = None


class DetectResponse(BaseModel):
    label: str
    confidence: float
    per_model_scores: PerModelScores
    ela_score: Optional[float] = None
    blur_score: Optional[float] = None
    media_type: str
    frames_analyzed: Optional[int] = None
    processing_time_ms: int


class HistoryEntry(BaseModel):
    id: str
    source: str
    label: str
    confidence: float
    timestamp: int


class HistoryResponse(BaseModel):
    entries: List[HistoryEntry]
