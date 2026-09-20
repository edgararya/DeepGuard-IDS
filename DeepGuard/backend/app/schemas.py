from pydantic import BaseModel
from typing import Optional, Dict, List, Any

class HealthResponse(BaseModel):
    success: bool
    message: str
    processing_time_ms: int
    status: str
    warnings: List[str]
    models_loaded: Dict[str, bool]

class ImageAnalysisResult(BaseModel):
    has_face: bool
    fake_prob: float
    verdict: str
    scores: Dict[str, Optional[float]]
    face_box: Optional[List[int]]

class ImageAnalysisResponse(BaseModel):
    success: bool
    message: str
    processing_time_ms: int
    warnings: List[str] = []
    result: Optional[ImageAnalysisResult] = None
    error: Optional[str] = None

class VideoAnalysisResult(BaseModel):
    fps: float
    duration_seconds: float
    total_frames_analyzed: int
    fake_frame_count: int
    real_frame_count: int
    no_face_count: int
    avg_fake_prob: float
    max_fake_prob: float
    verdict: str

class VideoAnalysisResponse(BaseModel):
    success: bool
    message: str
    processing_time_ms: int
    warnings: List[str] = []
    result: Optional[VideoAnalysisResult] = None
    error: Optional[str] = None
