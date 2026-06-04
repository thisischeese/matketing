"""Typed records used by the best-shot benchmark."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class VideoEntry:
    """Single input video entry."""

    video_id: str
    video_path: Path


@dataclass(frozen=True)
class SampledFrame:
    """Single sampled frame written to disk."""

    video_id: str
    frame_id: str
    video_path: Path
    frame_path: Path
    frame_index: int
    sample_index: int
    timestamp_ms: int
    width: int
    height: int


@dataclass(frozen=True)
class LabelEntry:
    """Human-selected label entry."""

    video_id: str
    frame_id: str


@dataclass(frozen=True)
class FrameScore:
    """Per-frame model score and latency."""

    model_name: str
    video_id: str
    frame_id: str
    sample_index: int
    frame_path: Path
    score: float
    latency_ms: float


@dataclass(frozen=True)
class PredictionRecord:
    """Selected frame prediction for one segment."""

    model_name: str
    video_id: str
    frame_id: str
    sample_index: int
    segment_index: int
    score: float


@dataclass(frozen=True)
class PerVideoSummary:
    """Per-video evaluation summary for one model."""

    model_name: str
    video_id: str
    sampled_frame_count: int
    hit_count: int
    precision_at_3: float
    total_inference_ms: float
    avg_frame_latency_ms: float
    p95_frame_latency_ms: float
    predicted_frame_ids: tuple[str, ...]
    labeled_frame_ids: tuple[str, ...]


@dataclass(frozen=True)
class ModelSummary:
    """Aggregate evaluation summary for one model."""

    model_name: str
    video_count: int
    overall_precision_at_3: float
    avg_frame_latency_ms: float
    p95_frame_latency_ms: float
    total_inference_ms: float
    total_runtime_ms: float
    eligible: bool

