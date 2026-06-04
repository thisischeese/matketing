"""Benchmark evaluation helpers."""

from __future__ import annotations

import math
import statistics

from bestshot_benchmark.types import FrameScore, LabelEntry, ModelSummary, PredictionRecord, SampledFrame


def group_frames_by_video(frames: list[SampledFrame]) -> dict[str, list[SampledFrame]]:
    """Group sampled frames by video while preserving timeline order."""

    grouped: dict[str, list[SampledFrame]] = {}
    for frame in sorted(frames, key=lambda item: (item.video_id, item.sample_index)):
        grouped.setdefault(frame.video_id, []).append(frame)
    return grouped


def build_label_index(
    labels: list[LabelEntry],
    frames: list[SampledFrame],
) -> dict[str, set[str]]:
    """Validate labels and return them as a video_id -> frame_id set mapping."""

    frame_ids_by_video = {
        video_id: {frame.frame_id for frame in video_frames}
        for video_id, video_frames in group_frames_by_video(frames).items()
    }
    labels_by_video: dict[str, set[str]] = {video_id: set() for video_id in frame_ids_by_video}

    for label in labels:
        if label.video_id not in frame_ids_by_video:
            raise ValueError(f"labels.csv references unknown video_id: {label.video_id}")
        if label.frame_id not in frame_ids_by_video[label.video_id]:
            raise ValueError(
                f"labels.csv references unknown frame_id '{label.frame_id}' for video '{label.video_id}'"
            )
        if label.frame_id in labels_by_video[label.video_id]:
            raise ValueError(
                f"labels.csv contains a duplicate frame_id '{label.frame_id}' for video '{label.video_id}'"
            )
        labels_by_video[label.video_id].add(label.frame_id)

    for video_id, frame_ids in labels_by_video.items():
        if len(frame_ids) != 3:
            raise ValueError(
                f"labels.csv must contain exactly 3 labels for video '{video_id}', found {len(frame_ids)}"
            )
    return labels_by_video


def percentile(values: list[float], p: float) -> float:
    """Return a simple nearest-rank percentile."""

    if not values:
        return 0.0
    if p <= 0:
        return min(values)
    if p >= 1:
        return max(values)
    ordered = sorted(values)
    rank = max(0, math.ceil(p * len(ordered)) - 1)
    return ordered[rank]


def select_segment_best(scored_frames: list[FrameScore], k: int = 3) -> list[PredictionRecord]:
    """Pick one highest-scoring frame from each timeline segment."""

    ordered = sorted(scored_frames, key=lambda item: item.sample_index)
    if not ordered or k <= 0:
        return []
    if len(ordered) <= k:
        return [
            PredictionRecord(
                model_name=item.model_name,
                video_id=item.video_id,
                frame_id=item.frame_id,
                sample_index=item.sample_index,
                segment_index=index,
                score=item.score,
            )
            for index, item in enumerate(ordered)
        ]

    predictions: list[PredictionRecord] = []
    total = len(ordered)
    for segment_index in range(k):
        start = (segment_index * total) // k
        end = ((segment_index + 1) * total) // k
        segment = ordered[start:end]
        if not segment:
            continue
        best = max(segment, key=lambda item: item.score)
        predictions.append(
            PredictionRecord(
                model_name=best.model_name,
                video_id=best.video_id,
                frame_id=best.frame_id,
                sample_index=best.sample_index,
                segment_index=segment_index,
                score=best.score,
            )
        )
    return predictions


def calculate_precision_at_3(predicted_frame_ids: list[str], labeled_frame_ids: set[str]) -> tuple[int, float]:
    """Return hit count and precision@3."""

    hits = len(set(predicted_frame_ids) & labeled_frame_ids)
    return hits, hits / 3.0


def derive_latency_cutoff_ms(
    sampled_frame_counts: list[int],
    total_budget_ms: float = 3000.0,
) -> float:
    """Compute the per-frame average latency cutoff from the sampled-frame median."""

    if not sampled_frame_counts:
        raise ValueError("Cannot derive a latency cutoff without sampled frame counts.")
    median_count = statistics.median(sampled_frame_counts)
    if median_count <= 0:
        raise ValueError("Median sampled frame count must be positive.")
    return total_budget_ms / float(median_count)


def select_winner(model_summaries: list[ModelSummary]) -> ModelSummary | None:
    """Pick the winning model under the latency eligibility rule."""

    eligible = [summary for summary in model_summaries if summary.eligible]
    if not eligible:
        return None
    return sorted(
        eligible,
        key=lambda item: (
            -item.overall_precision_at_3,
            item.p95_frame_latency_ms,
            item.avg_frame_latency_ms,
            item.total_runtime_ms,
        ),
    )[0]

