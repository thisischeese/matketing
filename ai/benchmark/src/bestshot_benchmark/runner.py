"""Benchmark execution and report generation."""

from __future__ import annotations

from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
import statistics
import time

from bestshot_benchmark.evaluation import (
    build_label_index,
    calculate_precision_at_3,
    derive_latency_cutoff_ms,
    group_frames_by_video,
    percentile,
    select_segment_best,
    select_winner,
)
from bestshot_benchmark.io import read_frames_csv, read_labels_csv, write_csv, write_json
from bestshot_benchmark.models import ScorerFactory, build_default_scorer_registry
from bestshot_benchmark.types import FrameScore, ModelSummary, PerVideoSummary, PredictionRecord, SampledFrame


def _sample_counts(grouped_frames: dict[str, list[SampledFrame]]) -> list[int]:
    return [len(frames) for frames in grouped_frames.values()]


def run_benchmark(
    frames_csv_path: Path,
    labels_csv_path: Path,
    workspace: Path,
    model_names: list[str] | None = None,
    device: str | None = None,
    scorer_registry: dict[str, ScorerFactory] | None = None,
    total_budget_ms: float = 3000.0,
) -> dict[str, object]:
    """Run the benchmark and emit all report artifacts."""

    workspace.mkdir(parents=True, exist_ok=True)
    frames = read_frames_csv(frames_csv_path)
    labels = read_labels_csv(labels_csv_path)
    grouped_frames = group_frames_by_video(frames)
    labels_by_video = build_label_index(labels, frames)
    sampled_frame_counts = _sample_counts(grouped_frames)
    latency_cutoff_ms = derive_latency_cutoff_ms(
        sampled_frame_counts,
        total_budget_ms=total_budget_ms,
    )

    registry = scorer_registry or build_default_scorer_registry()
    selected_models = model_names or ["nima", "clipiqa", "qalign"]
    unknown_models = [name for name in selected_models if name not in registry]
    if unknown_models:
        raise ValueError(f"Unsupported models requested: {', '.join(unknown_models)}")

    predictions_rows: list[dict[str, object]] = []
    latency_rows: list[dict[str, object]] = []
    per_video_rows: list[dict[str, object]] = []
    model_summaries: list[ModelSummary] = []

    for model_name in selected_models:
        scorer = registry[model_name](device)
        model_started_at = time.perf_counter()
        model_frame_scores: list[FrameScore] = []
        model_video_summaries: list[PerVideoSummary] = []

        for video_id, video_frames in grouped_frames.items():
            scored_frames: list[FrameScore] = []
            for frame in video_frames:
                started_at = time.perf_counter()
                score = scorer.score(frame.frame_path)
                latency_ms = (time.perf_counter() - started_at) * 1000.0
                frame_score = FrameScore(
                    model_name=model_name,
                    video_id=video_id,
                    frame_id=frame.frame_id,
                    sample_index=frame.sample_index,
                    frame_path=frame.frame_path,
                    score=float(score),
                    latency_ms=latency_ms,
                )
                scored_frames.append(frame_score)
                model_frame_scores.append(frame_score)
                latency_rows.append(
                    {
                        "model_name": model_name,
                        "video_id": video_id,
                        "frame_id": frame.frame_id,
                        "sample_index": frame.sample_index,
                        "score": round(frame_score.score, 6),
                        "latency_ms": round(frame_score.latency_ms, 3),
                    }
                )

            predictions = select_segment_best(scored_frames, k=3)
            predicted_frame_ids = [prediction.frame_id for prediction in predictions]
            hit_count, precision_at_3 = calculate_precision_at_3(
                predicted_frame_ids,
                labels_by_video[video_id],
            )
            video_latencies = [item.latency_ms for item in scored_frames]
            per_video_summary = PerVideoSummary(
                model_name=model_name,
                video_id=video_id,
                sampled_frame_count=len(scored_frames),
                hit_count=hit_count,
                precision_at_3=precision_at_3,
                total_inference_ms=sum(video_latencies),
                avg_frame_latency_ms=statistics.mean(video_latencies),
                p95_frame_latency_ms=percentile(video_latencies, 0.95),
                predicted_frame_ids=tuple(predicted_frame_ids),
                labeled_frame_ids=tuple(sorted(labels_by_video[video_id])),
            )
            model_video_summaries.append(per_video_summary)
            per_video_rows.append(
                {
                    "model_name": model_name,
                    "video_id": video_id,
                    "sampled_frame_count": per_video_summary.sampled_frame_count,
                    "hit_count": per_video_summary.hit_count,
                    "precision_at_3": round(per_video_summary.precision_at_3, 6),
                    "total_inference_ms": round(per_video_summary.total_inference_ms, 3),
                    "avg_frame_latency_ms": round(per_video_summary.avg_frame_latency_ms, 3),
                    "p95_frame_latency_ms": round(per_video_summary.p95_frame_latency_ms, 3),
                    "predicted_frame_ids": "|".join(per_video_summary.predicted_frame_ids),
                    "labeled_frame_ids": "|".join(per_video_summary.labeled_frame_ids),
                }
            )
            for prediction in predictions:
                predictions_rows.append(
                    {
                        "model_name": prediction.model_name,
                        "video_id": prediction.video_id,
                        "frame_id": prediction.frame_id,
                        "sample_index": prediction.sample_index,
                        "segment_index": prediction.segment_index,
                        "score": round(prediction.score, 6),
                    }
                )

        model_runtime_ms = (time.perf_counter() - model_started_at) * 1000.0
        all_latencies = [item.latency_ms for item in model_frame_scores]
        total_hits = sum(item.hit_count for item in model_video_summaries)
        eligible = statistics.mean(all_latencies) <= latency_cutoff_ms if all_latencies else False
        model_summaries.append(
            ModelSummary(
                model_name=model_name,
                video_count=len(model_video_summaries),
                overall_precision_at_3=total_hits / (len(model_video_summaries) * 3.0),
                avg_frame_latency_ms=statistics.mean(all_latencies) if all_latencies else 0.0,
                p95_frame_latency_ms=percentile(all_latencies, 0.95),
                total_inference_ms=sum(all_latencies),
                total_runtime_ms=model_runtime_ms,
                eligible=eligible,
            )
        )

    winner = select_winner(model_summaries)

    write_csv(
        workspace / "predictions.csv",
        ["model_name", "video_id", "frame_id", "sample_index", "segment_index", "score"],
        predictions_rows,
    )
    write_csv(
        workspace / "latency.csv",
        ["model_name", "video_id", "frame_id", "sample_index", "score", "latency_ms"],
        latency_rows,
    )
    write_csv(
        workspace / "per_video.csv",
        [
            "model_name",
            "video_id",
            "sampled_frame_count",
            "hit_count",
            "precision_at_3",
            "total_inference_ms",
            "avg_frame_latency_ms",
            "p95_frame_latency_ms",
            "predicted_frame_ids",
            "labeled_frame_ids",
        ],
        per_video_rows,
    )

    summary_payload: dict[str, object] = {
        "generated_at": datetime.now(UTC).isoformat(),
        "frames_csv": str(frames_csv_path.resolve()),
        "labels_csv": str(labels_csv_path.resolve()),
        "workspace": str(workspace.resolve()),
        "device": device,
        "models": [
            {
                **asdict(summary),
                "overall_precision_at_3": round(summary.overall_precision_at_3, 6),
                "avg_frame_latency_ms": round(summary.avg_frame_latency_ms, 3),
                "p95_frame_latency_ms": round(summary.p95_frame_latency_ms, 3),
                "total_inference_ms": round(summary.total_inference_ms, 3),
                "total_runtime_ms": round(summary.total_runtime_ms, 3),
            }
            for summary in model_summaries
        ],
        "latency_rule": {
            "total_budget_ms": total_budget_ms,
            "sampled_frame_count_median": statistics.median(sampled_frame_counts),
            "avg_frame_latency_cutoff_ms": round(latency_cutoff_ms, 6),
        },
        "winner": (
            {
                "model_name": winner.model_name,
                "overall_precision_at_3": round(winner.overall_precision_at_3, 6),
                "avg_frame_latency_ms": round(winner.avg_frame_latency_ms, 3),
                "p95_frame_latency_ms": round(winner.p95_frame_latency_ms, 3),
            }
            if winner is not None
            else None
        ),
    }
    write_json(workspace / "summary.json", summary_payload)
    return summary_payload

