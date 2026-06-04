from __future__ import annotations

from pathlib import Path

import pytest

from bestshot_benchmark.evaluation import (
    build_label_index,
    derive_latency_cutoff_ms,
    select_segment_best,
    select_winner,
)
from bestshot_benchmark.types import FrameScore, LabelEntry, ModelSummary, SampledFrame


def _frame(video_id: str, frame_id: str, sample_index: int) -> SampledFrame:
    return SampledFrame(
        video_id=video_id,
        frame_id=frame_id,
        video_path=Path(f"C:/{video_id}.mp4"),
        frame_path=Path(f"C:/{video_id}/{frame_id}.png"),
        frame_index=sample_index - 1,
        sample_index=sample_index,
        timestamp_ms=sample_index * 100,
        width=32,
        height=24,
    )


def test_build_label_index_requires_exactly_three_labels_per_video() -> None:
    frames = [_frame("video-1", f"frame-{index:05d}", index) for index in range(1, 6)]
    labels = [
        LabelEntry(video_id="video-1", frame_id="frame-00001"),
        LabelEntry(video_id="video-1", frame_id="frame-00002"),
    ]

    with pytest.raises(ValueError, match="exactly 3 labels"):
        build_label_index(labels, frames)


def test_select_segment_best_picks_one_best_frame_per_segment() -> None:
    scores = [
        FrameScore("model", "video-1", "frame-00001", 1, Path("1.png"), 1.0, 10.0),
        FrameScore("model", "video-1", "frame-00002", 2, Path("2.png"), 2.0, 10.0),
        FrameScore("model", "video-1", "frame-00003", 3, Path("3.png"), 10.0, 10.0),
        FrameScore("model", "video-1", "frame-00004", 4, Path("4.png"), 3.0, 10.0),
        FrameScore("model", "video-1", "frame-00005", 5, Path("5.png"), 4.0, 10.0),
        FrameScore("model", "video-1", "frame-00006", 6, Path("6.png"), 5.0, 10.0),
    ]

    predictions = select_segment_best(scores, k=3)

    assert [prediction.frame_id for prediction in predictions] == [
        "frame-00002",
        "frame-00003",
        "frame-00006",
    ]


def test_latency_cutoff_and_winner_selection_follow_plan_rule() -> None:
    cutoff_ms = derive_latency_cutoff_ms([9, 12, 15], total_budget_ms=3000.0)
    assert cutoff_ms == pytest.approx(250.0)

    winner = select_winner(
        [
            ModelSummary("fast-good", 3, 0.777777, 100.0, 120.0, 1200.0, 1500.0, True),
            ModelSummary("fast-best", 3, 0.888888, 110.0, 150.0, 1300.0, 1600.0, True),
            ModelSummary("slow-best", 3, 1.0, 320.0, 350.0, 4000.0, 4300.0, False),
        ]
    )

    assert winner is not None
    assert winner.model_name == "fast-best"

