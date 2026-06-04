from __future__ import annotations

from pathlib import Path
import time

import numpy as np

from bestshot_benchmark.contact_sheet import render_contact_sheets
from bestshot_benchmark.io import write_csv, write_frames_csv
from bestshot_benchmark.runner import run_benchmark
from bestshot_benchmark.sampling import _import_cv2, sample_videos
from bestshot_benchmark.types import VideoEntry


def _write_test_video(path: Path, frame_count: int = 6) -> None:
    cv2 = _import_cv2()
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    writer = cv2.VideoWriter(str(path), fourcc, 5.0, (32, 24))
    assert writer.isOpened()
    for index in range(frame_count):
        frame = np.full((24, 32, 3), index * 30, dtype=np.uint8)
        writer.write(frame)
    writer.release()


class _FakeScorer:
    def __init__(self, preferred: set[int], delay_ms: float = 0.0) -> None:
        self._preferred = preferred
        self._delay_ms = delay_ms

    def score(self, frame_path: Path) -> float:
        if self._delay_ms > 0:
            time.sleep(self._delay_ms / 1000.0)
        sample_index = int(frame_path.stem.split("-")[-1])
        if sample_index in self._preferred:
            return 100.0 + sample_index
        return float(sample_index)


def test_run_benchmark_writes_expected_artifacts_and_selects_winner(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    video_path = tmp_path / "video.avi"
    _write_test_video(video_path)

    frames = sample_videos(
        [VideoEntry(video_id="video-1", video_path=video_path)],
        workspace=workspace,
    )
    frames_csv_path = workspace / "frames.csv"
    write_frames_csv(frames_csv_path, frames)

    rendered = render_contact_sheets(frames, workspace / "contact_sheets")
    assert len(rendered) == 1
    assert rendered[0].exists()

    labels_csv_path = workspace / "labels.csv"
    write_csv(
        labels_csv_path,
        ["video_id", "frame_id"],
        [
            {"video_id": "video-1", "frame_id": "frame-00002"},
            {"video_id": "video-1", "frame_id": "frame-00004"},
            {"video_id": "video-1", "frame_id": "frame-00006"},
        ],
    )

    registry = {
        "perfect": lambda device: _FakeScorer({2, 4, 6}),
        "alternate": lambda device: _FakeScorer({1, 3, 5}),
        "slow-perfect": lambda device: _FakeScorer({2, 4, 6}, delay_ms=10.0),
    }

    summary = run_benchmark(
        frames_csv_path=frames_csv_path,
        labels_csv_path=labels_csv_path,
        workspace=workspace,
        model_names=["perfect", "alternate", "slow-perfect"],
        scorer_registry=registry,
        total_budget_ms=30.0,
    )

    assert summary["winner"] is not None
    assert summary["winner"]["model_name"] == "perfect"
    assert (workspace / "summary.json").exists()
    assert (workspace / "per_video.csv").exists()
    assert (workspace / "predictions.csv").exists()
    assert (workspace / "latency.csv").exists()

