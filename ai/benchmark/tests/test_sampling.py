from __future__ import annotations

from pathlib import Path

import numpy as np

from bestshot_benchmark.io import read_frames_csv, write_frames_csv
from bestshot_benchmark.sampling import _import_cv2, sample_video_frames
from bestshot_benchmark.types import VideoEntry


def _write_test_video(path: Path, fps: float, frame_count: int) -> None:
    cv2 = _import_cv2()
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (32, 24))
    assert writer.isOpened()
    for index in range(frame_count):
        frame = np.full((24, 32, 3), index * 20, dtype=np.uint8)
        writer.write(frame)
    writer.release()


def test_sample_video_frames_matches_service_sampling_policy(tmp_path: Path) -> None:
    video_path = tmp_path / "sample.avi"
    _write_test_video(video_path, fps=10.0, frame_count=10)

    frames = sample_video_frames(
        VideoEntry(video_id="video-1", video_path=video_path),
        frames_root=tmp_path / "frames",
    )

    assert len(frames) == 5
    assert [frame.frame_id for frame in frames] == [
        "frame-00001",
        "frame-00002",
        "frame-00003",
        "frame-00004",
        "frame-00005",
    ]
    assert [frame.frame_index for frame in frames] == [0, 2, 4, 6, 8]
    assert [frame.timestamp_ms for frame in frames] == [0, 200, 400, 600, 800]
    assert all(frame.frame_path.exists() for frame in frames)


def test_frames_csv_round_trip_preserves_manifest(tmp_path: Path) -> None:
    video_path = tmp_path / "sample.avi"
    _write_test_video(video_path, fps=5.0, frame_count=4)

    frames = sample_video_frames(
        VideoEntry(video_id="video-2", video_path=video_path),
        frames_root=tmp_path / "frames",
    )
    manifest_path = tmp_path / "frames.csv"
    write_frames_csv(manifest_path, frames)

    restored = read_frames_csv(manifest_path)

    assert restored == frames

