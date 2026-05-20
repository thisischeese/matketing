from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pytest

from app.perfectframe.video_processors import OpenCVVideo


def test_get_sampling_step_samples_30fps_video_at_5fps() -> None:
    assert OpenCVVideo._get_sampling_step(30) == 6


def test_get_sampling_step_samples_24fps_video_at_5fps() -> None:
    assert OpenCVVideo._get_sampling_step(24) == 5


def test_get_sampling_step_keeps_5fps_video_at_every_frame() -> None:
    assert OpenCVVideo._get_sampling_step(5) == 1


def test_get_sampling_step_keeps_lower_fps_video_at_every_frame() -> None:
    assert OpenCVVideo._get_sampling_step(4) == 1


def test_normalize_frame_rate_falls_back_for_invalid_or_extreme_values() -> None:
    assert OpenCVVideo._normalize_frame_rate(None) == 5
    assert OpenCVVideo._normalize_frame_rate(0) == 5
    assert OpenCVVideo._normalize_frame_rate(1000) == 5


def test_normalize_frame_count_returns_none_for_invalid_values() -> None:
    assert OpenCVVideo._normalize_frame_count(None) is None
    assert OpenCVVideo._normalize_frame_count(-1) is None
    assert OpenCVVideo._normalize_frame_count(20_000_000) is None


def test_sequential_sampling_yields_sampled_frames() -> None:
    class FakeVideo:
        def __init__(self) -> None:
            self.index = 0

        def read(self):
            if self.index >= 5:
                return False, None
            frame = np.full((2, 2, 3), self.index, dtype=np.uint8)
            self.index += 1
            return True, frame

    batches = list(
        OpenCVVideo._yield_sampled_frames_sequentially(
            FakeVideo(),
            sampling_step=2,
            frames_batch_size=2,
        )
    )

    assert len(batches) == 2
    assert [int(frame[0, 0, 0]) for frame in batches[0]] == [0, 2]
    assert [int(frame[0, 0, 0]) for frame in batches[1]] == [4]


def test_get_next_frames_always_uses_sequential_sampling(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeVideo:
        def __init__(self) -> None:
            self.index = 0

        def isOpened(self) -> bool:
            return True

        def release(self) -> None:
            return None

        def get(self, property_id: int) -> float:
            if property_id == 5:
                return 30.0
            if property_id == 7:
                return 120.0
            return 0.0

        def read(self):
            if self.index >= 13:
                return False, None
            frame = np.full((2, 2, 3), self.index, dtype=np.uint8)
            self.index += 1
            return True, frame

    @contextmanager
    def fake_capture(_video_path):
        yield FakeVideo()

    monkeypatch.setattr(OpenCVVideo, "_video_capture", fake_capture)
    batches = list(OpenCVVideo.get_next_frames(Path("dummy.webm"), frames_batch_size=2))

    assert len(batches) == 2
    assert [int(frame[0, 0, 0]) for frame in batches[0]] == [0, 6]
    assert [int(frame[0, 0, 0]) for frame in batches[1]] == [12]
    assert OpenCVVideo.last_scan_metadata["video_frame_scan_mode"] == "sequential"
    assert OpenCVVideo.last_scan_metadata["video_fps_effective"] == 30
    assert OpenCVVideo.last_scan_metadata["video_frame_count_effective"] == 120
    assert OpenCVVideo.last_scan_metadata["video_sampling_step"] == 6


def test_sequential_sampling_records_scan_metadata_for_invalid_fps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeVideo:
        def __init__(self) -> None:
            self.index = 0

        def isOpened(self) -> bool:
            return True

        def release(self) -> None:
            return None

        def get(self, property_id: int) -> float:
            if property_id == 5:
                return 1000.0
            if property_id == 7:
                return -9.223372036854776e18
            return 0.0

        def read(self):
            if self.index >= 3:
                return False, None
            frame = np.full((2, 2, 3), self.index, dtype=np.uint8)
            self.index += 1
            return True, frame

    @contextmanager
    def fake_capture(_video_path):
        yield FakeVideo()

    monkeypatch.setattr(OpenCVVideo, "_video_capture", fake_capture)
    batches = list(OpenCVVideo.get_next_frames(Path("dummy.webm"), frames_batch_size=4))

    assert len(batches) == 1
    assert len(batches[0]) == 3
    assert OpenCVVideo.last_scan_metadata["video_frame_scan_mode"] == "sequential"
    assert OpenCVVideo.last_scan_metadata["video_fps_effective"] == 5
    assert OpenCVVideo.last_scan_metadata["video_sampling_step"] == 1
