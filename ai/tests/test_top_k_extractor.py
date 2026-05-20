from __future__ import annotations

from pathlib import Path

import numpy as np

from app.perfectframe.extractors import BestFrameExtractor
from app.perfectframe.image_evaluators import ImageEvaluator
from app.perfectframe.image_processors import ImageProcessor
from app.perfectframe.schemas import ExtractorConfig, ImageResolution
from app.perfectframe.video_processors import VideoProcessor


class _StaticVideoProcessor(VideoProcessor):
    """Yield a fixed sequence of frame batches regardless of video path."""

    batches: list[list[np.ndarray]] = []

    @classmethod
    def get_next_frames(cls, video_path: Path, frames_batch_size: int):
        for batch in cls.batches:
            yield batch


class _PassThroughImageProcessor(ImageProcessor):
    """Skip resizing — frames are already tagged by their pixel value."""

    @staticmethod
    def normalize_images(images, target_size):
        return np.array(images)

    @staticmethod
    def save_image(*args, **kwargs):  # pragma: no cover - unused in these tests
        raise NotImplementedError


class _ScoreByPixelEvaluator(ImageEvaluator):
    """Use the (0,0,0) pixel of each frame as its aesthetic score."""

    def evaluate_images(self, normalized_images):
        return [float(image[0, 0, 0]) for image in normalized_images]


def _make_frame(score_tag: int) -> np.ndarray:
    return np.full((4, 4, 3), score_tag, dtype=np.uint8)


def _make_extractor(batches: list[list[np.ndarray]]) -> BestFrameExtractor:
    _StaticVideoProcessor.batches = batches
    config = ExtractorConfig(
        input_directory=Path("."),
        output_directory=Path("."),
        batch_size=10,
        input_size=ImageResolution(4, 4),
    )
    return BestFrameExtractor(
        config,
        _PassThroughImageProcessor,
        _StaticVideoProcessor,
        _ScoreByPixelEvaluator(),
    )


def test_extract_top_k_frames_returns_segment_best_in_timeline_order() -> None:
    # 9 frames split into 3 equal segments of length 3 each:
    #   [1, 9, 4] -> best 9
    #   [7, 2, 8] -> best 8
    #   [5, 6, 3] -> best 6
    batches = [
        [_make_frame(s) for s in (1, 9, 4)],
        [_make_frame(s) for s in (7, 2, 8)],
        [_make_frame(s) for s in (5, 6, 3)],
    ]
    extractor = _make_extractor(batches)

    results = extractor.extract_top_k_frames(Path("dummy.mp4"), k=3)

    assert [score for _, score in results] == [9.0, 8.0, 6.0]
    for frame, score in results:
        assert int(frame[0, 0, 0]) == int(score)


def test_extract_top_k_frames_ignores_high_score_when_outside_segment_max() -> None:
    # 6 frames, k=3 -> segments of length 2:
    #   [1, 2] -> best 2
    #   [10, 3] -> best 10
    #   [4, 5] -> best 5
    # A late frame scoring 5 cannot displace a higher-scoring 10 in an earlier
    # segment — segments are independent — so 10 wins its segment even though
    # the very-late frame scoring 9 (in batch 3) below also exists.
    batches = [
        [_make_frame(s) for s in (1, 2)],
        [_make_frame(s) for s in (10, 3)],
        [_make_frame(s) for s in (4, 5)],
    ]
    extractor = _make_extractor(batches)

    results = extractor.extract_top_k_frames(Path("dummy.mp4"), k=3)

    assert [score for _, score in results] == [2.0, 10.0, 5.0]


def test_extract_top_k_frames_returns_all_when_fewer_than_k_in_timeline_order() -> None:
    batches = [[_make_frame(2), _make_frame(5)]]
    extractor = _make_extractor(batches)

    results = extractor.extract_top_k_frames(Path("dummy.mp4"), k=3)

    # n=2 <= k=3 — return everything in timeline order, not score order.
    assert [score for _, score in results] == [2.0, 5.0]


def test_extract_top_k_frames_returns_empty_when_k_is_zero() -> None:
    batches = [[_make_frame(1), _make_frame(2)]]
    extractor = _make_extractor(batches)

    assert extractor.extract_top_k_frames(Path("dummy.mp4"), k=0) == []


def test_extract_top_k_frames_returns_empty_when_video_has_no_frames() -> None:
    extractor = _make_extractor([])
    assert extractor.extract_top_k_frames(Path("dummy.mp4"), k=3) == []


def test_extract_top_k_frames_skips_empty_batches() -> None:
    # Empty batches are skipped so candidates = [3, 8, 6].
    # With k=2 the candidates split into segments of length 1 and 2:
    #   [3] -> 3
    #   [8, 6] -> 8
    batches = [
        [],
        [_make_frame(3), _make_frame(8)],
        [],
        [_make_frame(6)],
    ]
    extractor = _make_extractor(batches)

    results = extractor.extract_top_k_frames(Path("dummy.mp4"), k=2)

    assert [score for _, score in results] == [3.0, 8.0]


def test_extract_top_k_frames_distributes_results_across_timeline() -> None:
    # 15 frames, k=3 -> three equal segments [0:5], [5:10], [10:15].
    # Pick scores so each segment has a distinct max.
    scores_in_order = [1, 2, 3, 9, 4, 5, 6, 8, 5, 4, 3, 2, 7, 1, 1]
    batches = [[_make_frame(s) for s in scores_in_order]]
    extractor = _make_extractor(batches)

    results = extractor.extract_top_k_frames(Path("dummy.mp4"), k=3)

    # Segment maxima: max(1,2,3,9,4)=9, max(5,6,8,5,4)=8, max(3,2,7,1,1)=7.
    assert [score for _, score in results] == [9.0, 8.0, 7.0]


def test_extract_top_k_frames_records_profile_metrics() -> None:
    batches = [[_make_frame(s) for s in (1, 9, 4, 7, 2, 8)]]
    extractor = _make_extractor(batches)

    results = extractor.extract_top_k_frames(Path("dummy.mp4"), k=3)

    assert len(results) == 3
    assert extractor.last_profile["extract_profile_batches"] == 1
    assert extractor.last_profile["extract_profile_sampled_frames"] == 6
    assert "extract_profile_decode_wait_ms" in extractor.last_profile
    assert "extract_profile_normalize_ms" in extractor.last_profile
    assert "extract_profile_inference_ms" in extractor.last_profile
    assert "extract_profile_candidate_collect_ms" in extractor.last_profile
    assert "extract_profile_segment_select_ms" in extractor.last_profile
