"""Provide a minimal BestFrameExtractor compatible with PerfectFrameAI."""

from __future__ import annotations

from pathlib import Path
import time

import numpy as np

from app.perfectframe.image_evaluators import ImageEvaluator
from app.perfectframe.image_processors import ImageProcessor
from app.perfectframe.schemas import ExtractorConfig, Image, Images, ScoresArray
from app.perfectframe.video_processors import VideoProcessor


class BestFrameExtractor:
    """Extractor that selects the single highest-scoring frame from a video."""

    def __init__(
        self,
        config: ExtractorConfig,
        image_processor: type[ImageProcessor],
        video_processor: type[VideoProcessor],
        image_evaluator: ImageEvaluator,
    ) -> None:
        self._config = config
        self._image_processor = image_processor
        self._video_processor = video_processor
        self._image_evaluator = image_evaluator
        self.last_profile: dict[str, int | float] = {}

    def extract_top_k_frames(
        self,
        video_path: Path,
        k: int,
    ) -> list[tuple[Image, float]]:
        """Return up to k frames spread across the video timeline.

        The video is split into k equal-length segments; the highest-scoring
        frame within each segment is selected. Results are returned in
        timeline order (segment 0 first), not score order.
        """

        if k <= 0:
            self.last_profile = {}
            return []

        candidates: list[tuple[float, Image]] = []
        frames_batch_generator = iter(self._video_processor.get_next_frames(
            video_path,
            self._config.batch_size,
        ))
        decode_wait_ms = 0.0
        normalize_ms = 0.0
        inference_ms = 0.0
        candidate_collect_ms = 0.0
        batch_count = 0
        sampled_frame_count = 0

        while True:
            batch_wait_started_at = time.perf_counter()
            try:
                frames = next(frames_batch_generator)
            except StopIteration:
                decode_wait_ms += (time.perf_counter() - batch_wait_started_at) * 1000
                break
            decode_wait_ms += (time.perf_counter() - batch_wait_started_at) * 1000
            if not frames:
                continue
            batch_count += 1
            sampled_frame_count += len(frames)

            normalize_started_at = time.perf_counter()
            normalized_images = self._image_processor.normalize_images(
                frames,
                self._config.input_size,
            )
            normalize_ms += (time.perf_counter() - normalize_started_at) * 1000

            inference_started_at = time.perf_counter()
            scores = self._image_evaluator.evaluate_images(normalized_images)
            inference_ms += (time.perf_counter() - inference_started_at) * 1000

            collect_started_at = time.perf_counter()
            for frame, score in zip(frames, scores):
                candidates.append((float(score), frame))
            candidate_collect_ms += (time.perf_counter() - collect_started_at) * 1000

        if not candidates:
            self.last_profile = {
                "extract_profile_decode_wait_ms": int(decode_wait_ms),
                "extract_profile_normalize_ms": int(normalize_ms),
                "extract_profile_inference_ms": int(inference_ms),
                "extract_profile_candidate_collect_ms": int(candidate_collect_ms),
                "extract_profile_segment_select_ms": 0,
                "extract_profile_batches": batch_count,
                "extract_profile_sampled_frames": sampled_frame_count,
            }
            return []
        if len(candidates) <= k:
            self.last_profile = {
                "extract_profile_decode_wait_ms": int(decode_wait_ms),
                "extract_profile_normalize_ms": int(normalize_ms),
                "extract_profile_inference_ms": int(inference_ms),
                "extract_profile_candidate_collect_ms": int(candidate_collect_ms),
                "extract_profile_segment_select_ms": 0,
                "extract_profile_batches": batch_count,
                "extract_profile_sampled_frames": sampled_frame_count,
            }
            return [(frame, score) for score, frame in candidates]

        n = len(candidates)
        selected: list[tuple[Image, float]] = []
        segment_select_started_at = time.perf_counter()
        for index in range(k):
            lo = (index * n) // k
            hi = ((index + 1) * n) // k
            segment = candidates[lo:hi]
            if not segment:
                continue
            best_score, best_frame = max(segment, key=lambda item: item[0])
            selected.append((best_frame, best_score))
        segment_select_ms = (time.perf_counter() - segment_select_started_at) * 1000
        self.last_profile = {
            "extract_profile_decode_wait_ms": int(decode_wait_ms),
            "extract_profile_normalize_ms": int(normalize_ms),
            "extract_profile_inference_ms": int(inference_ms),
            "extract_profile_candidate_collect_ms": int(candidate_collect_ms),
            "extract_profile_segment_select_ms": int(segment_select_ms),
            "extract_profile_batches": batch_count,
            "extract_profile_sampled_frames": sampled_frame_count,
        }
        return selected

    def extract_best_frame(self, video_path: Path) -> tuple[Image, float] | None:
        best_frame: Image | None = None
        best_score: float | None = None

        frames_batch_generator = self._video_processor.get_next_frames(
            video_path,
            self._config.batch_size,
        )
        for frames in frames_batch_generator:
            if not frames:
                continue
            frame, score = self._get_best_frame(frames)
            if frame is None or score is None:
                continue
            if best_score is None or score > best_score:
                best_frame = frame
                best_score = score

        if best_frame is None or best_score is None:
            return None
        return best_frame, best_score

    def _get_best_frame(self, frames: Images) -> tuple[Image | None, float | None]:
        normalized_images = self._image_processor.normalize_images(
            frames,
            self._config.input_size,
        )
        scores = self._evaluate_images(normalized_images)
        if len(scores) == 0:
            return None, None

        best_index = int(np.argmax(scores))
        return frames[best_index], float(scores[best_index])

    def _evaluate_images(self, normalized_images) -> ScoresArray:
        return np.array(self._image_evaluator.evaluate_images(normalized_images))


BestFramesExtractor = BestFrameExtractor
