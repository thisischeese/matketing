"""Video processor implementations used by the frame extractor."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Generator
from contextlib import contextmanager
import math
from pathlib import Path

from app.perfectframe.schemas import Images

TARGET_SAMPLE_FPS = 5
DEFAULT_FALLBACK_FPS = 5
MAX_REASONABLE_FPS = 240


def _import_cv2():
    try:
        import cv2  # type: ignore[import-not-found]
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "opencv-python-headless is required for frame extraction."
        ) from exc
    return cv2


class VideoProcessor(ABC):
    """Abstract video processor."""

    @classmethod
    @abstractmethod
    def get_next_frames(
        cls, video_path: Path, frames_batch_size: int
    ) -> Generator[Images, None, None]:
        """Yield batches of frames from a video."""


class OpenCVVideo(VideoProcessor):
    """OpenCV-based frame sampler."""

    class Error(Exception):
        """Raised when a video cannot be processed."""

    last_scan_metadata: dict[str, int | float | str | None] = {}

    @staticmethod
    @contextmanager
    def _video_capture(video_path: Path):
        cv2 = _import_cv2()
        video_cap = cv2.VideoCapture(str(video_path))
        try:
            if not video_cap.isOpened():
                raise OpenCVVideo.Error(f"Can't open video file: {video_path}")
            yield video_cap
        finally:
            video_cap.release()

    @classmethod
    def get_next_frames(
        cls, video_path: Path, frames_batch_size: int
    ) -> Generator[Images, None, None]:
        cv2 = _import_cv2()
        with cls._video_capture(video_path) as video:
            fps_raw = cls._get_video_property(video, cv2.CAP_PROP_FPS)
            fps_effective = cls._normalize_frame_rate(fps_raw)
            frame_count_raw = cls._get_video_property(video, cv2.CAP_PROP_FRAME_COUNT)
            frame_count = cls._normalize_frame_count(frame_count_raw)
            sampling_step = cls._get_sampling_step(fps_effective)

            cls.last_scan_metadata = {
                "video_fps_raw": round(fps_raw, 3) if fps_raw is not None else None,
                "video_fps_effective": fps_effective,
                "video_frame_count_raw": round(frame_count_raw)
                if frame_count_raw is not None and math.isfinite(frame_count_raw)
                else None,
                "video_frame_count_effective": frame_count,
                "video_sampling_step": sampling_step,
                "video_frame_scan_mode": "sequential",
            }

            yield from cls._yield_sampled_frames_sequentially(
                video,
                sampling_step,
                frames_batch_size,
            )

    @classmethod
    def _yield_sampled_frames_sequentially(
        cls,
        video,
        sampling_step: int,
        frames_batch_size: int,
    ) -> Generator[Images, None, None]:
        frames_batch: Images = []
        frame_index = 0

        while True:
            success, frame = video.read()
            if not success:
                break
            if frame_index % sampling_step == 0:
                frames_batch.append(frame)
                if len(frames_batch) == frames_batch_size:
                    yield frames_batch
                    frames_batch = []
            frame_index += 1

        if frames_batch:
            yield frames_batch

    @staticmethod
    def _get_sampling_step(frame_rate: int) -> int:
        # Cap sampling at 5 frames per second while keeping lower-FPS videos intact.
        return max(1, round(frame_rate / TARGET_SAMPLE_FPS))

    @staticmethod
    def _get_video_property(video, property_id: int) -> float | None:
        property_value = video.get(property_id)
        if not math.isfinite(property_value):
            return None
        return property_value

    @staticmethod
    def _normalize_frame_rate(frame_rate: float | None) -> int:
        if frame_rate is None or frame_rate <= 0 or frame_rate > MAX_REASONABLE_FPS:
            return DEFAULT_FALLBACK_FPS
        return round(frame_rate)

    @staticmethod
    def _normalize_frame_count(frame_count: float | None) -> int | None:
        if frame_count is None or frame_count <= 0:
            return None
        if frame_count > 10_000_000:
            return None
        return round(frame_count)
