"""Frame sampling aligned with the current service policy."""

from __future__ import annotations

import math
import re
from pathlib import Path

from bestshot_benchmark.types import SampledFrame, VideoEntry

TARGET_SAMPLE_FPS = 5
DEFAULT_FALLBACK_FPS = 5
MAX_REASONABLE_FPS = 240


def _import_cv2():
    try:
        import cv2  # type: ignore[import-not-found]
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "opencv-python-headless is required for frame sampling."
        ) from exc
    return cv2


def safe_path_component(value: str) -> str:
    """Return a filesystem-safe directory name."""

    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._")
    return sanitized or "video"


def normalize_frame_rate(frame_rate: float | None) -> int:
    """Mirror the service fallback logic for odd fps values."""

    if frame_rate is None or frame_rate <= 0 or frame_rate > MAX_REASONABLE_FPS:
        return DEFAULT_FALLBACK_FPS
    return round(frame_rate)


def get_sampling_step(frame_rate: int) -> int:
    """Cap sampling at 5fps while keeping slower videos intact."""

    return max(1, round(frame_rate / TARGET_SAMPLE_FPS))


def sample_video_frames(video: VideoEntry, frames_root: Path) -> list[SampledFrame]:
    """Sample frames from one video and persist them to disk."""

    cv2 = _import_cv2()
    frames_root.mkdir(parents=True, exist_ok=True)
    video_dir = frames_root / safe_path_component(video.video_id)
    video_dir.mkdir(parents=True, exist_ok=True)

    capture = cv2.VideoCapture(str(video.video_path))
    try:
        if not capture.isOpened():
            raise RuntimeError(f"Unable to open video: {video.video_path}")

        fps_raw = capture.get(cv2.CAP_PROP_FPS)
        fps_value = fps_raw if math.isfinite(fps_raw) else None
        fps_effective = normalize_frame_rate(fps_value)
        sampling_step = get_sampling_step(fps_effective)

        frame_index = 0
        sample_index = 0
        sampled_frames: list[SampledFrame] = []

        while True:
            success, frame = capture.read()
            if not success:
                break
            if frame_index % sampling_step == 0:
                sample_index += 1
                frame_id = f"frame-{sample_index:05d}"
                frame_path = video_dir / f"{frame_id}.png"
                if not cv2.imwrite(str(frame_path), frame):
                    raise RuntimeError(f"Failed to write sampled frame: {frame_path}")
                height, width = frame.shape[:2]
                sampled_frames.append(
                    SampledFrame(
                        video_id=video.video_id,
                        frame_id=frame_id,
                        video_path=video.video_path,
                        frame_path=frame_path.resolve(),
                        frame_index=frame_index,
                        sample_index=sample_index,
                        timestamp_ms=int(round((frame_index / fps_effective) * 1000)),
                        width=width,
                        height=height,
                    )
                )
            frame_index += 1
    finally:
        capture.release()

    if not sampled_frames:
        raise RuntimeError(f"No sampled frames were produced for {video.video_id}.")
    return sampled_frames


def sample_videos(videos: list[VideoEntry], workspace: Path) -> list[SampledFrame]:
    """Sample frames for every video in the manifest."""

    frames_root = workspace / "frames"
    sampled_frames: list[SampledFrame] = []
    for video in videos:
        if not video.video_path.exists():
            raise FileNotFoundError(f"Video file does not exist: {video.video_path}")
        sampled_frames.extend(sample_video_frames(video, frames_root))
    return sampled_frames

