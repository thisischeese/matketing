"""CSV and JSON helpers for the best-shot benchmark."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterable

from bestshot_benchmark.types import LabelEntry, SampledFrame, VideoEntry


def _require_columns(fieldnames: list[str] | None, required: set[str], source: Path) -> None:
    available = set(fieldnames or [])
    missing = sorted(required - available)
    if missing:
        raise ValueError(
            f"{source} is missing required columns: {', '.join(missing)}"
        )


def read_videos_csv(path: Path) -> list[VideoEntry]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        _require_columns(reader.fieldnames, {"video_id", "video_path"}, path)
        videos: list[VideoEntry] = []
        for row in reader:
            video_id = (row["video_id"] or "").strip()
            raw_path = (row["video_path"] or "").strip()
            if not video_id or not raw_path:
                raise ValueError(f"{path} contains an empty video_id or video_path.")
            video_path = Path(raw_path).expanduser().resolve()
            videos.append(VideoEntry(video_id=video_id, video_path=video_path))
    if not videos:
        raise ValueError(f"{path} does not contain any videos.")
    return videos


def write_frames_csv(path: Path, frames: Iterable[SampledFrame]) -> None:
    rows = list(frames)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "video_id",
                "frame_id",
                "video_path",
                "frame_path",
                "frame_index",
                "sample_index",
                "timestamp_ms",
                "width",
                "height",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "video_id": row.video_id,
                    "frame_id": row.frame_id,
                    "video_path": str(row.video_path),
                    "frame_path": str(row.frame_path),
                    "frame_index": row.frame_index,
                    "sample_index": row.sample_index,
                    "timestamp_ms": row.timestamp_ms,
                    "width": row.width,
                    "height": row.height,
                }
            )


def read_frames_csv(path: Path) -> list[SampledFrame]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        _require_columns(
            reader.fieldnames,
            {
                "video_id",
                "frame_id",
                "video_path",
                "frame_path",
                "frame_index",
                "sample_index",
                "timestamp_ms",
                "width",
                "height",
            },
            path,
        )
        frames: list[SampledFrame] = []
        for row in reader:
            frames.append(
                SampledFrame(
                    video_id=(row["video_id"] or "").strip(),
                    frame_id=(row["frame_id"] or "").strip(),
                    video_path=Path(row["video_path"]).expanduser().resolve(),
                    frame_path=Path(row["frame_path"]).expanduser().resolve(),
                    frame_index=int(row["frame_index"]),
                    sample_index=int(row["sample_index"]),
                    timestamp_ms=int(row["timestamp_ms"]),
                    width=int(row["width"]),
                    height=int(row["height"]),
                )
            )
    if not frames:
        raise ValueError(f"{path} does not contain any sampled frames.")
    return frames


def read_labels_csv(path: Path) -> list[LabelEntry]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        _require_columns(reader.fieldnames, {"video_id", "frame_id"}, path)
        labels: list[LabelEntry] = []
        for row in reader:
            video_id = (row["video_id"] or "").strip()
            frame_id = (row["frame_id"] or "").strip()
            if not video_id or not frame_id:
                raise ValueError(f"{path} contains an empty video_id or frame_id.")
            labels.append(LabelEntry(video_id=video_id, frame_id=frame_id))
    if not labels:
        raise ValueError(f"{path} does not contain any labels.")
    return labels


def write_csv(path: Path, fieldnames: list[str], rows: Iterable[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

