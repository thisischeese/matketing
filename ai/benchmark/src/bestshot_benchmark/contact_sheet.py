"""Contact sheet rendering for manual frame labeling."""

from __future__ import annotations

import math
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from bestshot_benchmark.types import SampledFrame

BACKGROUND = (248, 248, 248)
TEXT_COLOR = (20, 20, 20)
BORDER_COLOR = (220, 220, 220)


def _group_frames(frames: list[SampledFrame]) -> dict[str, list[SampledFrame]]:
    grouped: dict[str, list[SampledFrame]] = {}
    for frame in sorted(frames, key=lambda item: (item.video_id, item.sample_index)):
        grouped.setdefault(frame.video_id, []).append(frame)
    return grouped


def _ordered_frames(
    frames: list[SampledFrame],
    shuffle_seed: int | None,
    video_id: str,
) -> list[SampledFrame]:
    ordered = list(frames)
    if shuffle_seed is None:
        return ordered
    rng = random.Random(f"{shuffle_seed}:{video_id}")
    rng.shuffle(ordered)
    return ordered


def render_contact_sheets(
    frames: list[SampledFrame],
    output_dir: Path,
    columns: int = 4,
    thumb_width: int = 240,
    thumb_height: int = 180,
    shuffle_seed: int | None = None,
) -> list[Path]:
    """Render one contact sheet image per video."""

    output_dir.mkdir(parents=True, exist_ok=True)
    font = ImageFont.load_default()
    cell_padding = 12
    label_height = 34
    gutter = 12

    generated: list[Path] = []
    for video_id, video_frames in _group_frames(frames).items():
        ordered = _ordered_frames(video_frames, shuffle_seed=shuffle_seed, video_id=video_id)
        rows = math.ceil(len(ordered) / columns)
        cell_width = thumb_width + (cell_padding * 2)
        cell_height = thumb_height + label_height + (cell_padding * 2)
        canvas_width = (columns * cell_width) + ((columns + 1) * gutter)
        canvas_height = (rows * cell_height) + ((rows + 1) * gutter)

        sheet = Image.new("RGB", (canvas_width, canvas_height), BACKGROUND)
        draw = ImageDraw.Draw(sheet)

        for index, frame in enumerate(ordered):
            row = index // columns
            col = index % columns
            origin_x = gutter + (col * cell_width) + (col * gutter)
            origin_y = gutter + (row * cell_height) + (row * gutter)

            with Image.open(frame.frame_path) as image:
                thumb = image.convert("RGB")
                thumb.thumbnail((thumb_width, thumb_height))
                frame_box_x = origin_x + cell_padding + ((thumb_width - thumb.width) // 2)
                frame_box_y = origin_y + cell_padding + ((thumb_height - thumb.height) // 2)
                sheet.paste(thumb, (frame_box_x, frame_box_y))

            draw.rectangle(
                [
                    origin_x,
                    origin_y,
                    origin_x + cell_width,
                    origin_y + cell_height,
                ],
                outline=BORDER_COLOR,
                width=1,
            )
            draw.text(
                (origin_x + cell_padding, origin_y + thumb_height + cell_padding + 6),
                f"{frame.frame_id}  t={frame.timestamp_ms}ms",
                fill=TEXT_COLOR,
                font=font,
            )

        output_path = output_dir / f"{video_id}.png"
        sheet.save(output_path)
        generated.append(output_path.resolve())

    return generated

