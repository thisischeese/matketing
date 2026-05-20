from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, cast

import numpy as np
from PIL import Image

from app.core.config import settings
from app.services.final_edit_filter_catalog import (
    FinalEditFilterCatalogError,
    resolve_final_edit_filter_definition,
    resolve_final_edit_filter_name,
)
from app.services.final_edit_runtime import import_cv2
from app.services.final_edit_upscaler import get_realesrgan_upscaler

ToolName = Literal[
    "crop",
    "upscale",
    "denoise",
    "filter",
    "color_grading",
    "sharpen",
    "background_blur",
]
CropEdge = Literal["top", "bottom", "left", "right"]
CropOrientation = Literal["portrait", "square", "landscape"]

AESTHETIC_TARGET_TONE_PROFILE: dict[str, dict[str, float]] = {
    "brightness": {"target": 82.2, "tolerance": 35.89},
    "contrast": {"target": 53.12, "tolerance": 15.08},
    "black_point": {"target": 13.17, "tolerance": 15.56},
    "white_point": {"target": 176.67, "tolerance": 46.46},
    "temperature": {"target": 11.33, "tolerance": 9.05},
    "tint": {"target": 4.35, "tolerance": 4.74},
    "saturation": {"target": 108.55, "tolerance": 26.86},
}
AESTHETIC_TONE_METRIC_KEYS: tuple[str, ...] = (
    "brightness",
    "contrast",
    "black_point",
    "white_point",
    "temperature",
    "tint",
    "saturation",
)
_TRIMMED_BRIGHTNESS_LOWER_PERCENTILE = 10.0
_TRIMMED_BRIGHTNESS_UPPER_PERCENTILE = 90.0
_HIGHLIGHT_PIXEL_THRESHOLD = 235.0
_HIGHLIGHT_RATIO_LOW = 0.03
_HIGHLIGHT_RATIO_MEDIUM = 0.08
_SPOTLIGHT_BROAD_HIGHLIGHT_START = 0.62
_SPOTLIGHT_BROAD_HIGHLIGHT_END = 0.90
_SPOTLIGHT_SPECULAR_START = 0.82
_SPOTLIGHT_SPECULAR_END = 0.98
_PORTRAIT_TARGET_RATIO = 3.0 / 4.0
_SQUARE_TARGET_RATIO = 1.0
_CROP_MARGIN_RATIO = 0.08
_MIN_CROP_AREA_GAIN = 0.03
_MIN_CROP_GAIN_MULTIPLIER = 1.15
_MAX_CROP_AREA_LOSS_RATIO = 0.7
_CROP_VALID_EDGES: tuple[CropEdge, ...] = ("top", "bottom", "left", "right")


@dataclass(frozen=True)
class FinalEditToolSpec:
    name: ToolName
    description: str
    params_schema: dict[str, Any]


SUPPORTED_TOOL_SPECS: tuple[FinalEditToolSpec, ...] = (
    FinalEditToolSpec(
        name="crop",
        description="주 피사체를 보존하며 인스타 구도에 맞게 보수적으로 자른다.",
        params_schema={
            "subject_boxes": "[[x1,y1,x2,y2], ...] 1~3 boxes",
            "keep_edges": "top|bottom|left|right list",
        },
    ),
    FinalEditToolSpec(
        name="upscale",
        description="이미지를 2배 업스케일한다.",
        params_schema={"scale": "2 only"},
    ),
    FinalEditToolSpec(
        name="denoise",
        description="이미지 노이즈를 줄인다.",
        params_schema={"strength": "0.0~1.0"},
    ),
    FinalEditToolSpec(
        name="filter",
        description="Records the VLM-selected filter name for downstream LangGraph execution.",
        params_schema={"filter_name": "catalog filter name"},
    ),
    FinalEditToolSpec(
        name="color_grading",
        description="목표 톤에 맞춰 색온도, 채도, 명암을 미세 조정한다.",
        params_schema={
            "contrast": "-100~100",
            "highlights": "-100~100",
            "shadows": "-100~100",
            "vibrance": "-100~100",
            "saturation": "-100~100",
            "temperature": "warm or cool",
            "temperature_strength": "0.0~1.0 (internal)",
            "tint": "-100~100 (internal)",
            "tint_strength": "0.0~1.0 (internal)",
            "tone_curve_shadow_lift": "-100~100",
            "brightness": "-1.0~1.0 (legacy)",
        },
    ),
    FinalEditToolSpec(
        name="sharpen",
        description="디테일을 선명하게 한다.",
        params_schema={"strength": "0.0~1.0"},
    ),
    FinalEditToolSpec(
        name="background_blur",
        description="배경을 부드럽게 흐린다.",
        params_schema={"blur_radius": "1~20"},
    ),
)


def _resolve_upscale_enabled(upscale_enabled: bool | None) -> bool:
    if upscale_enabled is None:
        return settings.FINAL_EDIT_ENABLE_UPSCALE
    return upscale_enabled


def supported_tool_specs(
    upscale_enabled: bool | None = None,
) -> tuple[FinalEditToolSpec, ...]:
    if _resolve_upscale_enabled(upscale_enabled):
        return SUPPORTED_TOOL_SPECS
    return tuple(spec for spec in SUPPORTED_TOOL_SPECS if spec.name != "upscale")


def supported_tool_names(upscale_enabled: bool | None = None) -> tuple[ToolName, ...]:
    return tuple(spec.name for spec in supported_tool_specs(upscale_enabled))


def _clamp_float(value: Any, default: float, minimum: float, maximum: float) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return default
    return max(minimum, min(maximum, numeric))


def _normalize_temperature(value: Any) -> str:
    normalized = str(value).strip().lower()
    if normalized in {"warm", "cool"}:
        return normalized
    return "warm"


def _clamp_percent(value: Any, default: float = 0.0) -> float:
    return _clamp_float(value, default, -100.0, 100.0)


def _is_legacy_color_grading_params(params: dict[str, Any]) -> bool:
    known_legacy_keys = {"temperature", "saturation", "brightness"}
    known_new_keys = {
        "contrast",
        "highlights",
        "shadows",
        "vibrance",
        "tone_curve_shadow_lift",
        "temperature_strength",
        "tint",
        "tint_strength",
    }
    if any(key in params for key in known_new_keys):
        return False
    if not set(params).issubset(known_legacy_keys):
        return False
    try:
        saturation = float(params.get("saturation", 0.0))
    except (TypeError, ValueError):
        return True
    return abs(saturation) <= 1.0 or "brightness" in params


def _empty_color_grading_params() -> dict[str, Any]:
    return {
        "contrast": 0.0,
        "highlights": 0.0,
        "shadows": 0.0,
        "vibrance": 0.0,
        "temperature": "warm",
        "temperature_strength": 0.0,
        "tint": 0.0,
        "tint_strength": 0.0,
        "saturation": 0.0,
        "tone_curve_shadow_lift": 0.0,
        "brightness": 0.0,
    }


def _is_gap_within_skip_threshold(gap: float, tolerance: float) -> bool:
    return abs(gap) <= tolerance * 0.5


def _scale_gap_to_percent(gap: float, tolerance: float, multiplier: float) -> float:
    safe_tolerance = max(tolerance, 1.0)
    return _clamp_percent((gap / safe_tolerance) * multiplier)


def _scale_gap_to_unit_interval(
    gap: float,
    tolerance: float,
    multiplier: float,
) -> float:
    safe_tolerance = max(tolerance, 1.0)
    return _clamp_float((gap / safe_tolerance) * multiplier, 0.0, -1.0, 1.0)


def _trimmed_mean(
    values: np.ndarray,
    lower_percentile: float,
    upper_percentile: float,
) -> float:
    flattened = values.reshape(-1).astype(np.float64)
    lower_bound = float(np.percentile(flattened, lower_percentile))
    upper_bound = float(np.percentile(flattened, upper_percentile))
    trimmed = flattened[(flattened >= lower_bound) & (flattened <= upper_bound)]
    if trimmed.size == 0:
        return float(np.mean(flattened))
    return float(np.mean(trimmed))


def _attenuate_negative_highlights(
    highlight_adjustment: float,
    highlight_area_ratio: float,
) -> float:
    if highlight_adjustment >= 0.0:
        return highlight_adjustment
    if highlight_area_ratio < _HIGHLIGHT_RATIO_LOW:
        return 0.0
    if highlight_area_ratio < _HIGHLIGHT_RATIO_MEDIUM:
        return round(highlight_adjustment * 0.35, 2)
    return highlight_adjustment


def _attenuate_combined_global_darkening(
    brightness_adjustment: float,
    highlight_adjustment: float,
    highlight_area_ratio: float,
) -> float:
    if (
        brightness_adjustment < 0.0
        and highlight_adjustment < 0.0
        and highlight_area_ratio < _HIGHLIGHT_RATIO_MEDIUM
    ):
        return round(brightness_adjustment * 0.7, 4)
    return brightness_adjustment


def _smoothstep(edge0: float, edge1: float, values: np.ndarray) -> np.ndarray:
    if edge1 <= edge0:
        return np.where(values >= edge1, 1.0, 0.0).astype(np.float32)
    scaled = np.clip((values - edge0) / (edge1 - edge0), 0.0, 1.0)
    return (scaled * scaled * (3.0 - 2.0 * scaled)).astype(np.float32)


def normalize_tool_params(tool_name: ToolName, params: dict[str, Any]) -> dict[str, Any]:
    if tool_name == "crop":
        normalized_boxes: list[list[int]] = []
        subject_boxes = params.get("subject_boxes", [])
        if isinstance(subject_boxes, list):
            for raw_box in subject_boxes[:3]:
                if not isinstance(raw_box, (list, tuple)) or len(raw_box) != 4:
                    continue
                try:
                    x1, y1, x2, y2 = [int(round(float(value))) for value in raw_box]
                except (TypeError, ValueError):
                    continue
                left = min(x1, x2)
                top = min(y1, y2)
                right = max(x1, x2)
                bottom = max(y1, y2)
                if right - left <= 1 or bottom - top <= 1:
                    continue
                normalized_boxes.append([left, top, right, bottom])

        normalized_edges: list[CropEdge] = []
        keep_edges = params.get("keep_edges", [])
        if isinstance(keep_edges, list):
            for raw_edge in keep_edges:
                edge = str(raw_edge).strip().lower()
                if edge in _CROP_VALID_EDGES and edge not in normalized_edges:
                    normalized_edges.append(cast(CropEdge, edge))

        orientation = str(params.get("anchor_orientation", "")).strip().lower()
        anchor_orientation: CropOrientation | None = None
        if orientation in {"portrait", "square", "landscape"}:
            anchor_orientation = cast(CropOrientation, orientation)

        return {
            "subject_boxes": normalized_boxes,
            "keep_edges": normalized_edges,
            "anchor_orientation": anchor_orientation,
        }
    if tool_name == "upscale":
        return {"scale": 2}
    if tool_name == "denoise":
        return {"strength": _clamp_float(params.get("strength"), 0.5, 0.0, 1.0)}
    if tool_name == "filter":
        try:
            filter_name = resolve_final_edit_filter_name(params.get("filter_name", ""))
        except FinalEditFilterCatalogError as exc:
            raise ValueError(str(exc)) from exc
        return {"filter_name": filter_name}
    if tool_name == "color_grading":
        legacy_mode = _is_legacy_color_grading_params(params)
        normalized_saturation = (
            _clamp_float(params.get("saturation"), 0.0, -1.0, 1.0) * 100.0
            if legacy_mode
            else _clamp_percent(params.get("saturation"), 0.0)
        )
        normalized = {
            "contrast": _clamp_percent(params.get("contrast"), 0.0),
            "highlights": _clamp_percent(params.get("highlights"), 0.0),
            "shadows": _clamp_percent(params.get("shadows"), 0.0),
            "vibrance": _clamp_percent(params.get("vibrance"), 0.0),
            "temperature": _normalize_temperature(params.get("temperature", "warm")),
            "temperature_strength": _clamp_float(
                params.get("temperature_strength"),
                1.0 if "temperature" in params else 0.0,
                0.0,
                1.0,
            ),
            "tint": _clamp_percent(params.get("tint"), 0.0),
            "tint_strength": _clamp_float(
                params.get("tint_strength"),
                1.0 if "tint" in params else 0.0,
                0.0,
                1.0,
            ),
            "saturation": normalized_saturation,
            "tone_curve_shadow_lift": _clamp_percent(
                params.get("tone_curve_shadow_lift"),
                0.0,
            ),
            "brightness": _clamp_float(params.get("brightness"), 0.0, -1.0, 1.0),
        }
        if "spotlight_protection" in params:
            normalized["spotlight_protection"] = bool(params["spotlight_protection"])
        return normalized
    if tool_name == "sharpen":
        return {"strength": _clamp_float(params.get("strength"), 0.5, 0.0, 1.0)}
    if tool_name == "background_blur":
        blur_radius = params.get("blur_radius", 5)
        try:
            blur_radius_int = int(blur_radius)
        except (TypeError, ValueError):
            blur_radius_int = 5
        return {"blur_radius": max(1, min(20, blur_radius_int))}
    raise ValueError(f"Unsupported tool: {tool_name}")


def _classify_crop_orientation(
    width: int,
    height: int,
    anchor_orientation: CropOrientation | None = None,
) -> CropOrientation:
    if anchor_orientation is not None:
        return anchor_orientation
    if width == height:
        return "square"
    if height > width:
        return "portrait"
    return "landscape"


def _target_ratio_for_orientation(
    orientation: CropOrientation,
    width: int,
    height: int,
) -> float:
    if orientation == "portrait":
        return _PORTRAIT_TARGET_RATIO
    if orientation == "square":
        return _SQUARE_TARGET_RATIO
    return width / max(height, 1)


def _clamp_box_to_image(
    box: tuple[int, int, int, int],
    width: int,
    height: int,
) -> tuple[int, int, int, int] | None:
    x1, y1, x2, y2 = box
    x1 = max(0, min(width - 1, x1))
    y1 = max(0, min(height - 1, y1))
    x2 = max(1, min(width, x2))
    y2 = max(1, min(height, y2))
    if x2 - x1 <= 1 or y2 - y1 <= 1:
        return None
    return (x1, y1, x2, y2)


def _union_subject_boxes(
    subject_boxes: list[list[int]],
    width: int,
    height: int,
) -> tuple[int, int, int, int] | None:
    clamped_boxes: list[tuple[int, int, int, int]] = []
    for raw_box in subject_boxes:
        clamped = _clamp_box_to_image(
            (raw_box[0], raw_box[1], raw_box[2], raw_box[3]),
            width,
            height,
        )
        if clamped is not None:
            clamped_boxes.append(clamped)
    if not clamped_boxes:
        return None
    return (
        min(box[0] for box in clamped_boxes),
        min(box[1] for box in clamped_boxes),
        max(box[2] for box in clamped_boxes),
        max(box[3] for box in clamped_boxes),
    )


def _expand_subject_box(
    box: tuple[int, int, int, int],
    width: int,
    height: int,
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    margin = int(round(max(x2 - x1, y2 - y1) * _CROP_MARGIN_RATIO))
    if margin <= 0:
        return box
    return (
        max(0, x1 - margin),
        max(0, y1 - margin),
        min(width, x2 + margin),
        min(height, y2 + margin),
    )


def _adjust_crop_position(
    start: int,
    size: int,
    limit: int,
    box_start: int,
    box_end: int,
    pin_start: bool,
    pin_end: bool,
) -> int:
    if size >= limit:
        return 0
    if pin_start:
        return 0
    if pin_end:
        return limit - size
    adjusted = start
    if adjusted > box_start:
        adjusted = box_start
    if adjusted + size < box_end:
        adjusted = box_end - size
    return max(0, min(limit - size, adjusted))


def _build_crop_window(
    union_box: tuple[int, int, int, int],
    width: int,
    height: int,
    target_ratio: float,
    keep_edges: list[CropEdge],
) -> tuple[int, int, int, int] | None:
    x1, y1, x2, y2 = union_box
    box_width = x2 - x1
    box_height = y2 - y1
    if box_width <= 1 or box_height <= 1:
        return None

    crop_width = box_width
    crop_height = int(round(crop_width / target_ratio))
    if crop_height < box_height:
        crop_height = box_height
        crop_width = int(round(crop_height * target_ratio))
    crop_width = max(crop_width, box_width)
    crop_height = max(crop_height, box_height)

    if crop_width > width or crop_height > height:
        return None

    center_x = (x1 + x2) / 2.0
    center_y = (y1 + y2) / 2.0
    start_x = int(round(center_x - crop_width / 2.0))
    start_y = int(round(center_y - crop_height / 2.0))
    start_x = _adjust_crop_position(
        start_x,
        crop_width,
        width,
        x1,
        x2,
        "left" in keep_edges,
        "right" in keep_edges,
    )
    start_y = _adjust_crop_position(
        start_y,
        crop_height,
        height,
        y1,
        y2,
        "top" in keep_edges,
        "bottom" in keep_edges,
    )
    end_x = start_x + crop_width
    end_y = start_y + crop_height
    if start_x > x1 or start_y > y1 or end_x < x2 or end_y < y2:
        return None
    return (start_x, start_y, end_x, end_y)


def _crop_quality_gate(
    union_box: tuple[int, int, int, int],
    crop_box: tuple[int, int, int, int],
    image_width: int,
    image_height: int,
) -> tuple[bool, str]:
    subject_area = float((union_box[2] - union_box[0]) * (union_box[3] - union_box[1]))
    image_area = float(image_width * image_height)
    crop_area = float((crop_box[2] - crop_box[0]) * (crop_box[3] - crop_box[1]))
    if subject_area <= 0.0 or crop_area <= 0.0 or image_area <= 0.0:
        return False, "invalid_geometry"

    current_ratio = subject_area / image_area
    cropped_ratio = subject_area / crop_area
    if cropped_ratio - current_ratio < _MIN_CROP_AREA_GAIN:
        return False, "gain_too_small"
    if cropped_ratio < current_ratio * _MIN_CROP_GAIN_MULTIPLIER:
        return False, "gain_ratio_too_small"

    area_loss_ratio = 1.0 - (crop_area / image_area)
    if area_loss_ratio > _MAX_CROP_AREA_LOSS_RATIO:
        return False, "crop_too_aggressive"
    return True, "applied"


def _tool_crop(
    image: np.ndarray,
    params: dict[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    height, width = image.shape[:2]
    subject_boxes = cast(list[list[int]], params.get("subject_boxes", []))
    keep_edges = cast(list[CropEdge], params.get("keep_edges", []))
    anchor_orientation = cast(CropOrientation | None, params.get("anchor_orientation"))
    native_orientation = _classify_crop_orientation(width, height, None)

    if not subject_boxes:
        return image, {
            "crop_applied": False,
            "crop_reason": "missing_subject_boxes",
            "crop_output_ratio": round(width / max(height, 1), 4),
        }

    if anchor_orientation is not None and anchor_orientation != native_orientation:
        return image, {
            "crop_applied": False,
            "crop_reason": "orientation_locked",
            "crop_output_ratio": round(width / max(height, 1), 4),
        }

    union_box = _union_subject_boxes(subject_boxes, width, height)
    if union_box is None:
        return image, {
            "crop_applied": False,
            "crop_reason": "invalid_subject_boxes",
            "crop_output_ratio": round(width / max(height, 1), 4),
        }

    expanded_box = _expand_subject_box(union_box, width, height)
    orientation = _classify_crop_orientation(width, height, anchor_orientation)
    target_ratio = _target_ratio_for_orientation(orientation, width, height)
    crop_box = _build_crop_window(
        expanded_box,
        width,
        height,
        target_ratio,
        keep_edges,
    )
    if crop_box is None:
        return image, {
            "crop_applied": False,
            "crop_reason": "constraints_not_satisfied",
            "crop_output_ratio": round(width / max(height, 1), 4),
        }

    allowed, reason = _crop_quality_gate(expanded_box, crop_box, width, height)
    if not allowed:
        return image, {
            "crop_applied": False,
            "crop_reason": reason,
            "crop_output_ratio": round(width / max(height, 1), 4),
        }

    x1, y1, x2, y2 = crop_box
    cropped = image[y1:y2, x1:x2]
    return cropped, {
        "crop_applied": True,
        "crop_reason": reason,
        "crop_box": [x1, y1, x2, y2],
        "crop_output_ratio": round((x2 - x1) / max(y2 - y1, 1), 4),
        "crop_orientation": orientation,
    }


def _tool_upscale(image: np.ndarray, params: dict[str, Any]) -> np.ndarray:
    return get_realesrgan_upscaler().upscale(image, 2)


def _tool_denoise(image: np.ndarray, params: dict[str, Any]) -> np.ndarray:
    cv2 = import_cv2()
    strength = cast(float, params["strength"])
    h_value = int(3 + strength * 12)
    return cv2.fastNlMeansDenoisingColored(image, None, h_value, h_value, 7, 21)


def _import_pilgram2() -> Any:
    import pilgram2

    return pilgram2


def _coerce_uint8_image(image: np.ndarray) -> np.ndarray:
    if image.dtype == np.uint8:
        return image
    return np.clip(image, 0, 255).astype(np.uint8)


def _numpy_image_to_pil_rgb(image: np.ndarray) -> Image.Image:
    cv2 = import_cv2()
    working = _coerce_uint8_image(image)

    if working.ndim == 2:
        rgb = cv2.cvtColor(working, cv2.COLOR_GRAY2RGB)
        return Image.fromarray(rgb)

    if working.ndim != 3:
        raise ValueError(f"Unsupported image shape for filter tool: {working.shape!r}")

    channel_count = working.shape[2]
    if channel_count == 1:
        rgb = cv2.cvtColor(working, cv2.COLOR_GRAY2RGB)
    elif channel_count == 3:
        rgb = cv2.cvtColor(working, cv2.COLOR_BGR2RGB)
    elif channel_count == 4:
        bgr = cv2.cvtColor(working, cv2.COLOR_BGRA2BGR)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    else:
        raise ValueError(f"Unsupported channel count for filter tool: {channel_count}")
    return Image.fromarray(rgb)


def _pil_image_to_bgr_array(image: Image.Image) -> np.ndarray:
    cv2 = import_cv2()
    rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def _tool_filter(
    image: np.ndarray,
    params: dict[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    filter_name = cast(str, params["filter_name"])
    filter_definition = resolve_final_edit_filter_definition(filter_name)
    pilgram2 = _import_pilgram2()
    filter_callable = getattr(pilgram2, filter_definition.pilgram2_name, None)
    if filter_callable is None or not callable(filter_callable):
        raise ValueError(
            f"pilgram2 filter callable is unavailable: {filter_definition.pilgram2_name}"
        )

    pil_image = _numpy_image_to_pil_rgb(image)
    filtered_image = filter_callable(pil_image)
    if not isinstance(filtered_image, Image.Image):
        raise TypeError(
            f"Unexpected pilgram2 filter result type: {type(filtered_image)!r}"
        )
    filtered_bgr = _pil_image_to_bgr_array(filtered_image)
    return filtered_bgr, {
        "filter_applied": True,
        "selected_filter": filter_definition.name,
        "selected_filter_backend": "pilgram2",
        "selected_filter_callable": filter_definition.pilgram2_name,
    }


def analyze_image_tone(image: np.ndarray) -> dict[str, float]:
    cv2 = import_cv2()
    working = image
    height, width = working.shape[:2]
    if max(height, width) > 1024:
        scale = 1024.0 / float(max(height, width))
        working = cv2.resize(working, (int(width * scale), int(height * scale)))

    lab = cv2.cvtColor(working, cv2.COLOR_BGR2LAB)
    l_channel = lab[:, :, 0].astype(np.float64)
    a_channel = lab[:, :, 1].astype(np.float64) - 128.0
    b_channel = lab[:, :, 2].astype(np.float64) - 128.0

    hsv = cv2.cvtColor(working, cv2.COLOR_BGR2HSV)
    saturation_channel = hsv[:, :, 1].astype(np.float64)
    brightness = _trimmed_mean(
        l_channel,
        _TRIMMED_BRIGHTNESS_LOWER_PERCENTILE,
        _TRIMMED_BRIGHTNESS_UPPER_PERCENTILE,
    )
    highlight_area_ratio = float(np.mean(l_channel >= _HIGHLIGHT_PIXEL_THRESHOLD))

    return {
        "brightness": round(brightness, 2),
        "contrast": round(float(np.std(l_channel)), 2),
        "black_point": round(float(np.percentile(l_channel, 5)), 2),
        "white_point": round(float(np.percentile(l_channel, 95)), 2),
        "temperature": round(float(np.mean(b_channel)), 2),
        "tint": round(float(np.mean(a_channel)), 2),
        "saturation": round(float(np.mean(saturation_channel)), 2),
        "highlight_area_ratio": round(highlight_area_ratio, 4),
    }


def build_aesthetic_color_grading_params(
    image: np.ndarray,
) -> tuple[dict[str, Any], dict[str, Any]]:
    current_tone = analyze_image_tone(image)
    applied_params = _empty_color_grading_params()
    gap_by_metric: dict[str, float] = {}
    skip_by_metric: dict[str, bool] = {}
    threshold_by_metric: dict[str, float] = {}

    for metric_name in AESTHETIC_TONE_METRIC_KEYS:
        target_config = AESTHETIC_TARGET_TONE_PROFILE[metric_name]
        target = float(target_config["target"])
        tolerance = float(target_config["tolerance"])
        gap = round(target - current_tone[metric_name], 2)
        should_skip = _is_gap_within_skip_threshold(gap, tolerance)

        gap_by_metric[metric_name] = gap
        skip_by_metric[metric_name] = should_skip
        threshold_by_metric[metric_name] = round(tolerance * 0.5, 2)

        if should_skip:
            continue

        if metric_name == "brightness":
            applied_params["brightness"] = _scale_gap_to_unit_interval(
                gap,
                tolerance,
                0.35,
            )
        elif metric_name == "contrast":
            applied_params["contrast"] = _scale_gap_to_percent(gap, tolerance, 40.0)
        elif metric_name == "black_point":
            applied_params["tone_curve_shadow_lift"] = _scale_gap_to_percent(
                gap,
                tolerance,
                40.0,
            )
        elif metric_name == "white_point":
            applied_params["highlights"] = _scale_gap_to_percent(gap, tolerance, 45.0)
        elif metric_name == "temperature":
            applied_params["temperature"] = "warm" if gap >= 0.0 else "cool"
            applied_params["temperature_strength"] = _clamp_float(
                abs(gap) / max(tolerance, 1.0),
                0.0,
                0.0,
                1.0,
            )
        elif metric_name == "tint":
            applied_params["tint"] = _scale_gap_to_percent(gap, tolerance, 45.0)
            applied_params["tint_strength"] = _clamp_float(
                abs(gap) / max(tolerance, 1.0),
                0.0,
                0.0,
                1.0,
            )
        elif metric_name == "saturation":
            applied_params["saturation"] = _scale_gap_to_percent(gap, tolerance, 35.0)

    applied_params["brightness"] = _attenuate_combined_global_darkening(
        cast(float, applied_params["brightness"]),
        cast(float, applied_params["highlights"]),
        current_tone.get("highlight_area_ratio", 0.0),
    )

    spotlight_protection = False
    highlight_area_ratio = current_tone.get("highlight_area_ratio", 0.0)
    target_white_point = AESTHETIC_TARGET_TONE_PROFILE["white_point"]["target"]
    if (
        highlight_area_ratio < _HIGHLIGHT_RATIO_MEDIUM
        and current_tone.get("white_point", 0.0) >= target_white_point
        and cast(float, applied_params["brightness"]) < 0.0
    ):
        applied_params["brightness"] = 0.0
        spotlight_protection = True

    original_highlights = cast(float, applied_params["highlights"])
    protected_highlights = _attenuate_negative_highlights(
        original_highlights,
        highlight_area_ratio,
    )
    if protected_highlights != original_highlights:
        applied_params["highlights"] = protected_highlights
        spotlight_protection = True

    if spotlight_protection:
        applied_params["spotlight_protection"] = True

    normalized_params = normalize_tool_params("color_grading", applied_params)
    debug_payload = {
        "current_tone": current_tone,
        "target_tone": AESTHETIC_TARGET_TONE_PROFILE,
        "gap_by_metric": gap_by_metric,
        "skip_by_metric": skip_by_metric,
        "skip_threshold_by_metric": threshold_by_metric,
        "applied_params": normalized_params,
        "spotlight_protection_applied": spotlight_protection,
    }
    return normalized_params, debug_payload


def _tool_color_grading(image: np.ndarray, params: dict[str, Any]) -> np.ndarray:
    cv2 = import_cv2()
    contrast = cast(float, params["contrast"])
    highlights = cast(float, params["highlights"])
    shadows = cast(float, params["shadows"])
    vibrance = cast(float, params["vibrance"])
    temperature = cast(str, params["temperature"])
    temperature_strength = cast(float, params["temperature_strength"])
    tint = cast(float, params["tint"])
    tint_strength = cast(float, params["tint_strength"])
    saturation = cast(float, params["saturation"])
    tone_curve_shadow_lift = cast(float, params["tone_curve_shadow_lift"])
    brightness = cast(float, params["brightness"])
    spotlight_protection = bool(params.get("spotlight_protection", False))

    result = image.astype(np.float32)
    if temperature_strength > 0.0:
        if temperature == "warm":
            result[:, :, 2] += 8.0 * temperature_strength
            result[:, :, 1] += 2.0 * temperature_strength
            result[:, :, 0] -= 6.0 * temperature_strength
        else:
            result[:, :, 0] += 8.0 * temperature_strength
            result[:, :, 2] -= 6.0 * temperature_strength

    if tint != 0.0 and tint_strength > 0.0:
        tint_shift = (tint / 100.0) * 24.0 * tint_strength
        result[:, :, 2] += tint_shift * 0.65
        result[:, :, 0] += tint_shift * 0.35
        result[:, :, 1] -= tint_shift

    if contrast != 0.0:
        contrast_factor = 1.0 + (contrast / 100.0) * 0.7
        result = (result - 127.5) * contrast_factor + 127.5

    if brightness != 0.0:
        result = result * (1.0 + brightness)

    luminance = (
        0.114 * result[:, :, 0] + 0.587 * result[:, :, 1] + 0.299 * result[:, :, 2]
    )
    normalized_luminance = np.clip(luminance / 255.0, 0.0, 1.0)
    highlight_mask = normalized_luminance**2
    shadow_mask = (1.0 - normalized_luminance) ** 2
    specular_mask: np.ndarray | None = None
    if spotlight_protection:
        broad_highlight_mask = _smoothstep(
            _SPOTLIGHT_BROAD_HIGHLIGHT_START,
            _SPOTLIGHT_BROAD_HIGHLIGHT_END,
            normalized_luminance,
        ) ** 2
        specular_mask = _smoothstep(
            _SPOTLIGHT_SPECULAR_START,
            _SPOTLIGHT_SPECULAR_END,
            normalized_luminance,
        ) ** 3
    else:
        broad_highlight_mask = highlight_mask

    if highlights != 0.0:
        if spotlight_protection and highlights < 0.0 and specular_mask is not None:
            protected_highlight_mask = broad_highlight_mask * (1.0 - 0.85 * specular_mask)
            result += protected_highlight_mask[:, :, None] * (highlights * 0.8)
        else:
            result += highlight_mask[:, :, None] * (highlights * 0.8)

    if shadows != 0.0:
        result += shadow_mask[:, :, None] * (shadows * 0.8)

    if tone_curve_shadow_lift != 0.0:
        result += shadow_mask[:, :, None] * (tone_curve_shadow_lift * 0.7)

    if spotlight_protection and specular_mask is not None:
        specular_boost = float(
            np.clip(
                max(0.0, -highlights) * 0.18
                + max(0.0, -brightness * 255.0) * 0.12,
                0.0,
                18.0,
            )
        )
        if specular_boost > 0.0:
            result += specular_mask[:, :, None] * specular_boost

    if vibrance != 0.0 or saturation != 0.0:
        hsv = cv2.cvtColor(
            np.clip(result, 0, 255).astype(np.uint8),
            cv2.COLOR_BGR2HSV,
        ).astype(np.float32)
        saturation_channel = hsv[:, :, 1]
        if vibrance != 0.0:
            low_saturation_mask = 1.0 - (saturation_channel / 255.0)
            saturation_channel += low_saturation_mask * (vibrance * 1.1)
        if saturation != 0.0:
            saturation_channel *= 1.0 + (saturation / 100.0)
        hsv[:, :, 1] = np.clip(saturation_channel, 0, 255)
        result = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR).astype(
            np.float32
        )

    return np.clip(result, 0, 255).astype(np.uint8)


def _tool_sharpen(image: np.ndarray, params: dict[str, Any]) -> np.ndarray:
    cv2 = import_cv2()
    strength = cast(float, params["strength"])
    blurred = cv2.GaussianBlur(image, (0, 0), sigmaX=3)
    return cv2.addWeighted(image, 1.0 + strength, blurred, -strength, 0)


def _tool_background_blur(image: np.ndarray, params: dict[str, Any]) -> np.ndarray:
    cv2 = import_cv2()
    blur_radius = cast(int, params["blur_radius"])
    kernel = blur_radius * 2 + 1
    blurred = cv2.GaussianBlur(image, (kernel, kernel), 0)

    height, width = image.shape[:2]
    y1, y2 = height // 4, height * 3 // 4
    x1, x2 = width // 4, width * 3 // 4
    blurred[y1:y2, x1:x2] = image[y1:y2, x1:x2]
    return blurred


class FinalEditToolRegistry:
    """Registry for final-edit tool execution."""

    def __init__(self, upscale_enabled: bool | None = None) -> None:
        self._upscale_enabled = _resolve_upscale_enabled(upscale_enabled)
        self._tool_names = supported_tool_names(self._upscale_enabled)
        self._tool_functions = {
            "crop": _tool_crop,
            "denoise": _tool_denoise,
            "filter": _tool_filter,
            "color_grading": _tool_color_grading,
            "sharpen": _tool_sharpen,
            "background_blur": _tool_background_blur,
        }
        if self._upscale_enabled:
            self._tool_functions["upscale"] = _tool_upscale

    @property
    def tool_names(self) -> tuple[ToolName, ...]:
        return self._tool_names

    def execute(
        self,
        tool_name: ToolName,
        image: np.ndarray,
        params: dict[str, Any],
    ) -> tuple[np.ndarray, dict[str, Any]]:
        if tool_name not in self._tool_functions:
            raise ValueError(f"Unsupported tool: {tool_name}")
        normalized_params = normalize_tool_params(tool_name, params)
        result = self._tool_functions[tool_name](image, normalized_params)
        if (
            isinstance(result, tuple)
            and len(result) == 2
            and isinstance(result[0], np.ndarray)
            and isinstance(result[1], dict)
        ):
            return result
        if isinstance(result, np.ndarray):
            return result, {}
        raise TypeError(f"Unexpected tool result type for {tool_name}: {type(result)!r}")
