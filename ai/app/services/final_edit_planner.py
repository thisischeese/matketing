from __future__ import annotations

import base64
import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from fastapi import Request
from redis.asyncio import Redis

from app.core.config import LlamaModelClientSettings, settings
from app.logging import build_log_extra
from app.services.final_edit_filter_catalog import (
    FinalEditFilterCatalogError,
    load_final_edit_filter_catalog,
    resolve_final_edit_filter_name,
)
from app.services.final_edit_tools import (
    AESTHETIC_TARGET_TONE_PROFILE,
    ToolName,
)
from app.services.remote_model_client import RemoteModelClient
from app.services.sessions import session_key

_JSON_ARRAY_PATTERN = re.compile(r"\[[\s\S]*\]")
_JSON_OBJECT_PATTERN = re.compile(r"\{[\s\S]*\}")
logger = logging.getLogger(__name__)
DEFAULT_OWNER_PERSONA = "aesthetic"
_FOOD_OR_BEVERAGE_TOKENS = {
    "food",
    "beverage",
    "dessert",
    "cake",
    "coffee",
    "drink",
    "menu",
    "bakery",
    "bread",
    "pastry",
    "brunch",
    "pasta",
    "salad",
    "burger",
    "pizza",
    "latte",
    "tea",
    "smoothie",
    "sandwich",
    "dish",
    "meal",
    "eggtart",
    "egg tart",
    "음식",
    "디저트",
    "케이크",
    "커피",
    "음료",
    "메뉴",
    "베이커리",
    "빵",
    "브런치",
    "파스타",
    "샐러드",
    "버거",
    "피자",
    "라떼",
    "차",
    "스무디",
    "샌드위치",
    "에그타르트", 
    "에그 타르트"
}
PERSONA_TARGET_PRESETS: dict[str, dict[str, Any]] = {
    "friendly": {
        "contrast": -8,
        "highlights": -6,
        "shadows": 18,
        "vibrance": 10,
        "saturation": 8,
        "temperature": "warm",
        "tone_curve_shadow_lift": 8,
    },
    "professional": {
        "contrast": 12,
        "highlights": -10,
        "shadows": 8,
        "vibrance": -6,
        "saturation": -8,
        "temperature": "cool",
        "tone_curve_shadow_lift": 4,
    },
    "trendy": {
        "contrast": 18,
        "highlights": -16,
        "shadows": 10,
        "vibrance": 14,
        "saturation": 10,
        "temperature": "warm",
        "tone_curve_shadow_lift": 2,
    },
    "other": {
        "contrast": 0,
        "highlights": -8,
        "shadows": 10,
        "vibrance": -4,
        "saturation": -2,
        "temperature": "warm",
        "tone_curve_shadow_lift": 6,
    },
}
TARGET_PROFILE_KIND_TONE = "tone_profile"
TARGET_PROFILE_KIND_PRESET = "style_preset"
_TOOL_ORDER_GROUPS: dict[ToolName, int] = {
    "crop": -1,
    "denoise": 0,
    "filter": 1,
    "color_grading": 1,
    "sharpen": 1,
    "upscale": 2,
}


@dataclass(frozen=True)
class FinalEditSessionContext:
    owner_persona: str = "aesthetic"
    caption: str = ""
    keywords: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class ImageEditPlan:
    image_index: int
    content: str
    strategy: str
    tools: list[ToolName]
    params: dict[str, dict[str, Any]]


@dataclass(frozen=True)
class CarouselOrderPlan:
    ordered_image_indexes: list[int]
    hook_scores: list[float]
    overall_confidence: float
    reason: str


@dataclass(frozen=True)
class FinalEditBatchPlan:
    image_plans: list[ImageEditPlan]
    carousel_order: CarouselOrderPlan | None = None
    carousel_order_error_message: str | None = None
    carousel_order_raw_output: str | None = None


class FinalEditPlanningError(RuntimeError):
    """Raised when the final-edit planner cannot build a usable plan."""


def build_noop_final_edit_plan(
    image_index: int,
    *,
    content: str = "draft image",
    strategy: str = "Planner fallback; keep the original image unchanged.",
) -> ImageEditPlan:
    return ImageEditPlan(
        image_index=image_index,
        content=content,
        strategy=strategy,
        tools=[],
        params={},
    )


def build_default_final_edit_plan(
    image_index: int,
    *,
    owner_persona: str = DEFAULT_OWNER_PERSONA,
    content: str = "draft image",
    strategy: str = "Planner fallback; keep the original image unchanged.",
    upscale_enabled: bool | None = None,
) -> ImageEditPlan:
    _ = owner_persona
    _ = upscale_enabled
    return build_noop_final_edit_plan(
        image_index,
        content=content,
        strategy=strategy,
    )


def normalize_image_edit_plan(
    plan: ImageEditPlan,
    *,
    upscale_enabled: bool | None = None,
    owner_persona: str = DEFAULT_OWNER_PERSONA,
) -> ImageEditPlan:
    upscale_enabled = settings.FINAL_EDIT_ENABLE_UPSCALE if upscale_enabled is None else upscale_enabled
    seen_tools: set[ToolName] = set()
    deduplicated_tools: list[ToolName] = []
    for tool_name in plan.tools:
        if tool_name == "upscale" and not upscale_enabled:
            continue
        if tool_name in seen_tools:
            continue
        deduplicated_tools.append(tool_name)
        seen_tools.add(tool_name)

    if deduplicated_tools and upscale_enabled and "upscale" not in seen_tools:
        deduplicated_tools.append("upscale")
        seen_tools.add("upscale")

    indexed_tools = list(enumerate(deduplicated_tools))
    ordered_tools = [
        tool_name
        for _, tool_name in sorted(
            indexed_tools,
            key=lambda item: (
                _TOOL_ORDER_GROUPS.get(item[1], 1),
                item[0],
            ),
        )
    ]
    if not ordered_tools:
        return ImageEditPlan(
            image_index=plan.image_index,
            content=plan.content,
            strategy=plan.strategy,
            tools=[],
            params={},
        )
    normalized_params: dict[str, dict[str, Any]] = {}
    for tool_name in ordered_tools:
        if tool_name == "upscale":
            normalized_params[tool_name] = {"scale": 2}
            continue
        normalized_params[tool_name] = dict(plan.params.get(tool_name, {}))

    return ImageEditPlan(
        image_index=plan.image_index,
        content=plan.content,
        strategy=plan.strategy,
        tools=ordered_tools,
        params=normalized_params,
    )


def _normalize_redis_value(value: object) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if value is None:
        return ""
    return str(value)


def _sorted_prefixed_values(payload: dict[str, object], prefix: str) -> list[str]:
    indexed_values: list[tuple[int, str]] = []
    for key, value in payload.items():
        normalized_key = _normalize_redis_value(key)
        if not normalized_key.startswith(prefix):
            continue
        suffix = normalized_key.removeprefix(prefix)
        try:
            index = int(suffix)
        except ValueError:
            continue
        normalized_value = _normalize_redis_value(value).strip()
        if normalized_value:
            indexed_values.append((index, normalized_value))
    indexed_values.sort(key=lambda item: item[0])
    return [value for _, value in indexed_values]


def _humanize_final_keyword(value: str) -> str:
    _, separator, display_name = value.partition(":")
    if separator and display_name.strip():
        return display_name.strip()
    return value.strip()


def resolve_owner_persona_preset(
    owner_persona: str,
) -> tuple[str, dict[str, Any], bool]:
    resolved_persona, resolved_target, used_fallback, _target_kind = (
        resolve_owner_persona_target(owner_persona)
    )
    return resolved_persona, resolved_target, used_fallback


def resolve_owner_persona_target(
    owner_persona: str,
) -> tuple[str, dict[str, Any], bool, str]:
    normalized_owner_persona = owner_persona.strip().lower()
    if normalized_owner_persona == DEFAULT_OWNER_PERSONA:
        return (
            DEFAULT_OWNER_PERSONA,
            dict(AESTHETIC_TARGET_TONE_PROFILE),
            False,
            TARGET_PROFILE_KIND_TONE,
        )
    preset = PERSONA_TARGET_PRESETS.get(normalized_owner_persona)
    if preset is not None:
        return normalized_owner_persona, dict(preset), False, TARGET_PROFILE_KIND_PRESET
    return (
        DEFAULT_OWNER_PERSONA,
        dict(AESTHETIC_TARGET_TONE_PROFILE),
        True,
        TARGET_PROFILE_KIND_TONE,
    )


async def load_final_edit_session_context(
    redis: Redis,
    session_id: str,
) -> FinalEditSessionContext:
    payload = await redis.hgetall(session_key(session_id))
    normalized_payload = {
        _normalize_redis_value(key): _normalize_redis_value(value)
        for key, value in payload.items()
    }
    owner_persona = normalized_payload.get("owner_persona", "").strip().lower()
    if not owner_persona:
        owner_persona = DEFAULT_OWNER_PERSONA
    caption = normalized_payload.get("caption", "").strip()
    keywords = _sorted_prefixed_values(normalized_payload, "draft_keyword:")
    if not keywords:
        keywords = [
            _humanize_final_keyword(value)
            for value in _sorted_prefixed_values(normalized_payload, "final_keyword:")
        ]
    return FinalEditSessionContext(
        owner_persona=owner_persona,
        caption=caption,
        keywords=keywords,
    )


def _filter_catalog_text() -> str:
    filter_names = ", ".join(
        definition.name for definition in load_final_edit_filter_catalog()
    )
    return f"[filter catalog] {filter_names}"


def _target_preset_text(context: FinalEditSessionContext) -> str:
    preset_persona, preset, preset_fallback, target_kind = resolve_owner_persona_target(
        context.owner_persona
    )
    def _compact_target_value(value: Any) -> Any:
        if isinstance(value, dict):
            return value.get("target", "n/a")
        return value

    brightness = _compact_target_value(preset.get("brightness", "n/a"))
    contrast = _compact_target_value(preset.get("contrast", "n/a"))
    temperature = _compact_target_value(preset.get("temperature", "n/a"))
    return (
        f"[owner_persona] {context.owner_persona or DEFAULT_OWNER_PERSONA}\n"
        f"[target_preset_persona] {preset_persona}\n"
        f"[target_preset_fallback] {'true' if preset_fallback else 'false'}\n"
        f"[target_profile_kind] {target_kind}\n"
        f"[style_hint] brightness={brightness}, "
        f"contrast={contrast}, "
        f"temperature={temperature}"
    )


def _content_focus_hint(context: FinalEditSessionContext) -> str:
    normalized_text = " ".join([context.caption, *context.keywords]).lower()
    if any(token in normalized_text for token in _FOOD_OR_BEVERAGE_TOKENS):
        return "likely_food_or_beverage"
    return "general_scene"


def _combined_batch_few_shot_example() -> str:
    return """
{"image_plans":[
{"image_index":0,"content":"cake hero shot","strategy":"Strong hero; denoise, sharpen, Valencia.","tools":["denoise","sharpen","filter"],"params":{"denoise":{"strength":0.35},"sharpen":{"strength":0.2},"filter":{"filter_name":"Valencia"}}},
{"image_index":1,"content":"coffee texture close-up","strategy":"Tighter crop and light sharpen for detail.","tools":["crop","sharpen","filter"],"params":{"crop":{"subject_boxes":[[120,80,420,520]],"keep_edges":[]},"sharpen":{"strength":0.2},"filter":{"filter_name":"Valencia"}}},
{"image_index":2,"content":"cafe ambience shelf scene","strategy":"Light crop, then Valencia for mood.","tools":["crop","filter"],"params":{"crop":{"subject_boxes":[[80,60,460,560]],"keep_edges":[]},"filter":{"filter_name":"Valencia"}}}
],"carousel_order":{"ordered_image_indexes":[0,1,2],"hook_scores":[0.93,0.74,0.58],"overall_confidence":0.84,"reason":"0 strongest, 1 second, 2 weakest."}}
""".strip()


def build_final_edit_system_prompt(*, image_count: int = 1) -> str:
    return (
        "You are a photo filter selection planner for draft images.\n"
        "Return only a JSON object.\n"
        'Use exactly two top-level keys: "image_plans" and "carousel_order".\n'
        f'"image_plans" must contain exactly {image_count} plans.\n'
        'Allowed tools: crop, denoise, sharpen, filter.\n'
        'Allowed tool sequences: ["filter"], ["denoise","filter"], ["sharpen","filter"], ["denoise","sharpen","filter"], ["crop","filter"], ["crop","denoise","filter"], ["crop","sharpen","filter"], ["crop","denoise","sharpen","filter"].\n'
        'crop params: {"subject_boxes":[[x1,y1,x2,y2],...],"keep_edges":[...optional...]}; denoise params: {"strength":0.0~1.0}; sharpen params: {"strength":0.0~1.0}; filter params: {"filter_name":"<catalog filter name>"}.\n'
        'Do not output any other numeric edit params.\n'
        'If image_count is 3, carousel_order must be {"ordered_image_indexes":[...],"hook_scores":[high to low],"overall_confidence":0.0~1.0,"reason":"short text"}.\n'
        'If image_count is not 3, carousel_order must be null.'
    )


def build_final_edit_static_prefix(image_count: int = 1) -> str:
    last_index = max(0, image_count - 1)
    return (
        f"{_filter_catalog_text()}\n\n"
        "[instructions]\n"
        f"1. Plan all {image_count} input images.\n"
        f"2. image_plans must include image_index 0 to {last_index} exactly once.\n"
        "3. Choose one exact filter catalog name per image.\n"
        "4. filter is mandatory; crop before denoise/sharpen/filter; denoise before sharpen/filter; sharpen before filter.\n"
        "5. Use denoise only for likely noise; use sharpen only for slight softness.\n"
        "6. Keep strategy short and tie the edit to subject + caption mood.\n"
        "7. If there are 3 images, rank them by hook strength only with hook_scores sorted high to low; otherwise carousel_order=null.\n\n"
        f"{_combined_batch_few_shot_example()}"
    )


def build_final_edit_dynamic_suffix(context: FinalEditSessionContext) -> str:
    keywords_text = ", ".join(context.keywords) if context.keywords else "(none)"
    caption_text = context.caption or "(none)"
    return (
        f"{_target_preset_text(context)}\n"
        f"[caption] {caption_text}\n"
        f"[keywords] {keywords_text}"
    )


def build_final_edit_prompt(
    context: FinalEditSessionContext,
    image_count: int,
) -> str:
    return (
        f"{build_final_edit_system_prompt(image_count=image_count)}\n\n"
        f"{build_final_edit_static_prefix(image_count)}\n\n"
        f"{build_final_edit_dynamic_suffix(context)}"
    )


def build_final_edit_messages(
    context: FinalEditSessionContext,
    image_paths: list[str],
) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = [
        {
            "type": "text",
            "text": build_final_edit_static_prefix(len(image_paths)),
        },
        {
            "type": "text",
            "text": build_final_edit_dynamic_suffix(context),
        },
    ]
    content.extend(
        {"type": "image_url", "image_url": {"url": image_path_to_data_uri(path)}}
        for path in image_paths
    )
    return [
        {
            "role": "system",
            "content": build_final_edit_system_prompt(image_count=len(image_paths)),
        },
        {"role": "user", "content": content},
    ]


def build_carousel_order_system_prompt() -> str:
    return (
        "You are a carousel image ordering planner for Instagram-style food and cafe posts.\n"
        "Return only a JSON object.\n"
        'The JSON object must contain exactly these fields: "ordered_image_indexes", "hook_scores", "overall_confidence", and "reason".\n'
        '"ordered_image_indexes" must be a permutation of [0, 1, 2].\n'
        '"hook_scores" must be a list of three numbers from 0.0 to 1.0 aligned with "ordered_image_indexes".\n'
        '"hook_scores" must be sorted in non-increasing order because slot 0 must be the strongest hook, slot 1 the second strongest hook, and slot 2 the third strongest hook.\n'
        '"overall_confidence" must be a number from 0.0 to 1.0.\n'
        '"reason" must be a short explanation.\n'
        "Rank the three images by hooking power only.\n"
        "The first output slot must be the strongest hook, the second output slot must be the second strongest hook, and the third output slot must be the third strongest hook.\n"
        "Do not output any extra keys."
    )


def _carousel_order_few_shot_example() -> str:
    return """
Example output:
{
  "ordered_image_indexes": [0, 2, 1],
  "hook_scores": [0.93, 0.74, 0.58],
  "overall_confidence": 0.84,
  "reason": "Image 0 has the strongest immediate hook, image 2 is the next most attention-grabbing, and image 1 is the weakest hook of the three."
}
""".strip()


def build_carousel_order_dynamic_suffix(context: FinalEditSessionContext) -> str:
    keywords_text = ", ".join(context.keywords) if context.keywords else "(none)"
    caption_text = context.caption or "(none)"
    return (
        "[carousel_ordering]\n"
        "Rank all three images by hooking power only.\n"
        "Slot 1: strongest hook.\n"
        "Slot 2: second strongest hook.\n"
        "Slot 3: third strongest hook.\n"
        "Use immediate stopping power, visual clarity, appetite appeal, curiosity, and thumb-stop potential as the ranking criteria.\n"
        "If the caption mood is strong, prefer images whose hook is reinforced by that mood.\n"
        "Important: hook_scores must align with ordered_image_indexes in output order.\n"
        "Important: hook_scores must be sorted from highest to lowest.\n"
        f"{_carousel_order_few_shot_example()}\n"
        f"[caption] {caption_text}\n"
        f"[keywords] {keywords_text}"
    )


def build_carousel_order_messages(
    context: FinalEditSessionContext,
    image_paths: list[str],
) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = [
        {
            "type": "text",
            "text": build_carousel_order_dynamic_suffix(context),
        }
    ]
    content.extend(
        {"type": "image_url", "image_url": {"url": image_path_to_data_uri(path)}}
        for path in image_paths
    )
    return [
        {"role": "system", "content": build_carousel_order_system_prompt()},
        {"role": "user", "content": content},
    ]


def image_path_to_data_uri(image_path: str) -> str:
    data = Path(image_path).read_bytes()
    encoded = base64.b64encode(data).decode("utf-8")
    suffix = Path(image_path).suffix.lower()
    media_type = "image/png" if suffix == ".png" else "image/jpeg"
    return f"data:{media_type};base64,{encoded}"


def _measure_chat_payload(messages: list[dict[str, Any]]) -> dict[str, int]:
    system_text_chars = 0
    user_text_chars = 0
    image_data_uri_chars_total = 0
    image_data_uri_chars_max = 0
    image_count = 0

    for message in messages:
        role = str(message.get("role", ""))
        content = message.get("content")
        if isinstance(content, str):
            if role == "system":
                system_text_chars += len(content)
            else:
                user_text_chars += len(content)
            continue
        if not isinstance(content, list):
            continue
        for part in content:
            if not isinstance(part, dict):
                continue
            part_type = part.get("type")
            if part_type == "text":
                text = str(part.get("text", ""))
                if role == "system":
                    system_text_chars += len(text)
                else:
                    user_text_chars += len(text)
            elif part_type == "image_url":
                image_url = part.get("image_url", {})
                if isinstance(image_url, dict):
                    url = str(image_url.get("url", ""))
                else:
                    url = str(image_url)
                image_count += 1
                image_data_uri_chars_total += len(url)
                image_data_uri_chars_max = max(image_data_uri_chars_max, len(url))

    return {
        "prompt_system_text_chars": system_text_chars,
        "prompt_user_text_chars": user_text_chars,
        "prompt_text_chars_total": system_text_chars + user_text_chars,
        "prompt_image_count": image_count,
        "prompt_image_data_uri_chars_total": image_data_uri_chars_total,
        "prompt_image_data_uri_chars_max": image_data_uri_chars_max,
    }


def _strip_fenced_output(raw_output: str) -> str:
    cleaned = raw_output.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[1]
    if cleaned.endswith("```"):
        cleaned = cleaned.rsplit("```", 1)[0]
    return cleaned.strip()


def _extract_json_array(raw_output: str) -> str:
    cleaned = _strip_fenced_output(raw_output)
    if cleaned.startswith("[") and cleaned.endswith("]"):
        return cleaned

    match = _JSON_ARRAY_PATTERN.search(cleaned)
    if match is None:
        raise FinalEditPlanningError("Planner did not return a JSON array.")
    return match.group(0)


def _extract_json_object(raw_output: str) -> str:
    cleaned = _strip_fenced_output(raw_output)
    if cleaned.startswith("{") and cleaned.endswith("}"):
        return cleaned

    match = _JSON_OBJECT_PATTERN.search(cleaned)
    if match is None:
        raise FinalEditPlanningError("Planner did not return a JSON object.")
    return match.group(0)


def _parse_image_edit_plan_payload(
    payload: object,
    *,
    upscale_enabled: bool | None = None,
    owner_persona: str = DEFAULT_OWNER_PERSONA,
) -> list[ImageEditPlan]:
    upscale_enabled = settings.FINAL_EDIT_ENABLE_UPSCALE if upscale_enabled is None else upscale_enabled
    if not isinstance(payload, list):
        raise FinalEditPlanningError("Planner payload must be a list.")

    allowed_tools = {"crop", "denoise", "sharpen", "filter"}
    seen_indexes: set[int] = set()
    plans: list[ImageEditPlan] = []

    for item in payload:
        if not isinstance(item, dict):
            raise FinalEditPlanningError("Planner payload must contain objects only.")

        required_fields = {"image_index", "content", "strategy", "tools", "params"}
        missing_fields = required_fields - set(item.keys())
        if missing_fields:
            raise FinalEditPlanningError(
                f"Planner payload is missing fields: {sorted(missing_fields)}"
            )

        image_index = item["image_index"]
        if not isinstance(image_index, int):
            raise FinalEditPlanningError("image_index must be an integer.")
        if image_index in seen_indexes:
            raise FinalEditPlanningError("image_index values must be unique.")
        seen_indexes.add(image_index)

        tools = item["tools"]
        if not isinstance(tools, list) or not tools:
            raise FinalEditPlanningError("tools must be a non-empty list.")

        normalized_tools: list[ToolName] = []
        seen_tools_in_plan: set[str] = set()
        for tool_name in tools:
            if not isinstance(tool_name, str) or tool_name not in allowed_tools:
                raise FinalEditPlanningError(f"Unsupported tool requested: {tool_name}")
            if tool_name in seen_tools_in_plan:
                raise FinalEditPlanningError(f"Duplicate tool requested: {tool_name}")
            seen_tools_in_plan.add(tool_name)
            normalized_tools.append(tool_name)

        if "filter" not in seen_tools_in_plan:
            raise FinalEditPlanningError("filter must be included in every plan.")
        allowed_tool_sequences = {
            ("filter",),
            ("denoise", "filter"),
            ("sharpen", "filter"),
            ("denoise", "sharpen", "filter"),
            ("crop", "filter"),
            ("crop", "denoise", "filter"),
            ("crop", "sharpen", "filter"),
            ("crop", "denoise", "sharpen", "filter"),
        }
        if tuple(normalized_tools) not in allowed_tool_sequences:
            raise FinalEditPlanningError(
                'tools must be one of ["filter"], ["denoise", "filter"], ["sharpen", "filter"], ["denoise", "sharpen", "filter"], ["crop", "filter"], ["crop", "denoise", "filter"], ["crop", "sharpen", "filter"], or ["crop", "denoise", "sharpen", "filter"].'
            )

        params = item["params"]
        if not isinstance(params, dict):
            raise FinalEditPlanningError("params must be an object.")
        normalized_params: dict[str, dict[str, Any]] = {}
        for tool_name in normalized_tools:
            tool_params = params.get(tool_name, {})
            if not isinstance(tool_params, dict):
                raise FinalEditPlanningError(
                    f"params for tool '{tool_name}' must be an object."
                )
            if tool_name == "crop":
                subject_boxes = tool_params.get("subject_boxes")
                if not isinstance(subject_boxes, list) or not subject_boxes:
                    raise FinalEditPlanningError(
                        "crop params must include 1~3 subject_boxes."
                    )
                if len(subject_boxes) > 3:
                    raise FinalEditPlanningError(
                        "crop subject_boxes must contain at most 3 boxes."
                    )
                normalized_boxes: list[list[int]] = []
                for raw_box in subject_boxes:
                    if not isinstance(raw_box, (list, tuple)) or len(raw_box) != 4:
                        raise FinalEditPlanningError(
                            "Each crop subject box must be a list of four coordinates."
                        )
                    try:
                        x1, y1, x2, y2 = [
                            int(round(float(value))) for value in raw_box
                        ]
                    except (TypeError, ValueError) as exc:
                        raise FinalEditPlanningError(
                            "Crop subject box coordinates must be numeric."
                        ) from exc
                    left = min(x1, x2)
                    top = min(y1, y2)
                    right = max(x1, x2)
                    bottom = max(y1, y2)
                    if right - left <= 1 or bottom - top <= 1:
                        raise FinalEditPlanningError(
                            "Crop subject boxes must have width and height greater than 1."
                        )
                    normalized_boxes.append([left, top, right, bottom])

                keep_edges = tool_params.get("keep_edges", [])
                if not isinstance(keep_edges, list):
                    raise FinalEditPlanningError("crop keep_edges must be a list.")
                normalized_edges: list[str] = []
                valid_edges = {"top", "bottom", "left", "right"}
                for raw_edge in keep_edges:
                    edge = str(raw_edge).strip().lower()
                    if edge not in valid_edges:
                        raise FinalEditPlanningError(
                            "crop keep_edges values must be one of top, bottom, left, right."
                        )
                    if edge not in normalized_edges:
                        normalized_edges.append(edge)

                tool_params = {
                    "subject_boxes": normalized_boxes,
                    "keep_edges": normalized_edges,
                }
            if tool_name == "denoise":
                if "strength" not in tool_params:
                    raise FinalEditPlanningError(
                        "denoise params must include strength."
                    )
                try:
                    strength = float(tool_params["strength"])
                except (TypeError, ValueError) as exc:
                    raise FinalEditPlanningError(
                        "denoise strength must be numeric."
                    ) from exc
                tool_params = {
                    "strength": max(0.0, min(1.0, strength))
                }
            if tool_name == "sharpen":
                if "strength" not in tool_params:
                    raise FinalEditPlanningError(
                        "sharpen params must include strength."
                    )
                try:
                    strength = float(tool_params["strength"])
                except (TypeError, ValueError) as exc:
                    raise FinalEditPlanningError(
                        "sharpen strength must be numeric."
                    ) from exc
                tool_params = {
                    "strength": max(0.0, min(1.0, strength))
                }
            if tool_name == "filter":
                try:
                    normalized_filter_name = resolve_final_edit_filter_name(
                        tool_params.get("filter_name")
                    )
                except FinalEditFilterCatalogError as exc:
                    raise FinalEditPlanningError(str(exc)) from exc
                tool_params = {"filter_name": normalized_filter_name}
            normalized_params[tool_name] = tool_params

        content = str(item["content"]).strip()
        strategy = str(item["strategy"]).strip()
        if not content or not strategy:
            raise FinalEditPlanningError("content and strategy must be non-empty.")

        plans.append(
            normalize_image_edit_plan(
                ImageEditPlan(
                    image_index=image_index,
                    content=content,
                    strategy=strategy,
                    tools=normalized_tools,
                    params=normalized_params,
                ),
                upscale_enabled=upscale_enabled,
                owner_persona=owner_persona,
            )
        )

    return plans


def parse_image_edit_plans(
    raw_output: str,
    *,
    upscale_enabled: bool | None = None,
    owner_persona: str = DEFAULT_OWNER_PERSONA,
) -> list[ImageEditPlan]:
    try:
        payload = json.loads(_extract_json_array(raw_output))
    except json.JSONDecodeError as exc:
        raise FinalEditPlanningError("Planner returned invalid JSON.") from exc

    return _parse_image_edit_plan_payload(
        payload,
        upscale_enabled=upscale_enabled,
        owner_persona=owner_persona,
    )


def parse_final_edit_batch_plan(
    raw_output: str,
    *,
    image_count: int,
    upscale_enabled: bool | None = None,
    owner_persona: str = DEFAULT_OWNER_PERSONA,
) -> FinalEditBatchPlan:
    try:
        payload = json.loads(_extract_json_object(raw_output))
    except json.JSONDecodeError as exc:
        raise FinalEditPlanningError("Planner returned invalid JSON.") from exc

    if not isinstance(payload, dict):
        raise FinalEditPlanningError("Planner payload must be an object.")
    if set(payload.keys()) != {"image_plans", "carousel_order"}:
        raise FinalEditPlanningError(
            'Planner payload must contain exactly "image_plans" and "carousel_order".'
        )

    image_plans = _parse_image_edit_plan_payload(
        payload["image_plans"],
        upscale_enabled=upscale_enabled,
        owner_persona=owner_persona,
    )
    carousel_payload = payload["carousel_order"]
    if image_count != 3:
        if carousel_payload is not None:
            return FinalEditBatchPlan(
                image_plans=image_plans,
                carousel_order=None,
                carousel_order_error_message=(
                    "carousel_order must be null unless exactly three images are provided."
                ),
                carousel_order_raw_output=json.dumps(
                    carousel_payload,
                    ensure_ascii=False,
                ),
            )
        return FinalEditBatchPlan(image_plans=image_plans, carousel_order=None)

    if carousel_payload is None:
        return FinalEditBatchPlan(
            image_plans=image_plans,
            carousel_order=None,
            carousel_order_error_message=(
                "carousel_order must be an object when exactly three images are provided."
            ),
            carousel_order_raw_output="null",
        )

    raw_carousel_output = json.dumps(carousel_payload, ensure_ascii=False)
    try:
        carousel_order = parse_carousel_order_plan(raw_carousel_output)
    except FinalEditPlanningError as exc:
        return FinalEditBatchPlan(
            image_plans=image_plans,
            carousel_order=None,
            carousel_order_error_message=str(exc),
            carousel_order_raw_output=raw_carousel_output,
        )
    return FinalEditBatchPlan(
        image_plans=image_plans,
        carousel_order=carousel_order,
    )


def parse_carousel_order_plan(raw_output: str) -> CarouselOrderPlan:
    try:
        payload = json.loads(_extract_json_object(raw_output))
    except json.JSONDecodeError as exc:
        raise FinalEditPlanningError("Planner returned invalid JSON.") from exc

    if not isinstance(payload, dict):
        raise FinalEditPlanningError("Carousel order payload must be an object.")

    required_fields = {
        "ordered_image_indexes",
        "hook_scores",
        "overall_confidence",
        "reason",
    }
    missing_fields = required_fields - set(payload.keys())
    if missing_fields:
        raise FinalEditPlanningError(
            f"Carousel order payload is missing fields: {sorted(missing_fields)}"
        )

    ordered_image_indexes = payload["ordered_image_indexes"]
    if not isinstance(ordered_image_indexes, list) or len(ordered_image_indexes) != 3:
        raise FinalEditPlanningError(
            "ordered_image_indexes must be a list of three integers."
        )
    if any(not isinstance(index, int) for index in ordered_image_indexes):
        raise FinalEditPlanningError("ordered_image_indexes must contain integers only.")
    if sorted(ordered_image_indexes) != [0, 1, 2]:
        raise FinalEditPlanningError(
            "ordered_image_indexes must be a permutation of [0, 1, 2]."
        )

    hook_scores = payload["hook_scores"]
    if not isinstance(hook_scores, list) or len(hook_scores) != 3:
        raise FinalEditPlanningError(
            "hook_scores must be a list of three numbers."
        )
    normalized_hook_scores: list[float] = []
    previous_score: float | None = None
    for raw_score in hook_scores:
        try:
            score = float(raw_score)
        except (TypeError, ValueError) as exc:
            raise FinalEditPlanningError("hook_scores must contain numeric values only.") from exc
        score = max(0.0, min(1.0, score))
        if previous_score is not None and score > previous_score:
            raise FinalEditPlanningError(
                "hook_scores must be sorted from highest to lowest."
            )
        normalized_hook_scores.append(score)
        previous_score = score

    try:
        overall_confidence = float(payload["overall_confidence"])
    except (TypeError, ValueError) as exc:
        raise FinalEditPlanningError("overall_confidence must be numeric.") from exc
    overall_confidence = max(0.0, min(1.0, overall_confidence))

    reason = str(payload["reason"]).strip()
    if not reason:
        raise FinalEditPlanningError("reason must be non-empty.")

    return CarouselOrderPlan(
        ordered_image_indexes=list(ordered_image_indexes),
        hook_scores=normalized_hook_scores,
        overall_confidence=overall_confidence,
        reason=reason,
    )


class FinalEditPlannerClient:
    """Client for requesting image edit plans from an OpenAI-compatible VLM."""

    def __init__(
        self,
        model_settings: LlamaModelClientSettings,
        model_name: str,
    ) -> None:
        self.model_settings = model_settings
        self.model_name = model_name
        self._remote_client = RemoteModelClient(
            base_url=self.model_settings.base_url,
            chat_endpoint=self.model_settings.chat_endpoint,
            health_endpoint=self.model_settings.health_endpoint,
            api_key=self.model_settings.api_key,
            timeout_seconds=self.model_settings.timeout_seconds,
        )
        self._health_checked = False
        self.last_raw_output: str | None = None
        self.last_reorder_raw_output: str | None = None
        self.last_reorder_error_message: str | None = None
        self.last_request_stats: dict[str, int] = {}

    async def preload(self) -> None:
        await self._check_server_connection()

    async def build_batch_plan(
        self,
        image_paths: list[str],
        context: FinalEditSessionContext,
    ) -> FinalEditBatchPlan:
        if not self.model_settings.enabled:
            raise FinalEditPlanningError("Final-edit planner is disabled.")
        if not image_paths:
            raise FinalEditPlanningError("No image paths were provided to the planner.")

        raw_output = await self._request_plan(image_paths, context)
        self.last_raw_output = raw_output
        batch_plan = parse_final_edit_batch_plan(
            raw_output,
            image_count=len(image_paths),
            upscale_enabled=settings.FINAL_EDIT_ENABLE_UPSCALE,
            owner_persona=context.owner_persona,
        )
        self.last_reorder_error_message = batch_plan.carousel_order_error_message
        self.last_reorder_raw_output = batch_plan.carousel_order_raw_output
        if len(batch_plan.image_plans) > len(image_paths):
            raise FinalEditPlanningError("Planner returned too many image plans.")
        for plan in batch_plan.image_plans:
            if plan.image_index < 0 or plan.image_index >= len(image_paths):
                raise FinalEditPlanningError("Planner returned an out-of-range image_index.")
        return batch_plan

    async def _request_plan(
        self,
        image_paths: list[str],
        context: FinalEditSessionContext,
    ) -> str:
        messages = build_final_edit_messages(context, image_paths)
        payload = {
            "model": self.model_name,
            "messages": messages,
            "temperature": self.model_settings.temperature,
            "top_p": self.model_settings.top_p,
            "max_tokens": self.model_settings.max_tokens,
        }
        request_stats = _measure_chat_payload(messages)
        request_stats["prompt_payload_json_chars"] = len(
            json.dumps(payload, ensure_ascii=False)
        )
        self.last_request_stats = request_stats
        logger.info(
            "Submitting final edit planner request.",
            extra=build_log_extra(
                "final_edit.planner_request.started",
                component="final_edit_planner",
                stage="planner_request",
                outcome="started",
                image_count=len(image_paths),
                model_name=self.model_name,
                **request_stats,
            ),
        )
        response = await self._remote_client.post_chat_completion(payload)
        logger.info(
            "Completed final edit planner request.",
            extra=build_log_extra(
                "final_edit.planner_request.completed",
                component="final_edit_planner",
                stage="planner_request",
                outcome="succeeded" if response.is_success else "failed",
                image_count=len(image_paths),
                model_name=self.model_name,
                status_code=response.status_code,
                **request_stats,
            ),
        )
        response.raise_for_status()
        raw_output = self._remote_client.extract_message_content(response)
        if not isinstance(raw_output, str) or not raw_output.strip():
            raise FinalEditPlanningError("Planner returned an empty response.")
        return raw_output

    async def _request_carousel_order(
        self,
        image_paths: list[str],
        context: FinalEditSessionContext,
    ) -> str:
        payload = {
            "model": self.model_name,
            "messages": build_carousel_order_messages(context, image_paths),
            "temperature": self.model_settings.temperature,
            "top_p": self.model_settings.top_p,
            "max_tokens": self.model_settings.max_tokens,
        }
        response = await self._remote_client.post_chat_completion(payload)
        response.raise_for_status()
        raw_output = self._remote_client.extract_message_content(response)
        if not isinstance(raw_output, str) or not raw_output.strip():
            raise FinalEditPlanningError("Planner returned an empty response.")
        return raw_output

    async def _check_server_connection(self) -> None:
        if not self.model_settings.base_url:
            raise FinalEditPlanningError("Final-edit planner base URL is not configured.")
        await self._remote_client.check_health()
        self._health_checked = True


def build_final_edit_planner_client() -> FinalEditPlannerClient:
    return FinalEditPlannerClient(
        model_settings=settings.final_edit_model_client,
        model_name=settings.FINAL_EDIT_MODEL_NAME,
    )


def get_final_edit_planner_client(request: Request) -> FinalEditPlannerClient:
    service = getattr(request.app.state, "final_edit_planner_client", None)
    if service is None:
        raise RuntimeError("Final edit planner client is not initialized.")
    return service
