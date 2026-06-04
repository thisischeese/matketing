from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable
from pathlib import Path
from uuid import uuid4

import cv2
import httpx
import numpy as np

from app.services.caption_generation import (
    CaptionFallbackResult,
    CaptionGenerationRequest,
    CaptionGenerationResult,
    DEFAULT_FALLBACK_GUIDE_TEXT,
)
from app.services.canonical_keyword_resolver import (
    CanonicalKeywordMatch,
    CanonicalKeywordResolution,
)
from app.services.final_edit import FinalEditResult
from app.services.final_edit_agent import FinalEditImageState
from app.services.final_edit_planner import (
    CarouselOrderPlan,
    FinalEditBatchPlan,
    ImageEditPlan,
)
from app.services.frame_extraction import ExtractFramesResult
from app.services.keyword_extraction import KeywordExtractionResult
from app.services.menu_promotion_context import MatchedMenuContext
from app.services.reference_caption_retriever import (
    ReferenceCaptionRetrievalResult,
    RetrievedReferenceCaption,
)
from app.services.weather_tags import (
    PRECIP_CLEAR,
    PRECIP_CLOUDY,
    PRECIP_HEAVY_RAIN,
    PRECIP_RAIN,
)

VALID_PAYLOAD = {
    "store_id": str(uuid4()),
    "utterance": "warm lighting cozy table signature menu",
    "owner_persona": "aesthetic",
    "date": "2026-04-27",
    "weather": {
        "temperature": 18.5,
        "precipitation": 0.0,
        "cloud_cover": "\ub9d1\uc74c",
        "humidity": 45,
        "wind_speed": 2.5,
        "pm10": 85,
        "pm25": 35,
        "diurnal_range": 12.0,
        "discomfort_index": 63,
        "heavy_rain_warning": None,
        "typhoon_warning": None,
    },
}

VALID_EXTRACT_PAYLOAD = {
    "session_id": str(uuid4()),
    "video": "/inputs/test-session/test-video.mp4",
}


def _weather_context_for_tests(weather_tags: list[str]) -> str:
    if PRECIP_HEAVY_RAIN in weather_tags:
        return "\ube44\uc640 \uc798 \uc5b4\uc6b8\ub9ac\ub294 "
    if PRECIP_RAIN in weather_tags:
        return "\ube44 \uc624\ub294 \ub0a0 "
    if PRECIP_CLEAR in weather_tags:
        return "\ub9d1\uc740 \ub0a0 "
    if PRECIP_CLOUDY in weather_tags:
        return "\ud750\ub9b0 \ub0a0 "
    return ""


def _make_chat_response(
    status_code: int,
    *,
    content: str = "",
    url: str = "http://llama-server:8000/v1/chat/completions",
) -> httpx.Response:
    return httpx.Response(
        status_code,
        request=httpx.Request("POST", url),
        json={"choices": [{"message": {"content": content}}]},
    )


def _write_test_jpeg(destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    image = np.full((16, 16, 3), 120, dtype=np.uint8)
    image[:, :8, 2] = 220
    cv2.imwrite(str(destination), image)


class FakeUpscaler:
    def upscale(self, image: np.ndarray, scale: int) -> np.ndarray:
        height, width = image.shape[:2]
        return cv2.resize(
            image,
            (width * scale, height * scale),
            interpolation=cv2.INTER_LANCZOS4,
        )


def _run_immediate(awaitable: Awaitable[object]) -> object:
    iterator = awaitable.__await__()
    try:
        yielded = next(iterator)
    except StopIteration as exc:
        return exc.value

    while True:
        try:
            if hasattr(yielded, "__await__"):
                yielded = iterator.send(_run_immediate(yielded))
            else:
                yielded = iterator.send(None)
        except StopIteration as exc:
            return exc.value


class FakeFrameExtractionService:
    def __init__(self, result: ExtractFramesResult) -> None:
        self.result = result
        self.calls: list[tuple[str, str]] = []

    async def extract_and_upload(
        self,
        session_id: str,
        video_key: str,
    ) -> ExtractFramesResult:
        self.calls.append((session_id, video_key))
        return self.result


class FakeFinalEditService:
    def __init__(self, result: FinalEditResult) -> None:
        self.result = result
        self.calls: list[tuple[str, list[str]]] = []

    async def edit_and_upload(
        self,
        session_id: str,
        drafts: list[str],
        context=None,
    ) -> FinalEditResult:
        self.calls.append((session_id, drafts))
        return self.result


class FakeKeywordExtractionService:
    def __init__(
        self,
        draft_keywords: list[str] | None = None,
        final_keywords: list[str] | None = None,
        error: Exception | None = None,
        healthy: bool = True,
    ) -> None:
        self.draft_keywords = (
            draft_keywords
            if draft_keywords is not None
            else ["signature menu", "cozy table"]
        )
        self.final_keywords = final_keywords if final_keywords is not None else []
        self.error = error
        self.healthy = healthy
        self.calls: list[str] = []
        self.health_check_calls: int = 0

    async def is_healthy(self) -> bool:
        self.health_check_calls += 1
        return self.healthy

    async def extract_keywords(self, utterance: str) -> KeywordExtractionResult:
        self.calls.append(utterance)
        if self.error is not None:
            raise self.error
        return KeywordExtractionResult(
            draft_keywords=self.draft_keywords,
            final_keywords=self.final_keywords,
        )


class FakeCanonicalKeywordResolverService:
    def __init__(
        self,
        final_keywords: list[str] | None = None,
        display_names: list[str] | None = None,
        matched_indexes: set[int] | None = None,
    ) -> None:
        self.final_keywords = final_keywords
        self.display_names = display_names
        self.matched_indexes = matched_indexes
        self.calls: list[list[str]] = []

    @staticmethod
    def _parse_stored_final_keyword(value: str) -> tuple[int | None, str | None]:
        keyword_id, separator, display_name = value.partition(":")
        if not separator:
            return None, None
        try:
            return int(keyword_id), display_name or None
        except ValueError:
            return None, None

    async def resolve_keywords(
        self,
        draft_keywords: list[str],
    ) -> CanonicalKeywordResolution:
        self.calls.append(list(draft_keywords))
        final_keywords = (
            list(self.final_keywords)
            if self.final_keywords is not None
            else [f"{1000 + index}:{keyword}" for index, keyword in enumerate(draft_keywords)]
        )
        display_names = (
            list(self.display_names)
            if self.display_names is not None
            else [
                self._parse_stored_final_keyword(final_keyword)[1] or draft_keyword
                for draft_keyword, final_keyword in zip(
                    draft_keywords, final_keywords, strict=True
                )
            ]
        )
        matched_indexes = (
            set(self.matched_indexes)
            if self.matched_indexes is not None
            else set(range(len(draft_keywords)))
        )
        matches = []
        for index, draft_keyword in enumerate(draft_keywords):
            matched = index in matched_indexes
            stored_final_keyword = final_keywords[index] if matched else draft_keyword
            canonical_keyword_id, parsed_display_name = self._parse_stored_final_keyword(
                stored_final_keyword
            )
            matches.append(
                CanonicalKeywordMatch(
                    draft_keyword=draft_keyword,
                    canonical_keyword_id=canonical_keyword_id if matched else None,
                    final_keyword=stored_final_keyword,
                    display_name=(
                        display_names[index] if matched else parsed_display_name
                    )
                    if matched
                    else None,
                    score=0.99 if matched else None,
                    matched=matched,
                )
            )
        return CanonicalKeywordResolution(
            final_keywords=[match.stored_final_keyword for match in matches],
            matches=matches,
        )


class FakeCaptionGenerationService:
    def __init__(
        self,
        result: CaptionGenerationResult | None = None,
        error: Exception | None = None,
        healthy: bool = True,
    ) -> None:
        self.result = result or CaptionGenerationResult(
            guide_text="\uc74c\uc2dd\uc774 \uc798 \ubcf4\uc774\ub3c4\ub85d \uad6c\ub3c4\ub97c \uac00\uae4c\uc774 \uc7a1\uc544\ubcf4\uc138\uc694.",
            draft_caption="\uc624\ub298 \uba54\ub274\ub97c \uc790\uc5f0\uc2a4\ub7fd\uac8c \uc18c\uac1c\ud574\ubcf4\uc138\uc694.",
        )
        self.error = error
        self.healthy = healthy
        self.calls: list[dict[str, object]] = []
        self.fallback_calls: list[dict[str, object]] = []
        self.health_check_calls: int = 0

    async def is_healthy(self) -> bool:
        self.health_check_calls += 1
        return self.healthy

    async def generate_text(
        self,
        request: CaptionGenerationRequest,
    ) -> CaptionGenerationResult:
        self.calls.append(
            {
                "draft_keywords": list(request.draft_keywords),
                "owner_persona": request.owner_persona,
                "context_routing_variant": request.context_routing_variant,
                "context_profile": request.context_profile,
                "today": request.today,
                "utterance": request.utterance,
                "weather_tags": list(request.weather_tags),
                "menu_name": request.menu_name,
                "menu_description": request.menu_description,
                "matched_keyword": request.matched_keyword,
                "prompt_context": request.prompt_context,
            }
        )
        if self.error is not None:
            raise self.error
        return self.result

    def build_fallback_result(
        self,
        request: CaptionGenerationRequest,
        fallback_source: str | None,
    ) -> CaptionFallbackResult:
        self.fallback_calls.append(
            {
                "draft_keywords": list(request.draft_keywords),
                "owner_persona": request.owner_persona,
                "context_routing_variant": request.context_routing_variant,
                "context_profile": request.context_profile,
                "today": request.today,
                "utterance": request.utterance,
                "weather_tags": list(request.weather_tags),
                "menu_name": request.menu_name,
                "menu_description": request.menu_description,
                "matched_keyword": request.matched_keyword,
                "prompt_context": request.prompt_context,
                "fallback_source": fallback_source,
            }
        )
        fallback_keywords = list(request.draft_keywords)
        keyword_phrase = request.menu_name or ", ".join(fallback_keywords) or "\uc624\ub298 \ub9e4\uc7a5"
        weather_context = _weather_context_for_tests(request.weather_tags)
        caption = (
            f"{weather_context or request.owner_persona} \ubd84\uc704\uae30\uc5d0 {request.owner_persona} \ubb34\ub4dc\ub85c "
            f"{keyword_phrase}\ub97c \uc18c\uac1c\ud574\ubcf4\uc138\uc694."
        )
        guide_text = (
            DEFAULT_FALLBACK_GUIDE_TEXT
            if not fallback_keywords and not request.menu_name
            else f"\uc0ac\uc7a5\ub2d8, {keyword_phrase}\uc774 \ub354 \ubcf4\uc774\ub3c4\ub85d \uc601\uc0c1\uc744 \ucd2c\uc601\ud574\ubcf4\uc138\uc694."
        )
        effective_fallback_source = fallback_source
        if not fallback_keywords and not request.menu_name and effective_fallback_source is None:
            effective_fallback_source = "default_guide"
        return CaptionFallbackResult(
            result=CaptionGenerationResult(
                guide_text=guide_text,
                draft_caption=caption,
            ),
            fallback_source=effective_fallback_source,
        )


class FakeMenuPromotionContextService:
    def __init__(
        self,
        result: MatchedMenuContext | None = None,
    ) -> None:
        self.result = result or MatchedMenuContext()
        self.calls: list[dict[str, object]] = []

    async def match_menu(
        self,
        *,
        store_id,
        draft_keywords: list[str],
    ) -> MatchedMenuContext:
        self.calls.append(
            {
                "store_id": str(store_id),
                "draft_keywords": list(draft_keywords),
            }
        )
        return self.result


class FakeReferenceCaptionRetrieverService:
    def __init__(
        self,
        *,
        references: list[str] | None = None,
        fallback_reason: str | None = None,
        candidate_count: int = 0,
    ) -> None:
        self.references = list(references or [])
        self.fallback_reason = fallback_reason
        self.candidate_count = candidate_count
        self.calls: list[dict[str, object]] = []

    async def retrieve(
        self,
        *,
        owner_persona: str,
        utterance: str,
        canonical_matches: list[CanonicalKeywordMatch],
        max_references: int | None = None,
    ) -> ReferenceCaptionRetrievalResult:
        self.calls.append(
            {
                "owner_persona": owner_persona,
                "utterance": utterance,
                "canonical_matches": list(canonical_matches),
                "max_references": max_references,
            }
        )
        return ReferenceCaptionRetrievalResult(
            references=[
                RetrievedReferenceCaption(
                    caption_id=index + 1,
                    caption_content=reference,
                    score=0.91 - (index * 0.05),
                )
                for index, reference in enumerate(self.references)
            ],
            fallback_reason=self.fallback_reason,
            candidate_count=self.candidate_count,
        )


class StubDraftDownloader:
    is_configured = True

    async def download_draft(self, draft_key: str, destination) -> None:
        _write_test_jpeg(destination)


class RecordingDraftDownloader:
    is_configured = True

    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    async def download_draft(self, draft_key: str, destination) -> None:
        self.calls.append((draft_key, str(destination)))
        _write_test_jpeg(destination)


class DelayedRecordingDraftDownloader:
    is_configured = True

    def __init__(self, delays_by_key: dict[str, float]) -> None:
        self.delays_by_key = delays_by_key
        self.calls: list[tuple[str, str]] = []

    async def download_draft(self, draft_key: str, destination) -> None:
        self.calls.append((draft_key, str(destination)))
        await asyncio.sleep(self.delays_by_key.get(draft_key, 0.0))
        _write_test_jpeg(destination)


class CapturingFinalUploader:
    is_configured = True

    def __init__(self) -> None:
        self.uploaded_paths: list[str] = []
        self.source_paths: list[str] = []

    async def upload_final(self, session_id: str, image_path, final_index: int) -> str:
        self.source_paths.append(str(image_path))
        uploaded_path = f"/ai-finals/{session_id}/final-{final_index:03d}.png"
        self.uploaded_paths.append(uploaded_path)
        return uploaded_path

    async def delete_final(self, uploaded_path: str) -> None:
        return None


class DelayedCapturingFinalUploader:
    is_configured = True

    def __init__(self, delays_by_index: dict[int, float]) -> None:
        self.delays_by_index = delays_by_index
        self.uploaded_paths: list[str] = []
        self.source_paths: list[str] = []

    async def upload_final(self, session_id: str, image_path, final_index: int) -> str:
        self.source_paths.append(str(image_path))
        await asyncio.sleep(self.delays_by_index.get(final_index, 0.0))
        uploaded_path = f"/ai-finals/{session_id}/final-{final_index:03d}.png"
        self.uploaded_paths.append(uploaded_path)
        return uploaded_path

    async def delete_final(self, uploaded_path: str) -> None:
        return None


class FailingPlannerClient:
    last_raw_output = ""
    last_reorder_raw_output = ""
    last_reorder_error_message = None

    async def build_batch_plan(self, image_paths, context) -> FinalEditBatchPlan:
        raise RuntimeError("planner_failed")


class FallbackAgent:
    async def run(self, state: FinalEditImageState) -> FinalEditImageState:
        state.fallback_to_original = True
        state.failure_reason = "tool_failed:denoise:RuntimeError"
        state.output_path = state.image_path
        state.executed_tools = []
        return state


class FakePlannerClient:
    def __init__(
        self,
        plans: list[ImageEditPlan],
        *,
        reorder_plan: CarouselOrderPlan | None = None,
        reorder_error_message: str | None = None,
        reorder_raw_output: str | None = None,
    ) -> None:
        self.plans = plans
        self.reorder_plan = reorder_plan
        self.reorder_error_message = reorder_error_message
        self.last_raw_output = (
            '{"image_plans":[{"image_index":0,"content":"cake","strategy":"warm cafe mood",'
            '"tools":["filter"],"params":{"filter":{"filter_name":"Clarendon"}}}],"carousel_order":null}'
        )
        self.last_reorder_raw_output = reorder_raw_output
        self.last_reorder_error_message = reorder_error_message

    async def build_batch_plan(self, image_paths, context) -> FinalEditBatchPlan:
        reorder_plan = self.reorder_plan
        if reorder_plan is None and len(image_paths) == 3 and self.reorder_error_message is None:
            reorder_plan = CarouselOrderPlan(
                ordered_image_indexes=[0, 1, 2],
                hook_scores=[0.9, 0.7, 0.5],
                overall_confidence=0.0,
                reason="default identity order",
            )
        return FinalEditBatchPlan(
            image_plans=self.plans,
            carousel_order=reorder_plan,
            carousel_order_error_message=self.reorder_error_message,
            carousel_order_raw_output=self.last_reorder_raw_output,
        )


class RecordingPlannerClient:
    def __init__(
        self,
        plans_by_call: list[list[ImageEditPlan]],
        *,
        reorder_plan: CarouselOrderPlan | None = None,
        reorder_error_message: str | None = None,
        reorder_raw_output: str | None = None,
    ) -> None:
        self.plans_by_call = plans_by_call
        self.reorder_plan = reorder_plan
        self.reorder_error_message = reorder_error_message
        self.calls: list[list[str]] = []
        self.last_raw_output = (
            '{"image_plans":[{"image_index":0,"content":"cake","strategy":"warm cafe mood",'
            '"tools":["filter"],"params":{"filter":{"filter_name":"Clarendon"}}}],"carousel_order":null}'
        )
        self.last_reorder_raw_output = reorder_raw_output
        self.last_reorder_error_message = reorder_error_message

    async def build_batch_plan(self, image_paths, context) -> FinalEditBatchPlan:
        self.calls.append(list(image_paths))
        flattened_plans = [
            plan
            for plans in self.plans_by_call
            for plan in plans
        ]
        reorder_plan = self.reorder_plan
        if reorder_plan is None and len(image_paths) == 3 and self.reorder_error_message is None:
            reorder_plan = CarouselOrderPlan(
                ordered_image_indexes=[0, 1, 2],
                hook_scores=[0.9, 0.7, 0.5],
                overall_confidence=0.0,
                reason="default identity order",
            )
        return FinalEditBatchPlan(
            image_plans=flattened_plans,
            carousel_order=reorder_plan,
            carousel_order_error_message=self.reorder_error_message,
            carousel_order_raw_output=self.last_reorder_raw_output,
        )


class FailOnCallPlannerClient(RecordingPlannerClient):
    def __init__(
        self,
        plans_by_call: list[list[ImageEditPlan]],
        *,
        fail_on_call: int,
    ) -> None:
        super().__init__(plans_by_call)
        self.fail_on_call = fail_on_call

    async def build_batch_plan(self, image_paths, context) -> FinalEditBatchPlan:
        self.calls.append(list(image_paths))
        call_index = len(self.calls) - 1
        if call_index == self.fail_on_call:
            raise RuntimeError("planner_failed")
        flattened_plans = [
            plan
            for plans in self.plans_by_call
            for plan in plans
        ]
        return FinalEditBatchPlan(
            image_plans=flattened_plans,
            carousel_order=CarouselOrderPlan(
                ordered_image_indexes=[0, 1, 2],
                hook_scores=[0.9, 0.7, 0.5],
                overall_confidence=0.0,
                reason="default identity order",
            ) if len(image_paths) == 3 else None,
        )


class FailingUploader:
    is_configured = True

    def __init__(self) -> None:
        self.uploaded: list[str] = []
        self.deleted: list[str] = []

    async def upload_final(self, session_id: str, image_path, final_index: int) -> str:
        if final_index == 2:
            raise RuntimeError("upload_failed")
        uploaded_path = f"/ai-finals/{session_id}/final-{final_index:03d}.png"
        self.uploaded.append(uploaded_path)
        return uploaded_path

    async def delete_final(self, uploaded_path: str) -> None:
        self.deleted.append(uploaded_path)


def _find_reference_caption_log_record(records: list[logging.LogRecord]) -> logging.LogRecord:
    return next(
        record
        for record in records
        if getattr(record, "event", "")
        == "session.process_utterance.reference_caption_retrieval.completed"
    )
