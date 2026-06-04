from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
import logging
from typing import Literal

from langgraph.graph import END, START, StateGraph
from redis.asyncio import Redis

from app.core.config import settings
from app.logging import build_log_extra, preview_text
from app.schemas.sessions import ProcessUtteranceRequest
from app.services.caption_generation import (
    CaptionGenerationRequest,
    CaptionGenerationService,
    CaptionGenerationUnavailableError,
    CaptionPromptContext,
    estimate_prompt_char_count,
    build_weather_context,
)
from app.services.canonical_keyword_resolver import (
    CanonicalKeywordMatch,
    CanonicalKeywordResolution,
    CanonicalKeywordResolverService,
)
from app.services.content_purpose import ContentPurpose, MENU_PROMOTION_PURPOSE
from app.services.intent_classifier import (
    INTENT_CAPTION_GENERATION,
    INTENT_IMAGE_POSTPROCESS,
    INTENT_SHOOTING_GUIDE,
    IntentClassificationResult,
    IntentClassifierService,
)
from app.services.keyword_extraction import (
    KeywordExtractionService,
    KeywordExtractionUnavailableError,
)
from app.services.menu_promotion_context import (
    MatchedMenuContext,
    MenuPromotionContextService,
)
from app.services.reference_caption_retriever import (
    ReferenceCaptionRetrievalResult,
    ReferenceCaptionRetrieverService,
)
from app.services.session_store import RedisSessionStore, session_key
from app.services.weather_tags import evaluate_weather_tags

STATUS_STARTED = "STARTED"
STATUS_TEXT_GENERATED = "TEXT_GENERATED"

HEALTH_CHECK_FALLBACK_SOURCE = "health_check_fallback"
GUIDE_ONLY_FALLBACK_SOURCE = "guide_only"
FINAL_EDIT_CONTEXT_ONLY_SOURCE = "final_edit_context_only"
CONTEXT_ROUTING_VARIANT_BASELINE = "baseline"
CONTEXT_ROUTING_VARIANT_SELECTIVE = "selective"
CONTEXT_PROFILE_BASIC = "basic_context"
CONTEXT_PROFILE_MENU = "menu_context"
CONTEXT_PROFILE_REFERENCE = "reference_context"
HEALTH_CHECK_FAILURE_GUIDE_TEXT = (
    "사장님, 잠시 후 다시 시도해 주세요. "
    "지금은 가게의 분위기와 메뉴가 잘 보이도록 자유롭게 촬영해보세요."
)
HEALTH_CHECK_FAILURE_CAPTION = (
    "지금은 AI 캡션 생성에 필요한 내용이 충분하지 않아 잠시 후 다시 시도해 주세요."
)
FINAL_EDIT_CONTEXT_GUIDE_TEXT = (
    "보정 방향을 반영할 수 있도록 편집할 이미지를 이어서 선택해보세요."
)
logger = logging.getLogger(__name__)


@dataclass
class ProcessUtteranceResult:
    status: str
    purpose: ContentPurpose
    weather_tags: list[str]
    draft_keywords: list[str]
    final_keywords: list[str]
    draft_caption: str
    guide_text: str
    caption: str


@dataclass
class TextGenerationOutcome:
    draft_caption: str
    guide_text: str
    stored_caption: str
    fallback_source: str | None


@dataclass
class ProcessUtteranceServices:
    keyword_service: KeywordExtractionService
    caption_service: CaptionGenerationService
    menu_promotion_context_service: MenuPromotionContextService
    intent_classifier_service: IntentClassifierService
    canonical_keyword_resolver_service: CanonicalKeywordResolverService | None = None
    reference_caption_retriever_service: ReferenceCaptionRetrieverService | None = None


@dataclass
class ProcessUtteranceState:
    session_id: str
    payload: ProcessUtteranceRequest
    redis: Redis
    services: ProcessUtteranceServices
    status: str = STATUS_STARTED
    purpose: ContentPurpose = MENU_PROMOTION_PURPOSE
    weather_tags: list[str] = field(default_factory=list)
    intent: str = INTENT_CAPTION_GENERATION
    intent_result: IntentClassificationResult | None = None
    draft_keywords: list[str] = field(default_factory=list)
    final_keywords: list[str] = field(default_factory=list)
    canonical_resolution: CanonicalKeywordResolution | None = None
    canonical_matches: list[CanonicalKeywordMatch] = field(default_factory=list)
    matched_menu_context: MatchedMenuContext | None = None
    reference_caption_result: ReferenceCaptionRetrievalResult | None = None
    reference_captions: list[str] = field(default_factory=list)
    prompt_context: CaptionPromptContext | None = None
    caption_request: CaptionGenerationRequest | None = None
    context_routing_variant: Literal["baseline", "selective"] = (
        settings.PROCESS_UTTERANCE_CONTEXT_ROUTING_DEFAULT
    )
    context_profile: Literal[
        "basic_context",
        "menu_context",
        "reference_context",
    ] = CONTEXT_PROFILE_BASIC
    guide_text: str = ""
    caption: str = ""
    draft_caption: str = ""
    fallback_source: str | None = None
    graph_path: list[str] = field(default_factory=list)
    skipped_nodes: list[str] = field(default_factory=list)
    context_sources: list[str] = field(default_factory=list)
    metrics: dict[str, int] = field(
        default_factory=lambda: {
            "prompt_char_count": 0,
            "reference_caption_count": 0,
            "model_call_count": 0,
            "external_dependency_calls": 0,
        }
    )
    debug_fields: dict[str, str] = field(default_factory=dict)
    health_check_failed: bool = False
    persisted: bool = False


async def upsert_content_session(
    redis: Redis,
    session_id: str,
    scalar_fields: dict[str, str],
    draft_keywords: list[str] | None = None,
    final_keywords: list[str] | None = None,
    weather_tags: list[str] | None = None,
    drafts: list[str] | None = None,
    photos: list[str] | None = None,
    debug_fields: dict[str, str] | None = None,
) -> None:
    await RedisSessionStore(redis).upsert_content_session(
        session_id=session_id,
        scalar_fields=scalar_fields,
        draft_keywords=draft_keywords,
        final_keywords=final_keywords,
        weather_tags=weather_tags,
        drafts=drafts,
        photos=photos,
        debug_fields=debug_fields,
    )


def _session_store(redis: Redis) -> RedisSessionStore:
    return RedisSessionStore(redis)


async def _persist_process_utterance_started(
    store: RedisSessionStore,
    session_id: str,
    utterance: str,
    owner_persona: str,
    weather_tags: list[str],
) -> None:
    await store.upsert_content_session(
        session_id=session_id,
        scalar_fields={
            "caption": "",
            "owner_persona": owner_persona,
            "utterance": utterance,
        },
        weather_tags=weather_tags,
    )
    logger.info(
        "Stored initial process-utterance request fields in redis.",
        extra=build_log_extra(
            "session.process_utterance.redis_store_request.completed",
            component="session",
            stage="persist_redis",
            session_id=session_id,
            outcome="succeeded",
            utterance_length=len(utterance),
            weather_tag_count=len(weather_tags),
            initialized_field_count=2,
        ),
    )


def _build_keyword_state(
    extraction_result,
    session_id: str,
) -> list[str]:
    draft_keywords = list(extraction_result.draft_keywords)
    logger.info(
        "Keyword extraction finished.",
        extra=build_log_extra(
            "session.process_utterance.keyword_extraction.completed",
            component="session",
            stage="keyword_extraction",
            session_id=session_id,
            outcome="succeeded",
            purpose=MENU_PROMOTION_PURPOSE,
            draft_keyword_count=len(draft_keywords),
            draft_keywords_preview=", ".join(draft_keywords[:3]),
        ),
    )
    return draft_keywords


def _build_final_keyword_state(
    draft_keywords: list[str],
    session_id: str,
) -> list[str]:
    final_keywords = list(draft_keywords)
    logger.info(
        "Final keyword state fixed to draft keywords for menu promotion.",
        extra=build_log_extra(
            "session.process_utterance.final_keyword_state.completed",
            component="session",
            stage="final_keyword_state",
            session_id=session_id,
            outcome="succeeded",
            draft_keyword_count=len(draft_keywords),
            draft_keywords_preview=", ".join(draft_keywords[:3]),
            final_keyword_count=len(final_keywords),
            final_keywords_preview=", ".join(final_keywords[:3]),
        ),
    )
    return final_keywords


def _build_process_utterance_debug_fields(
    matched_menu_context: MatchedMenuContext | None = None,
) -> dict[str, str]:
    debug_fields = {
        "debug:purpose": MENU_PROMOTION_PURPOSE,
    }
    if matched_menu_context is not None:
        debug_fields["debug:menu_match_source"] = matched_menu_context.match_source
        debug_fields["debug:menu_candidate_count"] = (
            "1" if matched_menu_context.matched else "0"
        )
        if matched_menu_context.matched_keyword is not None:
            debug_fields["debug:matched_keyword"] = matched_menu_context.matched_keyword
        if matched_menu_context.menu_name is not None:
            debug_fields["debug:matched_menu_name"] = matched_menu_context.menu_name
    return debug_fields


async def _persist_process_utterance_result(
    store: RedisSessionStore,
    session_id: str,
    owner_persona: str,
    caption: str,
    weather_tags: list[str],
    draft_keywords: list[str],
    final_keywords: list[str],
    debug_fields: dict[str, str],
) -> None:
    await store.delete_fields(
        session_key(session_id),
        [
            "session_id",
            "store_id",
            "weather_condition",
            "weather_temperature",
            "date",
            "expires_at",
        ],
    )
    await store.upsert_content_session(
        session_id=session_id,
        scalar_fields={
            "status": STATUS_TEXT_GENERATED,
            "caption": caption,
            "owner_persona": owner_persona,
        },
        weather_tags=weather_tags,
        draft_keywords=draft_keywords,
        final_keywords=final_keywords,
        debug_fields=debug_fields,
    )
    logger.info(
        "Stored process-utterance result in redis.",
        extra=build_log_extra(
            "session.process_utterance.redis_store.completed",
            component="session",
            stage="persist_redis",
            session_id=session_id,
            outcome="succeeded",
            caption_length=len(caption),
        ),
    )


async def _build_health_check_fallback_result(
    store: RedisSessionStore,
    session_id: str,
    utterance: str,
    owner_persona: str,
    weather_tags: list[str],
    *,
    keyword_ok: bool,
    caption_ok: bool,
) -> ProcessUtteranceResult:
    logger.warning(
        "Pre-flight health check failed; returning fixed fallback caption.",
        extra=build_log_extra(
            "session.process_utterance.health_check.failed",
            component="session",
            stage="health_check",
            session_id=session_id,
            outcome="failed",
            keyword_server_healthy=keyword_ok,
            caption_server_healthy=caption_ok,
        ),
    )
    debug_fields = {
        "debug:purpose": MENU_PROMOTION_PURPOSE,
        "debug:text_generation_fallback_source": HEALTH_CHECK_FALLBACK_SOURCE,
        "debug:health_check_keyword_ok": "true" if keyword_ok else "false",
        "debug:health_check_caption_ok": "true" if caption_ok else "false",
    }
    await store.upsert_content_session(
        session_id=session_id,
        scalar_fields={
            "status": STATUS_TEXT_GENERATED,
            "caption": HEALTH_CHECK_FAILURE_CAPTION,
            "owner_persona": owner_persona,
            "utterance": utterance,
        },
        weather_tags=weather_tags,
        draft_keywords=[],
        final_keywords=[],
        debug_fields=debug_fields,
    )
    return ProcessUtteranceResult(
        status=STATUS_TEXT_GENERATED,
        purpose=MENU_PROMOTION_PURPOSE,
        weather_tags=weather_tags,
        draft_keywords=[],
        final_keywords=[],
        draft_caption=HEALTH_CHECK_FAILURE_CAPTION,
        guide_text=HEALTH_CHECK_FAILURE_GUIDE_TEXT,
        caption=HEALTH_CHECK_FAILURE_CAPTION,
    )


async def _run_preflight_health_checks(
    keyword_service: KeywordExtractionService,
    caption_service: CaptionGenerationService,
) -> tuple[bool, bool]:
    return await asyncio.gather(
        keyword_service.is_healthy(),
        caption_service.is_healthy(),
    )


def _log_keyword_extraction_start(
    session_id: str,
    payload: ProcessUtteranceRequest,
    weather_tags: list[str],
) -> None:
    logger.info(
        "Starting keyword extraction for process-utterance.",
        extra=build_log_extra(
            "session.process_utterance.keyword_extraction.started",
            component="session",
            stage="keyword_extraction",
            session_id=session_id,
            outcome="started",
            owner_persona=payload.owner_persona,
            weather_cloud_cover=payload.weather.cloud_cover,
            weather_tag_count=len(weather_tags),
            weather_tags_preview=", ".join(weather_tags[:3]),
            utterance_length=len(payload.utterance),
            utterance_preview=preview_text(
                payload.utterance,
                settings.LOG_EVENT_PREVIEW_MAX_LEN,
            )
            if settings.LOG_INCLUDE_RAW_IDENTIFIERS
            else None,
        ),
    )


def _fallback_text_generation_outcome(
    caption_service: CaptionGenerationService,
    caption_request: CaptionGenerationRequest,
    fallback_source: str | None,
) -> TextGenerationOutcome:
    fallback_result = caption_service.build_fallback_result(
        caption_request,
        fallback_source=fallback_source,
    )
    return TextGenerationOutcome(
        draft_caption=fallback_result.result.draft_caption,
        guide_text=fallback_result.result.guide_text,
        stored_caption=fallback_result.result.stored_caption,
        fallback_source=fallback_result.fallback_source,
    )


def resolve_context_routing_variant(
    requested_variant: str | None,
) -> Literal["baseline", "selective"]:
    normalized = (requested_variant or "").strip().lower()
    if normalized in {
        CONTEXT_ROUTING_VARIANT_BASELINE,
        CONTEXT_ROUTING_VARIANT_SELECTIVE,
    }:
        return normalized  # type: ignore[return-value]
    return settings.PROCESS_UTTERANCE_CONTEXT_ROUTING_DEFAULT


def _combine_prompt_menu_description(
    menu_description: str | None,
    reference_captions: list[str],
) -> str | None:
    parts: list[str] = []
    if menu_description:
        parts.append(menu_description.strip())
    normalized_references = [
        preview_text(reference.strip(), 120)
        for reference in reference_captions[:2]
        if reference.strip()
    ]
    if normalized_references:
        parts.append("reference_captions: " + " | ".join(normalized_references))
    combined = " ".join(part for part in parts if part).strip()
    return combined or None


class ProcessUtteranceGraphRunner:
    def __init__(self) -> None:
        self.graph = self._build_graph()

    def _build_graph(self):
        graph = StateGraph(ProcessUtteranceState)
        graph.add_node("preflight_health_check", self._preflight_health_check)
        graph.add_node("classify_intent", self._classify_intent)
        graph.add_node("extract_keywords", self._extract_keywords)
        graph.add_node("fetch_menu_context", self._fetch_menu_context)
        graph.add_node("fetch_reference_captions", self._fetch_reference_captions)
        graph.add_node(
            "build_caption_prompt_context",
            self._build_caption_prompt_context,
        )
        graph.add_node("generate_caption", self._generate_caption)
        graph.add_node("build_guide_only", self._build_guide_only)
        graph.add_node(
            "route_to_final_edit_context",
            self._route_to_final_edit_context,
        )
        graph.add_node("finalize_and_persist", self._finalize_and_persist)
        graph.add_edge(START, "preflight_health_check")
        graph.add_conditional_edges(
            "preflight_health_check",
            self._route_after_preflight,
            {
                "classify_intent": "classify_intent",
                "finalize_and_persist": "finalize_and_persist",
            },
        )
        graph.add_conditional_edges(
            "classify_intent",
            self._route_after_classification,
            {
                "extract_keywords": "extract_keywords",
                "build_guide_only": "build_guide_only",
                "route_to_final_edit_context": "route_to_final_edit_context",
            },
        )
        graph.add_edge("extract_keywords", "fetch_menu_context")
        graph.add_conditional_edges(
            "fetch_menu_context",
            self._route_after_menu_context,
            {
                "fetch_reference_captions": "fetch_reference_captions",
                "build_caption_prompt_context": "build_caption_prompt_context",
            },
        )
        graph.add_edge("fetch_reference_captions", "build_caption_prompt_context")
        graph.add_edge("build_caption_prompt_context", "generate_caption")
        graph.add_edge("generate_caption", "finalize_and_persist")
        graph.add_edge("build_guide_only", "finalize_and_persist")
        graph.add_edge("route_to_final_edit_context", "finalize_and_persist")
        graph.add_edge("finalize_and_persist", END)
        return graph.compile()

    async def _preflight_health_check(
        self,
        state: ProcessUtteranceState,
    ) -> ProcessUtteranceState:
        state.graph_path.append("preflight_health_check")
        state.weather_tags = evaluate_weather_tags(
            state.payload.weather,
            target_date=state.payload.date,
        )
        keyword_ok, caption_ok = await _run_preflight_health_checks(
            state.services.keyword_service,
            state.services.caption_service,
        )
        state.metrics["external_dependency_calls"] += 2
        if not (keyword_ok and caption_ok):
            state.health_check_failed = True
            state.fallback_source = HEALTH_CHECK_FALLBACK_SOURCE
            state.status = STATUS_TEXT_GENERATED
            state.guide_text = HEALTH_CHECK_FAILURE_GUIDE_TEXT
            state.caption = HEALTH_CHECK_FAILURE_CAPTION
            state.draft_caption = HEALTH_CHECK_FAILURE_CAPTION
            state.debug_fields.update(
                {
                    "debug:purpose": MENU_PROMOTION_PURPOSE,
                    "debug:context_routing_variant": state.context_routing_variant,
                    "debug:context_profile": state.context_profile,
                    "debug:text_generation_fallback_source": HEALTH_CHECK_FALLBACK_SOURCE,
                    "debug:health_check_keyword_ok": "true" if keyword_ok else "false",
                    "debug:health_check_caption_ok": "true" if caption_ok else "false",
                }
            )
            await upsert_content_session(
                state.redis,
                state.session_id,
                scalar_fields={
                    "status": STATUS_TEXT_GENERATED,
                    "caption": HEALTH_CHECK_FAILURE_CAPTION,
                    "owner_persona": state.payload.owner_persona,
                    "utterance": state.payload.utterance,
                },
                weather_tags=state.weather_tags,
                draft_keywords=[],
                final_keywords=[],
                debug_fields=state.debug_fields,
            )
            state.persisted = True
        else:
            await _persist_process_utterance_started(
                _session_store(state.redis),
                state.session_id,
                state.payload.utterance,
                state.payload.owner_persona,
                state.weather_tags,
            )
        return state

    def _route_after_preflight(
        self,
        state: ProcessUtteranceState,
    ) -> Literal["classify_intent", "finalize_and_persist"]:
        if state.health_check_failed:
            return "finalize_and_persist"
        return "classify_intent"

    async def _classify_intent(
        self,
        state: ProcessUtteranceState,
    ) -> ProcessUtteranceState:
        state.graph_path.append("classify_intent")
        state.intent_result = state.services.intent_classifier_service.classify(
            state.payload.utterance
        )
        state.intent = state.intent_result.intent
        logger.info(
            "Intent classification completed.",
            extra=build_log_extra(
                "session.process_utterance.intent_classification.completed",
                component="session",
                stage="intent_classification",
                session_id=state.session_id,
                outcome="succeeded",
                intent=state.intent,
                intent_rule=state.intent_result.rule,
                intent_confidence=state.intent_result.confidence,
            ),
        )
        return state

    def _route_after_classification(
        self,
        state: ProcessUtteranceState,
    ) -> Literal[
        "extract_keywords",
        "build_guide_only",
        "route_to_final_edit_context",
    ]:
        if state.intent == INTENT_SHOOTING_GUIDE:
            self._mark_skipped_node(state, "extract_keywords", "guide_path")
            self._mark_skipped_node(state, "fetch_menu_context", "guide_path")
            self._mark_skipped_node(state, "fetch_reference_captions", "guide_path")
            self._mark_skipped_node(
                state,
                "build_caption_prompt_context",
                "guide_path",
            )
            self._mark_skipped_node(state, "generate_caption", "guide_path")
            route = "build_guide_only"
        elif state.intent == INTENT_IMAGE_POSTPROCESS:
            self._mark_skipped_node(state, "extract_keywords", "image_postprocess_path")
            self._mark_skipped_node(state, "fetch_menu_context", "image_postprocess_path")
            self._mark_skipped_node(
                state,
                "fetch_reference_captions",
                "image_postprocess_path",
            )
            self._mark_skipped_node(
                state,
                "build_caption_prompt_context",
                "image_postprocess_path",
            )
            self._mark_skipped_node(state, "generate_caption", "image_postprocess_path")
            route = "route_to_final_edit_context"
        else:
            self._mark_skipped_node(state, "build_guide_only", "caption_path")
            self._mark_skipped_node(
                state,
                "route_to_final_edit_context",
                "caption_path",
            )
            route = "extract_keywords"
        logger.info(
            "Graph route selected.",
            extra=build_log_extra(
                "session.process_utterance.graph_route.completed",
                component="session",
                stage="graph_route",
                session_id=state.session_id,
                outcome="succeeded",
                intent=state.intent,
                route=route,
            ),
        )
        return route

    async def _extract_keywords(
        self,
        state: ProcessUtteranceState,
    ) -> ProcessUtteranceState:
        state.graph_path.append("extract_keywords")
        _log_keyword_extraction_start(
            state.session_id,
            state.payload,
            state.weather_tags,
        )
        extraction_result = await state.services.keyword_service.extract_keywords(
            state.payload.utterance
        )
        state.metrics["external_dependency_calls"] += 1
        state.draft_keywords = _build_keyword_state(
            extraction_result=extraction_result,
            session_id=state.session_id,
        )
        state.final_keywords = _build_final_keyword_state(
            draft_keywords=state.draft_keywords,
            session_id=state.session_id,
        )
        resolver = state.services.canonical_keyword_resolver_service
        if resolver is not None:
            state.canonical_resolution = await resolver.resolve_keywords(
                state.draft_keywords
            )
            state.metrics["external_dependency_calls"] += 1
            state.final_keywords = list(state.canonical_resolution.final_keywords)
            state.canonical_matches = list(state.canonical_resolution.matches)
        return state

    async def _fetch_menu_context(
        self,
        state: ProcessUtteranceState,
    ) -> ProcessUtteranceState:
        state.graph_path.append("fetch_menu_context")
        if not state.draft_keywords:
            self._mark_skipped_node(state, "fetch_menu_context", "no_draft_keywords")
            return state
        matched_menu_context = await state.services.menu_promotion_context_service.match_menu(
            store_id=state.payload.store_id,
            draft_keywords=state.draft_keywords,
        )
        state.metrics["external_dependency_calls"] += 1
        state.matched_menu_context = matched_menu_context
        logger.info(
            "Matched menu retrieval completed.",
            extra=build_log_extra(
                "session.process_utterance.menu_match.completed",
                component="session",
                stage="menu_match",
                session_id=state.session_id,
                outcome="succeeded" if matched_menu_context.matched else "skipped",
                menu_match_source=matched_menu_context.match_source,
                matched_keyword=matched_menu_context.matched_keyword,
                matched_menu_name=matched_menu_context.menu_name,
            ),
        )
        return state

    def _route_after_menu_context(
        self,
        state: ProcessUtteranceState,
    ) -> Literal["fetch_reference_captions", "build_caption_prompt_context"]:
        if state.intent != INTENT_CAPTION_GENERATION:
            self._mark_skipped_node(state, "fetch_reference_captions", "non_caption_path")
            return "build_caption_prompt_context"
        if state.services.reference_caption_retriever_service is None:
            self._mark_skipped_node(state, "fetch_reference_captions", "retriever_missing")
            return "build_caption_prompt_context"
        if not any(
            match.canonical_keyword_id is not None and match.matched
            for match in state.canonical_matches
        ):
            self._mark_skipped_node(
                state,
                "fetch_reference_captions",
                "no_canonical_keyword_ids",
            )
            return "build_caption_prompt_context"
        if (
            state.matched_menu_context is not None
            and state.matched_menu_context.matched
            and bool(state.matched_menu_context.menu_description)
        ):
            self._mark_skipped_node(
                state,
                "fetch_reference_captions",
                "menu_context_sufficient",
            )
            return "build_caption_prompt_context"
        return "fetch_reference_captions"

    async def _fetch_reference_captions(
        self,
        state: ProcessUtteranceState,
    ) -> ProcessUtteranceState:
        state.graph_path.append("fetch_reference_captions")
        retriever = state.services.reference_caption_retriever_service
        if retriever is None:
            self._mark_skipped_node(state, "fetch_reference_captions", "retriever_missing")
            return state
        state.reference_caption_result = await retriever.retrieve(
            owner_persona=state.payload.owner_persona,
            utterance=state.payload.utterance,
            canonical_matches=state.canonical_matches,
        )
        state.metrics["external_dependency_calls"] += 1
        state.reference_captions = list(state.reference_caption_result.captions)
        state.metrics["reference_caption_count"] = len(state.reference_captions)
        logger.info(
            "Reference caption retrieval completed.",
            extra=build_log_extra(
                "session.process_utterance.reference_caption_retrieval.completed",
                component="session",
                stage="reference_caption_retrieval",
                session_id=state.session_id,
                outcome=(
                    "succeeded"
                    if state.reference_captions
                    else "fallback"
                ),
                reference_caption_count=len(state.reference_captions),
                reference_caption_candidate_count=(
                    state.reference_caption_result.candidate_count
                ),
                reference_caption_fallback_reason=(
                    state.reference_caption_result.fallback_reason
                ),
            ),
        )
        return state

    def _resolve_context_profile(
        self,
        state: ProcessUtteranceState,
        matched_menu: MatchedMenuContext,
    ) -> Literal["basic_context", "menu_context", "reference_context"]:
        if matched_menu.matched and bool(matched_menu.menu_description):
            return CONTEXT_PROFILE_MENU
        if state.reference_captions:
            return CONTEXT_PROFILE_REFERENCE
        return CONTEXT_PROFILE_BASIC

    async def _build_caption_prompt_context(
        self,
        state: ProcessUtteranceState,
    ) -> ProcessUtteranceState:
        state.graph_path.append("build_caption_prompt_context")
        matched_menu = state.matched_menu_context or MatchedMenuContext()
        context_profile = self._resolve_context_profile(state, matched_menu)
        combined_menu_description = _combine_prompt_menu_description(
            matched_menu.menu_description,
            state.reference_captions,
        )
        prompt_menu_description = combined_menu_description
        prompt_matched_keyword = matched_menu.matched_keyword
        prompt_menu_name = matched_menu.menu_name
        prompt_reference_captions = list(state.reference_captions[:2])
        if state.context_routing_variant == CONTEXT_ROUTING_VARIANT_SELECTIVE:
            if context_profile == CONTEXT_PROFILE_MENU:
                prompt_reference_captions = []
            elif context_profile == CONTEXT_PROFILE_REFERENCE:
                prompt_matched_keyword = None
                prompt_menu_name = None
                prompt_menu_description = None
            else:
                prompt_matched_keyword = None
                prompt_menu_name = None
                prompt_menu_description = None
                prompt_reference_captions = []
        state.context_profile = context_profile
        state.prompt_context = CaptionPromptContext(
            owner_persona=state.payload.owner_persona,
            today=state.payload.date.isoformat(),
            weather_context=build_weather_context(state.weather_tags),
            utterance_summary=state.payload.utterance,
            context_profile=context_profile,
            draft_keywords=list(state.draft_keywords),
            matched_keyword=prompt_matched_keyword,
            menu_name=prompt_menu_name,
            menu_description=prompt_menu_description,
            reference_captions=prompt_reference_captions,
        )
        state.caption_request = CaptionGenerationRequest(
            draft_keywords=list(state.draft_keywords),
            owner_persona=state.payload.owner_persona,
            context_routing_variant=state.context_routing_variant,
            context_profile=context_profile,
            today=state.payload.date.isoformat(),
            utterance=state.prompt_context.utterance_summary,
            weather_tags=list(state.weather_tags),
            menu_name=matched_menu.menu_name,
            menu_description=combined_menu_description,
            matched_keyword=matched_menu.matched_keyword,
            prompt_context=state.prompt_context,
        )
        context_sources = ["draft_keywords", "weather"]
        if context_profile == CONTEXT_PROFILE_MENU:
            context_sources.append("menu_context")
        if context_profile == CONTEXT_PROFILE_REFERENCE:
            context_sources.append("reference_captions")
        state.context_sources = context_sources
        state.metrics["prompt_char_count"] = estimate_prompt_char_count(
            state.caption_request
        )
        logger.info(
            "Caption prompt context assembled.",
            extra=build_log_extra(
                "session.process_utterance.context_assembly.completed",
                component="session",
                stage="context_assembly",
                session_id=state.session_id,
                outcome="succeeded",
                intent=state.intent,
                context_routing_variant=state.context_routing_variant,
                context_profile=context_profile,
                context_sources=context_sources,
                prompt_char_count=state.metrics["prompt_char_count"],
                reference_caption_count=len(state.reference_captions),
            ),
        )
        return state

    async def _generate_caption(
        self,
        state: ProcessUtteranceState,
    ) -> ProcessUtteranceState:
        state.graph_path.append("generate_caption")
        assert state.caption_request is not None
        if not state.caption_request.draft_keywords:
            outcome = _fallback_text_generation_outcome(
                state.services.caption_service,
                state.caption_request,
                fallback_source=None,
            )
            logger.info(
                "Built text generation result from default guide fallback.",
                extra=build_log_extra(
                    "session.process_utterance.text_generation.completed",
                    component="session",
                    stage="text_generation",
                    session_id=state.session_id,
                    outcome="fallback",
                    text_generation_source="default_guide",
                    purpose=MENU_PROMOTION_PURPOSE,
                    weather_tag_count=len(state.weather_tags),
                    weather_tags_preview=", ".join(state.weather_tags[:3]),
                    draft_caption_length=len(outcome.draft_caption),
                    guide_text_length=len(outcome.guide_text),
                ),
            )
        else:
            try:
                state.metrics["model_call_count"] += 1
                state.metrics["external_dependency_calls"] += 1
                result = await state.services.caption_service.generate_text(
                    state.caption_request
                )
                outcome = TextGenerationOutcome(
                    draft_caption=result.draft_caption,
                    guide_text=result.guide_text,
                    stored_caption=result.stored_caption,
                    fallback_source=None,
                )
                logger.info(
                    "Built text generation result from caption model.",
                    extra=build_log_extra(
                        "session.process_utterance.text_generation.completed",
                        component="session",
                        stage="text_generation",
                        session_id=state.session_id,
                        outcome="succeeded",
                        text_generation_source="caption_model",
                        purpose=MENU_PROMOTION_PURPOSE,
                        weather_tag_count=len(state.weather_tags),
                        weather_tags_preview=", ".join(state.weather_tags[:3]),
                        draft_caption_length=len(outcome.draft_caption),
                        guide_text_length=len(outcome.guide_text),
                    ),
                )
            except CaptionGenerationUnavailableError as exc:
                outcome = _fallback_text_generation_outcome(
                    state.services.caption_service,
                    state.caption_request,
                    fallback_source="caption_model_fallback",
                )
                logger.warning(
                    "Caption model generation failed; falling back to rule-based text generation: %s",
                    exc,
                    exc_info=True,
                    extra=build_log_extra(
                        "session.process_utterance.text_generation.completed",
                        component="session",
                        stage="text_generation",
                        session_id=state.session_id,
                        outcome="fallback",
                        text_generation_source="rule_based_fallback",
                        error_type=exc.__class__.__name__,
                        purpose=MENU_PROMOTION_PURPOSE,
                        weather_tag_count=len(state.weather_tags),
                        weather_tags_preview=", ".join(state.weather_tags[:3]),
                        draft_caption_length=len(outcome.draft_caption),
                        guide_text_length=len(outcome.guide_text),
                    ),
                )
        state.guide_text = outcome.guide_text
        state.caption = outcome.stored_caption
        state.draft_caption = outcome.draft_caption
        state.fallback_source = outcome.fallback_source
        state.status = STATUS_TEXT_GENERATED
        return state

    async def _build_guide_only(
        self,
        state: ProcessUtteranceState,
    ) -> ProcessUtteranceState:
        state.graph_path.append("build_guide_only")
        caption_request = CaptionGenerationRequest(
            draft_keywords=[],
            owner_persona=state.payload.owner_persona,
            today=state.payload.date.isoformat(),
            utterance=state.payload.utterance,
            weather_tags=list(state.weather_tags),
        )
        outcome = _fallback_text_generation_outcome(
            state.services.caption_service,
            caption_request,
            fallback_source=GUIDE_ONLY_FALLBACK_SOURCE,
        )
        state.guide_text = outcome.guide_text
        state.caption = outcome.stored_caption
        state.draft_caption = outcome.draft_caption
        state.fallback_source = outcome.fallback_source
        state.context_sources = ["weather", "utterance"]
        state.status = STATUS_TEXT_GENERATED
        return state

    async def _route_to_final_edit_context(
        self,
        state: ProcessUtteranceState,
    ) -> ProcessUtteranceState:
        state.graph_path.append("route_to_final_edit_context")
        state.guide_text = FINAL_EDIT_CONTEXT_GUIDE_TEXT
        state.caption = state.payload.utterance.strip()
        state.draft_caption = state.caption
        state.fallback_source = FINAL_EDIT_CONTEXT_ONLY_SOURCE
        state.context_sources = ["utterance"]
        state.status = STATUS_TEXT_GENERATED
        return state

    async def _finalize_and_persist(
        self,
        state: ProcessUtteranceState,
    ) -> ProcessUtteranceState:
        state.graph_path.append("finalize_and_persist")
        if not state.persisted:
            debug_fields = _build_process_utterance_debug_fields(
                matched_menu_context=state.matched_menu_context,
            )
            debug_fields.update(state.debug_fields)
            debug_fields["debug:intent"] = state.intent
            debug_fields["debug:context_routing_variant"] = (
                state.context_routing_variant
            )
            debug_fields["debug:context_profile"] = state.context_profile
            debug_fields["debug:graph_path"] = ",".join(state.graph_path)
            debug_fields["debug:skipped_nodes"] = ",".join(state.skipped_nodes)
            debug_fields["debug:context_sources"] = ",".join(state.context_sources)
            debug_fields["debug:prompt_char_count"] = str(
                state.metrics["prompt_char_count"]
            )
            debug_fields["debug:reference_caption_count"] = str(
                state.metrics["reference_caption_count"]
            )
            debug_fields["debug:model_call_count"] = str(
                state.metrics["model_call_count"]
            )
            debug_fields["debug:external_dependency_calls"] = str(
                state.metrics["external_dependency_calls"]
            )
            if state.fallback_source is not None:
                debug_fields["debug:text_generation_fallback_source"] = (
                    state.fallback_source
                )
            await _persist_process_utterance_result(
                store=_session_store(state.redis),
                session_id=state.session_id,
                owner_persona=state.payload.owner_persona,
                caption=state.caption,
                weather_tags=state.weather_tags,
                draft_keywords=state.draft_keywords,
                final_keywords=state.final_keywords,
                debug_fields=debug_fields,
            )
            state.persisted = True
            state.debug_fields = debug_fields
        logger.info(
            "Process utterance graph completed.",
            extra=build_log_extra(
                "session.process_utterance.graph.completed",
                component="session",
                stage="graph_complete",
                session_id=state.session_id,
                outcome="succeeded",
                intent=state.intent,
                graph_path=state.graph_path,
                skipped_nodes=state.skipped_nodes,
            ),
        )
        return state

    def _mark_skipped_node(
        self,
        state: ProcessUtteranceState,
        node_name: str,
        reason: str,
    ) -> None:
        state.skipped_nodes.append(node_name)
        logger.info(
            "Graph node skipped.",
            extra=build_log_extra(
                "session.process_utterance.node_skipped",
                component="session",
                stage="graph_route",
                session_id=state.session_id,
                outcome="skipped",
                node_name=node_name,
                skip_reason=reason,
            ),
        )

    async def run(
        self,
        state: ProcessUtteranceState,
    ) -> ProcessUtteranceResult:
        result = await self.graph.ainvoke(state)
        final_state = result if isinstance(result, ProcessUtteranceState) else ProcessUtteranceState(**result)
        return ProcessUtteranceResult(
            status=STATUS_TEXT_GENERATED,
            purpose=MENU_PROMOTION_PURPOSE,
            weather_tags=list(final_state.weather_tags),
            draft_keywords=list(final_state.draft_keywords),
            final_keywords=list(final_state.final_keywords),
            draft_caption=final_state.draft_caption,
            guide_text=final_state.guide_text,
            caption=final_state.caption,
        )


async def process_utterance(
    session_id: str,
    payload: ProcessUtteranceRequest,
    redis: Redis,
    keyword_service: KeywordExtractionService,
    caption_service: CaptionGenerationService,
    menu_promotion_context_service: MenuPromotionContextService,
    intent_classifier_service: IntentClassifierService,
    canonical_keyword_resolver_service: CanonicalKeywordResolverService | None = None,
    reference_caption_retriever_service: ReferenceCaptionRetrieverService | None = None,
    context_routing_variant: Literal["baseline", "selective"] | None = None,
) -> ProcessUtteranceResult:
    runner = ProcessUtteranceGraphRunner()
    state = ProcessUtteranceState(
        session_id=session_id,
        payload=payload,
        redis=redis,
        context_routing_variant=resolve_context_routing_variant(context_routing_variant),
        services=ProcessUtteranceServices(
            keyword_service=keyword_service,
            caption_service=caption_service,
            menu_promotion_context_service=menu_promotion_context_service,
            intent_classifier_service=intent_classifier_service,
            canonical_keyword_resolver_service=canonical_keyword_resolver_service,
            reference_caption_retriever_service=reference_caption_retriever_service,
        ),
    )
    try:
        return await runner.run(state)
    except KeywordExtractionUnavailableError:
        raise
