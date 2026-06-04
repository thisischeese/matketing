import logging
from typing import TYPE_CHECKING

from fastapi import APIRouter, Depends, HTTPException, Request
from redis.asyncio import Redis

from app.db.redis import get_redis
from app.schemas.sessions import (
    ExtractFramesRequest,
    ExtractFramesResponse,
    FinalEditRequest,
    FinalEditResponse,
    ProcessUtteranceRequest,
    ProcessUtteranceResponse,
)
from app.services.frame_extraction import (
    get_frame_extraction_service,
    process_extract_frames,
)
from app.services.keyword_extraction import (
    KeywordExtractionUnavailableError,
    get_keyword_extraction_service,
)
from app.services.caption_generation import get_caption_generation_service
from app.services.canonical_keyword_resolver import (
    get_canonical_keyword_resolver_service,
)
from app.services.intent_classifier import build_intent_classifier_service
from app.services.menu_promotion_context import (
    get_menu_promotion_context_service,
)
from app.services.reference_caption_retriever import (
    get_reference_caption_retriever_service,
)
from app.services.sessions import process_utterance, resolve_context_routing_variant

if TYPE_CHECKING:
    from app.services.final_edit import FinalEditUnavailableError

router = APIRouter(prefix="/sessions", tags=["ai-sessions"])
logger = logging.getLogger(__name__)


def _final_edit_handlers():
    from app.services.final_edit import (
        FinalEditUnavailableError,
        get_final_edit_service,
        process_final_edit,
    )

    return FinalEditUnavailableError, get_final_edit_service, process_final_edit


@router.post(
    "/{session_id}/process-utterance",
    response_model=ProcessUtteranceResponse,
    response_model_exclude_none=True,
    summary="AI 기반 게시물 캡션 생성",
)
async def process_utterance_endpoint(
    session_id: str,
    payload: ProcessUtteranceRequest,
    request: Request,
    redis: Redis = Depends(get_redis),
) -> ProcessUtteranceResponse:
    context_routing_variant = resolve_context_routing_variant(
        request.headers.get("X-Context-Routing-Variant")
    )
    keyword_service = get_keyword_extraction_service(request)
    caption_service = get_caption_generation_service(request)
    menu_promotion_context_service = get_menu_promotion_context_service(request)
    intent_classifier_service = getattr(
        request.app.state,
        "intent_classifier_service",
        build_intent_classifier_service(),
    )
    canonical_keyword_resolver_service = get_canonical_keyword_resolver_service(request)
    reference_caption_retriever_service = get_reference_caption_retriever_service(
        request
    )
    logger.info(
        "process-utterance request received.",
        extra={
            "session_id": session_id,
            "context_routing_variant": context_routing_variant,
            "payload_owner_persona": payload.owner_persona,
            "payload_weather_cloud_cover": payload.weather.cloud_cover,
            "payload_weather_temperature": payload.weather.temperature,
            "payload_weather_precipitation": payload.weather.precipitation,
            "utterance_length": len(payload.utterance),
        },
    )
    try:
        result = await process_utterance(
            session_id,
            payload,
            redis,
            keyword_service,
            caption_service,
            menu_promotion_context_service,
            intent_classifier_service,
            canonical_keyword_resolver_service,
            reference_caption_retriever_service,
            context_routing_variant=context_routing_variant,
        )
    except KeywordExtractionUnavailableError as exc:
        cause = exc.__cause__
        logger.warning(
            "Keyword extraction failed during process-utterance: %s | cause=%r",
            exc,
            cause,
            extra={"session_id": session_id},
            exc_info=True,
        )
        raise HTTPException(
            status_code=503,
            detail="Keyword extraction is unavailable.",
        ) from exc
    logger.info(
        "process-utterance completed successfully.",
        extra={
            "session_id": session_id,
            "weather_tag_count": len(result.weather_tags),
            "draft_keyword_count": len(result.draft_keywords),
            "final_keyword_count": len(result.final_keywords),
        },
    )
    return ProcessUtteranceResponse(
        session_id=session_id,
        status=result.status,
        guide_text=result.guide_text,
        caption=result.caption,
    )


@router.post(
    "/{session_id}/extract-frames",
    response_model=ExtractFramesResponse,
    summary="가장 점수가 높은 프레임을 추출해 초안으로 업로드",
)
async def extract_frames_endpoint(
    session_id: str,
    payload: ExtractFramesRequest,
    request: Request,
    redis: Redis = Depends(get_redis),
) -> ExtractFramesResponse:
    if str(payload.session_id) != session_id:
        raise HTTPException(
            status_code=422,
            detail="Path session_id and body session_id must match.",
        )

    frame_service = get_frame_extraction_service(request)
    result = await process_extract_frames(session_id, payload, redis, frame_service)
    return ExtractFramesResponse(
        session_id=session_id,
        status=result.status,
        drafts=result.drafts,
    )


@router.post(
    "/{session_id}/final-edit",
    response_model=FinalEditResponse,
    summary="초안 이미지 방향을 보정하고 최종 이미지를 업로드",
)
async def final_edit_endpoint(
    session_id: str,
    payload: FinalEditRequest,
    request: Request,
    redis: Redis = Depends(get_redis),
) -> FinalEditResponse:
    if str(payload.session_id) != session_id:
        raise HTTPException(
            status_code=422,
            detail="Path session_id and body session_id must match.",
        )

    (
        FinalEditUnavailableError,
        get_final_edit_service,
        process_final_edit,
    ) = _final_edit_handlers()
    try:
        final_edit_service = get_final_edit_service(request)
        result = await process_final_edit(session_id, payload, redis, final_edit_service)
    except FinalEditUnavailableError as exc:
        logger.warning(
            "Final edit is unavailable.",
            extra={
                "session_id": session_id,
                "final_edit_unavailable_reason": str(exc),
            },
            exc_info=True,
        )
        raise HTTPException(
            status_code=503,
            detail="Final edit is unavailable.",
        ) from exc
    return FinalEditResponse(
        session_id=session_id,
        status=result.status,
        results=result.results,
    )
