from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
import logging
from pathlib import Path
import shutil
from tempfile import mkdtemp

from fastapi import FastAPI

from app.core.config import settings
from app.db.postgres import dispose_engine
from app.db.redis import close_redis, get_redis_client
from app.logging import build_log_extra
from app.perfectframe.dependencies import get_dependencies
from app.perfectframe.extractors import BestFrameExtractor
from app.perfectframe.image_evaluators import NIMAEvaluator, build_image_evaluator
from app.perfectframe.schemas import ExtractorConfig
from app.perfectframe.weights import (
    NimaWeightsUnavailableError,
    ensure_nima_weights_available,
)
from app.services.canonical_keyword_resolver import (
    build_canonical_keyword_resolver_service,
)
from app.services.caption_generation import (
    CaptionGenerationUnavailableError,
    build_caption_generation_service,
)
from app.services.final_edit_runtime import import_cv2, summarize_unavailable_reason
from app.services.final_edit_upscaler import ensure_final_edit_upscaler_available
from app.services.frame_extraction import (
    FrameExtractionService,
    S3DraftUploader,
    S3VideoDownloader,
)
from app.services.intent_classifier import build_intent_classifier_service
from app.services.keyword_extraction import (
    KeywordExtractionUnavailableError,
    build_keyword_extraction_service,
)
from app.services.menu_promotion_context import (
    build_menu_promotion_context_service,
)
from app.services.reference_caption_retriever import (
    build_reference_caption_retriever_service,
)

logger = logging.getLogger(__name__)


def _build_temp_root() -> Path:
    temp_root = Path(mkdtemp(prefix="ai-frame-extractor-"))
    logger.info(
        "AI service startup initialized temporary workspace.",
        extra=build_log_extra(
            "app.startup.temp_root_created",
            component="startup",
            outcome="started",
            temp_root=str(temp_root),
        ),
    )
    return temp_root


async def _build_image_evaluator(app: FastAPI) -> object:
    nima_weights_path: Path | None = None
    try:
        nima_weights_path = await ensure_nima_weights_available()
        logger.info(
            "NIMA weights ready.",
            extra=build_log_extra(
                "app.startup.nima_weights.ready",
                component="startup",
                stage="nima_weights_prepare",
                outcome="succeeded",
                weights_path=str(nima_weights_path),
            ),
        )
    except NimaWeightsUnavailableError:
        logger.warning(
            "NIMA weights unavailable; sharpness fallback will be used.",
            exc_info=True,
            extra=build_log_extra(
                "app.startup.nima_weights.unavailable",
                component="startup",
                stage="nima_weights_prepare",
                outcome="failed",
                error_type="nima_weights_unavailable",
            ),
        )

    image_evaluator = build_image_evaluator(nima_weights_path)
    app.state.frame_evaluator_kind = (
        "nima" if isinstance(image_evaluator, NIMAEvaluator) else "sharpness"
    )
    logger.info(
        "Frame evaluator selected.",
        extra=build_log_extra(
            "app.startup.frame_evaluator_selected",
            component="startup",
            stage="frame_evaluator_select",
            outcome="succeeded",
            evaluator_kind=app.state.frame_evaluator_kind,
        ),
    )
    return image_evaluator


def _initialize_frame_extraction(app: FastAPI, temp_root: Path) -> None:
    extractor_config = ExtractorConfig(
        input_directory=temp_root,
        output_directory=temp_root,
    )
    dependencies = get_dependencies(extractor_config)
    app.state.frame_extractor_config = extractor_config
    app.state.best_frame_extractor = BestFrameExtractor(
        dependencies.config,
        dependencies.image_processor,
        dependencies.video_processor,
        app.state.image_evaluator,
    )
    app.state.frame_extraction_temp_root = temp_root
    app.state.frame_extraction_service = FrameExtractionService(
        extractor_config=extractor_config,
        extractor=app.state.best_frame_extractor,
        downloader=S3VideoDownloader(),
        uploader=S3DraftUploader(),
        temp_root=temp_root,
    )


def _mark_final_edit_unavailable(app: FastAPI, reason: str) -> None:
    app.state.final_edit_service = None
    app.state.final_edit_planner_client = None
    app.state.final_edit_available = False
    app.state.final_edit_upscale_enabled = settings.FINAL_EDIT_ENABLE_UPSCALE
    app.state.final_edit_unavailable_reason = reason


def _initialize_final_edit_service(app: FastAPI, temp_root: Path) -> None:
    try:
        import_cv2()
        if settings.FINAL_EDIT_ENABLE_UPSCALE:
            ensure_final_edit_upscaler_available()
        from app.services.final_edit import (
            FinalEditService,
            S3DraftImageDownloader,
            S3FinalImageUploader,
        )

        service = FinalEditService(
            downloader=S3DraftImageDownloader(),
            uploader=S3FinalImageUploader(),
            temp_root=temp_root / "final-edit",
            upscale_enabled=settings.FINAL_EDIT_ENABLE_UPSCALE,
        )
    except Exception as exc:
        reason = summarize_unavailable_reason(exc)
        _mark_final_edit_unavailable(app, reason)
        logger.warning(
            "Final edit service is unavailable at startup: %s. "
            "The app will continue, but final-edit will return 503 until the issue is fixed.",
            reason,
            exc_info=True,
            extra=build_log_extra(
                "app.startup.final_edit_unavailable",
                component="startup",
                stage="final_edit_initialize",
                outcome="failed",
                error_type=exc.__class__.__name__,
                final_edit_unavailable_reason=reason,
            ),
        )
        return

    app.state.final_edit_service = service
    app.state.final_edit_planner_client = service.planner_client
    app.state.final_edit_available = True
    app.state.final_edit_upscale_enabled = settings.FINAL_EDIT_ENABLE_UPSCALE
    app.state.final_edit_unavailable_reason = None


def _initialize_text_services(app: FastAPI, temp_root: Path) -> None:
    _initialize_final_edit_service(app, temp_root)
    app.state.intent_classifier_service = build_intent_classifier_service()
    app.state.keyword_extraction_service = build_keyword_extraction_service()
    app.state.caption_generation_service = build_caption_generation_service()
    app.state.canonical_keyword_resolver_service = (
        build_canonical_keyword_resolver_service()
    )
    app.state.reference_caption_retriever_service = (
        build_reference_caption_retriever_service(
            app.state.canonical_keyword_resolver_service,
            enabled=settings.REFERENCE_CAPTION_RAG_ENABLED,
            max_references=settings.REFERENCE_CAPTION_MAX_REFERENCES,
        )
    )
    app.state.menu_promotion_context_service = (
        build_menu_promotion_context_service()
    )


def _log_service_configuration(app: FastAPI) -> None:
    caption_client = settings.caption_model_client
    keyword_client = settings.keyword_model_client
    logger.info(
        "Keyword extraction configured.",
        extra=build_log_extra(
            "app.startup.keyword_extraction_configured",
            component="startup",
            keyword_model_base_url=keyword_client.base_url,
            keyword_chat_endpoint=keyword_client.chat_endpoint,
            keyword_health_endpoint=keyword_client.health_endpoint,
            keyword_timeout_seconds=keyword_client.timeout_seconds,
        ),
    )
    logger.info(
        "Caption generation configured.",
        extra=build_log_extra(
            "app.startup.caption_generation_configured",
            component="startup",
            caption_model_base_url=caption_client.base_url,
            caption_chat_endpoint=caption_client.chat_endpoint,
            caption_health_endpoint=caption_client.health_endpoint,
            caption_timeout_seconds=caption_client.timeout_seconds,
        ),
    )
    if getattr(app.state, "final_edit_available", False):
        logger.info(
            (
                "Final edit planner configured."
                if getattr(app.state, "final_edit_upscale_enabled", True)
                else "Final edit planner configured with upscale disabled."
            ),
            extra=build_log_extra(
                "app.startup.final_edit_planner_configured",
                component="startup",
                final_edit_model_base_url=(
                    app.state.final_edit_planner_client.model_settings.base_url
                ),
                final_edit_chat_endpoint=(
                    app.state.final_edit_planner_client.model_settings.chat_endpoint
                ),
                final_edit_health_endpoint=(
                    app.state.final_edit_planner_client.model_settings.health_endpoint
                ),
                final_edit_timeout_seconds=(
                    app.state.final_edit_planner_client.model_settings.timeout_seconds
                ),
                final_edit_model_name=settings.FINAL_EDIT_MODEL_NAME,
                final_edit_available="true",
                final_edit_upscale_enabled=str(
                    getattr(app.state, "final_edit_upscale_enabled", True)
                ).lower(),
            ),
        )
    else:
        logger.warning(
            "Final edit service is unavailable.",
            extra=build_log_extra(
                "app.startup.final_edit_unavailable",
                component="startup",
                stage="final_edit_config",
                outcome="failed",
                final_edit_available="false",
                final_edit_upscale_enabled=str(
                    getattr(app.state, "final_edit_upscale_enabled", True)
                ).lower(),
                final_edit_unavailable_reason=getattr(
                    app.state,
                    "final_edit_unavailable_reason",
                    None,
                ),
            ),
        )
    logger.info(
        "Canonical keyword resolver configured.",
        extra=build_log_extra(
            "app.startup.canonical_keyword_resolver_configured",
            component="startup",
            canonical_embedding_model_name=(
                settings.CANONICAL_KEYWORD_EMBEDDING_MODEL_NAME
            ),
            canonical_embedding_dim=settings.CANONICAL_KEYWORD_EMBEDDING_DIM,
            canonical_embedding_cache_dir=str(
                settings.CANONICAL_KEYWORD_EMBEDDING_MODEL_CACHE_DIR
            ),
        ),
    )
    logger.info(
        "Reference caption retriever configured.",
        extra=build_log_extra(
            "app.startup.reference_caption_retriever_configured",
            component="startup",
            reference_caption_rag_enabled=settings.REFERENCE_CAPTION_RAG_ENABLED,
            reference_caption_max_references=settings.REFERENCE_CAPTION_MAX_REFERENCES,
        ),
    )
    logger.info(
        "Menu promotion context service configured.",
        extra=build_log_extra(
            "app.startup.menu_promotion_context_configured",
            component="startup",
        ),
    )


async def _preload_services(app: FastAPI) -> None:
    caption_client = settings.caption_model_client
    keyword_client = settings.keyword_model_client

    try:
        await app.state.keyword_extraction_service.preload()
        logger.info(
            "Keyword extraction server connectivity check completed.",
            extra=build_log_extra(
                "app.startup.keyword_preload",
                component="startup",
                stage="keyword_preload",
                outcome="succeeded",
                keyword_model_base_url=keyword_client.base_url,
                keyword_health_endpoint=keyword_client.health_endpoint,
            ),
        )
    except KeywordExtractionUnavailableError as exc:
        logger.warning(
            "Keyword extraction server is unavailable at startup: %s. "
            "The app will continue, but process-utterance may return 503 until the server is reachable.",
            exc,
            exc_info=True,
            extra=build_log_extra(
                "app.startup.keyword_preload",
                component="startup",
                stage="keyword_preload",
                outcome="failed",
                error_type=exc.__class__.__name__,
                keyword_model_base_url=keyword_client.base_url,
                keyword_health_endpoint=keyword_client.health_endpoint,
            ),
        )
    except Exception:
        logger.warning(
            "Keyword extraction server is unavailable at startup. "
            "The app will continue, but process-utterance may return 503 until the server is reachable.",
            exc_info=True,
            extra=build_log_extra(
                "app.startup.keyword_preload",
                component="startup",
                stage="keyword_preload",
                outcome="failed",
                error_type="unexpected_startup_error",
                keyword_model_base_url=keyword_client.base_url,
                keyword_health_endpoint=keyword_client.health_endpoint,
            ),
        )

    try:
        await app.state.caption_generation_service.preload()
        logger.info(
            "Caption generation server connectivity check completed.",
            extra=build_log_extra(
                "app.startup.caption_preload",
                component="startup",
                stage="caption_preload",
                outcome="succeeded",
                caption_model_base_url=caption_client.base_url,
                caption_health_endpoint=caption_client.health_endpoint,
            ),
        )
    except CaptionGenerationUnavailableError as exc:
        logger.warning(
            "Caption generation server is unavailable at startup: %s. "
            "The app will continue and fall back to rule-based caption text until the server is reachable.",
            exc,
            exc_info=True,
            extra=build_log_extra(
                "app.startup.caption_preload",
                component="startup",
                stage="caption_preload",
                outcome="failed",
                error_type=exc.__class__.__name__,
                caption_model_base_url=caption_client.base_url,
                caption_health_endpoint=caption_client.health_endpoint,
            ),
        )
    except Exception:
        logger.warning(
            "Caption generation server is unavailable at startup. "
            "The app will continue and fall back to rule-based caption text until the server is reachable.",
            exc_info=True,
            extra=build_log_extra(
                "app.startup.caption_preload",
                component="startup",
                stage="caption_preload",
                outcome="failed",
                error_type="unexpected_startup_error",
                caption_model_base_url=caption_client.base_url,
                caption_health_endpoint=caption_client.health_endpoint,
            ),
        )

    try:
        await app.state.canonical_keyword_resolver_service.preload()
        logger.info(
            "Canonical keyword resolver preload completed.",
            extra=build_log_extra(
                "app.startup.canonical_keyword_resolver_preload",
                component="startup",
                stage="canonical_keyword_preload",
                outcome="succeeded",
                canonical_embedding_model_name=(
                    settings.CANONICAL_KEYWORD_EMBEDDING_MODEL_NAME
                ),
            ),
        )
    except Exception as exc:
        logger.warning(
            "Canonical keyword resolver is unavailable at startup: %s. "
            "The app will continue and fall back to draft keywords for final keyword storage.",
            exc,
            exc_info=True,
            extra=build_log_extra(
                "app.startup.canonical_keyword_resolver_preload",
                component="startup",
                stage="canonical_keyword_preload",
                outcome="failed",
                error_type=exc.__class__.__name__,
                canonical_embedding_model_name=(
                    settings.CANONICAL_KEYWORD_EMBEDDING_MODEL_NAME
                ),
            ),
        )

    try:
        await app.state.reference_caption_retriever_service.preload()
        logger.info(
            "Reference caption retriever preload completed.",
            extra=build_log_extra(
                "app.startup.reference_caption_retriever_preload",
                component="startup",
                stage="reference_caption_retriever_preload",
                outcome="succeeded",
                reference_caption_rag_enabled=settings.REFERENCE_CAPTION_RAG_ENABLED,
                reference_caption_max_references=(
                    settings.REFERENCE_CAPTION_MAX_REFERENCES
                ),
            ),
        )
    except Exception as exc:
        logger.warning(
            "Reference caption retriever is unavailable at startup: %s. "
            "The app will continue and skip caption RAG until the retriever is reachable.",
            exc,
            exc_info=True,
            extra=build_log_extra(
                "app.startup.reference_caption_retriever_preload",
                component="startup",
                stage="reference_caption_retriever_preload",
                outcome="failed",
                error_type=exc.__class__.__name__,
                reference_caption_rag_enabled=settings.REFERENCE_CAPTION_RAG_ENABLED,
                reference_caption_max_references=(
                    settings.REFERENCE_CAPTION_MAX_REFERENCES
                ),
            ),
        )

    try:
        await app.state.menu_promotion_context_service.preload()
        logger.info(
            "Menu promotion context service preload completed.",
            extra=build_log_extra(
                "app.startup.menu_promotion_context_preload",
                component="startup",
                stage="menu_promotion_context_preload",
                outcome="succeeded",
            ),
        )
    except Exception as exc:
        logger.warning(
            "Menu promotion context service is unavailable at startup: %s. "
            "The app will continue and skip menu candidate lookup until the database is reachable.",
            exc,
            exc_info=True,
            extra=build_log_extra(
                "app.startup.menu_promotion_context_preload",
                component="startup",
                stage="menu_promotion_context_preload",
                outcome="failed",
                error_type=exc.__class__.__name__,
            ),
        )


async def _ping_redis() -> None:
    redis = get_redis_client()
    try:
        await redis.ping()
        logger.info(
            "Redis connectivity check succeeded.",
            extra=build_log_extra(
                "app.startup.redis_ping",
                component="startup",
                stage="redis_ping",
                outcome="succeeded",
            ),
        )
    except Exception:
        logger.warning(
            "Redis connectivity check failed during startup.",
            exc_info=True,
            extra=build_log_extra(
                "app.startup.redis_ping",
                component="startup",
                stage="redis_ping",
                outcome="failed",
                error_type="redis_ping_failed",
            ),
        )


async def _cleanup_temp_root(temp_root: Path) -> None:
    try:
        shutil.rmtree(temp_root, ignore_errors=True)
        logger.info(
            "Temporary workspace cleanup completed.",
            extra=build_log_extra(
                "app.shutdown.temp_root_cleanup",
                component="shutdown",
                stage="temp_root_cleanup",
                outcome="succeeded",
                temp_root=str(temp_root),
            ),
        )
    except Exception:
        logger.warning(
            "Temporary workspace cleanup failed.",
            exc_info=True,
            extra=build_log_extra(
                "app.shutdown.temp_root_cleanup",
                component="shutdown",
                stage="temp_root_cleanup",
                outcome="failed",
                error_type="temp_root_cleanup_failed",
                temp_root=str(temp_root),
            ),
        )


async def _close_shared_clients() -> None:
    try:
        await close_redis()
        logger.info(
            "Redis client closed.",
            extra=build_log_extra(
                "app.shutdown.redis_close",
                component="shutdown",
                stage="redis_close",
                outcome="succeeded",
            ),
        )
    except Exception:
        logger.warning(
            "Redis client close failed.",
            exc_info=True,
            extra=build_log_extra(
                "app.shutdown.redis_close",
                component="shutdown",
                stage="redis_close",
                outcome="failed",
                error_type="redis_close_failed",
            ),
        )
    try:
        await dispose_engine()
        logger.info(
            "Database engine disposed.",
            extra=build_log_extra(
                "app.shutdown.dispose_engine",
                component="shutdown",
                stage="dispose_engine",
                outcome="succeeded",
            ),
        )
    except Exception:
        logger.warning(
            "Database engine dispose failed.",
            exc_info=True,
            extra=build_log_extra(
                "app.shutdown.dispose_engine",
                component="shutdown",
                stage="dispose_engine",
                outcome="failed",
                error_type="dispose_engine_failed",
            ),
        )


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    temp_root = _build_temp_root()
    app.state.image_evaluator = await _build_image_evaluator(app)
    _initialize_frame_extraction(app, temp_root)
    _initialize_text_services(app, temp_root)
    _log_service_configuration(app)
    await _preload_services(app)
    await _ping_redis()

    try:
        yield
    finally:
        logger.info(
            "AI service shutdown started.",
            extra=build_log_extra(
                "app.shutdown.started",
                component="shutdown",
                outcome="started",
                temp_root=str(temp_root),
            ),
        )
        await _cleanup_temp_root(temp_root)
        await _close_shared_clients()
