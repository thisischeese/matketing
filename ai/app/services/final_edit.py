"""Services for uploading final images from extracted draft frames."""

from __future__ import annotations

import asyncio
import logging
import shutil
from dataclasses import dataclass
from pathlib import Path
import time
from typing import TYPE_CHECKING

from fastapi import Request
from redis.asyncio import Redis

from app.core.config import settings
from app.logging import build_log_extra, preview_text
from app.schemas.sessions import FinalEditRequest
from app.services.final_edit_planner import (
    CarouselOrderPlan,
    FinalEditPlannerClient,
    FinalEditPlanningError,
    FinalEditSessionContext,
    ImageEditPlan,
    build_noop_final_edit_plan,
    build_final_edit_planner_client,
    load_final_edit_session_context,
    normalize_image_edit_plan,
    resolve_owner_persona_preset,
)
from app.services.final_edit_runtime import (
    FinalEditUnavailableError,
    import_cv2,
)
from app.services.s3_support import build_s3_client, normalize_s3_key
from app.services.sessions import session_key, upsert_content_session

if TYPE_CHECKING:
    from app.services.final_edit_agent import FinalEditAgent, FinalEditImageState

STATUS_FAIL = "FRAME_EXTRACTED"
STATUS_PHOTO_EDITED = "PHOTO_EDITED"
logger = logging.getLogger(__name__)
_PREVIEW_MAX_LENGTH = 120
_PRE_UPSCALE_FILTER_TOOLS = {"filter", "sharpen"}


def _build_default_agent(tool_event_sink, upscale_enabled: bool):
    from app.services.final_edit_agent import FinalEditAgent
    from app.services.final_edit_tools import FinalEditToolRegistry

    return FinalEditAgent(
        FinalEditToolRegistry(upscale_enabled=upscale_enabled),
        tool_event_sink=tool_event_sink,
        upscale_enabled=upscale_enabled,
    )


@dataclass
class FinalEditResult:
    status: str
    results: list[str]
    failure_reason: str | None = None
    debug_fields: dict[str, str] | None = None


def _preview_or_empty(value: str, max_length: int = _PREVIEW_MAX_LENGTH) -> str:
    normalized = value.strip()
    if not normalized:
        return ""
    return preview_text(normalized, max_length)


def _preview_keywords(keywords: list[str], max_length: int = _PREVIEW_MAX_LENGTH) -> str:
    if not keywords:
        return ""
    return preview_text(", ".join(keywords), max_length)


def _determine_output_source(output_path: Path, draft_path: Path) -> str:
    try:
        if output_path.resolve() == draft_path.resolve():
            return "original_fallback"
    except OSError:
        if output_path == draft_path:
            return "original_fallback"
    return "edited"


def _format_reorder_mapping(
    *,
    original_index: int,
    reordered_slot_index: int,
    uploaded_path: str,
) -> str:
    return (
        f"original={original_index},"
        f"slot={reordered_slot_index},"
        f"uploaded={uploaded_path}"
    )


def _read_image_dimensions(image_path: Path) -> tuple[int | None, int | None]:
    cv2 = import_cv2()
    image = cv2.imread(str(image_path))
    if image is None:
        return None, None
    height, width = image.shape[:2]
    return width, height


def _classify_layout_orientation(width: int | None, height: int | None) -> str | None:
    if width is None or height is None:
        return None
    if width == height:
        return "square"
    if height > width:
        return "portrait"
    return "landscape"


def _attach_crop_context(
    plan: ImageEditPlan,
    *,
    anchor_orientation: str | None,
) -> ImageEditPlan:
    if "crop" not in plan.tools:
        return plan
    merged_params = {tool_name: dict(tool_params) for tool_name, tool_params in plan.params.items()}
    crop_params = dict(merged_params.get("crop", {}))
    if anchor_orientation:
        crop_params["anchor_orientation"] = anchor_orientation
    merged_params["crop"] = crop_params
    return ImageEditPlan(
        image_index=plan.image_index,
        content=plan.content,
        strategy=plan.strategy,
        tools=list(plan.tools),
        params=merged_params,
    )


def _resolve_image_suffix(image_path: str, default_suffix: str = ".png") -> str:
    suffix = Path(normalize_s3_key(image_path)).suffix.lower()
    return suffix or default_suffix


def _ensure_filter_before_upscale(
    plan: ImageEditPlan,
    owner_persona: str,
    *,
    upscale_enabled: bool,
) -> ImageEditPlan:
    plan = normalize_image_edit_plan(
        plan,
        upscale_enabled=upscale_enabled,
        owner_persona=owner_persona,
    )
    if not upscale_enabled:
        return plan
    if "upscale" not in plan.tools:
        return plan

    upscale_index = plan.tools.index("upscale")
    has_filter_before_upscale = any(
        tool_name in _PRE_UPSCALE_FILTER_TOOLS for tool_name in plan.tools[:upscale_index]
    )
    if has_filter_before_upscale:
        return plan

    fallback_plan = ImageEditPlan(
        plan.image_index,
        content=plan.content,
        strategy=plan.strategy,
        tools=["filter"],
        params={"filter": {"filter_name": "Clarendon"}},
    )
    merged_params = dict(plan.params)
    merged_params["filter"] = dict(
        merged_params.get("filter", fallback_plan.params["filter"])
    )
    merged_params["upscale"] = dict(merged_params.get("upscale", {"scale": 2}))

    enforced_tools: list[str] = []
    inserted_filter = False
    for tool_name in plan.tools:
        if tool_name == "upscale" and not inserted_filter:
            enforced_tools.append("filter")
            inserted_filter = True
        enforced_tools.append(tool_name)

    return normalize_image_edit_plan(
        ImageEditPlan(
            image_index=plan.image_index,
            content=plan.content,
            strategy=plan.strategy,
            tools=enforced_tools,
            params=merged_params,
        ),
        upscale_enabled=upscale_enabled,
        owner_persona=owner_persona,
    )


class S3DraftImageDownloader:
    """Download extracted draft images from S3."""

    def __init__(self) -> None:
        self._settings = settings
        self._client = build_s3_client() if self._settings.s3_configured else None

    @property
    def is_configured(self) -> bool:
        return self._settings.s3_configured

    async def download_draft(self, draft_key: str, destination: Path) -> None:
        await asyncio.to_thread(self._download_draft, draft_key, destination)

    def _download_draft(self, draft_key: str, destination: Path) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        assert self._client is not None
        self._client.download_file(
            self._settings.S3_BUCKET_NAME,
            normalize_s3_key(draft_key),
            str(destination),
        )


class S3FinalImageUploader:
    """Upload corrected final images to S3 and delete them during rollback."""

    def __init__(self) -> None:
        self._settings = settings
        self._client = build_s3_client() if self._settings.s3_configured else None

    @property
    def is_configured(self) -> bool:
        return self._settings.s3_configured

    async def upload_final(
        self,
        session_id: str,
        image_path: Path,
        final_index: int,
    ) -> str:
        return await asyncio.to_thread(
            self._upload_final,
            session_id,
            image_path,
            final_index,
        )

    async def delete_final(self, uploaded_path: str) -> None:
        await asyncio.to_thread(self._delete_final, uploaded_path)

    def _upload_final(
        self,
        session_id: str,
        image_path: Path,
        final_index: int,
    ) -> str:
        object_key = f"ai-finals/{session_id}/final-{final_index:03d}.png"
        assert self._client is not None
        self._client.upload_file(
            str(image_path),
            self._settings.S3_BUCKET_NAME,
            object_key,
            ExtraArgs={"ContentType": "image/png"},
        )
        return f"/{object_key}"

    def _delete_final(self, uploaded_path: str) -> None:
        assert self._client is not None
        self._client.delete_object(
            Bucket=self._settings.S3_BUCKET_NAME,
            Key=normalize_s3_key(uploaded_path),
        )


class FinalEditService:
    """Upload final images from extracted draft frames."""

    def __init__(
        self,
        downloader: S3DraftImageDownloader,
        uploader: S3FinalImageUploader,
        temp_root: Path,
        planner_client: FinalEditPlannerClient | None = None,
        agent: FinalEditAgent | None = None,
        upscale_enabled: bool | None = None,
    ) -> None:
        self.downloader = downloader
        self.uploader = uploader
        self.temp_root = temp_root
        self.upscale_enabled = (
            settings.FINAL_EDIT_ENABLE_UPSCALE
            if upscale_enabled is None
            else upscale_enabled
        )
        self.planner_client = planner_client or build_final_edit_planner_client()
        self.agent = agent or _build_default_agent(
            self._log_tool_event,
            self.upscale_enabled,
        )

    def _build_debug_fields(
        self,
        session_dir: Path,
        drafts: list[str],
        context: FinalEditSessionContext,
    ) -> dict[str, str]:
        preset_persona, _, preset_fallback = resolve_owner_persona_preset(
            context.owner_persona
        )
        return {
            "debug:draft_count": str(len(drafts)),
            "debug:session_dir": str(session_dir),
            "debug:owner_persona": context.owner_persona,
            "debug:target_preset_persona": preset_persona,
            "debug:target_preset_fallback": str(preset_fallback).lower(),
            "debug:caption_present": str(bool(context.caption)).lower(),
            "debug:keyword_count": str(len(context.keywords)),
            "debug:planner_mode": "multi_image_single_call",
            "debug:planner_response_received": "false",
            "debug:planned_indexes": "",
            "debug:upscale_enabled": str(self.upscale_enabled).lower(),
            "debug:reorder_attempted": "false",
            "debug:reorder_applied": "false",
        }

    def _s3_failure_result(
        self,
        session_id: str,
        drafts: list[str],
        debug_fields: dict[str, str],
    ) -> FinalEditResult:
        debug_fields["debug:stage"] = "s3_config_check"
        debug_fields["debug:s3_downloader_configured"] = str(
            self.downloader.is_configured
        ).lower()
        debug_fields["debug:s3_uploader_configured"] = str(
            self.uploader.is_configured
        ).lower()
        logger.warning(
            "Final edit aborted because S3 is unavailable.",
            extra=build_log_extra(
                "final_edit.s3_config_check",
                component="final_edit",
                stage="s3_config_check",
                session_id=session_id,
                outcome="failed",
                error_type="s3_unavailable",
                draft_count=len(drafts),
            ),
        )
        return FinalEditResult(
            status=STATUS_FAIL,
            results=[],
            failure_reason="s3_unavailable",
            debug_fields=debug_fields,
        )

    async def _download_drafts(
        self,
        session_id: str,
        drafts: list[str],
        session_dir: Path,
        debug_fields: dict[str, str],
    ) -> list[Path]:
        async def _download_single_draft(
            final_index: int,
            draft_key: str,
        ) -> tuple[int, str, Path, int]:
            suffix = _resolve_image_suffix(draft_key)
            draft_path = session_dir / f"draft-{final_index:03d}{suffix}"
            download_started_at = time.perf_counter()
            logger.info(
                "Downloading draft image for final edit.",
                extra=build_log_extra(
                    "final_edit.download_draft.started",
                    component="final_edit",
                    stage="download_draft",
                    session_id=session_id,
                    final_index=final_index,
                    draft_key=draft_key,
                    outcome="started",
                ),
            )
            await self.downloader.download_draft(draft_key, draft_path)
            elapsed_ms = int((time.perf_counter() - download_started_at) * 1000)
            logger.info(
                "Downloaded draft image for final edit.",
                extra=build_log_extra(
                    "final_edit.download_draft.completed",
                    component="final_edit",
                    stage="download_draft",
                    session_id=session_id,
                    final_index=final_index,
                    draft_key=draft_key,
                    outcome="succeeded",
                    elapsed_ms=elapsed_ms,
                ),
            )
            return final_index, draft_key, draft_path, elapsed_ms

        results = await asyncio.gather(
            *[
                _download_single_draft(final_index, draft_key)
                for final_index, draft_key in enumerate(drafts, start=1)
            ]
        )
        ordered_results = sorted(results, key=lambda item: item[0])
        debug_fields["debug:downloaded_draft_count"] = str(len(ordered_results))
        debug_fields["debug:download_task_count"] = str(len(ordered_results))
        total_download_elapsed_ms = 0
        downloaded_drafts: list[Path] = []
        for final_index, draft_key, draft_path, elapsed_ms in ordered_results:
            debug_fields["debug:stage"] = f"download_draft_{final_index}"
            debug_fields[f"debug:draft_key:{final_index}"] = draft_key
            debug_fields[f"debug:download_elapsed_ms:{final_index}"] = str(elapsed_ms)
            total_download_elapsed_ms += elapsed_ms
            downloaded_drafts.append(draft_path)
        debug_fields["debug:total_download_elapsed_ms"] = str(total_download_elapsed_ms)
        return downloaded_drafts

    async def _plan_images_and_order(
        self,
        session_id: str,
        context: FinalEditSessionContext,
        downloaded_drafts: list[Path],
        debug_fields: dict[str, str],
    ) -> tuple[bool, dict[int, object], CarouselOrderPlan | None]:
        planner_fallback = not bool(context.caption or context.keywords)
        plans_by_index: dict[int, object] = {}
        precomputed_reorder_plan: CarouselOrderPlan | None = None

        logger.info(
            "Loading final edit session context.",
            extra=build_log_extra(
                "final_edit.load_context.completed",
                component="final_edit",
                stage="load_context",
                session_id=session_id,
                outcome="succeeded",
                caption_present=bool(context.caption),
                keyword_count=len(context.keywords),
                caption_preview=_preview_or_empty(context.caption),
                keyword_preview=_preview_keywords(context.keywords),
            ),
        )

        if planner_fallback:
            logger.info(
                "Skipping final edit planner because no text context is available.",
                extra=build_log_extra(
                    "final_edit.planner_fallback",
                    component="final_edit",
                    stage="plan",
                    session_id=session_id,
                    outcome="fallback",
                    error_type="missing_context",
                    planner_fallback_reason="missing_context",
                    caption_present=bool(context.caption),
                    keyword_count=len(context.keywords),
                    downloaded_draft_count=len(downloaded_drafts),
                ),
            )
        else:
            plan_started_at = time.perf_counter()
            logger.info(
                "Planning final edit sequence.",
                extra=build_log_extra(
                    "final_edit.plan.started",
                    component="final_edit",
                    stage="plan",
                    session_id=session_id,
                    outcome="started",
                    draft_count=len(downloaded_drafts),
                    keyword_count=len(context.keywords),
                ),
            )
            try:
                batch_plan = await self.planner_client.build_batch_plan(
                    [str(draft_path) for draft_path in downloaded_drafts],
                    context,
                )
                debug_fields["debug:planner_response_received"] = str(
                    bool(getattr(self.planner_client, "last_raw_output", None))
                ).lower()
                for plan in batch_plan.image_plans:
                    plans_by_index[plan.image_index] = normalize_image_edit_plan(
                        plan,
                        upscale_enabled=self.upscale_enabled,
                        owner_persona=context.owner_persona,
                    )
                precomputed_reorder_plan = batch_plan.carousel_order
                if batch_plan.carousel_order_error_message:
                    reorder_failure_raw_preview = _preview_or_empty(
                        batch_plan.carousel_order_raw_output or ""
                    )
                    debug_fields["debug:reorder_failure_type"] = (
                        FinalEditPlanningError.__name__
                    )
                    debug_fields["debug:reorder_failure_message"] = (
                        batch_plan.carousel_order_error_message
                    )
                    debug_fields["debug:reorder_failure_raw_preview"] = (
                        reorder_failure_raw_preview
                    )
                    logger.warning(
                        "Final edit carousel reorder failed. Preserving original order.",
                        extra=build_log_extra(
                            "final_edit.reorder.completed",
                            component="final_edit",
                            stage="reorder",
                            session_id=session_id,
                            outcome="fallback",
                            error_type=FinalEditPlanningError.__name__,
                            reorder_failure_message=batch_plan.carousel_order_error_message,
                            reorder_failure_raw_preview=reorder_failure_raw_preview,
                        ),
                    )
            except Exception as exc:
                planner_fallback = True
                planner_failure_message = str(exc)
                planner_failure_raw_preview = _preview_or_empty(
                    getattr(self.planner_client, "last_raw_output", "") or ""
                )
                debug_fields["debug:planner_failure_type"] = exc.__class__.__name__
                debug_fields["debug:planner_failure_message"] = (
                    planner_failure_message
                )
                debug_fields["debug:planner_failure_raw_preview"] = (
                    planner_failure_raw_preview
                )
                logger.warning(
                    "Final edit planner failed. Falling back for all images.",
                    extra=build_log_extra(
                        "final_edit.planner_fallback",
                        component="final_edit",
                        stage="plan",
                        session_id=session_id,
                        outcome="fallback",
                        error_type=exc.__class__.__name__,
                        planner_fallback_reason="planner_exception",
                        planner_failure_message=planner_failure_message,
                        planner_failure_raw_preview=planner_failure_raw_preview,
                        caption_present=bool(context.caption),
                        keyword_count=len(context.keywords),
                        downloaded_draft_count=len(downloaded_drafts),
                    ),
                )
                debug_fields["debug:planner_response_received"] = "false"

            request_stats = getattr(self.planner_client, "last_request_stats", {}) or {}
            for key, value in request_stats.items():
                debug_fields[f"debug:{key}"] = str(value)
            debug_fields["debug:planned_indexes"] = ",".join(
                str(image_index) for image_index in sorted(plans_by_index)
            )
            logger.info(
                "Planned final edit sequence.",
                extra=build_log_extra(
                    "final_edit.plan.completed",
                    component="final_edit",
                    stage="plan",
                    session_id=session_id,
                    outcome="succeeded",
                    planned_image_count=len(plans_by_index),
                    planned_indexes=sorted(plans_by_index),
                    planned_tools_by_image={
                        str(image_index): plan.tools
                        for image_index, plan in sorted(plans_by_index.items())
                    },
                    planned_strategy_preview_by_image={
                        str(image_index): _preview_or_empty(plan.strategy)
                        for image_index, plan in sorted(plans_by_index.items())
                    },
                    caption_preview=_preview_or_empty(context.caption),
                    keyword_preview=_preview_keywords(context.keywords),
                    planner_response_preview=_preview_or_empty(
                        getattr(self.planner_client, "last_raw_output", "") or ""
                    ),
                    prompt_text_chars_total=request_stats.get("prompt_text_chars_total"),
                    prompt_image_count=request_stats.get("prompt_image_count"),
                    prompt_image_data_uri_chars_total=request_stats.get(
                        "prompt_image_data_uri_chars_total"
                    ),
                    prompt_payload_json_chars=request_stats.get(
                        "prompt_payload_json_chars"
                    ),
                    reorder_precomputed=precomputed_reorder_plan is not None,
                    planner_partial_fallback=planner_fallback,
                    elapsed_ms=int((time.perf_counter() - plan_started_at) * 1000),
                ),
            )

        debug_fields["debug:planner_fallback"] = str(planner_fallback).lower()
        debug_fields["debug:planned_image_count"] = str(len(plans_by_index))
        return planner_fallback, plans_by_index, precomputed_reorder_plan

    async def _process_images(
        self,
        session_id: str,
        session_dir: Path,
        downloaded_drafts: list[Path],
        plans_by_index: dict[int, object],
        planner_fallback: bool,
        context: FinalEditSessionContext,
        debug_fields: dict[str, str],
    ) -> list[Path]:
        from app.services.final_edit_agent import FinalEditImageState

        edited_paths: list[Path] = []
        anchor_width, anchor_height = _read_image_dimensions(downloaded_drafts[0])
        anchor_orientation = _classify_layout_orientation(anchor_width, anchor_height)
        for image_index, draft_path in enumerate(downloaded_drafts):
            plan = plans_by_index.get(image_index)
            if plan is None:
                fallback_reason = (
                    "planner_fallback" if planner_fallback else "plan_missing_for_image"
                )
                plan = build_noop_final_edit_plan(
                    image_index,
                    content="draft image",
                    strategy=(
                        "Planner fallback; keep the original image unchanged."
                        if planner_fallback
                        else "No plan returned for this image; keep the original image unchanged."
                    ),
                )
                debug_fields[f"debug:plan_source:{image_index + 1}"] = fallback_reason
            else:
                plan = _ensure_filter_before_upscale(
                    plan,
                    context.owner_persona,
                    upscale_enabled=self.upscale_enabled,
                )

            plan = normalize_image_edit_plan(
                plan,
                upscale_enabled=self.upscale_enabled,
                owner_persona=context.owner_persona,
            )
            plan = _attach_crop_context(
                plan,
                anchor_orientation=anchor_orientation,
            )

            debug_fields[f"debug:tool_count:{image_index + 1}"] = str(len(plan.tools))
            debug_fields[f"debug:tools:{image_index + 1}"] = ",".join(plan.tools)
            image_state = await self.agent.run(
                FinalEditImageState(
                    image_path=draft_path,
                    working_dir=session_dir / "edited",
                    plan=plan,
                    owner_persona=context.owner_persona,
                    session_id=session_id,
                    final_index=image_index + 1,
                )
            )
            output_path = image_state.output_path or draft_path
            output_source = _determine_output_source(output_path, draft_path)
            debug_fields[f"debug:executed_tools:{image_index + 1}"] = ",".join(
                image_state.executed_tools
            )
            debug_fields[f"debug:upload_source_kind:{image_index + 1}"] = output_source
            debug_fields[f"debug:output_path:{image_index + 1}"] = str(output_path)
            debug_fields[f"debug:crop_applied:{image_index + 1}"] = str(
                bool(image_state.metadata.get("crop_applied"))
            ).lower()
            debug_fields[f"debug:filter_applied:{image_index + 1}"] = str(
                bool(image_state.metadata.get("filter_applied"))
            ).lower()
            selected_filter = image_state.metadata.get("selected_filter")
            if selected_filter is not None:
                debug_fields[f"debug:selected_filter:{image_index + 1}"] = str(
                    selected_filter
                )
                debug_fields[f"debug:filter_source:{image_index + 1}"] = str(
                    debug_fields.get(f"debug:plan_source:{image_index + 1}", "planner")
                )
            if image_state.metadata.get("selected_filter_backend") is not None:
                debug_fields[
                    f"debug:selected_filter_backend:{image_index + 1}"
                ] = str(image_state.metadata.get("selected_filter_backend"))
            if image_state.metadata.get("selected_filter_callable") is not None:
                debug_fields[
                    f"debug:selected_filter_callable:{image_index + 1}"
                ] = str(image_state.metadata.get("selected_filter_callable"))
            if image_state.metadata.get("crop_reason") is not None:
                debug_fields[f"debug:crop_reason:{image_index + 1}"] = str(
                    image_state.metadata.get("crop_reason")
                )
            if image_state.metadata.get("crop_output_ratio") is not None:
                debug_fields[f"debug:crop_output_ratio:{image_index + 1}"] = str(
                    image_state.metadata.get("crop_output_ratio")
                )
            if image_state.fallback_to_original:
                debug_fields[f"debug:image_fallback:{image_index + 1}"] = "true"
                if image_state.failure_reason:
                    debug_fields[f"debug:image_failure:{image_index + 1}"] = (
                        image_state.failure_reason
                    )
            logger.info(
                "Completed final edit image processing.",
                extra=build_log_extra(
                    "final_edit.image.completed",
                    component="final_edit",
                    stage="image",
                    session_id=session_id,
                    final_index=image_index + 1,
                    image_index=image_index,
                    planned_tool_count=len(plan.tools),
                    executed_tool_count=len(image_state.executed_tools),
                    executed_tools=image_state.executed_tools,
                    used_fallback=image_state.fallback_to_original,
                    failure_reason=image_state.failure_reason,
                    input_path=str(draft_path),
                    output_path=str(output_path),
                    output_source=output_source,
                ),
            )
            edited_paths.append(output_path)
        return edited_paths

    async def _reorder_edited_images(
        self,
        session_id: str,
        drafts: list[str],
        downloaded_drafts: list[Path],
        edited_paths: list[Path],
        precomputed_reorder_plan: CarouselOrderPlan | None,
        debug_fields: dict[str, str],
    ) -> tuple[list[str], list[Path], list[Path], list[int]]:
        original_indexes = list(range(len(edited_paths)))
        if len(edited_paths) != 3:
            return drafts, downloaded_drafts, edited_paths, original_indexes

        debug_fields["debug:reorder_attempted"] = "true"
        logger.info(
            "Attempting final edit carousel reorder.",
            extra=build_log_extra(
                "final_edit.reorder.started",
                component="final_edit",
                stage="reorder",
                session_id=session_id,
                outcome="started",
                image_count=len(edited_paths),
            ),
        )
        if precomputed_reorder_plan is None:
            debug_fields["debug:reorder_applied"] = "false"
            return drafts, downloaded_drafts, edited_paths, original_indexes

        return self._apply_reorder_plan(
            session_id=session_id,
            drafts=drafts,
            downloaded_drafts=downloaded_drafts,
            edited_paths=edited_paths,
            reorder_plan=precomputed_reorder_plan,
            debug_fields=debug_fields,
        )

    def _apply_reorder_plan(
        self,
        *,
        session_id: str,
        drafts: list[str],
        downloaded_drafts: list[Path],
        edited_paths: list[Path],
        reorder_plan: CarouselOrderPlan,
        debug_fields: dict[str, str],
    ) -> tuple[list[str], list[Path], list[Path], list[int]]:
        debug_fields["debug:reorder_order"] = ",".join(
            str(index) for index in reorder_plan.ordered_image_indexes
        )
        debug_fields["debug:reorder_hook_scores"] = ",".join(
            f"{score:.2f}" for score in reorder_plan.hook_scores
        )
        debug_fields["debug:reorder_confidence"] = (
            f"{reorder_plan.overall_confidence:.2f}"
        )
        debug_fields["debug:reorder_reason"] = reorder_plan.reason

        is_valid_order = sorted(reorder_plan.ordered_image_indexes) == [0, 1, 2]
        is_valid_hook_scores = (
            len(reorder_plan.hook_scores) == 3
            and all(
                reorder_plan.hook_scores[index] >= reorder_plan.hook_scores[index + 1]
                for index in range(2)
            )
        )
        should_apply = (
            is_valid_order
            and is_valid_hook_scores
            and reorder_plan.overall_confidence >= 0.70
        )
        if not should_apply:
            debug_fields["debug:reorder_applied"] = "false"
            if not is_valid_order:
                skip_reason = "invalid_order"
            elif not is_valid_hook_scores:
                skip_reason = "invalid_hook_scores"
            else:
                skip_reason = "low_confidence"
            debug_fields["debug:reorder_skip_reason"] = skip_reason
            logger.info(
                "Final edit carousel reorder skipped due to low confidence or invalid output.",
                extra=build_log_extra(
                    "final_edit.reorder.completed",
                    component="final_edit",
                    stage="reorder",
                    session_id=session_id,
                    outcome="skipped",
                    reorder_skip_reason=skip_reason,
                    reorder_confidence=reorder_plan.overall_confidence,
                    reorder_order=reorder_plan.ordered_image_indexes,
                    reorder_hook_scores=reorder_plan.hook_scores,
                    reorder_reason_preview=_preview_or_empty(reorder_plan.reason),
                ),
            )
            return drafts, downloaded_drafts, edited_paths, list(range(len(edited_paths)))

        reordered_drafts = [drafts[index] for index in reorder_plan.ordered_image_indexes]
        reordered_downloaded_drafts = [
            downloaded_drafts[index] for index in reorder_plan.ordered_image_indexes
        ]
        reordered_edited_paths = [
            edited_paths[index] for index in reorder_plan.ordered_image_indexes
        ]
        debug_fields["debug:reorder_applied"] = "true"
        logger.info(
            "Applied final edit carousel reorder.",
            extra=build_log_extra(
                "final_edit.reorder.completed",
                component="final_edit",
                stage="reorder",
                session_id=session_id,
                outcome="succeeded",
                reorder_confidence=reorder_plan.overall_confidence,
                reorder_order=reorder_plan.ordered_image_indexes,
                reorder_hook_scores=reorder_plan.hook_scores,
                reorder_reason_preview=_preview_or_empty(reorder_plan.reason),
            ),
        )
        return (
            reordered_drafts,
            reordered_downloaded_drafts,
            reordered_edited_paths,
            list(reorder_plan.ordered_image_indexes),
        )

    async def _upload_edited_images(
        self,
        session_id: str,
        drafts: list[str],
        downloaded_drafts: list[Path],
        edited_paths: list[Path],
        original_indexes: list[int],
        debug_fields: dict[str, str],
        uploaded_results: list[str],
    ) -> None:
        async def _upload_single_final(
            final_index: int,
            original_index: int,
            draft_key: str,
            output_path: Path,
            draft_path: Path,
        ) -> tuple[int, int, str, str, Path, bool, int | None, int | None, int, str]:
            upload_source_kind = _determine_output_source(output_path, draft_path)
            upload_source_exists = output_path.exists()
            upload_width, upload_height = _read_image_dimensions(output_path)
            upload_started_at = time.perf_counter()
            logger.info(
                "Uploading final edited image.",
                extra=build_log_extra(
                    "final_edit.upload_final.started",
                    component="final_edit",
                    stage="upload_final",
                    session_id=session_id,
                    final_index=final_index,
                    original_image_index=original_index + 1,
                    reordered_slot_index=final_index,
                    draft_key=draft_key,
                    outcome="started",
                    upload_source_path=str(output_path),
                    upload_source_kind=upload_source_kind,
                    upload_source_exists=upload_source_exists,
                    upload_image_width=upload_width,
                    upload_image_height=upload_height,
                ),
            )
            uploaded_path = await self.uploader.upload_final(
                session_id,
                output_path,
                final_index,
            )
            elapsed_ms = int((time.perf_counter() - upload_started_at) * 1000)
            logger.info(
                "Uploaded final edited image.",
                extra=build_log_extra(
                    "final_edit.upload_final.completed",
                    component="final_edit",
                    stage="upload_final",
                    session_id=session_id,
                    final_index=final_index,
                    original_image_index=original_index + 1,
                    reordered_slot_index=final_index,
                    draft_key=draft_key,
                    outcome="succeeded",
                    uploaded_path=uploaded_path,
                    upload_source_path=str(output_path),
                    upload_source_kind=upload_source_kind,
                    upload_source_exists=upload_source_exists,
                    upload_image_width=upload_width,
                    upload_image_height=upload_height,
                    elapsed_ms=elapsed_ms,
                ),
            )
            return (
                final_index,
                original_index,
                draft_key,
                uploaded_path,
                output_path,
                upload_source_exists,
                upload_width,
                upload_height,
                elapsed_ms,
                upload_source_kind,
            )

        tasks = [
            _upload_single_final(
                final_index,
                original_indexes[final_index - 1],
                draft_key,
                edited_paths[final_index - 1],
                downloaded_drafts[final_index - 1],
            )
            for final_index, draft_key in enumerate(drafts, start=1)
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        successful_results: list[
            tuple[int, str, str, Path, bool, int | None, int | None, int, str]
        ] = []
        first_exception: Exception | None = None
        for result in results:
            if isinstance(result, Exception):
                if first_exception is None:
                    first_exception = result
                continue
            successful_results.append(result)

        ordered_results = sorted(successful_results, key=lambda item: item[0])
        debug_fields["debug:uploaded_final_count"] = str(len(ordered_results))
        debug_fields["debug:upload_task_count"] = str(len(drafts))
        total_upload_elapsed_ms = 0
        for (
            final_index,
            original_index,
            _draft_key,
            uploaded_path,
            _output_path,
            _upload_source_exists,
            _upload_width,
            _upload_height,
            elapsed_ms,
            _upload_source_kind,
        ) in ordered_results:
            debug_fields["debug:stage"] = f"upload_final_{final_index}"
            debug_fields[f"debug:upload_elapsed_ms:{final_index}"] = str(elapsed_ms)
            debug_fields[f"debug:reorder_mapping:{final_index}"] = _format_reorder_mapping(
                original_index=original_index + 1,
                reordered_slot_index=final_index,
                uploaded_path=uploaded_path,
            )
            total_upload_elapsed_ms += elapsed_ms
            uploaded_results.append(uploaded_path)
        debug_fields["debug:total_upload_elapsed_ms"] = str(total_upload_elapsed_ms)
        if first_exception is not None:
            raise first_exception

    async def edit_and_upload(
        self,
        session_id: str,
        drafts: list[str],
        context: FinalEditSessionContext | None = None,
    ) -> FinalEditResult:
        session_dir = self.temp_root / session_id
        session_dir.mkdir(parents=True, exist_ok=True)
        uploaded_results: list[str] = []
        context = context or FinalEditSessionContext(caption="", keywords=[])
        debug_fields = self._build_debug_fields(session_dir, drafts, context)

        try:
            if not self.downloader.is_configured or not self.uploader.is_configured:
                return self._s3_failure_result(session_id, drafts, debug_fields)

            downloaded_drafts = await self._download_drafts(
                session_id,
                drafts,
                session_dir,
                debug_fields,
            )
            planner_fallback, plans_by_index, precomputed_reorder_plan = await self._plan_images_and_order(
                session_id,
                context,
                downloaded_drafts,
                debug_fields,
            )

            edited_paths = await self._process_images(
                session_id,
                session_dir,
                downloaded_drafts,
                plans_by_index,
                planner_fallback,
                context,
                debug_fields,
            )
            drafts, downloaded_drafts, edited_paths, original_indexes = await self._reorder_edited_images(
                session_id,
                drafts,
                downloaded_drafts,
                edited_paths,
                precomputed_reorder_plan,
                debug_fields,
            )
            await self._upload_edited_images(
                session_id,
                drafts,
                downloaded_drafts,
                edited_paths,
                original_indexes,
                debug_fields,
                uploaded_results,
            )

            debug_fields["debug:stage"] = "completed"
            logger.info(
                "Completed final edit workflow.",
                extra=build_log_extra(
                    "final_edit.completed",
                    component="final_edit",
                    stage="completed",
                    session_id=session_id,
                    outcome="succeeded",
                    downloaded_draft_count=len(downloaded_drafts),
                    uploaded_final_count=len(uploaded_results),
                    reorder_applied=debug_fields.get("debug:reorder_applied"),
                    total_download_elapsed_ms=debug_fields.get(
                        "debug:total_download_elapsed_ms"
                    ),
                    total_upload_elapsed_ms=debug_fields.get(
                        "debug:total_upload_elapsed_ms"
                    ),
                ),
            )
            return FinalEditResult(
                status=STATUS_PHOTO_EDITED,
                results=uploaded_results,
                debug_fields=debug_fields,
            )
        except Exception as exc:
            debug_fields["debug:stage"] = "exception"
            debug_fields["debug:exception_type"] = exc.__class__.__name__
            debug_fields["debug:exception_message"] = str(exc)
            logger.exception(
                "Final edit failed with an exception.",
                extra=build_log_extra(
                    "final_edit.exception",
                    component="final_edit",
                    stage=debug_fields["debug:stage"],
                    session_id=session_id,
                    draft_count=len(drafts),
                    outcome="failed",
                    error_type=exc.__class__.__name__,
                ),
            )
            await self._rollback_uploaded(uploaded_results)
            return FinalEditResult(
                status=STATUS_FAIL,
                results=[],
                failure_reason=str(exc),
                debug_fields=debug_fields,
            )
        finally:
            logger.info(
                "Cleaning up final edit temp directory.",
                extra=build_log_extra(
                    "final_edit.cleanup_tempdir",
                    component="final_edit",
                    stage="cleanup_tempdir",
                    session_id=session_id,
                    outcome="completed",
                    session_dir=str(session_dir),
                ),
            )
            shutil.rmtree(session_dir, ignore_errors=True)

    def _log_tool_event(
        self,
        phase: str,
        state: FinalEditImageState,
        tool_name: str,
    ) -> None:
        event_name = f"final_edit.tool.{phase}"
        logger_method = logger.info if phase != "failed" else logger.warning
        logger_method(
            "Final edit tool event.",
            extra=build_log_extra(
                event_name,
                component="final_edit",
                stage="tool",
                session_id=state.session_id,
                final_index=state.final_index,
                tool_name=tool_name,
                tool_order=state.metadata.get("current_tool_order"),
                tool_params=state.metadata.get("current_tool_params"),
                input_shape=state.metadata.get("current_tool_input_shape"),
                output_shape=state.metadata.get("current_tool_output_shape")
                if phase == "completed"
                else None,
                used_fallback=state.fallback_to_original if phase == "failed" else None,
                outcome="failed" if phase == "failed" else phase,
            ),
        )

    async def _rollback_uploaded(self, uploaded_results: list[str]) -> None:
        if not uploaded_results:
            return

        for uploaded_path in uploaded_results:
            try:
                logger.info(
                    "Rolling back uploaded final image.",
                    extra=build_log_extra(
                        "final_edit.rollback_delete_final.started",
                        component="final_edit",
                        stage="rollback_delete_final",
                        outcome="started",
                        uploaded_path=uploaded_path,
                    ),
                )
                await self.uploader.delete_final(uploaded_path)
                logger.info(
                    "Rolled back uploaded final image.",
                    extra=build_log_extra(
                        "final_edit.rollback_delete_final.completed",
                        component="final_edit",
                        stage="rollback_delete_final",
                        outcome="succeeded",
                        uploaded_path=uploaded_path,
                    ),
                )
            except Exception:
                logger.warning(
                    "Failed to delete uploaded final image during rollback.",
                    extra=build_log_extra(
                        "final_edit.rollback_delete_final.completed",
                        component="final_edit",
                        stage="rollback_delete_final",
                        outcome="failed",
                        error_type="rollback_delete_failed",
                        uploaded_path=uploaded_path,
                    ),
                    exc_info=True,
                )


def get_final_edit_service(request: Request) -> FinalEditService:
    service = getattr(request.app.state, "final_edit_service", None)
    if service is None:
        reason = getattr(request.app.state, "final_edit_unavailable_reason", None)
        if reason:
            raise FinalEditUnavailableError(reason)
        raise FinalEditUnavailableError("Final edit service is not initialized.")
    return service


async def process_final_edit(
    session_id: str,
    payload: FinalEditRequest,
    redis: Redis,
    final_edit_service: FinalEditService,
) -> FinalEditResult:
    logger.info(
        "Loading final edit session context.",
        extra=build_log_extra(
            "final_edit.load_context.started",
            component="final_edit",
            stage="load_context",
            session_id=session_id,
            outcome="started",
        ),
    )
    context = await load_final_edit_session_context(redis, session_id)
    result = await final_edit_service.edit_and_upload(
        session_id=session_id,
        drafts=payload.drafts,
        context=context,
    )

    redis_payload = {
        "session_id": session_id,
        "status": result.status,
    }
    if payload.drafts:
        redis_payload["draft_count"] = str(len(payload.drafts))
    logger.info(
        "Persisting final edit result to redis.",
        extra=build_log_extra(
            "final_edit.persist_redis.started",
            component="final_edit",
            stage="persist_redis",
            session_id=session_id,
            outcome="started",
            draft_count=len(payload.drafts),
            result_status=result.status,
        ),
    )

    await upsert_content_session(
        redis,
        session_id,
        scalar_fields=redis_payload,
        photos=result.results,
        debug_fields=result.debug_fields,
    )
    logger.info(
        "Persisted final edit result to redis.",
        extra=build_log_extra(
            "final_edit.persist_redis.completed",
            component="final_edit",
            stage="persist_redis",
            session_id=session_id,
            outcome="succeeded",
            photo_count=len(result.results),
            result_status=result.status,
        ),
    )

    if result.status != STATUS_PHOTO_EDITED:
        await redis.hset(session_key(session_id), mapping={"status": STATUS_FAIL})

    return result
