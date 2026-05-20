"""Services for extracting the best frame and uploading draft images."""

from __future__ import annotations

import asyncio
import logging
import shutil
from dataclasses import dataclass
from pathlib import Path
import time

from fastapi import Request
from redis.asyncio import Redis

from app.core.config import settings
from app.logging import build_log_extra
from app.perfectframe.extractors import BestFrameExtractor
from app.perfectframe.image_processors import OpenCVImage
from app.perfectframe.schemas import ExtractorConfig, Image, ImageExtension
from app.perfectframe.video_processors import OpenCVVideo, _import_cv2
from app.schemas.sessions import ExtractFramesRequest
from app.services.s3_support import build_s3_client, normalize_s3_key
from app.services.sessions import STATUS_STARTED, STATUS_TEXT_GENERATED, session_key, upsert_content_session

STATUS_FAIL = STATUS_TEXT_GENERATED
STATUS_FRAME_EXTRACTED = "FRAME_EXTRACTED"
DRAFT_TOP_K = 3
logger = logging.getLogger(__name__)


@dataclass
class ExtractFramesResult:
    """Result returned by the frame extraction workflow."""

    status: str
    drafts: list[str]
    failure_reason: str | None = None
    debug_fields: dict[str, str] | None = None


class S3DraftUploader:
    """Upload extracted draft images to S3 when configured."""

    def __init__(self) -> None:
        self._settings = settings
        self._client = build_s3_client() if self._settings.s3_configured else None

    @property
    def is_configured(self) -> bool:
        return self._settings.s3_configured

    async def upload_frame(
        self,
        session_id: str,
        frame: Image,
        workdir: Path,
        draft_index: int = 1,
    ) -> str | None:
        if not self.is_configured:
            return None

        filename = f"draft-{draft_index:03d}.png"
        image_path = await asyncio.to_thread(
            OpenCVImage.save_image,
            frame,
            workdir,
            ImageExtension.PNG,
            filename,
        )
        return await asyncio.to_thread(
            self._upload_file,
            session_id,
            image_path,
            draft_index,
        )

    def _upload_file(self, session_id: str, image_path: Path, draft_index: int) -> str:
        object_key = f"ai-drafts/{session_id}/draft-{draft_index:03d}.png"
        assert self._client is not None
        self._client.upload_file(
            str(image_path),
            self._settings.S3_BUCKET_NAME,
            object_key,
            ExtraArgs={"ContentType": "image/png"},
        )
        return f"/{object_key}"


class S3VideoDownloader:
    """Download source videos from S3 using configured credentials."""

    def __init__(self) -> None:
        self._settings = settings
        self._client = build_s3_client() if self._settings.s3_configured else None

    @property
    def is_configured(self) -> bool:
        return self._settings.s3_configured

    async def download_video(self, video_key: str, destination: Path) -> None:
        await asyncio.to_thread(self._download_video, video_key, destination)

    def _download_video(self, video_key: str, destination: Path) -> None:
        assert self._client is not None
        self._client.download_file(
            self._settings.S3_BUCKET_NAME,
            normalize_s3_key(video_key),
            str(destination),
        )


class FrameExtractionService:
    """Download videos, extract the best frame, and upload it as a draft."""

    def __init__(
        self,
        extractor_config: ExtractorConfig,
        extractor: BestFrameExtractor,
        downloader: S3VideoDownloader,
        uploader: S3DraftUploader,
        temp_root: Path,
    ) -> None:
        self.config = extractor_config
        self.extractor = extractor
        self.downloader = downloader
        self.uploader = uploader
        self.temp_root = temp_root

    async def _upload_single_draft(
        self,
        session_id: str,
        video_key: str,
        session_dir: Path,
        draft_index: int,
        frame: Image,
        frame_score: float,
    ) -> tuple[int, str | None, dict[str, str]]:
        source_shape = "x".join(str(dimension) for dimension in frame.shape)
        draft_debug_fields = {
            f"debug:frame_shape:source:{draft_index}": source_shape,
            f"debug:frame_score:{draft_index}": f"{frame_score:.6f}",
        }
        logger.info(
            "Selected frame for draft.",
            extra=build_log_extra(
                "frame_extraction.select_frame",
                component="frame_extraction",
                stage="select_frame",
                session_id=session_id,
                outcome="succeeded",
                video_key=video_key,
                draft_index=draft_index,
                frame_shape=source_shape,
                frame_score=f"{frame_score:.6f}",
            ),
        )

        upload_started_at = time.perf_counter()
        logger.info(
            "Uploading extracted source frame as draft.",
            extra=build_log_extra(
                "frame_extraction.upload_frame.started",
                component="frame_extraction",
                stage="upload_frame",
                session_id=session_id,
                outcome="started",
                video_key=video_key,
                draft_index=draft_index,
                frame_shape=source_shape,
            ),
        )
        uploaded_path = await self.uploader.upload_frame(
            session_id,
            frame,
            session_dir,
            draft_index=draft_index,
        )
        elapsed_ms = int((time.perf_counter() - upload_started_at) * 1000)
        if uploaded_path is None:
            draft_debug_fields[f"debug:upload_result:{draft_index}"] = "none"
            logger.warning(
                "Frame upload returned no draft path.",
                extra=build_log_extra(
                    "frame_extraction.upload_frame.completed",
                    component="frame_extraction",
                    stage="upload_frame",
                    session_id=session_id,
                    outcome="failed",
                    error_type="s3_upload_unavailable",
                    video_key=video_key,
                    draft_index=draft_index,
                    elapsed_ms=elapsed_ms,
                ),
            )
            return draft_index, None, draft_debug_fields

        draft_debug_fields[f"debug:upload_result:{draft_index}"] = uploaded_path
        logger.info(
            "Draft frame uploaded.",
            extra=build_log_extra(
                "frame_extraction.upload_frame.completed",
                component="frame_extraction",
                stage="upload_frame",
                session_id=session_id,
                outcome="succeeded",
                video_key=video_key,
                draft_index=draft_index,
                draft_path=uploaded_path,
                frame_shape=source_shape,
                elapsed_ms=elapsed_ms,
            ),
        )
        return draft_index, uploaded_path, draft_debug_fields

    @staticmethod
    def _resolve_download_video_path(session_dir: Path, video_key: str) -> Path:
        suffix = Path(video_key).suffix.lower()
        if suffix in {"", "."}:
            suffix = ".mp4"
        return session_dir / f"input-video{suffix}"

    @staticmethod
    def _build_debug_fields(session_dir: Path, video_key: str, video_path: Path) -> dict[str, str]:
        return {
            "debug:video_key": video_key,
            "debug:session_dir": str(session_dir),
            "debug:video_path": str(video_path),
        }

    @staticmethod
    def _inspect_downloaded_video(video_path: Path) -> dict[str, object]:
        metadata: dict[str, object] = {
            "video_extension": video_path.suffix.lower() or ".mp4",
        }

        if not video_path.exists():
            metadata["video_metadata_status"] = "missing"
            return metadata

        metadata["video_size_bytes"] = video_path.stat().st_size
        metadata["video_size_mb"] = round(video_path.stat().st_size / (1024 * 1024), 2)

        try:
            cv2 = _import_cv2()
            capture = cv2.VideoCapture(str(video_path))
        except Exception as exc:
            metadata["video_metadata_status"] = "probe_error"
            metadata["video_metadata_error"] = exc.__class__.__name__
            return metadata

        try:
            if not capture.isOpened():
                metadata["video_metadata_status"] = "open_failed"
                return metadata

            width = round(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = capture.get(cv2.CAP_PROP_FPS)
            frame_count = capture.get(cv2.CAP_PROP_FRAME_COUNT)

            metadata["video_metadata_status"] = "ok"
            if width > 0 and height > 0:
                metadata["video_resolution"] = f"{width}x{height}"
                metadata["video_width"] = width
                metadata["video_height"] = height
            if fps and fps > 0:
                metadata["video_fps"] = round(fps, 3)
            if frame_count and frame_count > 0:
                metadata["video_frame_count"] = round(frame_count)
        finally:
            capture.release()

        return metadata

    def _s3_failure_result(
        self,
        session_id: str,
        video_key: str,
        debug_fields: dict[str, str],
    ) -> ExtractFramesResult:
        debug_fields["debug:stage"] = "s3_config_check"
        debug_fields["debug:s3_downloader_configured"] = str(
            self.downloader.is_configured
        ).lower()
        debug_fields["debug:s3_uploader_configured"] = str(
            self.uploader.is_configured
        ).lower()
        logger.warning(
            "Frame extraction aborted because S3 is unavailable.",
            extra=build_log_extra(
                "frame_extraction.s3_config_check",
                component="frame_extraction",
                stage="s3_config_check",
                session_id=session_id,
                outcome="failed",
                error_type="s3_unavailable",
                video_key=video_key,
                downloader_configured=self.downloader.is_configured,
                uploader_configured=self.uploader.is_configured,
            ),
        )
        return ExtractFramesResult(
            status=STATUS_FAIL,
            drafts=[],
            failure_reason="s3_unavailable",
            debug_fields=debug_fields,
        )

    async def _download_source_video(
        self,
        session_id: str,
        video_key: str,
        video_path: Path,
        debug_fields: dict[str, str],
    ) -> None:
        debug_fields["debug:stage"] = "download_video"
        download_started_at = time.perf_counter()
        logger.info(
            "Downloading source video for frame extraction.",
            extra=build_log_extra(
                "frame_extraction.download_video.started",
                component="frame_extraction",
                stage="download_video",
                session_id=session_id,
                outcome="started",
                video_key=video_key,
                video_path=str(video_path),
            ),
        )
        await self.downloader.download_video(video_key, video_path)
        debug_fields["debug:downloaded"] = str(video_path.exists()).lower()
        metadata = self._inspect_downloaded_video(video_path)
        if "video_size_bytes" in metadata:
            debug_fields["debug:video_size_bytes"] = str(metadata["video_size_bytes"])
        if "video_resolution" in metadata:
            debug_fields["debug:video_resolution"] = str(metadata["video_resolution"])
        if "video_fps" in metadata:
            debug_fields["debug:video_fps"] = str(metadata["video_fps"])
        if "video_frame_count" in metadata:
            debug_fields["debug:video_frame_count"] = str(metadata["video_frame_count"])
        if "video_metadata_status" in metadata:
            debug_fields["debug:video_metadata_status"] = str(metadata["video_metadata_status"])

        logger.info(
            "Video downloaded for frame extraction.",
            extra=build_log_extra(
                "frame_extraction.download_video.completed",
                component="frame_extraction",
                stage="download_video",
                session_id=session_id,
                outcome="succeeded",
                video_key=video_key,
                video_path=str(video_path),
                video_exists=video_path.exists(),
                **metadata,
                elapsed_ms=int((time.perf_counter() - download_started_at) * 1000),
            ),
        )

    async def _extract_top_k_frames(
        self,
        session_id: str,
        video_key: str,
        video_path: Path,
        debug_fields: dict[str, str],
    ) -> list[tuple[Image, float]] | None:
        debug_fields["debug:stage"] = "extract_frame"
        extract_started_at = time.perf_counter()
        metadata = self._inspect_downloaded_video(video_path)
        logger.info(
            "Running top-k frame extraction.",
            extra=build_log_extra(
                "frame_extraction.extract_frame.started",
                component="frame_extraction",
                stage="extract_frame",
                session_id=session_id,
                outcome="started",
                video_key=video_key,
                video_path=str(video_path),
                top_k=DRAFT_TOP_K,
                **metadata,
            ),
        )
        extracted = await asyncio.to_thread(
            self.extractor.extract_top_k_frames,
            video_path,
            DRAFT_TOP_K,
        )
        extractor_profile = getattr(self.extractor, "last_profile", {}) or {}
        for key, value in extractor_profile.items():
            debug_fields[f"debug:{key}"] = str(value)
        if not extracted:
            debug_fields["debug:extractor_result"] = "none"
            logger.warning(
                "Top-k frame extraction returned no frames.",
                extra=build_log_extra(
                    "frame_extraction.extract_frame.completed",
                    component="frame_extraction",
                    stage="extract_frame",
                    session_id=session_id,
                    outcome="failed",
                    error_type="frame_extraction_failed",
                    video_key=video_key,
                    video_path=str(video_path),
                    top_k=DRAFT_TOP_K,
                    **{
                        key: value
                        for key, value in OpenCVVideo.last_scan_metadata.items()
                        if value is not None
                    },
                    **extractor_profile,
                    elapsed_ms=int((time.perf_counter() - extract_started_at) * 1000),
                ),
            )
            return None

        debug_fields["debug:extractor_result"] = "frame_found"
        debug_fields["debug:extracted_count"] = str(len(extracted))
        logger.info(
            "Top-k frame extraction completed.",
            extra=build_log_extra(
                "frame_extraction.extract_frame.completed",
                component="frame_extraction",
                stage="extract_frame",
                session_id=session_id,
                outcome="succeeded",
                video_key=video_key,
                extracted_count=len(extracted),
                top_k=DRAFT_TOP_K,
                **{
                    key: value
                    for key, value in OpenCVVideo.last_scan_metadata.items()
                    if value is not None
                },
                **extractor_profile,
                elapsed_ms=int((time.perf_counter() - extract_started_at) * 1000),
            ),
        )
        return extracted

    async def _upload_drafts(
        self,
        session_id: str,
        video_key: str,
        session_dir: Path,
        extracted: list[tuple[Image, float]],
        debug_fields: dict[str, str],
    ) -> list[str] | None:
        tasks = [
            self._upload_single_draft(
                session_id,
                video_key,
                session_dir,
                draft_index,
                frame,
                frame_score,
            )
            for draft_index, (frame, frame_score) in enumerate(extracted, start=1)
        ]
        results = await asyncio.gather(*tasks)
        ordered_results = sorted(results, key=lambda item: item[0])
        uploaded_paths: list[str] = []
        for draft_index, uploaded_path, draft_debug_fields in ordered_results:
            debug_fields["debug:stage"] = f"upload_frame_{draft_index}"
            debug_fields.update(draft_debug_fields)
            if uploaded_path is None:
                return None
            uploaded_paths.append(uploaded_path)
        return uploaded_paths

    async def extract_and_upload(
        self,
        session_id: str,
        video_key: str,
    ) -> ExtractFramesResult:
        session_dir = self.temp_root / session_id
        session_dir.mkdir(parents=True, exist_ok=True)
        video_path = self._resolve_download_video_path(session_dir, video_key)
        debug_fields = self._build_debug_fields(session_dir, video_key, video_path)

        try:
            if not self.downloader.is_configured or not self.uploader.is_configured:
                return self._s3_failure_result(session_id, video_key, debug_fields)

            await self._download_source_video(
                session_id,
                video_key,
                video_path,
                debug_fields,
            )

            extracted = await self._extract_top_k_frames(
                session_id,
                video_key,
                video_path,
                debug_fields,
            )
            if extracted is None:
                return ExtractFramesResult(
                    status=STATUS_FAIL,
                    drafts=[],
                    failure_reason="frame_extraction_failed",
                    debug_fields=debug_fields,
                )
            uploaded_paths = await self._upload_drafts(
                session_id,
                video_key,
                session_dir,
                extracted,
                debug_fields,
            )
            if uploaded_paths is None:
                return ExtractFramesResult(
                    status=STATUS_FAIL,
                    drafts=[],
                    failure_reason="s3_upload_unavailable",
                    debug_fields=debug_fields,
                )

            debug_fields["debug:stage"] = "completed"
            debug_fields["debug:uploaded_count"] = str(len(uploaded_paths))
            logger.info(
                "Frame extraction completed successfully.",
                extra=build_log_extra(
                    "frame_extraction.completed",
                    component="frame_extraction",
                    stage="completed",
                    session_id=session_id,
                    outcome="succeeded",
                    video_key=video_key,
                    uploaded_count=len(uploaded_paths),
                ),
            )
            return ExtractFramesResult(
                status=STATUS_FRAME_EXTRACTED,
                drafts=uploaded_paths,
                debug_fields=debug_fields,
            )
        except Exception as exc:
            debug_fields["debug:stage"] = "exception"
            debug_fields["debug:exception_type"] = exc.__class__.__name__
            debug_fields["debug:exception_message"] = str(exc)
            logger.exception(
                "Frame extraction failed with an exception.",
                extra=build_log_extra(
                    "frame_extraction.exception",
                    component="frame_extraction",
                    stage=debug_fields["debug:stage"],
                    session_id=session_id,
                    outcome="failed",
                    error_type=exc.__class__.__name__,
                    video_key=video_key,
                ),
            )
            return ExtractFramesResult(
                status=STATUS_FAIL,
                drafts=[],
                failure_reason=str(exc),
                debug_fields=debug_fields,
            )
        finally:
            logger.info(
                "Cleaning up frame extraction temp directory.",
                extra=build_log_extra(
                    "frame_extraction.cleanup_tempdir",
                    component="frame_extraction",
                    stage="cleanup_tempdir",
                    session_id=session_id,
                    outcome="completed",
                    session_dir=str(session_dir),
                ),
            )
            shutil.rmtree(session_dir, ignore_errors=True)


def get_frame_extraction_service(request: Request) -> FrameExtractionService:
    """Return the shared frame extraction service from FastAPI state."""

    service = getattr(request.app.state, "frame_extraction_service", None)
    if service is None:
        raise RuntimeError("Frame extraction service is not initialized.")
    return service


async def process_extract_frames(
    session_id: str,
    payload: ExtractFramesRequest,
    redis: Redis,
    frame_service: FrameExtractionService,
) -> ExtractFramesResult:
    """Process a best-frame extraction request and persist session metadata."""

    result = await frame_service.extract_and_upload(
        session_id=session_id,
        video_key=payload.video,
    )

    existing_status = await redis.hget(session_key(session_id), "status")
    redis_payload = {
        "session_id": session_id,
        "video": payload.video,
        "status": result.status if result.status == STATUS_FRAME_EXTRACTED else (existing_status or STATUS_TEXT_GENERATED),
    }
    logger.info(
        "Persisting frame extraction result to redis.",
        extra=build_log_extra(
            "frame_extraction.persist_redis.started",
            component="frame_extraction",
            stage="persist_redis",
            session_id=session_id,
            outcome="started",
            video_key=payload.video,
            result_status=result.status,
            draft_count=len(result.drafts),
        ),
    )

    await upsert_content_session(
        redis,
        session_id,
        scalar_fields=redis_payload,
        drafts=result.drafts,
        debug_fields=result.debug_fields,
    )
    logger.info(
        "Persisted frame extraction result to redis.",
        extra=build_log_extra(
            "frame_extraction.persist_redis.completed",
            component="frame_extraction",
            stage="persist_redis",
            session_id=session_id,
            outcome="succeeded",
            result_status=result.status,
            draft_count=len(result.drafts),
        ),
    )
    return result
