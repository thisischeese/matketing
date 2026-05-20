import io
import logging
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
import re

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.core.config import settings
from app.logging import build_log_extra
from app.main import app
from app.perfectframe.schemas import ExtractorConfig
from app.services.final_edit_planner import FinalEditSessionContext, ImageEditPlan
from app.services.final_edit import FinalEditService
from app.services.frame_extraction import FrameExtractionService
from tests.utils.session_support import FakeUpscaler


VALID_PAYLOAD = {
    "store_id": "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa",
    "utterance": "warm lighting cozy table signature menu",
    "owner_persona": "aesthetic",
    "date": "2026-04-27",
    "weather": {
        "temperature": 18.5,
        "precipitation": 0.0,
        "cloud_cover": "맑음",
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


@contextmanager
def capture_app_logs() -> Iterator[io.StringIO]:
    app_logger = logging.getLogger("app")
    assert app_logger.handlers
    handler = app_logger.handlers[0]
    original_stream = handler.stream
    stream = io.StringIO()
    handler.setStream(stream)
    try:
        yield stream
    finally:
        handler.flush()
        handler.setStream(original_stream)


class StubVideoDownloader:
    is_configured = True

    async def download_video(self, video_key: str, destination: Path) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(video_key.encode("utf-8"))


class StubDraftUploader:
    is_configured = True

    async def upload_frame(self, session_id: str, frame, workdir: Path, draft_index: int = 1) -> str:
        return f"/ai-drafts/{session_id}/draft-{draft_index:03d}.png"


class StubFrameExtractor:
    last_profile = {
        "extract_profile_decode_wait_ms": 11,
        "extract_profile_normalize_ms": 22,
        "extract_profile_inference_ms": 33,
        "extract_profile_candidate_collect_ms": 4,
        "extract_profile_segment_select_ms": 5,
        "extract_profile_batches": 1,
        "extract_profile_sampled_frames": 3,
    }

    def extract_top_k_frames(self, video_path: Path, k: int):
        return [
            (np.zeros((8, 8, 3), dtype=np.uint8), float(5 - index))
            for index in range(k)
        ]


class StubDraftDownloader:
    is_configured = True

    async def download_draft(self, draft_key: str, destination: Path) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        image = np.full((16, 16, 3), 120, dtype=np.uint8)
        image[:, :8, 1] = 220
        cv2.imwrite(str(destination), image)


class StubFinalUploader:
    is_configured = True

    async def upload_final(self, session_id: str, image_path: Path, final_index: int) -> str:
        return f"/ai-finals/{session_id}/final-{final_index:03d}.png"

    async def delete_final(self, uploaded_path: str) -> None:
        return None


class StubPlannerClient:
    def __init__(self, plans: list[ImageEditPlan]) -> None:
        self.plans = plans
        self.last_raw_output = (
            '[{"image_index":0,"content":"cake","strategy":"denoise then sharpen",'
            '"tools":["denoise","sharpen"],"params":{"denoise":{"strength":0.4},'
            '"sharpen":{"strength":0.3}}}]'
        )

    async def build_plans(self, image_paths, context) -> list[ImageEditPlan]:
        return self.plans


@pytest.fixture(autouse=True)
def fake_realesrgan_upscaler(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "app.services.final_edit_tools.get_realesrgan_upscaler",
        lambda: FakeUpscaler(),
    )


def test_formatter_includes_extra_fields_and_redacts_secrets() -> None:
    logger = logging.getLogger("app.tests.logging")

    with capture_app_logs() as stream:
        logger.info(
            "formatter test",
            extra=build_log_extra(
                "test.logging.formatter",
                video_key="/inputs/demo.mp4",
                keyword_count=2,
                s3_secret_key="top-secret",
            ),
        )

    output = stream.getvalue()
    assert 'event="test.logging.formatter"' in output
    assert 'video_key="/inputs/demo.mp4"' in output
    assert "keyword_count=2" in output
    assert 's3_secret_key="<redacted>"' in output


def test_request_logging_uses_same_request_id_for_start_and_finish(client: TestClient) -> None:
    with capture_app_logs() as stream:
        response = client.get("/ai/health")

    assert response.status_code == 200
    request_id = response.headers["x-request-id"]
    output = stream.getvalue()
    started_match = re.search(r'event="http.request.started".*request_id="([^"]+)"', output)
    completed_match = re.search(r'event="http.request.completed".*request_id="([^"]+)"', output)

    assert started_match is not None
    assert completed_match is not None
    assert started_match.group(1) == request_id
    assert completed_match.group(1) == request_id


def test_request_logging_hides_raw_identifiers_when_disabled(client: TestClient) -> None:
    original = settings.LOG_INCLUDE_RAW_IDENTIFIERS
    settings.LOG_INCLUDE_RAW_IDENTIFIERS = False

    try:
        with capture_app_logs() as stream:
            response = client.post(
                "/ai/sessions/log-session-1/process-utterance",
                json=VALID_PAYLOAD,
            )
    finally:
        settings.LOG_INCLUDE_RAW_IDENTIFIERS = original

    assert response.status_code == 200
    output = stream.getvalue()
    assert "utterance_length=39" in output
    assert "utterance_preview=" not in output
    assert "store_id=" not in output


def test_request_logging_includes_raw_identifiers_when_enabled(client: TestClient) -> None:
    original = settings.LOG_INCLUDE_RAW_IDENTIFIERS
    settings.LOG_INCLUDE_RAW_IDENTIFIERS = True

    try:
        with capture_app_logs() as stream:
            response = client.post(
                "/ai/sessions/log-session-2/process-utterance",
                json=VALID_PAYLOAD,
            )
    finally:
        settings.LOG_INCLUDE_RAW_IDENTIFIERS = original

    assert response.status_code == 200
    output = stream.getvalue()
    assert 'utterance_preview="warm lighting cozy table signature menu"' in output
    assert 'store_id="aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"' in output


@pytest.mark.asyncio
async def test_frame_extraction_emits_stage_logs(tmp_path: Path) -> None:
    service = FrameExtractionService(
        extractor_config=ExtractorConfig(input_directory=tmp_path, output_directory=tmp_path),
        extractor=StubFrameExtractor(),
        downloader=StubVideoDownloader(),
        uploader=StubDraftUploader(),
        temp_root=tmp_path,
    )

    with capture_app_logs() as stream:
        result = await service.extract_and_upload(
            session_id="frame-log-session",
            video_key="/inputs/frame-log.mp4",
        )

    assert result.status == "FRAME_EXTRACTED"
    assert len(result.drafts) == 3
    output = stream.getvalue()
    assert 'event="frame_extraction.download_video.started"' in output
    assert 'event="frame_extraction.download_video.completed"' in output
    assert 'event="frame_extraction.extract_frame.started"' in output
    assert 'event="frame_extraction.extract_frame.completed"' in output
    assert 'frame_score="5.000000"' in output
    assert 'frame_score="4.000000"' in output
    assert 'frame_score="3.000000"' in output
    assert "extract_profile_decode_wait_ms=11" in output
    assert "extract_profile_normalize_ms=22" in output
    assert "extract_profile_inference_ms=33" in output
    assert "extract_profile_candidate_collect_ms=4" in output
    assert "extract_profile_segment_select_ms=5" in output
    assert "extract_profile_batches=1" in output
    assert "extract_profile_sampled_frames=3" in output
    assert output.count('event="frame_extraction.upload_frame.started"') == 3
    assert output.count('event="frame_extraction.upload_frame.completed"') == 3
    assert 'frame_shape="8x8x3"' in output
    assert 'event="frame_extraction.cleanup_tempdir"' in output


@pytest.mark.asyncio
async def test_frame_extraction_logs_video_metadata(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    service = FrameExtractionService(
        extractor_config=ExtractorConfig(input_directory=tmp_path, output_directory=tmp_path),
        extractor=StubFrameExtractor(),
        downloader=StubVideoDownloader(),
        uploader=StubDraftUploader(),
        temp_root=tmp_path,
    )

    monkeypatch.setattr(
        service,
        "_inspect_downloaded_video",
        lambda path: {
            "video_extension": ".webm",
            "video_size_bytes": 7340032,
            "video_size_mb": 7.0,
            "video_metadata_status": "ok",
            "video_resolution": "3840x2160",
            "video_width": 3840,
            "video_height": 2160,
            "video_fps": 59.94,
            "video_frame_count": 1800,
        },
    )

    with capture_app_logs() as stream:
        await service.extract_and_upload(
            session_id="frame-meta-session",
            video_key="/inputs/frame-meta.webm",
        )

    output = stream.getvalue()
    assert 'video_extension=".webm"' in output
    assert "video_size_bytes=7340032" in output
    assert "video_size_mb=7.0" in output
    assert 'video_resolution="3840x2160"' in output
    assert "video_fps=59.94" in output
    assert "video_frame_count=1800" in output
    assert 'video_metadata_status="ok"' in output


def test_resolve_download_video_path_preserves_known_extension(tmp_path: Path) -> None:
    path = FrameExtractionService._resolve_download_video_path(
        tmp_path,
        "/inputs/test-session/draft.webm",
    )

    assert path == tmp_path / "input-video.webm"


def test_resolve_download_video_path_falls_back_to_mp4_without_extension(tmp_path: Path) -> None:
    path = FrameExtractionService._resolve_download_video_path(
        tmp_path,
        "/inputs/test-session/draft",
    )

    assert path == tmp_path / "input-video.mp4"


@pytest.mark.asyncio
async def test_final_edit_emits_stage_logs(tmp_path: Path) -> None:
    service = FinalEditService(
        downloader=StubDraftDownloader(),
        uploader=StubFinalUploader(),
        temp_root=tmp_path,
        planner_client=StubPlannerClient(
            [
                ImageEditPlan(
                    image_index=0,
                    content="케이크",
                    strategy="디노이즈 후 샤프닝",
                    tools=["filter"],
                    params={"filter": {"filter_name": "Valencia"}},
                )
            ]
        ),
        upscale_enabled=True,
    )

    with capture_app_logs() as stream:
        result = await service.edit_and_upload(
            session_id="final-log-session",
            drafts=["/ai-drafts/final-log-session/draft-001.png"],
            context=FinalEditSessionContext(
                caption="연말 케이크 소개",
                keywords=["케이크"],
            ),
        )

    assert result.status == "PHOTO_EDITED"
    output = stream.getvalue()
    assert 'event="final_edit.load_context.completed"' in output
    assert 'event="final_edit.plan.started"' in output
    assert 'event="final_edit.plan.completed"' in output
    assert 'planned_indexes=[0]' in output
    assert 'planned_tools_by_image=' in output
    assert 'planned_strategy_preview_by_image=' in output
    assert 'planner_response_preview=' in output
    assert 'event="final_edit.download_draft.started"' in output
    assert 'event="final_edit.download_draft.completed"' in output
    assert 'event="final_edit.tool.started"' in output
    assert 'tool_order=1' in output
    assert 'tool_params=' in output
    assert 'Valencia' in output
    assert 'valencia' in output
    assert 'input_shape="16x16x3"' in output
    assert 'event="final_edit.tool.completed"' in output
    assert 'output_shape="32x32x3"' in output
    assert 'event="final_edit.image.completed"' in output
    assert 'output_source="edited"' in output
    assert 'event="final_edit.upload_final.started"' in output
    assert 'upload_source_kind="edited"' in output
    assert 'upload_source_path=' in output
    assert 'event="final_edit.upload_final.completed"' in output
    assert 'event="final_edit.cleanup_tempdir"' in output
