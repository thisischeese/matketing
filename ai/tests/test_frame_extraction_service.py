from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from app.perfectframe.schemas import ExtractorConfig
from app.services.frame_extraction import (
    FrameExtractionService,
    S3DraftUploader,
    S3VideoDownloader,
)


class _NoopExtractor:
    def extract_top_k_frames(self, video_path: Path, k: int):
        return []


@pytest.fixture
def extractor_config(tmp_path: Path) -> ExtractorConfig:
    return ExtractorConfig(input_directory=tmp_path, output_directory=tmp_path)


def test_s3_video_downloader_reuses_client(monkeypatch: pytest.MonkeyPatch) -> None:
    client = MagicMock()
    monkeypatch.setattr("app.services.frame_extraction.settings.S3_BUCKET_NAME", "bucket")
    monkeypatch.setattr("app.services.frame_extraction.settings.S3_REGION", "ap-southeast-2")
    monkeypatch.setattr("app.services.frame_extraction.settings.S3_ACCESS_KEY", "access")
    monkeypatch.setattr("app.services.frame_extraction.settings.S3_SECRET_KEY", "secret")
    monkeypatch.setattr("app.services.frame_extraction.build_s3_client", lambda: client)

    downloader = S3VideoDownloader()
    assert downloader._client is client
    downloader._download_video("/inputs/video.webm", Path("video.webm"))

    client.download_file.assert_called_once_with("bucket", "inputs/video.webm", "video.webm")


def test_s3_draft_uploader_reuses_client(monkeypatch: pytest.MonkeyPatch) -> None:
    client = MagicMock()
    monkeypatch.setattr("app.services.frame_extraction.settings.S3_BUCKET_NAME", "bucket")
    monkeypatch.setattr("app.services.frame_extraction.settings.S3_REGION", "ap-southeast-2")
    monkeypatch.setattr("app.services.frame_extraction.settings.S3_ACCESS_KEY", "access")
    monkeypatch.setattr("app.services.frame_extraction.settings.S3_SECRET_KEY", "secret")
    monkeypatch.setattr("app.services.frame_extraction.build_s3_client", lambda: client)

    uploader = S3DraftUploader()
    assert uploader._client is client
    uploader._upload_file("session", Path("draft-001.png"), 1)

    client.upload_file.assert_called_once_with(
        "draft-001.png",
        "bucket",
        "ai-drafts/session/draft-001.png",
        ExtraArgs={"ContentType": "image/png"},
    )


@pytest.mark.asyncio
async def test_upload_drafts_preserves_draft_order(
    extractor_config: ExtractorConfig,
    tmp_path: Path,
) -> None:
    class SlowUploader:
        is_configured = True

        async def upload_frame(self, session_id: str, frame, workdir: Path, draft_index: int = 1) -> str:
            await asyncio.sleep({1: 0.02, 2: 0.0, 3: 0.01}[draft_index])
            return f"/ai-drafts/{session_id}/draft-{draft_index:03d}.png"

    service = FrameExtractionService(
        extractor_config=extractor_config,
        extractor=_NoopExtractor(),
        downloader=MagicMock(),
        uploader=SlowUploader(),
        temp_root=tmp_path,
    )
    extracted = [
        (np.zeros((4, 4, 3), dtype=np.uint8), 5.0),
        (np.zeros((4, 4, 3), dtype=np.uint8), 4.0),
        (np.zeros((4, 4, 3), dtype=np.uint8), 3.0),
    ]

    result = await service._upload_drafts(
        "session-1",
        "/inputs/video.webm",
        tmp_path,
        extracted,
        {},
    )

    assert result == [
        "/ai-drafts/session-1/draft-001.png",
        "/ai-drafts/session-1/draft-002.png",
        "/ai-drafts/session-1/draft-003.png",
    ]


@pytest.mark.asyncio
async def test_upload_drafts_returns_none_when_one_upload_fails(
    extractor_config: ExtractorConfig,
    tmp_path: Path,
) -> None:
    class PartiallyFailingUploader:
        is_configured = True

        async def upload_frame(self, session_id: str, frame, workdir: Path, draft_index: int = 1) -> str | None:
            if draft_index == 2:
                return None
            return f"/ai-drafts/{session_id}/draft-{draft_index:03d}.png"

    service = FrameExtractionService(
        extractor_config=extractor_config,
        extractor=_NoopExtractor(),
        downloader=MagicMock(),
        uploader=PartiallyFailingUploader(),
        temp_root=tmp_path,
    )
    extracted = [
        (np.zeros((4, 4, 3), dtype=np.uint8), 5.0),
        (np.zeros((4, 4, 3), dtype=np.uint8), 4.0),
        (np.zeros((4, 4, 3), dtype=np.uint8), 3.0),
    ]
    debug_fields: dict[str, str] = {}

    result = await service._upload_drafts(
        "session-2",
        "/inputs/video.webm",
        tmp_path,
        extracted,
        debug_fields,
    )

    assert result is None
    assert debug_fields["debug:upload_result:2"] == "none"
