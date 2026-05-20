from uuid import uuid4

import fakeredis
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from app.main import app
from app.schemas.sessions import ExtractFramesResponse
from app.services.frame_extraction import ExtractFramesResult
from app.services.sessions import session_key
from tests.utils.session_support import (
    FakeFrameExtractionService,
    VALID_EXTRACT_PAYLOAD,
    VALID_PAYLOAD,
)


def test_extract_frames_returns_success_and_persists_result(
    client: TestClient,
    fake_redis_sync: fakeredis.FakeStrictRedis,
) -> None:
    original_service = app.state.frame_extraction_service
    session_id = str(uuid4())
    fake_service = FakeFrameExtractionService(
        ExtractFramesResult(
            status="FRAME_EXTRACTED",
            drafts=["/ai-drafts/session-123/draft-001.png"],
        )
    )
    app.state.frame_extraction_service = fake_service
    payload = {"session_id": session_id, "video": "/inputs/test-session/test-video.mp4"}

    try:
        response = client.post(
            f"/ai/sessions/{session_id}/extract-frames",
            json=payload,
        )
    finally:
        app.state.frame_extraction_service = original_service

    assert response.status_code == 200
    body = response.json()
    assert body["session_id"] == session_id
    assert body["status"] == "FRAME_EXTRACTED"
    assert body["drafts"] == ["/ai-drafts/session-123/draft-001.png"]
    assert fake_service.calls == [(session_id, payload["video"])]

    saved = fake_redis_sync.hgetall(session_key(session_id))
    assert saved["status"] == "FRAME_EXTRACTED"
    assert saved["draft:1"] == "/ai-drafts/session-123/draft-001.png"
    assert saved["video"] == payload["video"]


def test_extract_frames_preserves_existing_final_keywords(
    client: TestClient,
    fake_redis_sync: fakeredis.FakeStrictRedis,
) -> None:
    original_service = app.state.frame_extraction_service
    session_id = str(uuid4())
    fake_service = FakeFrameExtractionService(
        ExtractFramesResult(
            status="FRAME_EXTRACTED",
            drafts=["/ai-drafts/session-123/draft-001.png"],
        )
    )
    app.state.frame_extraction_service = fake_service

    try:
        process_response = client.post(
            f"/ai/sessions/{session_id}/process-utterance",
            json=VALID_PAYLOAD,
        )
        assert process_response.status_code == 200

        extract_response = client.post(
            f"/ai/sessions/{session_id}/extract-frames",
            json={"session_id": session_id, "video": "/inputs/test-session/test-video.mp4"},
        )
    finally:
        app.state.frame_extraction_service = original_service

    assert extract_response.status_code == 200
    saved = fake_redis_sync.hgetall(session_key(session_id))
    assert saved["final_keyword:1"] == "signature menu"
    assert saved["final_keyword:2"] == "cozy table"


def test_extract_frames_returns_fail_when_service_fails(
    client: TestClient,
    fake_redis_sync: fakeredis.FakeStrictRedis,
) -> None:
    original_service = app.state.frame_extraction_service
    session_id = str(uuid4())
    fake_service = FakeFrameExtractionService(
        ExtractFramesResult(
            status="TEXT_GENERATED",
            drafts=[],
            failure_reason="s3_upload_unavailable",
        )
    )
    app.state.frame_extraction_service = fake_service
    payload = {"session_id": session_id, "video": "/inputs/test-session/test-video.mp4"}

    try:
        response = client.post(
            f"/ai/sessions/{session_id}/extract-frames",
            json=payload,
        )
    finally:
        app.state.frame_extraction_service = original_service

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "TEXT_GENERATED"
    assert body["drafts"] == []

    saved = fake_redis_sync.hgetall(session_key(session_id))
    assert saved["status"] == "TEXT_GENERATED"
    assert "draft:1" not in saved
    assert saved["video"] == payload["video"]


def test_extract_frames_persists_webm_video_key(
    client: TestClient,
    fake_redis_sync: fakeredis.FakeStrictRedis,
) -> None:
    original_service = app.state.frame_extraction_service
    session_id = str(uuid4())
    payload = {"session_id": session_id, "video": "/inputs/test-session/test-video.webm"}
    fake_service = FakeFrameExtractionService(
        ExtractFramesResult(
            status="FRAME_EXTRACTED",
            drafts=["/ai-drafts/session-123/draft-001.png"],
        )
    )
    app.state.frame_extraction_service = fake_service

    try:
        response = client.post(
            f"/ai/sessions/{session_id}/extract-frames",
            json=payload,
        )
    finally:
        app.state.frame_extraction_service = original_service

    assert response.status_code == 200
    assert fake_service.calls == [(session_id, payload["video"])]

    saved = fake_redis_sync.hgetall(session_key(session_id))
    assert saved["video"] == payload["video"]


def test_extract_frames_rejects_mismatched_session_id(client: TestClient) -> None:
    payload = dict(VALID_EXTRACT_PAYLOAD)
    payload["session_id"] = str(uuid4())

    response = client.post("/ai/sessions/extract-session-3/extract-frames", json=payload)

    assert response.status_code == 422


def test_extract_frames_response_limits_drafts_to_three() -> None:
    with pytest.raises(ValidationError):
        ExtractFramesResponse(
            session_id="sess",
            status="FRAME_EXTRACTED",
            drafts=["1", "2", "3", "4"],
        )


def test_frame_extraction_singletons_initialized_on_app_state(
    client: TestClient,
) -> None:
    assert app.state.frame_extractor_config is not None
    assert app.state.best_frame_extractor is not None
    assert app.state.frame_extraction_service is not None
    assert app.state.frame_extraction_service.extractor is app.state.best_frame_extractor
    assert hasattr(app.state, "final_edit_available")
    if app.state.final_edit_available:
        assert app.state.final_edit_service is not None
        assert app.state.final_edit_unavailable_reason is None
    else:
        assert app.state.final_edit_service is None
        assert app.state.final_edit_unavailable_reason is not None
    assert app.state.keyword_extraction_service is not None
    assert app.state.caption_generation_service is not None
    assert app.state.canonical_keyword_resolver_service is not None
