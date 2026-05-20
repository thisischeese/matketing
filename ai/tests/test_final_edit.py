import asyncio
from pathlib import Path
from uuid import uuid4

import fakeredis
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import ValidationError

from app.main import app
from app.core.config import settings
from app.schemas.sessions import FinalEditResponse
from app.bootstrap import _initialize_final_edit_service
from app.services.final_edit import FinalEditResult, FinalEditService
from app.services.final_edit_runtime import FinalEditUnavailableError
from app.services.final_edit_planner import (
    FinalEditPlanningError,
    FinalEditSessionContext,
    ImageEditPlan,
)
from app.services.sessions import session_key
from tests.utils.session_support import (
    CapturingFinalUploader,
    FailOnCallPlannerClient,
    FakeFinalEditService,
    FakeUpscaler,
    FakePlannerClient,
    FailingPlannerClient,
    FailingUploader,
    FallbackAgent,
    RecordingPlannerClient,
    StubDraftDownloader,
    VALID_PAYLOAD,
)


@pytest.fixture(autouse=True)
def fake_realesrgan_upscaler(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "app.services.final_edit_tools.get_realesrgan_upscaler",
        lambda: FakeUpscaler(),
    )


@pytest.fixture(scope="module")
def service_loop() -> object:
    loop = asyncio.new_event_loop()
    try:
        yield loop
    finally:
        loop.close()


def _run_service(loop, awaitable):
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(awaitable)
    finally:
        asyncio.set_event_loop(None)


def test_final_edit_returns_success_and_persists_result(
    client: TestClient,
    fake_redis_sync: fakeredis.FakeStrictRedis,
) -> None:
    original_service = app.state.final_edit_service
    session_id = str(uuid4())
    drafts = [
        "/ai-drafts/session-123/draft-001.png",
        "/ai-drafts/session-123/draft-002.png",
    ]
    fake_service = FakeFinalEditService(
        FinalEditResult(
            status="PHOTO_EDITED",
            results=[
                "/ai-finals/session-123/final-001.png",
                "/ai-finals/session-123/final-002.png",
            ],
        )
    )
    app.state.final_edit_service = fake_service
    payload = {"session_id": session_id, "drafts": drafts}

    try:
        response = client.post(f"/ai/sessions/{session_id}/final-edit", json=payload)
    finally:
        app.state.final_edit_service = original_service

    assert response.status_code == 200
    body = response.json()
    assert body["session_id"] == session_id
    assert body["status"] == "PHOTO_EDITED"
    assert body["results"] == [
        "/ai-finals/session-123/final-001.png",
        "/ai-finals/session-123/final-002.png",
    ]
    assert fake_service.calls == [(session_id, drafts)]

    saved = fake_redis_sync.hgetall(session_key(session_id))
    assert saved["status"] == "PHOTO_EDITED"
    assert saved["photo:1"] == "/ai-finals/session-123/final-001.png"
    assert saved["photo:2"] == "/ai-finals/session-123/final-002.png"


def test_final_edit_preserves_existing_final_keywords(
    client: TestClient,
    fake_redis_sync: fakeredis.FakeStrictRedis,
) -> None:
    original_service = app.state.final_edit_service
    session_id = str(uuid4())
    fake_service = FakeFinalEditService(
        FinalEditResult(
            status="PHOTO_EDITED",
            results=["/ai-finals/session-123/final-001.png"],
        )
    )
    app.state.final_edit_service = fake_service

    try:
        process_response = client.post(
            f"/ai/sessions/{session_id}/process-utterance",
            json=VALID_PAYLOAD,
        )
        assert process_response.status_code == 200

        edit_response = client.post(
            f"/ai/sessions/{session_id}/final-edit",
            json={
                "session_id": session_id,
                "drafts": ["/ai-drafts/session-123/draft-001.png"],
            },
        )
    finally:
        app.state.final_edit_service = original_service

    assert edit_response.status_code == 200
    saved = fake_redis_sync.hgetall(session_key(session_id))
    assert saved["final_keyword:1"] == "signature menu"
    assert saved["final_keyword:2"] == "cozy table"


def test_final_edit_reuses_owner_persona_from_session_context(
    client: TestClient,
) -> None:
    class CapturingContextFinalEditService:
        def __init__(self) -> None:
            self.context = None

        async def edit_and_upload(self, session_id, drafts, context=None) -> FinalEditResult:
            self.context = context
            return FinalEditResult(
                status="PHOTO_EDITED",
                results=["/ai-finals/session-123/final-001.png"],
            )

    original_service = app.state.final_edit_service
    session_id = str(uuid4())
    process_payload = dict(VALID_PAYLOAD)
    process_payload["owner_persona"] = "friendly"
    capturing_service = CapturingContextFinalEditService()
    app.state.final_edit_service = capturing_service

    try:
        process_response = client.post(
            f"/ai/sessions/{session_id}/process-utterance",
            json=process_payload,
        )
        assert process_response.status_code == 200

        edit_response = client.post(
            f"/ai/sessions/{session_id}/final-edit",
            json={
                "session_id": session_id,
                "drafts": ["/ai-drafts/session-123/draft-001.png"],
            },
        )
    finally:
        app.state.final_edit_service = original_service

    assert edit_response.status_code == 200
    assert capturing_service.context is not None
    assert capturing_service.context.owner_persona == "friendly"


def test_final_edit_returns_fail_and_clears_photo_results(
    client: TestClient,
    fake_redis_sync: fakeredis.FakeStrictRedis,
) -> None:
    original_service = app.state.final_edit_service
    session_id = str(uuid4())
    fake_service = FakeFinalEditService(
        FinalEditResult(
            status="FRAME_EXTRACTED",
            results=[],
            failure_reason="final_edit_failed",
        )
    )
    app.state.final_edit_service = fake_service
    payload = {
        "session_id": session_id,
        "drafts": ["/ai-drafts/session-123/draft-001.png"],
    }

    try:
        response = client.post(f"/ai/sessions/{session_id}/final-edit", json=payload)
    finally:
        app.state.final_edit_service = original_service

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "FRAME_EXTRACTED"
    assert body["results"] == []

    saved = fake_redis_sync.hgetall(session_key(session_id))
    assert saved["status"] == "FRAME_EXTRACTED"
    assert "photo:1" not in saved


def test_final_edit_rejects_mismatched_session_id(client: TestClient) -> None:
    payload = {
        "session_id": str(uuid4()),
        "drafts": ["/ai-drafts/session-123/draft-001.png"],
    }

    response = client.post("/ai/sessions/edit-session-3/final-edit", json=payload)

    assert response.status_code == 422


def test_final_edit_returns_503_when_service_is_unavailable(
    client: TestClient,
) -> None:
    original_service = getattr(app.state, "final_edit_service", None)
    original_available = getattr(app.state, "final_edit_available", True)
    original_reason = getattr(app.state, "final_edit_unavailable_reason", None)
    session_id = str(uuid4())
    payload = {
        "session_id": session_id,
        "drafts": ["/ai-drafts/session-123/draft-001.png"],
    }
    app.state.final_edit_service = None
    app.state.final_edit_available = False
    app.state.final_edit_unavailable_reason = "FinalEditUnavailableError: OpenCV is unavailable"

    try:
        response = client.post(f"/ai/sessions/{session_id}/final-edit", json=payload)
    finally:
        app.state.final_edit_service = original_service
        app.state.final_edit_available = original_available
        app.state.final_edit_unavailable_reason = original_reason

    assert response.status_code == 503
    assert response.json()["detail"] == "Final edit is unavailable."


def test_initialize_final_edit_service_marks_unavailable_on_opencv_error(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    startup_app = FastAPI()

    def fail_import_cv2():
        raise FinalEditUnavailableError("OpenCV is unavailable for final edit.")

    monkeypatch.setattr("app.bootstrap.import_cv2", fail_import_cv2)

    _initialize_final_edit_service(startup_app, tmp_path)

    assert startup_app.state.final_edit_service is None
    assert startup_app.state.final_edit_planner_client is None
    assert startup_app.state.final_edit_available is False
    assert (
        startup_app.state.final_edit_unavailable_reason
        == "FinalEditUnavailableError: OpenCV is unavailable for final edit."
    )


def test_initialize_final_edit_service_marks_unavailable_on_upscaler_error(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    startup_app = FastAPI()

    monkeypatch.setattr(settings, "FINAL_EDIT_ENABLE_UPSCALE", True)
    monkeypatch.setattr("app.bootstrap.import_cv2", lambda: object())

    def fail_upscaler():
        raise FinalEditUnavailableError(
            "Real-ESRGAN runtime is unavailable for final edit."
        )

    monkeypatch.setattr(
        "app.bootstrap.ensure_final_edit_upscaler_available",
        fail_upscaler,
    )

    _initialize_final_edit_service(startup_app, tmp_path)

    assert startup_app.state.final_edit_service is None
    assert startup_app.state.final_edit_planner_client is None
    assert startup_app.state.final_edit_available is False
    assert (
        startup_app.state.final_edit_unavailable_reason
        == "FinalEditUnavailableError: Real-ESRGAN runtime is unavailable for final edit."
    )


def test_initialize_final_edit_service_skips_upscaler_when_disabled(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    startup_app = FastAPI()

    monkeypatch.setattr(settings, "FINAL_EDIT_ENABLE_UPSCALE", False)
    monkeypatch.setattr("app.bootstrap.import_cv2", lambda: object())

    def fail_upscaler():
        raise AssertionError("upscaler should not be preloaded when disabled")

    monkeypatch.setattr(
        "app.bootstrap.ensure_final_edit_upscaler_available",
        fail_upscaler,
    )

    _initialize_final_edit_service(startup_app, tmp_path)

    assert startup_app.state.final_edit_service is not None
    assert startup_app.state.final_edit_available is True
    assert startup_app.state.final_edit_upscale_enabled is False


def test_final_edit_response_limits_results_to_three() -> None:
    with pytest.raises(ValidationError):
        FinalEditResponse(
            session_id="sess",
            status="PHOTO_EDITED",
            results=["1", "2", "3", "4"],
        )


def test_final_edit_service_rolls_back_uploaded_results(tmp_path: Path, service_loop) -> None:
    uploader = FailingUploader()
    service = FinalEditService(
        downloader=StubDraftDownloader(),
        uploader=uploader,
        temp_root=tmp_path,
    )
    session_id = "session-rollback-1"

    result = _run_service(
        service_loop,
        service.edit_and_upload(
            session_id=session_id,
            drafts=[
                "/ai-drafts/session-rollback-1/draft-001.png",
                "/ai-drafts/session-rollback-1/draft-002.png",
            ],
        ),
    )

    assert result.status == "FRAME_EXTRACTED"
    assert result.results == []
    assert uploader.uploaded == ["/ai-finals/session-rollback-1/final-001.png"]
    assert uploader.deleted == ["/ai-finals/session-rollback-1/final-001.png"]
    assert not (tmp_path / session_id).exists()


def test_final_edit_service_marks_planner_fallback_on_planner_error(
    tmp_path: Path,
    service_loop,
) -> None:
    uploader = CapturingFinalUploader()
    service = FinalEditService(
        downloader=StubDraftDownloader(),
        uploader=uploader,
        temp_root=tmp_path,
        planner_client=FailingPlannerClient(),
        upscale_enabled=True,
    )

    result = _run_service(
        service_loop,
        service.edit_and_upload(
            session_id="session-planner-fallback-1",
            drafts=["/ai-drafts/session-planner-fallback-1/draft-001.png"],
            context=FinalEditSessionContext(
                caption="\ucfc4\ub81b \ucf00\uc774\ud06c \uc18c\uac1c",
                keywords=["\ucf00\uc774\ud06c"],
            ),
        ),
    )

    assert result.status == "PHOTO_EDITED"
    assert result.results == ["/ai-finals/session-planner-fallback-1/final-001.png"]
    assert result.debug_fields is not None
    assert result.debug_fields["debug:planner_fallback"] == "true"
    assert result.debug_fields["debug:owner_persona"] == "aesthetic"
    assert result.debug_fields["debug:target_preset_persona"] == "aesthetic"
    assert result.debug_fields["debug:target_preset_fallback"] == "false"
    assert result.debug_fields["debug:planner_failure_type"] == "RuntimeError"
    assert result.debug_fields["debug:planner_response_received"] == "false"
    assert result.debug_fields["debug:planned_indexes"] == ""
    assert result.debug_fields["debug:upscale_enabled"] == "true"
    assert result.debug_fields["debug:tools:1"] == ""
    assert result.debug_fields["debug:executed_tools:1"] == ""
    assert result.debug_fields["debug:plan_source:1"] == "planner_fallback"
    assert result.debug_fields["debug:filter_applied:1"] == "false"
    assert result.debug_fields["debug:upload_source_kind:1"] == "original_fallback"
    assert result.debug_fields["debug:output_path:1"].endswith("draft-001.png")
    assert uploader.source_paths[0].endswith("draft-001.png")


def test_final_edit_service_records_planner_failure_details(
    tmp_path: Path,
    service_loop,
) -> None:
    class RawPreviewFailingPlannerClient:
        def __init__(self) -> None:
            self.last_raw_output = '[{"image_index":0,"content":"cake"}]'

        async def build_plans(self, image_paths, context) -> list[ImageEditPlan]:
            raise FinalEditPlanningError("Planner payload is missing fields: ['params']")

    uploader = CapturingFinalUploader()
    service = FinalEditService(
        downloader=StubDraftDownloader(),
        uploader=uploader,
        temp_root=tmp_path,
        planner_client=RawPreviewFailingPlannerClient(),
        upscale_enabled=False,
    )

    result = _run_service(
        service_loop,
        service.edit_and_upload(
            session_id="session-planner-failure-debug-1",
            drafts=["/ai-drafts/session-planner-failure-debug-1/draft-001.png"],
            context=FinalEditSessionContext(
                caption="dessert showcase",
                keywords=["cake"],
            ),
        ),
    )

    assert result.status == "PHOTO_EDITED"
    assert result.debug_fields is not None
    assert result.debug_fields["debug:planner_failure_type"] == "FinalEditPlanningError"
    assert (
        result.debug_fields["debug:planner_failure_message"]
        == "Planner payload is missing fields: ['params']"
    )
    assert result.debug_fields["debug:planner_failure_image_index"] == "0"
    assert result.debug_fields["debug:planner_failure_raw_preview"] == (
        '[{"image_index":0,"content":"cake"}]'
    )
    assert result.debug_fields["debug:planner_response_received"] == "false"


def test_final_edit_service_marks_image_fallback_when_agent_falls_back(
    tmp_path: Path,
    service_loop,
) -> None:
    uploader = CapturingFinalUploader()
    service = FinalEditService(
        downloader=StubDraftDownloader(),
        uploader=uploader,
        temp_root=tmp_path,
        planner_client=FakePlannerClient(
            [
                ImageEditPlan(
                    image_index=0,
                    content="\ucf00\uc774\ud06c",
                    strategy="\ub178\uc774\uc988 \uc815\ub9ac",
                    tools=["denoise"],
                    params={"denoise": {"strength": 0.4}},
                )
            ]
        ),
        agent=FallbackAgent(),
    )

    result = _run_service(
        service_loop,
        service.edit_and_upload(
            session_id="session-image-fallback-1",
            drafts=["/ai-drafts/session-image-fallback-1/draft-001.png"],
            context=FinalEditSessionContext(
                caption="\ucfc4\ub81b \ucf00\uc774\ud06c \uc18c\uac1c",
                keywords=["\ucf00\uc774\ud06c"],
            ),
        ),
    )

    assert result.status == "PHOTO_EDITED"
    assert result.debug_fields is not None
    assert result.debug_fields["debug:planner_response_received"] == "true"
    assert result.debug_fields["debug:owner_persona"] == "aesthetic"
    assert result.debug_fields["debug:planned_indexes"] == "0"
    assert result.debug_fields["debug:image_fallback:1"] == "true"
    assert result.debug_fields["debug:image_failure:1"] == "tool_failed:denoise:RuntimeError"
    assert result.debug_fields["debug:executed_tools:1"] == ""
    assert result.debug_fields["debug:upload_source_kind:1"] == "original_fallback"
    assert result.debug_fields["debug:output_path:1"].endswith("draft-001.png")
    assert uploader.source_paths[0].endswith("draft-001.png")


def test_final_edit_service_marks_edited_upload_source_when_agent_succeeds(
    tmp_path: Path,
    service_loop,
) -> None:
    uploader = CapturingFinalUploader()
    service = FinalEditService(
        downloader=StubDraftDownloader(),
        uploader=uploader,
        temp_root=tmp_path,
        planner_client=FakePlannerClient(
            [
                ImageEditPlan(
                    image_index=0,
                    content="\ucf00\uc774\ud06c",
                    strategy="\ub178\uc774\uc988\ub97c \uc904\uc774\uace0 \uc120\uba85\ub3c4\ub97c \ub192\uc774\uae30",
                    tools=["denoise", "sharpen"],
                    params={
                        "denoise": {"strength": 0.4},
                        "sharpen": {"strength": 0.3},
                    },
                )
            ]
        ),
        upscale_enabled=True,
    )

    result = _run_service(
        service_loop,
        service.edit_and_upload(
            session_id="session-edited-upload-1",
            drafts=["/ai-drafts/session-edited-upload-1/draft-001.png"],
            context=FinalEditSessionContext(
                caption="\ub514\uc800\ud2b8 \ucf00\uc774\ud06c \uc18c\uac1c",
                keywords=["\ucf00\uc774\ud06c"],
            ),
        ),
    )

    assert result.status == "PHOTO_EDITED"
    assert result.debug_fields is not None
    assert result.debug_fields["debug:planner_response_received"] == "true"
    assert result.debug_fields["debug:owner_persona"] == "aesthetic"
    assert result.debug_fields["debug:planned_indexes"] == "0"
    assert result.debug_fields["debug:tools:1"] == "denoise,sharpen,upscale"
    assert result.debug_fields["debug:executed_tools:1"] == "denoise,sharpen,upscale"
    assert result.debug_fields["debug:upload_source_kind:1"] == "edited"
    assert "edited-000.png" in result.debug_fields["debug:output_path:1"]
    assert "edited-000.png" in uploader.source_paths[0]


def test_final_edit_service_calls_planner_once_per_image_and_remaps_indexes(
    tmp_path: Path,
    service_loop,
) -> None:
    uploader = CapturingFinalUploader()
    planner_client = RecordingPlannerClient(
        [
            [
                ImageEditPlan(
                    image_index=0,
                    content="first image",
                    strategy="denoise first",
                    tools=["denoise"],
                    params={"denoise": {"strength": 0.4}},
                )
            ],
            [
                ImageEditPlan(
                    image_index=0,
                    content="second image",
                    strategy="sharpen second",
                    tools=["sharpen"],
                    params={"sharpen": {"strength": 0.3}},
                )
            ],
            [
                ImageEditPlan(
                    image_index=0,
                    content="third image",
                    strategy="color grade third",
                    tools=["color_grading"],
                    params={"color_grading": {"contrast": -6}},
                )
            ],
        ]
    )
    service = FinalEditService(
        downloader=StubDraftDownloader(),
        uploader=uploader,
        temp_root=tmp_path,
        planner_client=planner_client,
        upscale_enabled=False,
    )

    drafts = [
        "/ai-drafts/session-multi-plan-1/draft-001.png",
        "/ai-drafts/session-multi-plan-1/draft-002.png",
        "/ai-drafts/session-multi-plan-1/draft-003.png",
    ]
    result = _run_service(
        service_loop,
        service.edit_and_upload(
            session_id="session-multi-plan-1",
            drafts=drafts,
            context=FinalEditSessionContext(
                caption="dessert showcase",
                keywords=["cake"],
            ),
        ),
    )

    assert result.status == "PHOTO_EDITED"
    assert len(planner_client.calls) == 3
    assert all(len(call) == 1 for call in planner_client.calls)
    assert planner_client.calls[0][0].endswith("draft-001.png")
    assert planner_client.calls[1][0].endswith("draft-002.png")
    assert planner_client.calls[2][0].endswith("draft-003.png")
    assert result.debug_fields is not None
    assert result.debug_fields["debug:planner_fallback"] == "false"
    assert result.debug_fields["debug:planned_image_count"] == "3"
    assert result.debug_fields["debug:planned_indexes"] == "0,1,2"
    assert result.debug_fields["debug:tools:1"] == "denoise"
    assert result.debug_fields["debug:tools:2"] == "sharpen"
    assert result.debug_fields["debug:tools:3"] == "color_grading"


def test_final_edit_service_falls_back_only_failed_images_when_any_planner_call_fails(
    tmp_path: Path,
    service_loop,
) -> None:
    uploader = CapturingFinalUploader()
    planner_client = FailOnCallPlannerClient(
        [
            [
                ImageEditPlan(
                    image_index=0,
                    content="first image",
                    strategy="denoise first",
                    tools=["denoise"],
                    params={"denoise": {"strength": 0.4}},
                )
            ],
            [
                ImageEditPlan(
                    image_index=0,
                    content="second image",
                    strategy="sharpen second",
                    tools=["sharpen"],
                    params={"sharpen": {"strength": 0.3}},
                )
            ],
            [
                ImageEditPlan(
                    image_index=0,
                    content="third image",
                    strategy="sharpen third",
                    tools=["sharpen"],
                    params={"sharpen": {"strength": 0.2}},
                )
            ],
        ],
        fail_on_call=1,
    )
    service = FinalEditService(
        downloader=StubDraftDownloader(),
        uploader=uploader,
        temp_root=tmp_path,
        planner_client=planner_client,
        upscale_enabled=False,
    )

    drafts = [
        "/ai-drafts/session-plan-fail-1/draft-001.png",
        "/ai-drafts/session-plan-fail-1/draft-002.png",
        "/ai-drafts/session-plan-fail-1/draft-003.png",
    ]
    result = _run_service(
        service_loop,
        service.edit_and_upload(
            session_id="session-plan-fail-1",
            drafts=drafts,
            context=FinalEditSessionContext(
                caption="dessert showcase",
                keywords=["cake"],
            ),
        ),
    )

    assert result.status == "PHOTO_EDITED"
    assert len(planner_client.calls) == 3
    assert all(len(call) == 1 for call in planner_client.calls)
    assert planner_client.calls[0][0].endswith("draft-001.png")
    assert planner_client.calls[1][0].endswith("draft-002.png")
    assert planner_client.calls[2][0].endswith("draft-003.png")
    assert result.debug_fields is not None
    assert result.debug_fields["debug:planner_fallback"] == "true"
    assert result.debug_fields["debug:planned_image_count"] == "2"
    assert result.debug_fields["debug:planned_indexes"] == "0,2"
    assert result.debug_fields["debug:tools:1"] == "denoise"
    assert result.debug_fields["debug:plan_source:2"] == "planner_fallback"
    assert result.debug_fields["debug:tools:2"] == ""
    assert result.debug_fields["debug:tools:3"] == "sharpen"
    assert result.debug_fields["debug:planner_failure_image_index"] == "1"
    assert result.debug_fields["debug:planner_failure_type:2"] == "RuntimeError"
    assert result.debug_fields["debug:planner_failure_message:2"] == "planner_failed"


def test_final_edit_service_applies_crop_and_records_debug_fields(
    tmp_path: Path,
    service_loop,
) -> None:
    uploader = CapturingFinalUploader()
    service = FinalEditService(
        downloader=StubDraftDownloader(),
        uploader=uploader,
        temp_root=tmp_path,
        planner_client=FakePlannerClient(
            [
                ImageEditPlan(
                    image_index=0,
                    content="cake",
                    strategy="crop first",
                    tools=["crop", "color_grading"],
                    params={
                        "crop": {
                            "subject_boxes": [[4, 4, 12, 12]],
                            "keep_edges": [],
                        },
                        "color_grading": {"contrast": -4},
                    },
                )
            ]
        ),
        upscale_enabled=True,
    )

    result = _run_service(
        service_loop,
        service.edit_and_upload(
            session_id="session-crop-debug-1",
            drafts=["/ai-drafts/session-crop-debug-1/draft-001.png"],
            context=FinalEditSessionContext(
                caption="dessert cake",
                keywords=["cake"],
            ),
        ),
    )

    assert result.status == "PHOTO_EDITED"
    assert result.debug_fields is not None
    assert result.debug_fields["debug:tools:1"] == "crop,color_grading,filter,upscale"
    assert result.debug_fields["debug:executed_tools:1"] == "crop,color_grading,filter,upscale"
    assert result.debug_fields["debug:crop_applied:1"] == "true"
    assert result.debug_fields["debug:crop_reason:1"] == "applied"
    assert result.debug_fields["debug:selected_filter:1"] == "Clarendon"
    assert float(result.debug_fields["debug:crop_output_ratio:1"]) == pytest.approx(1.0)


def test_final_edit_service_reorders_tools_before_execution(
    tmp_path: Path,
    service_loop,
) -> None:
    uploader = CapturingFinalUploader()
    service = FinalEditService(
        downloader=StubDraftDownloader(),
        uploader=uploader,
        temp_root=tmp_path,
        planner_client=FakePlannerClient(
            [
                ImageEditPlan(
                    image_index=0,
                    content="\ucf00\uc774\ud06c",
                    strategy="\uc798\ubabb\ub41c \uc21c\uc11c",
                    tools=["upscale", "sharpen", "denoise"],
                    params={
                        "upscale": {"scale": 4},
                        "sharpen": {"strength": 0.3},
                        "denoise": {"strength": 0.4},
                    },
                )
            ]
        ),
        upscale_enabled=True,
    )

    result = _run_service(
        service_loop,
        service.edit_and_upload(
            session_id="session-tool-order-1",
            drafts=["/ai-drafts/session-tool-order-1/draft-001.png"],
            context=FinalEditSessionContext(
                caption="\ub514\uc800\ud2b8 \ucf00\uc774\ud06c \uc18c\uac1c",
                keywords=["\ucf00\uc774\ud06c"],
            ),
        ),
    )

    assert result.status == "PHOTO_EDITED"
    assert result.debug_fields is not None
    assert result.debug_fields["debug:tools:1"] == "denoise,sharpen,upscale"
    assert result.debug_fields["debug:executed_tools:1"] == "denoise,sharpen,upscale"


def test_final_edit_service_deduplicates_upscale_to_single_execution(
    tmp_path: Path,
    service_loop,
) -> None:
    uploader = CapturingFinalUploader()
    service = FinalEditService(
        downloader=StubDraftDownloader(),
        uploader=uploader,
        temp_root=tmp_path,
        planner_client=FakePlannerClient(
            [
                ImageEditPlan(
                    image_index=0,
                    content="케이크",
                    strategy="duplicate upscale",
                    tools=["upscale", "sharpen", "upscale", "denoise"],
                    params={
                        "upscale": {"scale": 4},
                        "sharpen": {"strength": 0.3},
                        "denoise": {"strength": 0.4},
                    },
                )
            ]
        ),
        upscale_enabled=True,
    )

    result = _run_service(
        service_loop,
        service.edit_and_upload(
            session_id="session-duplicate-upscale-1",
            drafts=["/ai-drafts/session-duplicate-upscale-1/draft-001.png"],
            context=FinalEditSessionContext(
                caption="디저트 케이크 소개",
                keywords=["케이크"],
            ),
        ),
    )

    assert result.status == "PHOTO_EDITED"
    assert result.debug_fields is not None
    assert result.debug_fields["debug:tools:1"] == "denoise,sharpen,upscale"
    assert result.debug_fields["debug:executed_tools:1"] == "denoise,sharpen,upscale"


def test_final_edit_service_uses_upscale_only_when_plan_is_missing(
    tmp_path: Path,
    service_loop,
) -> None:
    uploader = CapturingFinalUploader()
    service = FinalEditService(
        downloader=StubDraftDownloader(),
        uploader=uploader,
        temp_root=tmp_path,
        planner_client=FakePlannerClient([]),
    )

    result = _run_service(
        service_loop,
        service.edit_and_upload(
            session_id="session-missing-plan-1",
            drafts=["/ai-drafts/session-missing-plan-1/draft-001.png"],
            context=FinalEditSessionContext(
                caption="\ub514\uc800\ud2b8 \ucf00\uc774\ud06c \uc18c\uac1c",
                keywords=["\ucf00\uc774\ud06c"],
            ),
        ),
    )

    assert result.status == "PHOTO_EDITED"
    assert result.debug_fields is not None
    assert result.debug_fields["debug:plan_source:1"] == "plan_missing_for_image"
    assert result.debug_fields["debug:tools:1"] == ""
    assert result.debug_fields["debug:executed_tools:1"] == ""
    assert result.debug_fields["debug:upload_source_kind:1"] == "original_fallback"


def test_final_edit_service_inserts_filter_before_upscale_only_plan(
    tmp_path: Path,
    service_loop,
) -> None:
    uploader = CapturingFinalUploader()
    service = FinalEditService(
        downloader=StubDraftDownloader(),
        uploader=uploader,
        temp_root=tmp_path,
        planner_client=FakePlannerClient(
            [
                ImageEditPlan(
                    image_index=0,
                    content="\ucf00\uc774\ud06c",
                    strategy="upscale only",
                    tools=["upscale"],
                    params={"upscale": {"scale": 2}},
                )
            ]
        ),
        upscale_enabled=True,
    )

    result = _run_service(
        service_loop,
        service.edit_and_upload(
            session_id="session-upscale-only-1",
            drafts=["/ai-drafts/session-upscale-only-1/draft-001.png"],
            context=FinalEditSessionContext(
                caption="\ub514\uc800\ud2b8 \ucf00\uc774\ud06c \uc18c\uac1c",
                keywords=["\ucf00\uc774\ud06c"],
            ),
        ),
    )

    assert result.status == "PHOTO_EDITED"
    assert result.debug_fields is not None
    assert result.debug_fields["debug:tools:1"] == "filter,upscale"
    assert result.debug_fields["debug:executed_tools:1"] == "filter,upscale"
    assert result.debug_fields["debug:selected_filter:1"] == "Clarendon"
    assert result.debug_fields["debug:selected_filter_backend:1"] == "pilgram2"
    assert result.debug_fields["debug:selected_filter_callable:1"] == "clarendon"


def test_final_edit_service_inserts_filter_between_denoise_and_upscale(
    tmp_path: Path,
    service_loop,
) -> None:
    uploader = CapturingFinalUploader()
    service = FinalEditService(
        downloader=StubDraftDownloader(),
        uploader=uploader,
        temp_root=tmp_path,
        planner_client=FakePlannerClient(
            [
                ImageEditPlan(
                    image_index=0,
                    content="\ucf00\uc774\ud06c",
                    strategy="denoise then upscale",
                    tools=["denoise", "upscale"],
                    params={
                        "denoise": {"strength": 0.4},
                        "upscale": {"scale": 2},
                    },
                )
            ]
        ),
        upscale_enabled=True,
    )

    result = _run_service(
        service_loop,
        service.edit_and_upload(
            session_id="session-denoise-upscale-1",
            drafts=["/ai-drafts/session-denoise-upscale-1/draft-001.png"],
            context=FinalEditSessionContext(
                caption="\ub514\uc800\ud2b8 \ucf00\uc774\ud06c \uc18c\uac1c",
                keywords=["\ucf00\uc774\ud06c"],
            ),
        ),
    )

    assert result.status == "PHOTO_EDITED"
    assert result.debug_fields is not None
    assert result.debug_fields["debug:tools:1"] == "denoise,filter,upscale"
    assert result.debug_fields["debug:executed_tools:1"] == "denoise,filter,upscale"
    assert result.debug_fields["debug:selected_filter:1"] == "Clarendon"
    assert result.debug_fields["debug:selected_filter_backend:1"] == "pilgram2"
    assert result.debug_fields["debug:selected_filter_callable:1"] == "clarendon"


def test_final_edit_service_marks_persona_preset_fallback_for_unknown_persona(
    tmp_path: Path,
    service_loop,
) -> None:
    uploader = CapturingFinalUploader()
    service = FinalEditService(
        downloader=StubDraftDownloader(),
        uploader=uploader,
        temp_root=tmp_path,
        planner_client=FailingPlannerClient(),
    )

    result = _run_service(
        service_loop,
        service.edit_and_upload(
            session_id="session-preset-fallback-1",
            drafts=["/ai-drafts/session-preset-fallback-1/draft-001.png"],
            context=FinalEditSessionContext(
                owner_persona="mystery",
                caption="디저트 소개",
                keywords=["케이크"],
            ),
        ),
    )

    assert result.status == "PHOTO_EDITED"
    assert result.debug_fields is not None
    assert result.debug_fields["debug:owner_persona"] == "mystery"
    assert result.debug_fields["debug:target_preset_persona"] == "aesthetic"
    assert result.debug_fields["debug:target_preset_fallback"] == "true"


def test_final_edit_service_omits_upscale_in_fallback_when_disabled(
    tmp_path: Path,
    service_loop,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "FINAL_EDIT_ENABLE_UPSCALE", False)
    uploader = CapturingFinalUploader()
    service = FinalEditService(
        downloader=StubDraftDownloader(),
        uploader=uploader,
        temp_root=tmp_path,
        planner_client=FailingPlannerClient(),
        upscale_enabled=False,
    )

    result = _run_service(
        service_loop,
        service.edit_and_upload(
            session_id="session-no-upscale-fallback-1",
            drafts=["/ai-drafts/session-no-upscale-fallback-1/draft-001.png"],
            context=FinalEditSessionContext(
                owner_persona="friendly",
                caption="cake",
                keywords=["dessert"],
            ),
        ),
    )

    assert result.status == "PHOTO_EDITED"
    assert result.debug_fields is not None
    assert result.debug_fields["debug:upscale_enabled"] == "false"
    assert result.debug_fields["debug:tools:1"] == ""
    assert result.debug_fields["debug:executed_tools:1"] == ""


def test_final_edit_service_strips_upscale_from_planner_plan_when_disabled(
    tmp_path: Path,
    service_loop,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "FINAL_EDIT_ENABLE_UPSCALE", False)
    uploader = CapturingFinalUploader()
    service = FinalEditService(
        downloader=StubDraftDownloader(),
        uploader=uploader,
        temp_root=tmp_path,
        planner_client=FakePlannerClient(
            [
                ImageEditPlan(
                    image_index=0,
                    content="cake",
                    strategy="denoise then upscale",
                    tools=["denoise", "upscale"],
                    params={
                        "denoise": {"strength": 0.4},
                        "upscale": {"scale": 2},
                    },
                )
            ]
        ),
        upscale_enabled=False,
    )

    result = _run_service(
        service_loop,
        service.edit_and_upload(
            session_id="session-no-upscale-plan-1",
            drafts=["/ai-drafts/session-no-upscale-plan-1/draft-001.png"],
            context=FinalEditSessionContext(
                caption="cake",
                keywords=["dessert"],
            ),
        ),
    )

    assert result.status == "PHOTO_EDITED"
    assert result.debug_fields is not None
    assert result.debug_fields["debug:upscale_enabled"] == "false"
    assert result.debug_fields["debug:tools:1"] == "denoise"
    assert result.debug_fields["debug:executed_tools:1"] == "denoise"


def test_final_edit_service_records_selected_filter_debug_fields(
    tmp_path: Path,
    service_loop,
) -> None:
    uploader = CapturingFinalUploader()
    service = FinalEditService(
        downloader=StubDraftDownloader(),
        uploader=uploader,
        temp_root=tmp_path,
        planner_client=FakePlannerClient(
            [
                ImageEditPlan(
                    image_index=0,
                    content="케이크",
                    strategy="aesthetic tone gap 보정",
                    tools=["filter"],
                    params={"filter": {"filter_name": "Valencia"}},
                )
            ]
        ),
    )

    result = _run_service(
        service_loop,
        service.edit_and_upload(
            session_id="session-selected-filter-1",
            drafts=["/ai-drafts/session-selected-filter-1/draft-001.png"],
            context=FinalEditSessionContext(
                owner_persona="aesthetic",
                caption="톤 보정",
                keywords=["케이크"],
            ),
        ),
    )

    assert result.status == "PHOTO_EDITED"
    assert result.debug_fields is not None
    assert result.debug_fields["debug:filter_applied:1"] == "true"
    assert result.debug_fields["debug:selected_filter:1"] == "Valencia"
    assert result.debug_fields["debug:selected_filter_backend:1"] == "pilgram2"
    assert result.debug_fields["debug:selected_filter_callable:1"] == "valencia"
    assert result.debug_fields["debug:filter_source:1"] == "planner"
    assert result.debug_fields["debug:upload_source_kind:1"] == "edited"
