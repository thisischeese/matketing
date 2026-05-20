import pytest

from app.core.config import settings
from app.services import final_edit_filter_catalog as filter_catalog_module
from app.services.final_edit_filter_catalog import FinalEditFilterCatalogError
from app.services.final_edit_planner import (
    TARGET_PROFILE_KIND_TONE,
    FinalEditPlannerClient,
    FinalEditPlanningError,
    FinalEditSessionContext,
    ImageEditPlan,
    build_default_final_edit_plan,
    build_final_edit_prompt,
    load_final_edit_filter_catalog,
    load_final_edit_session_context,
    normalize_image_edit_plan,
    parse_image_edit_plans,
    resolve_owner_persona_preset,
    resolve_owner_persona_target,
)
from app.services.final_edit_tools import AESTHETIC_TARGET_TONE_PROFILE
from app.services.sessions import session_key


def _planner_client() -> FinalEditPlannerClient:
    return FinalEditPlannerClient(
        model_settings=type(
            "Settings",
            (),
            {
                "base_url": "http://planner.test",
                "chat_endpoint": "/v1/chat/completions",
                "health_endpoint": "/health",
                "api_key": None,
                "timeout_seconds": 30.0,
                "max_tokens": 1024,
                "temperature": 0.2,
                "top_p": 0.9,
                "enabled": True,
            },
        )(),
        model_name="test-model",
    )


@pytest.mark.asyncio
async def test_load_final_edit_session_context_uses_draft_keywords(
    fake_redis_async,
) -> None:
    await fake_redis_async.hset(
        session_key("sess-1"),
        mapping={
            "owner_persona": "friendly",
            "caption": "dessert intro",
            "draft_keyword:2": "warm cafe",
            "draft_keyword:1": "signature cake",
            "final_keyword:1": "1001:cake",
        },
    )

    context = await load_final_edit_session_context(fake_redis_async, "sess-1")

    assert context.owner_persona == "friendly"
    assert context.caption == "dessert intro"
    assert context.keywords == ["signature cake", "warm cafe"]


@pytest.mark.asyncio
async def test_load_final_edit_session_context_falls_back_to_final_keywords(
    fake_redis_async,
) -> None:
    await fake_redis_async.hset(
        session_key("sess-2"),
        mapping={
            "caption": "store mood",
            "final_keyword:2": "2002:table",
            "final_keyword:1": "1001:cake",
        },
    )

    context = await load_final_edit_session_context(fake_redis_async, "sess-2")

    assert context.owner_persona == "aesthetic"
    assert context.keywords == ["cake", "table"]


def test_load_final_edit_filter_catalog_returns_named_filters() -> None:
    catalog = load_final_edit_filter_catalog()

    assert catalog
    assert catalog[0].name
    assert catalog[0].description
    assert catalog[0].pilgram2_name


def test_load_final_edit_filter_catalog_maps_display_name_to_callable_name() -> None:
    catalog = {definition.name: definition for definition in load_final_edit_filter_catalog()}

    assert catalog["X-Pro II"].pilgram2_name == "xpro2"
    assert catalog["Lo-Fi"].pilgram2_name == "lofi"


def test_load_final_edit_filter_catalog_rejects_missing_pilgram2_name(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    invalid_catalog = tmp_path / "invalid_filters.json"
    invalid_catalog.write_text(
        '[{"name":"Clarendon","description":"desc"}]',
        encoding="utf-8",
    )
    load_final_edit_filter_catalog.cache_clear()
    monkeypatch.setattr(
        filter_catalog_module,
        "FINAL_EDIT_FILTER_CATALOG_PATH",
        invalid_catalog,
    )

    with pytest.raises(FinalEditFilterCatalogError, match="pilgram2_name"):
        load_final_edit_filter_catalog()

    monkeypatch.undo()
    load_final_edit_filter_catalog.cache_clear()


def test_build_final_edit_prompt_includes_filter_catalog_and_rules() -> None:
    prompt = build_final_edit_prompt(
        FinalEditSessionContext(
            owner_persona="friendly",
            caption="cozy cake intro",
            keywords=["dessert", "latte"],
        ),
        3,
    )

    assert "[owner_persona] friendly" in prompt
    assert "[caption] cozy cake intro" in prompt
    assert '[filter catalog]' in prompt
    assert 'tools must be exactly ["filter"]' in prompt
    assert '"filter_name": "<catalog filter name>"' in prompt
    assert "Do not output any numeric editing parameters" in prompt
    assert "Valencia" in prompt
    assert "pilgram2_name" not in prompt
    assert "all 3 input images" in prompt
    assert "image_index from 0 to 2" in prompt
    assert "color_grading" not in prompt


def test_build_final_edit_prompt_keeps_aesthetic_target_profile_context() -> None:
    prompt = build_final_edit_prompt(
        FinalEditSessionContext(
            owner_persona="aesthetic",
            caption="airy lifestyle post",
            keywords=["morning"],
        ),
        1,
    )

    assert "[target_profile_kind] tone_profile" in prompt
    assert '"brightness"' in prompt
    assert '"tolerance"' in prompt
    assert (
        f"\"target\": {AESTHETIC_TARGET_TONE_PROFILE['brightness']['target']}"
        in prompt
    )


def test_resolve_owner_persona_preset_falls_back_to_aesthetic() -> None:
    persona, preset, used_fallback = resolve_owner_persona_preset("unknown")

    assert persona == "aesthetic"
    assert used_fallback is True
    assert preset == AESTHETIC_TARGET_TONE_PROFILE


def test_resolve_owner_persona_target_marks_aesthetic_as_tone_profile() -> None:
    persona, target, used_fallback, target_kind = resolve_owner_persona_target(
        "aesthetic"
    )

    assert persona == "aesthetic"
    assert used_fallback is False
    assert target == AESTHETIC_TARGET_TONE_PROFILE
    assert target_kind == TARGET_PROFILE_KIND_TONE


def test_normalize_image_edit_plan_orders_filter_before_upscale() -> None:
    normalized = normalize_image_edit_plan(
        ImageEditPlan(
            image_index=0,
            content="cake",
            strategy="unordered",
            tools=["upscale", "sharpen", "filter", "denoise"],
            params={
                "upscale": {"scale": 4},
                "sharpen": {"strength": 0.3},
                "filter": {"filter_name": "Valencia"},
                "denoise": {"strength": 0.4},
            },
        ),
        upscale_enabled=True,
    )

    assert normalized.tools == ["denoise", "sharpen", "filter", "upscale"]
    assert normalized.params["upscale"] == {"scale": 2}


def test_normalize_image_edit_plan_appends_default_upscale_to_non_empty_plan() -> None:
    normalized = normalize_image_edit_plan(
        ImageEditPlan(
            image_index=0,
            content="cake",
            strategy="filter only",
            tools=["filter"],
            params={"filter": {"filter_name": "Clarendon"}},
        ),
        upscale_enabled=True,
    )

    assert normalized.tools == ["filter", "upscale"]
    assert normalized.params["filter"] == {"filter_name": "Clarendon"}
    assert normalized.params["upscale"] == {"scale": 2}


def test_normalize_image_edit_plan_keeps_noop_plan_empty() -> None:
    normalized = normalize_image_edit_plan(
        ImageEditPlan(
            image_index=0,
            content="draft image",
            strategy="keep original",
            tools=[],
            params={},
        )
    )

    assert normalized.tools == []
    assert normalized.params == {}


def test_build_default_final_edit_plan_is_noop() -> None:
    plan = build_default_final_edit_plan(1)

    assert plan.image_index == 1
    assert plan.tools == []
    assert plan.params == {}


def test_build_final_edit_prompt_is_stable_when_upscale_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "FINAL_EDIT_ENABLE_UPSCALE", False)

    prompt = build_final_edit_prompt(
        FinalEditSessionContext(
            owner_persona="friendly",
            caption="cake",
            keywords=["dessert"],
        ),
        1,
    )

    assert 'tools must be exactly ["filter"]' in prompt
    assert "Upscale must be included for every image" not in prompt


def test_normalize_image_edit_plan_drops_upscale_when_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "FINAL_EDIT_ENABLE_UPSCALE", False)

    normalized = normalize_image_edit_plan(
        ImageEditPlan(
            image_index=0,
            content="cake",
            strategy="filter then upscale",
            tools=["upscale", "filter"],
            params={
                "upscale": {"scale": 4},
                "filter": {"filter_name": "Juno"},
            },
        )
    )

    assert normalized.tools == ["filter"]
    assert "upscale" not in normalized.params


def test_parse_image_edit_plans_accepts_filter_and_appends_upscale() -> None:
    plans = parse_image_edit_plans(
        """
        [
          {
            "image_index": 0,
            "content": "cake and latte on a wooden table",
            "strategy": "Warm cafe mood fits Valencia.",
            "tools": ["filter"],
            "params": {"filter": {"filter_name": "valencia"}}
          }
        ]
        """,
        upscale_enabled=True,
        owner_persona="friendly",
    )

    assert len(plans) == 1
    assert plans[0].tools == ["filter", "upscale"]
    assert plans[0].params["filter"]["filter_name"] == "Valencia"


def test_parse_image_edit_plans_rejects_catalog_miss() -> None:
    with pytest.raises(FinalEditPlanningError, match="Unsupported filter requested"):
        parse_image_edit_plans(
            """
            [
              {
                "image_index": 0,
                "content": "cake",
                "strategy": "unknown filter",
                "tools": ["filter"],
                "params": {"filter": {"filter_name": "Unknown Filter"}}
              }
            ]
            """
        )


def test_parse_image_edit_plans_requires_filter_name() -> None:
    with pytest.raises(FinalEditPlanningError, match="filter_name"):
        parse_image_edit_plans(
            """
            [
              {
                "image_index": 0,
                "content": "cake",
                "strategy": "missing filter",
                "tools": ["filter"],
                "params": {"filter": {}}
              }
            ]
            """
        )


def test_parse_image_edit_plans_rejects_color_grading_tool() -> None:
    with pytest.raises(FinalEditPlanningError, match='tools must be exactly \\["filter"\\]'):
        parse_image_edit_plans(
            """
            [
              {
                "image_index": 0,
                "content": "cake",
                "strategy": "grade it",
                "tools": ["color_grading"],
                "params": {"color_grading": {"contrast": -4}}
              }
            ]
            """
        )


@pytest.mark.asyncio
async def test_final_edit_planner_client_builds_validated_plans() -> None:
    client = _planner_client()
    original = settings.FINAL_EDIT_ENABLE_UPSCALE
    settings.FINAL_EDIT_ENABLE_UPSCALE = True

    try:
        async def fake_request_plan(image_paths, context):
            assert image_paths == ["a.jpg", "b.jpg"]
            assert context.owner_persona == "friendly"
            assert context.caption == "dessert intro"
            return """
            [
              {
                "image_index": 0,
                "content": "signature cake on a cafe table",
                "strategy": "The warm cafe mood and dessert subject fit Valencia best.",
                "tools": ["filter"],
                "params": {
                  "filter": {"filter_name": "Valencia"}
                }
              }
            ]
            """

        client._request_plan = fake_request_plan  # type: ignore[method-assign]

        plans = await client.build_plans(
            ["a.jpg", "b.jpg"],
            FinalEditSessionContext(
                owner_persona="friendly",
                caption="dessert intro",
                keywords=["cake"],
            ),
        )
    finally:
        settings.FINAL_EDIT_ENABLE_UPSCALE = original

    assert len(plans) == 1
    assert plans[0].image_index == 0
    assert plans[0].tools == ["filter", "upscale"]
    assert plans[0].params["filter"]["filter_name"] == "Valencia"


@pytest.mark.asyncio
async def test_final_edit_planner_client_rejects_out_of_range_index() -> None:
    client = _planner_client()

    async def fake_request_plan(image_paths, context):
        return """
        [
          {
            "image_index": 2,
            "content": "store interior",
            "strategy": "Soft clean mood fits Reyes.",
            "tools": ["filter"],
            "params": {
              "filter": {"filter_name": "Reyes"}
            }
          }
        ]
        """

    client._request_plan = fake_request_plan  # type: ignore[method-assign]

    with pytest.raises(FinalEditPlanningError, match="out-of-range"):
        await client.build_plans(
            ["a.jpg", "b.jpg"],
            FinalEditSessionContext(caption="store intro", keywords=["interior"]),
        )
