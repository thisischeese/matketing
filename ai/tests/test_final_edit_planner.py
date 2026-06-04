import pytest

from app.core.config import settings
from app.services import final_edit_filter_catalog as filter_catalog_module
from app.services.final_edit_filter_catalog import FinalEditFilterCatalogError
from app.services.final_edit_planner import (
    TARGET_PROFILE_KIND_TONE,
    CarouselOrderPlan,
    FinalEditPlannerClient,
    FinalEditPlanningError,
    FinalEditSessionContext,
    ImageEditPlan,
    build_carousel_order_dynamic_suffix,
    build_carousel_order_system_prompt,
    build_default_final_edit_plan,
    build_final_edit_dynamic_suffix,
    build_final_edit_messages,
    build_final_edit_prompt,
    build_final_edit_static_prefix,
    build_final_edit_system_prompt,
    load_final_edit_filter_catalog,
    load_final_edit_session_context,
    normalize_image_edit_plan,
    parse_carousel_order_plan,
    parse_final_edit_batch_plan,
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
    assert 'Allowed tool sequences: ["filter"], ["denoise","filter"], ["sharpen","filter"], ["denoise","sharpen","filter"], ["crop","filter"], ["crop","denoise","filter"], ["crop","sharpen","filter"], ["crop","denoise","sharpen","filter"].' in prompt
    assert 'Use exactly two top-level keys: "image_plans" and "carousel_order".' in prompt
    assert 'Allowed tools: crop, denoise, sharpen, filter.' in prompt
    assert '"filter_name":"<catalog filter name>"' in prompt
    assert '"subject_boxes":[[x1,y1,x2,y2],...]' in prompt
    assert 'sharpen params: {"strength":0.0~1.0}' in prompt
    assert "Do not output any other numeric edit params." in prompt
    assert "Valencia" in prompt
    assert "pilgram2_name" not in prompt
    assert "Plan all 3 input images." in prompt
    assert "image_plans must include image_index 0 to 2 exactly once." in prompt
    assert '"ordered_image_indexes":[0,1,2]' in prompt
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
    assert "[style_hint]" in prompt
    assert f"brightness={AESTHETIC_TARGET_TONE_PROFILE['brightness']['target']}" in prompt


def test_build_final_edit_static_prefix_is_stable() -> None:
    assert build_final_edit_static_prefix(3) == build_final_edit_static_prefix(3)
    assert "[filter catalog]" in build_final_edit_static_prefix(3)
    assert "Plan all 3 input images." in build_final_edit_static_prefix(3)
    assert "image_plans must include image_index 0 to 2 exactly once." in build_final_edit_static_prefix(3)


def test_build_final_edit_dynamic_suffix_contains_only_runtime_context() -> None:
    suffix = build_final_edit_dynamic_suffix(
        FinalEditSessionContext(
            owner_persona="friendly",
            caption="cozy cake intro",
            keywords=["dessert", "latte"],
        )
    )

    assert "[owner_persona] friendly" in suffix
    assert "[caption] cozy cake intro" in suffix
    assert "[keywords] dessert, latte" in suffix
    assert "[filter catalog]" not in suffix
    assert "Example output" not in suffix


def test_build_final_edit_system_prompt_keeps_strict_rules() -> None:
    system_prompt = build_final_edit_system_prompt(image_count=3)

    assert "Return only a JSON object." in system_prompt
    assert '"image_plans" must contain exactly 3 plans.' in system_prompt
    assert 'Allowed tool sequences: ["filter"], ["denoise","filter"], ["sharpen","filter"], ["denoise","sharpen","filter"], ["crop","filter"], ["crop","denoise","filter"], ["crop","sharpen","filter"], ["crop","denoise","sharpen","filter"].' in system_prompt
    assert '"subject_boxes":[[x1,y1,x2,y2],...]' in system_prompt
    assert 'sharpen params: {"strength":0.0~1.0}' in system_prompt
    assert "Do not output any other numeric edit params." in system_prompt


def test_build_final_edit_messages_places_fixed_text_before_images() -> None:
    from app.services import final_edit_planner as planner_module

    original = planner_module.image_path_to_data_uri
    planner_module.image_path_to_data_uri = lambda path: f"data:image/mock;base64,{path}"
    try:
        messages = build_final_edit_messages(
            FinalEditSessionContext(
                owner_persona="friendly",
                caption="cake",
                keywords=["dessert"],
            ),
            ["a.jpg"],
        )
    finally:
        planner_module.image_path_to_data_uri = original

    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"
    user_content = messages[1]["content"]
    assert user_content[0]["type"] == "text"
    assert "[filter catalog]" in user_content[0]["text"]
    assert "Plan all 1 input images." in user_content[0]["text"]
    assert user_content[1]["type"] == "text"
    assert "[caption] cake" in user_content[1]["text"]
    assert user_content[2]["type"] == "image_url"
    assert user_content[2]["image_url"]["url"] == "data:image/mock;base64,a.jpg"


def test_build_carousel_order_system_prompt_keeps_slot_roles_fixed() -> None:
    system_prompt = build_carousel_order_system_prompt()

    assert '"hook_scores" must be a list of three numbers from 0.0 to 1.0 aligned with "ordered_image_indexes".' in system_prompt
    assert '"hook_scores" must be sorted in non-increasing order' in system_prompt
    assert "Rank the three images by hooking power only." in system_prompt


def test_build_carousel_order_dynamic_suffix_includes_non_identity_example() -> None:
    suffix = build_carousel_order_dynamic_suffix(
        FinalEditSessionContext(
            owner_persona="friendly",
            caption="cozy cake intro",
            keywords=["dessert", "latte"],
        )
    )

    assert "Important: hook_scores must align with ordered_image_indexes in output order." in suffix
    assert "Important: hook_scores must be sorted from highest to lowest." in suffix
    assert '"ordered_image_indexes": [0, 2, 1]' in suffix
    assert '"hook_scores": [0.93, 0.74, 0.58]' in suffix
    assert "logger = logging.getLogger(__name__)" not in suffix


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

    assert 'Allowed tool sequences: ["filter"], ["denoise","filter"], ["sharpen","filter"], ["denoise","sharpen","filter"], ["crop","filter"], ["crop","denoise","filter"], ["crop","sharpen","filter"], ["crop","denoise","sharpen","filter"].' in prompt
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
    with pytest.raises(FinalEditPlanningError, match="Unsupported tool requested"):
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


def test_parse_image_edit_plans_accepts_crop_and_filter_and_appends_upscale() -> None:
    plans = parse_image_edit_plans(
        """
        [
          {
            "image_index": 0,
            "content": "cake and latte on a wooden table",
            "strategy": "Crop in on the dessert and apply a warm nostalgic filter.",
            "tools": ["crop", "filter"],
            "params": {
              "crop": {
                "subject_boxes": [[120, 80, 420, 520]],
                "keep_edges": ["top"]
              },
              "filter": {"filter_name": "valencia"}
            }
          }
        ]
        """,
        upscale_enabled=True,
        owner_persona="friendly",
    )

    assert len(plans) == 1
    assert plans[0].tools == ["crop", "filter", "upscale"]
    assert plans[0].params["crop"] == {
        "subject_boxes": [[120, 80, 420, 520]],
        "keep_edges": ["top"],
    }
    assert plans[0].params["filter"]["filter_name"] == "Valencia"


def test_parse_image_edit_plans_accepts_denoise_and_filter_and_appends_upscale() -> None:
    plans = parse_image_edit_plans(
        """
        [
          {
            "image_index": 0,
            "content": "night cafe cake",
            "strategy": "The low-light scene looks noisy, so denoise lightly before applying Reyes.",
            "tools": ["denoise", "filter"],
            "params": {
              "denoise": {"strength": 0.4},
              "filter": {"filter_name": "reyes"}
            }
          }
        ]
        """,
        upscale_enabled=True,
        owner_persona="friendly",
    )

    assert len(plans) == 1
    assert plans[0].tools == ["denoise", "filter", "upscale"]
    assert plans[0].params["denoise"] == {"strength": 0.4}
    assert plans[0].params["filter"]["filter_name"] == "Reyes"


def test_parse_image_edit_plans_accepts_sharpen_and_filter_and_appends_upscale() -> None:
    plans = parse_image_edit_plans(
        """
        [
          {
            "image_index": 0,
            "content": "cake close-up",
            "strategy": "The main subject looks a little soft, so sharpen it lightly before applying Reyes.",
            "tools": ["sharpen", "filter"],
            "params": {
              "sharpen": {"strength": 0.25},
              "filter": {"filter_name": "reyes"}
            }
          }
        ]
        """,
        upscale_enabled=True,
        owner_persona="friendly",
    )

    assert len(plans) == 1
    assert plans[0].tools == ["sharpen", "filter", "upscale"]
    assert plans[0].params["sharpen"] == {"strength": 0.25}
    assert plans[0].params["filter"]["filter_name"] == "Reyes"


def test_parse_image_edit_plans_accepts_crop_denoise_sharpen_and_filter_and_appends_upscale() -> None:
    plans = parse_image_edit_plans(
        """
        [
          {
            "image_index": 0,
            "content": "night cafe cake",
            "strategy": "Crop in on the dessert, reduce visible noise, sharpen lightly, then apply a warm filter.",
            "tools": ["crop", "denoise", "sharpen", "filter"],
            "params": {
              "crop": {
                "subject_boxes": [[120, 80, 420, 520]],
                "keep_edges": ["top"]
              },
              "denoise": {"strength": 0.35},
              "sharpen": {"strength": 0.2},
              "filter": {"filter_name": "valencia"}
            }
          }
        ]
        """,
        upscale_enabled=True,
        owner_persona="friendly",
    )

    assert len(plans) == 1
    assert plans[0].tools == ["crop", "denoise", "sharpen", "filter", "upscale"]
    assert plans[0].params["crop"] == {
        "subject_boxes": [[120, 80, 420, 520]],
        "keep_edges": ["top"],
    }
    assert plans[0].params["denoise"] == {"strength": 0.35}
    assert plans[0].params["sharpen"] == {"strength": 0.2}
    assert plans[0].params["filter"]["filter_name"] == "Valencia"


def test_parse_image_edit_plans_accepts_crop_denoise_and_filter_and_appends_upscale() -> None:
    plans = parse_image_edit_plans(
        """
        [
          {
            "image_index": 0,
            "content": "night cafe cake",
            "strategy": "Crop in on the dessert, reduce the visible noise, then apply a warm filter.",
            "tools": ["crop", "denoise", "filter"],
            "params": {
              "crop": {
                "subject_boxes": [[120, 80, 420, 520]],
                "keep_edges": ["top"]
              },
              "denoise": {"strength": 0.35},
              "filter": {"filter_name": "valencia"}
            }
          }
        ]
        """,
        upscale_enabled=True,
        owner_persona="friendly",
    )

    assert len(plans) == 1
    assert plans[0].tools == ["crop", "denoise", "filter", "upscale"]
    assert plans[0].params["crop"] == {
        "subject_boxes": [[120, 80, 420, 520]],
        "keep_edges": ["top"],
    }
    assert plans[0].params["denoise"] == {"strength": 0.35}
    assert plans[0].params["filter"]["filter_name"] == "Valencia"


def test_parse_image_edit_plans_requires_crop_subject_boxes() -> None:
    with pytest.raises(FinalEditPlanningError, match="subject_boxes"):
        parse_image_edit_plans(
            """
            [
              {
                "image_index": 0,
                "content": "cake",
                "strategy": "crop it",
                "tools": ["crop", "filter"],
                "params": {
                  "crop": {},
                  "filter": {"filter_name": "Reyes"}
                }
              }
            ]
            """
        )


def test_parse_image_edit_plans_rejects_empty_crop_subject_boxes() -> None:
    with pytest.raises(FinalEditPlanningError, match="subject_boxes"):
        parse_image_edit_plans(
            """
            [
              {
                "image_index": 0,
                "content": "cake",
                "strategy": "crop it",
                "tools": ["crop", "filter"],
                "params": {
                  "crop": {"subject_boxes": []},
                  "filter": {"filter_name": "Reyes"}
                }
              }
            ]
            """
        )


def test_parse_image_edit_plans_rejects_invalid_crop_box_shape() -> None:
    with pytest.raises(FinalEditPlanningError, match="list of four coordinates"):
        parse_image_edit_plans(
            """
            [
              {
                "image_index": 0,
                "content": "cake",
                "strategy": "crop it",
                "tools": ["crop", "filter"],
                "params": {
                  "crop": {"subject_boxes": [[1, 2, 3]]},
                  "filter": {"filter_name": "Reyes"}
                }
              }
            ]
            """
        )


def test_parse_image_edit_plans_rejects_invalid_crop_keep_edges() -> None:
    with pytest.raises(FinalEditPlanningError, match="keep_edges"):
        parse_image_edit_plans(
            """
            [
              {
                "image_index": 0,
                "content": "cake",
                "strategy": "crop it",
                "tools": ["crop", "filter"],
                "params": {
                  "crop": {
                    "subject_boxes": [[10, 10, 60, 100]],
                    "keep_edges": ["diagonal"]
                  },
                  "filter": {"filter_name": "Reyes"}
                }
              }
            ]
            """
        )


def test_parse_image_edit_plans_requires_filter_when_crop_is_used() -> None:
    with pytest.raises(FinalEditPlanningError, match="filter must be included"):
        parse_image_edit_plans(
            """
            [
              {
                "image_index": 0,
                "content": "cake",
                "strategy": "crop it",
                "tools": ["crop"],
                "params": {
                  "crop": {"subject_boxes": [[10, 10, 60, 100]]}
                }
              }
            ]
            """
        )


def test_parse_image_edit_plans_requires_denoise_strength() -> None:
    with pytest.raises(FinalEditPlanningError, match="denoise params must include strength"):
        parse_image_edit_plans(
            """
            [
              {
                "image_index": 0,
                "content": "cake",
                "strategy": "denoise it",
                "tools": ["denoise", "filter"],
                "params": {
                  "denoise": {},
                  "filter": {"filter_name": "Reyes"}
                }
              }
            ]
            """
        )


def test_parse_image_edit_plans_rejects_non_numeric_denoise_strength() -> None:
    with pytest.raises(FinalEditPlanningError, match="denoise strength must be numeric"):
        parse_image_edit_plans(
            """
            [
              {
                "image_index": 0,
                "content": "cake",
                "strategy": "denoise it",
                "tools": ["denoise", "filter"],
                "params": {
                  "denoise": {"strength": "high"},
                  "filter": {"filter_name": "Reyes"}
                }
              }
            ]
            """
        )


def test_parse_image_edit_plans_requires_sharpen_strength() -> None:
    with pytest.raises(FinalEditPlanningError, match="sharpen params must include strength"):
        parse_image_edit_plans(
            """
            [
              {
                "image_index": 0,
                "content": "cake",
                "strategy": "sharpen it",
                "tools": ["sharpen", "filter"],
                "params": {
                  "sharpen": {},
                  "filter": {"filter_name": "Reyes"}
                }
              }
            ]
            """
        )


def test_parse_image_edit_plans_rejects_non_numeric_sharpen_strength() -> None:
    with pytest.raises(FinalEditPlanningError, match="sharpen strength must be numeric"):
        parse_image_edit_plans(
            """
            [
              {
                "image_index": 0,
                "content": "cake",
                "strategy": "sharpen it",
                "tools": ["sharpen", "filter"],
                "params": {
                  "sharpen": {"strength": "high"},
                  "filter": {"filter_name": "Reyes"}
                }
              }
            ]
            """
        )


def test_parse_image_edit_plans_rejects_crop_after_filter() -> None:
    with pytest.raises(FinalEditPlanningError, match='tools must be one of'):
        parse_image_edit_plans(
            """
            [
              {
                "image_index": 0,
                "content": "cake",
                "strategy": "filter then crop",
                "tools": ["filter", "crop"],
                "params": {
                  "crop": {"subject_boxes": [[10, 10, 60, 100]]},
                  "filter": {"filter_name": "Reyes"}
                }
              }
            ]
            """
        )


def test_parse_image_edit_plans_rejects_denoise_after_filter() -> None:
    with pytest.raises(FinalEditPlanningError, match='tools must be one of'):
        parse_image_edit_plans(
            """
            [
              {
                "image_index": 0,
                "content": "cake",
                "strategy": "filter then denoise",
                "tools": ["filter", "denoise"],
                "params": {
                  "denoise": {"strength": 0.4},
                  "filter": {"filter_name": "Reyes"}
                }
              }
            ]
            """
        )


def test_parse_image_edit_plans_rejects_sharpen_after_filter() -> None:
    with pytest.raises(FinalEditPlanningError, match='tools must be one of'):
        parse_image_edit_plans(
            """
            [
              {
                "image_index": 0,
                "content": "cake",
                "strategy": "filter then sharpen",
                "tools": ["filter", "sharpen"],
                "params": {
                  "sharpen": {"strength": 0.2},
                  "filter": {"filter_name": "Reyes"}
                }
              }
            ]
            """
        )


def test_parse_carousel_order_plan_accepts_valid_payload() -> None:
    plan = parse_carousel_order_plan(
        """
          {
            "ordered_image_indexes": [2, 0, 1],
            "hook_scores": [0.91, 0.73, 0.52],
            "overall_confidence": 0.82,
            "reason": "The third image hooks best, the first is the second strongest hook, and the second is the weakest hook."
          }
        """
    )

    assert plan == CarouselOrderPlan(
        ordered_image_indexes=[2, 0, 1],
        hook_scores=[0.91, 0.73, 0.52],
        overall_confidence=0.82,
        reason="The third image hooks best, the first is the second strongest hook, and the second is the weakest hook.",
    )


def test_parse_carousel_order_plan_rejects_duplicate_indexes() -> None:
    with pytest.raises(FinalEditPlanningError, match="permutation"):
        parse_carousel_order_plan(
            """
              {
                "ordered_image_indexes": [0, 0, 2],
                "hook_scores": [0.91, 0.73, 0.52],
                "overall_confidence": 0.82,
                "reason": "invalid order"
              }
            """
        )


def test_parse_carousel_order_plan_rejects_non_descending_hook_scores() -> None:
    with pytest.raises(FinalEditPlanningError, match="hook_scores"):
        parse_carousel_order_plan(
            """
            {
              "ordered_image_indexes": [0, 1, 2],
              "hook_scores": [0.70, 0.91, 0.50],
              "overall_confidence": 0.82,
              "reason": "invalid hook score order"
            }
            """
        )


def test_parse_carousel_order_plan_rejects_non_numeric_confidence() -> None:
    with pytest.raises(FinalEditPlanningError, match="overall_confidence"):
        parse_carousel_order_plan(
            """
            {
              "ordered_image_indexes": [0, 1, 2],
              "hook_scores": [0.91, 0.73, 0.52],
              "overall_confidence": "high",
              "reason": "invalid confidence"
            }
            """
        )


def test_parse_final_edit_batch_plan_accepts_image_plans_and_carousel_order() -> None:
    batch_plan = parse_final_edit_batch_plan(
        """
        {
          "image_plans": [
            {
              "image_index": 0,
              "content": "cake hero shot",
              "strategy": "Keep the warm cafe mood and add a crisp nostalgic finish.",
              "tools": ["denoise", "sharpen", "filter"],
              "params": {
                "denoise": {"strength": 0.35},
                "sharpen": {"strength": 0.2},
                "filter": {"filter_name": "Valencia"}
              }
            },
            {
              "image_index": 1,
              "content": "texture close-up",
              "strategy": "Crop tighter and keep the warm cafe finish consistent.",
              "tools": ["crop", "filter"],
              "params": {
                "crop": {"subject_boxes": [[10, 10, 60, 100]], "keep_edges": []},
                "filter": {"filter_name": "Valencia"}
              }
            },
            {
              "image_index": 2,
              "content": "ambience shelf scene",
              "strategy": "Use a gentle warm finish for consistency.",
              "tools": ["filter"],
              "params": {
                "filter": {"filter_name": "Valencia"}
              }
            }
          ],
          "carousel_order": {
            "ordered_image_indexes": [2, 0, 1],
            "hook_scores": [0.91, 0.73, 0.52],
            "overall_confidence": 0.82,
            "reason": "The third image hooks best, the first is the second strongest hook, and the second is the weakest hook."
          }
        }
        """,
        image_count=3,
        upscale_enabled=False,
        owner_persona="friendly",
    )

    assert [plan.image_index for plan in batch_plan.image_plans] == [0, 1, 2]
    assert batch_plan.carousel_order is not None
    assert batch_plan.carousel_order.ordered_image_indexes == [2, 0, 1]
    assert batch_plan.carousel_order.hook_scores == [0.91, 0.73, 0.52]


def test_parse_final_edit_batch_plan_keeps_image_plans_when_only_carousel_order_is_invalid() -> None:
    batch_plan = parse_final_edit_batch_plan(
        """
        {
          "image_plans": [
            {
              "image_index": 0,
              "content": "cake hero shot",
              "strategy": "Keep the warm cafe mood.",
              "tools": ["filter"],
              "params": {"filter": {"filter_name": "Valencia"}}
            }
          ],
          "carousel_order": {
            "ordered_image_indexes": [0, 1, 2],
            "hook_scores": [0.52, 0.91, 0.73],
            "overall_confidence": 0.82,
            "reason": "invalid hook score order"
          }
        }
        """,
        image_count=3,
        upscale_enabled=False,
        owner_persona="friendly",
    )

    assert [plan.image_index for plan in batch_plan.image_plans] == [0]
    assert batch_plan.carousel_order is None
    assert batch_plan.carousel_order_error_message == "hook_scores must be sorted from highest to lowest."


@pytest.mark.asyncio
async def test_final_edit_planner_client_builds_validated_batch_plan() -> None:
    client = _planner_client()
    original = settings.FINAL_EDIT_ENABLE_UPSCALE
    settings.FINAL_EDIT_ENABLE_UPSCALE = True

    try:
        async def fake_request_plan(image_paths, context):
            assert image_paths == ["a.jpg", "b.jpg", "c.jpg"]
            assert context.owner_persona == "friendly"
            assert context.caption == "dessert intro"
            return """
            {
              "image_plans": [
                {
                  "image_index": 0,
                  "content": "signature cake on a cafe table",
                  "strategy": "Crop in on the cake, reduce low-light noise, sharpen lightly, and apply a warm cafe filter.",
                  "tools": ["crop", "denoise", "sharpen", "filter"],
                  "params": {
                    "crop": {
                      "subject_boxes": [[120, 80, 420, 520]],
                      "keep_edges": []
                    },
                    "denoise": {"strength": 0.35},
                    "sharpen": {"strength": 0.2},
                    "filter": {"filter_name": "Valencia"}
                  }
                }
              ],
              "carousel_order": {
                "ordered_image_indexes": [2, 0, 1],
                "hook_scores": [0.88, 0.70, 0.49],
                "overall_confidence": 0.78,
                "reason": "The third image is the strongest hook, the first is the second strongest hook, and the second is the weakest hook."
              }
            }
            """

        client._request_plan = fake_request_plan  # type: ignore[method-assign]

        batch_plan = await client.build_batch_plan(
            ["a.jpg", "b.jpg", "c.jpg"],
            FinalEditSessionContext(
                owner_persona="friendly",
                caption="dessert intro",
                keywords=["cake"],
            ),
        )
    finally:
        settings.FINAL_EDIT_ENABLE_UPSCALE = original

    assert len(batch_plan.image_plans) == 1
    assert batch_plan.image_plans[0].image_index == 0
    assert batch_plan.image_plans[0].tools == ["crop", "denoise", "sharpen", "filter", "upscale"]
    assert batch_plan.image_plans[0].params["crop"] == {
        "subject_boxes": [[120, 80, 420, 520]],
        "keep_edges": [],
    }
    assert batch_plan.image_plans[0].params["denoise"] == {"strength": 0.35}
    assert batch_plan.image_plans[0].params["sharpen"] == {"strength": 0.2}
    assert batch_plan.image_plans[0].params["filter"]["filter_name"] == "Valencia"
    assert batch_plan.carousel_order is not None
    assert batch_plan.carousel_order.ordered_image_indexes == [2, 0, 1]
    assert batch_plan.carousel_order.hook_scores == [0.88, 0.70, 0.49]


@pytest.mark.asyncio
async def test_final_edit_planner_client_rejects_out_of_range_index() -> None:
    client = _planner_client()

    async def fake_request_plan(image_paths, context):
        return """
        {
          "image_plans": [
            {
              "image_index": 2,
              "content": "store interior",
              "strategy": "Soft clean mood fits Reyes.",
              "tools": ["filter"],
              "params": {
                "filter": {"filter_name": "Reyes"}
              }
            }
          ],
          "carousel_order": null
        }
        """

    client._request_plan = fake_request_plan  # type: ignore[method-assign]

    with pytest.raises(FinalEditPlanningError, match="out-of-range"):
        await client.build_batch_plan(
            ["a.jpg", "b.jpg"],
            FinalEditSessionContext(caption="store intro", keywords=["interior"]),
        )
