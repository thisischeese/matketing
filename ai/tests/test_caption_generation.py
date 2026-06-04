import httpx

import pytest

from app.services.caption_generation import (
    DEFAULT_FALLBACK_GUIDE_TEXT,
    CaptionGenerationRequest,
    CaptionGenerationService,
    CaptionGenerationUnavailableError,
    CaptionPromptContext,
    build_caption_generation_dynamic_suffix,
    build_caption_generation_selective_suffix,
    build_caption_generation_static_prefix,
)


def _make_chat_response(
    status_code: int,
    *,
    content: str = "",
    url: str = "http://caption-server:8002/v1/chat/completions",
) -> httpx.Response:
    return httpx.Response(
        status_code,
        request=httpx.Request("POST", url),
        json={"choices": [{"message": {"content": content}}]},
    )


def _run_immediate(awaitable):
    iterator = awaitable.__await__()
    try:
        yielded = next(iterator)
    except StopIteration as exc:
        return exc.value

    while True:
        try:
            if hasattr(yielded, "__await__"):
                yielded = iterator.send(_run_immediate(yielded))
            else:
                yielded = iterator.send(None)
        except StopIteration as exc:
            return exc.value


def _build_request(**overrides) -> CaptionGenerationRequest:
    payload = {
        "draft_keywords": ["egg tart", "cozy cafe"],
        "owner_persona": "warm",
        "today": "2026-05-18",
        "utterance": "promote today's egg tart with a cozy tone",
        "weather_tags": ["PRECIP_CLEAR"],
        "menu_name": "egg tart",
        "menu_description": "buttery pastry with a crisp shell",
        "matched_keyword": "egg tart",
    }
    payload.update(overrides)
    return CaptionGenerationRequest(**payload)


def _build_prompt_context(**overrides) -> CaptionPromptContext:
    payload = {
        "owner_persona": "warm",
        "today": "2026-05-18",
        "weather_context": "clear day",
        "utterance_summary": "promote today's egg tart with a cozy tone",
        "context_profile": "basic_context",
        "draft_keywords": ["egg tart", "cozy cafe"],
        "matched_keyword": "egg tart",
        "menu_name": "egg tart",
        "menu_description": "buttery pastry with a crisp shell",
        "reference_captions": [
            "Warm dessert copy with cozy evening mood.",
            "Highlight the signature menu with a soft cafe tone.",
        ],
    }
    payload.update(overrides)
    return CaptionPromptContext(**payload)


def test_caption_generation_service_calls_remote_server_successfully(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = CaptionGenerationService(base_url="http://caption-server:8002")
    calls: list[tuple[bool, bool]] = []

    async def fake_post_chat_completion(
        prompt: str,
        *,
        include_response_format: bool,
        strict_language: bool,
    ):
        calls.append((include_response_format, strict_language))
        return _make_chat_response(
            200,
            content=(
                '{"guide_text":"메뉴가 잘 보이도록 가까이 촬영해보세요.",'
                '"caption":"오늘의 에그타르트를 따뜻하게 소개해보세요."}'
            ),
        )

    monkeypatch.setattr(service, "_post_chat_completion", fake_post_chat_completion)

    result = _run_immediate(service.generate_text(_build_request()))

    assert calls == [(True, False)]
    assert result.guide_text == "메뉴가 잘 보이도록 가까이 촬영해보세요."
    assert result.draft_caption == "오늘의 에그타르트를 따뜻하게 소개해보세요."
    assert result.stored_caption == result.draft_caption


def test_caption_generation_service_includes_model_name_in_payload() -> None:
    service = CaptionGenerationService(
        base_url="http://caption-server:8002",
        model_name="exaone-text",
    )

    dynamic_suffix = service._pipeline.build_dynamic_suffix(_build_request())
    payload = service._build_request_payload(
        dynamic_suffix,
        include_response_format=False,
        strict_language=False,
    )

    assert payload["model"] == "exaone-text"
    assert [message["role"] for message in payload["messages"]] == [
        "system",
        "user",
        "user",
    ]


def test_caption_generation_service_retries_without_response_format(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = CaptionGenerationService(base_url="http://caption-server:8002")
    calls: list[tuple[bool, bool]] = []

    async def fake_post_chat_completion(
        prompt: str,
        *,
        include_response_format: bool,
        strict_language: bool,
    ):
        calls.append((include_response_format, strict_language))
        if include_response_format:
            return httpx.Response(
                400,
                request=httpx.Request(
                    "POST",
                    "http://caption-server:8002/v1/chat/completions",
                ),
                text="response_format unsupported",
            )
        return _make_chat_response(
            200,
            content='{"guide_text":"구도를 조금 더 가까이 잡아보세요.","caption":"비 오는 날 메뉴를 소개합니다."}',
        )

    monkeypatch.setattr(service, "_post_chat_completion", fake_post_chat_completion)

    result = _run_immediate(
        service.generate_text(_build_request(weather_tags=["PRECIP_RAIN"]))
    )

    assert calls == [(True, False), (False, False)]
    assert result.draft_caption == "비 오는 날 메뉴를 소개합니다."


def test_caption_generation_service_does_not_retry_on_validation_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = CaptionGenerationService(base_url="http://caption-server:8002")
    calls: list[tuple[bool, bool]] = []

    async def fake_post_chat_completion(
        prompt: str,
        *,
        include_response_format: bool,
        strict_language: bool,
    ):
        calls.append((include_response_format, strict_language))
        return httpx.Response(
            400,
            request=httpx.Request(
                "POST",
                "http://caption-server:8002/v1/chat/completions",
            ),
            text="schema validation failed: caption is required",
        )

    monkeypatch.setattr(service, "_post_chat_completion", fake_post_chat_completion)

    with pytest.raises(CaptionGenerationUnavailableError, match="HTTP 400"):
        _run_immediate(service.generate_text(_build_request()))

    assert calls == [(True, False)]
    assert service._response_format_supported is True


def test_caption_generation_service_caches_response_format_unsupported(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = CaptionGenerationService(base_url="http://caption-server:8002")
    calls: list[tuple[bool, bool]] = []

    async def fake_post_chat_completion(
        prompt: str,
        *,
        include_response_format: bool,
        strict_language: bool,
    ):
        calls.append((include_response_format, strict_language))
        if include_response_format:
            return httpx.Response(
                400,
                request=httpx.Request(
                    "POST",
                    "http://caption-server:8002/v1/chat/completions",
                ),
                text="response_format unsupported",
            )
        return _make_chat_response(
            200,
            content='{"guide_text":"가이드입니다.","caption":"캡션입니다."}',
        )

    monkeypatch.setattr(service, "_post_chat_completion", fake_post_chat_completion)

    request = _build_request()
    _run_immediate(service.generate_text(request))
    _run_immediate(service.generate_text(request))

    assert calls == [(True, False), (False, False), (False, False)]
    assert service._response_format_supported is False


def test_caption_generation_service_uses_menu_promotion_prompt_only() -> None:
    service = CaptionGenerationService(base_url="http://caption-server:8002")

    prompt = service._pipeline.build_prompt(_build_request())

    assert "today: 2026-05-18" in prompt
    assert "draft_keywords: egg tart, cozy cafe" in prompt
    assert "menu_name: egg tart" in prompt
    assert "menu_description: buttery pastry with a crisp shell" in prompt
    assert "reference_captions:" not in prompt
    assert "selected_menu_name" not in prompt


def test_caption_generation_static_prefix_excludes_runtime_fields() -> None:
    prefix = build_caption_generation_static_prefix()

    assert '"guide_text": "..."' in prefix
    assert '"caption": "..."' in prefix
    assert "owner_persona:" not in prefix
    assert "draft_keywords:" not in prefix
    assert "menu_description" not in prefix


def test_caption_generation_dynamic_suffix_contains_only_runtime_fields() -> None:
    request = _build_request()
    suffix = build_caption_generation_dynamic_suffix(
        owner_persona=request.owner_persona,
        today=request.today,
        weather_context="clear day",
        utterance=request.utterance,
        draft_keywords=", ".join(request.draft_keywords),
        matched_keyword=request.matched_keyword or "",
        menu_name=request.menu_name or "",
        menu_description=request.menu_description or "",
    )

    assert "[filter catalog]" not in suffix
    assert "owner_persona: warm" in suffix
    assert "today: 2026-05-18" in suffix
    assert "weather_context: clear day" in suffix
    assert "draft_keywords:" in suffix
    assert "menu_description:" in suffix


def test_selective_suffix_omits_menu_fields_for_basic_context() -> None:
    suffix = build_caption_generation_selective_suffix(
        owner_persona="warm",
        today="2026-05-18",
        weather_context="clear day",
        utterance="highlight the dessert",
        draft_keywords="egg tart, beans",
        context_profile="basic_context",
        matched_keyword="egg tart",
        menu_name="egg tart",
        menu_description="buttery and crisp",
        reference_captions="reference one | reference two",
    )

    assert "owner_persona: warm" in suffix
    assert "draft_keywords: egg tart, beans" in suffix
    assert "menu_name:" not in suffix
    assert "menu_description:" not in suffix
    assert "reference_captions:" not in suffix


def test_selective_pipeline_uses_reference_context_only() -> None:
    service = CaptionGenerationService(base_url="http://caption-server:8002")
    request = _build_request(
        context_routing_variant="selective",
        context_profile="reference_context",
        menu_name="egg tart",
        menu_description=None,
        matched_keyword="egg tart",
        prompt_context=_build_prompt_context(
            context_profile="reference_context",
            menu_name=None,
            menu_description=None,
            matched_keyword=None,
        ),
    )

    prompt = service._pipeline.build_prompt(request)

    assert "reference_captions:" in prompt
    assert "menu_name:" not in prompt
    assert "menu_description:" not in prompt
    assert "(없음)" not in prompt


def test_selective_pipeline_uses_menu_context_without_reference_captions() -> None:
    service = CaptionGenerationService(base_url="http://caption-server:8002")
    request = _build_request(
        context_routing_variant="selective",
        context_profile="menu_context",
        prompt_context=_build_prompt_context(
            context_profile="menu_context",
            reference_captions=[],
        ),
    )

    prompt = service._pipeline.build_prompt(request)

    assert "menu_name:" in prompt
    assert "menu_description:" in prompt
    assert "reference_captions:" not in prompt


def test_caption_generation_payload_uses_static_and_dynamic_messages() -> None:
    service = CaptionGenerationService(base_url="http://caption-server:8002")
    request = _build_request()
    dynamic_suffix = service._pipeline.build_dynamic_suffix(request)

    payload = service._build_request_payload(
        dynamic_suffix,
        include_response_format=False,
        strict_language=False,
    )

    assert payload["messages"][0]["role"] == "system"
    assert payload["messages"][1] == {
        "role": "user",
        "content": build_caption_generation_static_prefix(),
    }
    assert payload["messages"][2] == {
        "role": "user",
        "content": dynamic_suffix,
    }


def test_caption_generation_strict_retry_adds_only_final_user_message() -> None:
    service = CaptionGenerationService(base_url="http://caption-server:8002")
    dynamic_suffix = service._pipeline.build_dynamic_suffix(_build_request())

    default_payload = service._build_request_payload(
        dynamic_suffix,
        include_response_format=False,
        strict_language=False,
    )
    strict_payload = service._build_request_payload(
        dynamic_suffix,
        include_response_format=False,
        strict_language=True,
    )

    assert strict_payload["messages"][:3] == default_payload["messages"]
    assert len(strict_payload["messages"]) == 4
    assert strict_payload["messages"][3] == {
        "role": "user",
        "content": service._build_user_retry_suffix(),
    }


def test_caption_generation_service_retries_with_stricter_korean_prompt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = CaptionGenerationService(base_url="http://caption-server:8002")
    calls: list[tuple[bool, bool]] = []

    async def fake_post_chat_completion(
        prompt: str,
        *,
        include_response_format: bool,
        strict_language: bool,
    ):
        calls.append((include_response_format, strict_language))
        if len(calls) == 1:
            return _make_chat_response(
                200,
                content=(
                    '{"guide_text":"Show the signature menu clearly.",'
                    '"caption":"Fresh pastry for today."}'
                ),
            )
        return _make_chat_response(
            200,
            content=(
                '{"guide_text":"에그타르트가 잘 보이도록 촬영해보세요.",'
                '"caption":"오늘의 에그타르트를 따뜻하게 소개해보세요."}'
            ),
        )

    monkeypatch.setattr(service, "_post_chat_completion", fake_post_chat_completion)

    result = _run_immediate(service.generate_text(_build_request()))

    assert calls == [(True, False), (True, True)]
    assert result.guide_text == "에그타르트가 잘 보이도록 촬영해보세요."


def test_caption_generation_service_raises_when_retry_still_non_korean(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = CaptionGenerationService(base_url="http://caption-server:8002")

    async def fake_post_chat_completion(
        prompt: str,
        *,
        include_response_format: bool,
        strict_language: bool,
    ):
        return _make_chat_response(
            200,
            content='{"guide_text":"Show the dish clearly.","caption":"Fresh pastry today."}',
        )

    monkeypatch.setattr(service, "_post_chat_completion", fake_post_chat_completion)

    with pytest.raises(
        CaptionGenerationUnavailableError,
        match="not sufficiently Korean",
    ):
        _run_immediate(service.generate_text(_build_request()))


def test_caption_generation_prompt_shrinks_long_menu_description() -> None:
    service = CaptionGenerationService(base_url="http://caption-server:8002")

    prompt = service._pipeline.build_prompt(
        _build_request(
            utterance="promote egg tart " * 80,
            menu_description=("buttery pastry with a crisp shell " * 80).strip(),
        )
    )

    assert len(prompt) <= 3200
    assert "menu_description:" in prompt


def test_caption_generation_fallback_uses_menu_name_first() -> None:
    service = CaptionGenerationService(base_url="http://caption-server:8002")

    fallback_result = service.build_fallback_result(
        _build_request(),
        fallback_source="caption_model_fallback",
    )

    assert "egg tart" in fallback_result.result.guide_text
    assert "egg tart" in fallback_result.result.draft_caption


def test_caption_generation_fallback_uses_default_guide_without_keywords_or_menu() -> None:
    service = CaptionGenerationService(base_url="http://caption-server:8002")

    fallback_result = service.build_fallback_result(
        _build_request(
            draft_keywords=[],
            menu_name=None,
            menu_description=None,
            matched_keyword=None,
        ),
        fallback_source=None,
    )

    assert fallback_result.result.guide_text == DEFAULT_FALLBACK_GUIDE_TEXT
    assert fallback_result.fallback_source == "rule_based_fallback"
