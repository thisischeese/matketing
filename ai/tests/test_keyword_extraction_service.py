import httpx

import pytest

from app.services.keyword_extraction import (
    KeywordExtractionService,
    KeywordExtractionUnavailableError,
    build_keyword_extraction_dynamic_suffix,
    build_keyword_extraction_static_prefix,
)
from tests.utils.session_support import _make_chat_response, _run_immediate


def test_keyword_extraction_service_normalizes_and_limits_keywords() -> None:
    service = KeywordExtractionService()

    keywords = service._parse_keywords(
        "["
        '"\\ub9c9\\uac78\\ub9ac", '
        '" \\ud30c\\uc804 ", '
        '"\\ub9c9\\uac78\\ub9ac", '
        '"\\uc624\\ub298", '
        '"\\ucc3b\\uc794", '
        '"\\ub514\\uc800\\ud2b8"'
        "]"
    )

    assert keywords == ["\ub9c9\uac78\ub9ac", "\ud30c\uc804", "\ucc3b\uc794"]


def test_keyword_extraction_service_strips_particles_from_keywords() -> None:
    service = KeywordExtractionService()

    assert service._normalize_keyword("\ubc24\ud638\ubc15\uc774") == "\ubc24\ud638\ubc15"
    assert (
        service._normalize_keyword("\uc81c\ucca0 \uc74c\uc2dd\uc740")
        == "\uc81c\ucca0 \uc74c\uc2dd"
    )


def test_keyword_extraction_service_rejects_sentence_like_keywords() -> None:
    service = KeywordExtractionService()

    assert service._normalize_keyword("\uba39\uace0 \uc0b4\uc544\uc57c\uc9c0") == ""
    assert service._normalize_keyword("\uc81c\ucca0\uc774\uc57c") == ""
    assert service._normalize_keyword("\ud64d\ubcf4\ud558\uace0 \uc2f6\uc5b4") == ""


def test_keyword_extraction_service_rejects_generic_promotion_keywords() -> None:
    service = KeywordExtractionService()

    assert service._normalize_keyword("\ud64d\ubcf4") == ""
    assert service._normalize_keyword("\ucd94\ucc9c") == ""
    assert service._normalize_keyword("\ubd84\uc704\uae30") == ""


def test_keyword_extraction_service_parses_keywords_only() -> None:
    service = KeywordExtractionService()

    result = service._parse_extraction_result(
        "{"
        '"keywords": ['
        '"\\ubc24\\ud638\\ubc15", '
        '"\\uc81c\\ucca0 \\uc74c\\uc2dd", '
        '"\\ub9e4\\uc7a5"'
        "]"
        "}"
    )

    assert result.draft_keywords == ["\ubc24\ud638\ubc15", "\uc81c\ucca0 \uc74c\uc2dd"]
    assert result.final_keywords == []


def test_keyword_extraction_service_ignores_extra_fields() -> None:
    service = KeywordExtractionService()

    result = service._parse_extraction_result(
        '{'
        '"purpose": "\\uae30\\ud0c0", '
        '"keywords": ["\\ub9c9\\uac78\\ub9ac"]'
        "}"
    )

    assert result.draft_keywords == ["\ub9c9\uac78\ub9ac"]


def test_keyword_extraction_service_raises_when_disabled() -> None:
    service = KeywordExtractionService(enabled=False)

    with pytest.raises(KeywordExtractionUnavailableError):
        _run_immediate(
            service.extract_keywords(
                "\uc624\ub298 \ub9c9\uac78\ub9ac\ub791 \ud30c\uc804\uc774 \ub531\uc774\ub2e4"
            )
        )


def test_keyword_extraction_service_calls_remote_server_successfully(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = KeywordExtractionService(base_url="http://llama-server:8000")
    calls: list[bool] = []

    async def fake_post_chat_completion(
        messages: list[dict[str, str]],
        *,
        include_response_format: bool,
    ):
        calls.append(include_response_format)
        assert messages == service._build_messages(
            "\uc624\ub298 \ube44\uc640\uc11c \ub9c9\uac78\ub9ac\uac00 \ub561\uae34\ub2e4"
        )
        return _make_chat_response(
            200,
            content='{"keywords":["\\ub9c9\\uac78\\ub9ac"]}',
        )

    monkeypatch.setattr(service, "_post_chat_completion", fake_post_chat_completion)

    result = _run_immediate(
        service.extract_keywords(
            "\uc624\ub298 \ube44\uc640\uc11c \ub9c9\uac78\ub9ac\uac00 \ub561\uae34\ub2e4"
        )
    )

    assert calls == [True]
    assert result.draft_keywords == ["\ub9c9\uac78\ub9ac"]


def test_keyword_extraction_service_includes_model_name_in_payload() -> None:
    service = KeywordExtractionService(
        base_url="http://llama-server:8000",
        model_name="exaone-text",
    )

    payload = service._build_request_payload(
        service._build_messages("prompt"),
        include_response_format=False,
    )

    assert payload["model"] == "exaone-text"
    assert [message["role"] for message in payload["messages"]] == [
        "system",
        "user",
        "user",
    ]


def test_keyword_extraction_service_retries_without_response_format(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = KeywordExtractionService(base_url="http://llama-server:8000")
    calls: list[bool] = []
    seen_messages: list[list[dict[str, str]]] = []

    async def fake_post_chat_completion(
        messages: list[dict[str, str]],
        *,
        include_response_format: bool,
    ):
        calls.append(include_response_format)
        seen_messages.append(messages)
        if include_response_format:
            return httpx.Response(
                400,
                request=httpx.Request(
                    "POST",
                    "http://llama-server:8000/v1/chat/completions",
                ),
                text="response_format is unsupported",
            )
        return _make_chat_response(
            200,
            content='{"keywords":["\\ud30c\\uc804"]}',
        )

    monkeypatch.setattr(service, "_post_chat_completion", fake_post_chat_completion)

    result = _run_immediate(
        service.extract_keywords(
            "\uc624\ub298\uc740 \ud30c\uc804\uc774 \uc798 \ub098\uac08 \uac83 \uac19\ub2e4"
        )
    )

    assert calls == [True, False]
    assert seen_messages[0] == seen_messages[1]
    assert result.draft_keywords == ["\ud30c\uc804"]


def test_keyword_extraction_service_raises_on_remote_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = KeywordExtractionService(base_url="http://llama-server:8000")

    async def fake_post_chat_completion(
        messages: list[dict[str, str]],
        *,
        include_response_format: bool,
    ):
        raise httpx.ReadTimeout("timed out")

    monkeypatch.setattr(service, "_post_chat_completion", fake_post_chat_completion)

    with pytest.raises(KeywordExtractionUnavailableError, match="timed out"):
        _run_immediate(
            service.extract_keywords(
                "\uc624\ub298 \ub9c9\uac78\ub9ac\uac00 \ub561\uae34\ub2e4"
            )
        )


def test_keyword_extraction_service_preload_uses_health_endpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = KeywordExtractionService(
        base_url="http://llama-server:8000",
        health_endpoint="/health",
    )
    calls: list[str] = []

    class RecordingAsyncClient:
        def __init__(self, *args, **kwargs) -> None:
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb) -> None:
            return None

        async def get(self, url: str) -> httpx.Response:
            calls.append(url)
            return httpx.Response(200, request=httpx.Request("GET", url), text="ok")

    monkeypatch.setattr(
        "app.services.keyword_extraction.httpx.AsyncClient",
        RecordingAsyncClient,
    )

    _run_immediate(service.preload())

    assert calls == ["http://llama-server:8000/health"]


def test_keyword_extraction_static_prefix_excludes_runtime_utterance() -> None:
    prefix = build_keyword_extraction_static_prefix()

    assert "\uc785\ub825: {utterance}" not in prefix
    assert "\ub9c9\uac78\ub9ac" in prefix
    assert "\ud30c\uc804" in prefix


def test_keyword_extraction_dynamic_suffix_contains_only_runtime_utterance() -> None:
    suffix = build_keyword_extraction_dynamic_suffix(
        "  \uc624\ub298 \ub9c9\uac78\ub9ac\uac00 \ub561\uae34\ub2e4  "
    )

    assert suffix == "\uc785\ub825: \uc624\ub298 \ub9c9\uac78\ub9ac\uac00 \ub561\uae34\ub2e4\n\ucd9c\ub825:"


def test_keyword_extraction_payload_uses_static_and_dynamic_messages() -> None:
    service = KeywordExtractionService(base_url="http://llama-server:8000")

    payload = service._build_request_payload(
        service._build_messages("\uc624\ub298 \ud30c\uc804\uc774 \uc798 \ub098\uac08 \uac83 \uac19\ub2e4"),
        include_response_format=False,
    )

    assert payload["messages"][0] == {
        "role": "system",
        "content": "Return only a JSON object with a keywords field.",
    }
    assert payload["messages"][1] == {
        "role": "user",
        "content": build_keyword_extraction_static_prefix(),
    }
    assert payload["messages"][2] == {
        "role": "user",
        "content": build_keyword_extraction_dynamic_suffix(
            "\uc624\ub298 \ud30c\uc804\uc774 \uc798 \ub098\uac08 \uac83 \uac19\ub2e4"
        ),
    }
