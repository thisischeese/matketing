from __future__ import annotations
from dataclasses import dataclass, field
import json
import logging
import re
import threading
import time
from typing import Any

from fastapi import Request
import httpx

from app.core.config import (
    settings,
)
from app.logging import build_log_extra, preview_text
from app.services.remote_model_client import RemoteModelClient
# Legacy local GGUF bootstrap is intentionally disabled during llama-server migration.
# from app.keyword_model_download import ensure_keyword_model_available

logger = logging.getLogger(__name__)
# TOODO : 조사, 어미, 접미사 등 키워드 코드 외부로 빼고 모듈로 만들기 -> 모듈 가져와서 사용하는 방식으로 리팩토링하기 
_GENERIC_KEYWORDS = {
    "오늘",
    "요즘",
    "그래도",
    "손님",
    "매장",
    "가게",
    "메뉴",
    "음식",
    "음료",
    "디저트",
    "분위기",
    "기분",
    "하늘",
    "날씨",
    "상황",
    "진열장",
    "먹고 살기",
    "만만치 않다",
    "만만치",
    "제철",
    "홍보",
    "추천",
    "권유",
    "게시물",
    "사진",
    "포스팅",
    "이야기",
    "소개",
}

_WEATHER_SIGNAL_HINTS = (
    "맑",
    "비",
    "눈",
    "흐",
    "바람",
    "쌀쌀",
    "선선",
    "후텁",
    "무더",
    "포근",
    "서늘",
    "따뜻",
    "추위",
    "더위",
    "장마",
    "봄",
    "여름",
    "가을",
    "겨울",
    "초여름",
    "초가을",
)

_PURE_WEATHER_SIGNALS = {
    "맑음",
    "비",
    "눈",
    "흐림",
    "흐린 날",
    "맑은 날",
    "비 오는 날",
    "눈 오는 날",
    "바람",
    "봄바람",
    "쌀쌀함",
    "선선함",
    "후텁지근함",
    "무더위",
    "포근함",
    "서늘함",
    "따뜻함",
    "추위",
    "더위",
    "장마",
    "봄",
    "여름",
    "가을",
    "겨울",
    "초여름",
    "초가을",
}

_MORPH_ALLOWED_KEYWORD_TAG_PREFIXES = ("N", "SL", "SH", "SN", "XR")
_MORPH_ALLOWED_WEATHER_TAG_PREFIXES = ("N", "SL", "SH", "SN", "XR", "V")
_MORPH_DISALLOWED_TAIL_TAGS = {
    "JKS",
    "JKC",
    "JKG",
    "JKO",
    "JKB",
    "JKV",
    "JKQ",
    "JX",
    "JC",
    "EP",
    "EF",
    "EC",
    "ETN",
    "ETM",
    "SF",
    "SP",
    "SS",
    "SE",
    "SO",
    "SW",
}
_WEATHER_CANONICAL_BY_HINT = (
    ("봄바람", "봄바람"),
    ("초여름", "초여름"),
    ("초가을", "초가을"),
    ("장마", "장마"),
    ("쌀쌀", "쌀쌀함"),
    ("선선", "선선함"),
    ("후텁", "후텁지근함"),
    ("무더", "무더위"),
    ("포근", "포근함"),
    ("서늘", "서늘함"),
    ("따뜻", "따뜻함"),
    ("추위", "추위"),
    ("더위", "더위"),
    ("맑", "맑음"),
    ("흐", "흐림"),
    ("비", "비"),
    ("눈", "눈"),
    ("봄", "봄"),
    ("여름", "여름"),
    ("가을", "가을"),
    ("겨울", "겨울"),
)

_STATIC_PROMPT_PREFIX = """당신은 한국어로 입력된 자영업자 발화를 읽고 메뉴 홍보 게시물에 사용할 메뉴 관련 키워드 1~3개를 추출하는 도우미다.

규칙:
- 출력은 JSON 객체 텍스트만 반환한다.
- 반드시 다음 스키마를 사용한다: {{"keywords": ["..."]}}.
- "keywords"는 발화에서 그대로 가져오거나 가볍게 정규화한 짧은 한국어 표현 1~3개여야 한다.
- "keywords"는 실제 판매 메뉴, 음식명, 음료명, 디저트명, 식재료, 원두, 토핑, 소스처럼 메뉴와 직접 연결되는 표현만 고른다.
- "홍보", "추천", "손님", "매장", "가게", "분위기", "오늘", "날씨"처럼 일반적인 홍보 표현이나 상황 설명은 제외한다.
- 감정 표현, 행동 표현, 바람이나 소망, 추상적인 분위기 설명은 제외한다.
- 유용한 표현은 보존하고, 중복은 제거한다.
- 추론 과정은 설명하지 않는다.

예시:
입력: 오늘 아주 비싼 바닐라로 버터 쿠키를 구웠는데 손님들이 좋아하셨으면 좋겠네
출력: {{"keywords": ["바닐라", "버터 쿠키"]}}

입력: 비 오는 날이라 막걸리랑 파전 생각나서 오늘은 이 조합 추천하고 싶어요
출력: {{"keywords": ["막걸리", "파전"]}}

"""


def build_keyword_extraction_static_prefix() -> str:
    return _STATIC_PROMPT_PREFIX


def build_keyword_extraction_dynamic_suffix(utterance: str) -> str:
    return f"입력: {utterance.strip()}\n출력:"


class KeywordExtractionUnavailableError(RuntimeError):
    """Raised when the local keyword extraction model is unavailable."""


@dataclass
class KeywordExtractionResult:
    draft_keywords: list[str]
    final_keywords: list[str] = field(default_factory=list)


class KeywordExtractionService:
    def __init__(
        self,
        enabled: bool = True,
        max_tokens: int = 64,
        temperature: float = 0.1,
        top_p: float = 0.9,
        timeout_seconds: float = 10.0,
        base_url: str | None = None,
        model_name: str | None = None,
        chat_endpoint: str | None = None,
        health_endpoint: str | None = None,
        api_key: str | None = None,
    ) -> None:
        self.enabled = enabled
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.timeout_seconds = timeout_seconds
        self.base_url = (base_url or "").rstrip("/")
        self.model_name = model_name.strip() if model_name else None
        self.chat_endpoint = chat_endpoint or "/v1/chat/completions"
        self.health_endpoint = health_endpoint or "/health"
        self.api_key = api_key
        self._remote_client = RemoteModelClient(
            base_url=self.base_url,
            chat_endpoint=self.chat_endpoint,
            health_endpoint=self.health_endpoint,
            api_key=self.api_key,
            timeout_seconds=self.timeout_seconds,
        )
        self._server_checked = False
        self._morph_analyzer: Any = None
        self._morph_analyzer_initialized = False
        self._lock = threading.Lock()

    async def preload(self) -> None:
        await self._check_server_connection()

    async def is_healthy(self) -> bool:
        try:
            await self._check_server_connection()
        except KeywordExtractionUnavailableError:
            return False
        return True

    async def extract_keywords(self, utterance: str) -> KeywordExtractionResult:
        if not self.enabled:
            raise KeywordExtractionUnavailableError(
                "Keyword extraction model is disabled."
            )

        messages = self._build_messages(utterance)
        started_at = time.perf_counter()

        logger.info(
            "Keyword extraction inference started.",
            extra=build_log_extra(
                "keyword_extraction.inference.started",
                component="keyword_extraction",
                stage="inference",
                outcome="started",
                keyword_timeout_seconds=self.timeout_seconds,
                keyword_model_base_url=self.base_url,
                keyword_chat_endpoint=self.chat_endpoint,
                keyword_health_endpoint=self.health_endpoint,
                keyword_server_checked=self._server_checked,
                utterance_length=len(utterance),
                utterance_preview=preview_text(
                    utterance,
                    settings.LOG_EVENT_PREVIEW_MAX_LEN,
                )
                if settings.LOG_INCLUDE_RAW_IDENTIFIERS
                else None,
            ),
        )

        try:
            raw_output = await self._generate_keywords(messages)
        except TimeoutError as exc:
            logger.warning(
                "Keyword extraction timed out while waiting for llama-server completion.",
                extra=build_log_extra(
                    "keyword_extraction.inference.completed",
                    component="keyword_extraction",
                    stage="inference",
                    outcome="failed",
                    error_type="TimeoutError",
                    keyword_timeout_seconds=self.timeout_seconds,
                    elapsed_ms=int((time.perf_counter() - started_at) * 1000),
                    keyword_model_base_url=self.base_url,
                    keyword_chat_endpoint=self.chat_endpoint,
                    keyword_health_endpoint=self.health_endpoint,
                ),
            )
            raise KeywordExtractionUnavailableError(
                "Keyword extraction timed out."
            ) from exc
        except KeywordExtractionUnavailableError:
            raise
        except Exception as exc:
            logger.warning(
                "Keyword extraction inference raised an unexpected exception.",
                extra=build_log_extra(
                    "keyword_extraction.inference.completed",
                    component="keyword_extraction",
                    stage="inference",
                    outcome="failed",
                    error_type=exc.__class__.__name__,
                    elapsed_ms=int((time.perf_counter() - started_at) * 1000),
                    keyword_model_base_url=self.base_url,
                    keyword_chat_endpoint=self.chat_endpoint,
                    keyword_health_endpoint=self.health_endpoint,
                ),
                exc_info=True,
            )
            raise KeywordExtractionUnavailableError(
                "Keyword extraction inference failed."
            ) from exc

        result = self._parse_extraction_result(raw_output)
        logger.info(
            "Keyword extraction inference finished.",
            extra=build_log_extra(
                "keyword_extraction.inference.completed",
                component="keyword_extraction",
                stage="inference",
                outcome="succeeded",
                elapsed_ms=int((time.perf_counter() - started_at) * 1000),
                draft_keyword_count=len(result.draft_keywords),
                draft_keywords_preview=", ".join(result.draft_keywords),
            ),
        )
        return result

    def _build_messages(self, utterance: str) -> list[dict[str, str]]:
        return [
            {
                "role": "system",
                "content": "Return only a JSON object with a keywords field.",
            },
            {
                "role": "user",
                "content": build_keyword_extraction_static_prefix(),
            },
            {
                "role": "user",
                "content": build_keyword_extraction_dynamic_suffix(utterance),
            },
        ]

    def _build_request_payload(
        self,
        messages: list[dict[str, str]],
        include_response_format: bool,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "messages": messages,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_tokens,
        }
        if self.model_name:
            payload["model"] = self.model_name
        if include_response_format:
            payload["response_format"] = {
                "type": "json_object",
                "schema": {
                    "type": "object",
                    "properties": {
                        "keywords": {
                            "type": "array",
                            "minItems": 1,
                            "maxItems": 3,
                            "items": {"type": "string"},
                        },
                    },
                    "required": ["keywords"],
                },
            }
        return payload

    def _build_request_headers(self) -> dict[str, str]:
        return self._remote_client.build_headers()

    @property
    def _chat_url(self) -> str:
        return self._remote_client.chat_url

    @property
    def _health_url(self) -> str:
        return self._remote_client.health_url

    async def _post_chat_completion(
        self,
        messages: list[dict[str, str]],
        *,
        include_response_format: bool,
    ) -> httpx.Response:
        payload = self._build_request_payload(messages, include_response_format)
        return await self._remote_client.post_chat_completion(payload)

    async def _check_server_connection(self) -> None:
        if not self.base_url:
            raise KeywordExtractionUnavailableError(
                "Keyword extraction server base URL is not configured."
            )

        try:
            response = await self._remote_client.check_health()
        except httpx.TimeoutException as exc:
            raise KeywordExtractionUnavailableError(
                "Keyword extraction server connectivity check timed out."
            ) from exc
        except httpx.HTTPStatusError as exc:
            raise KeywordExtractionUnavailableError(
                "Keyword extraction server connectivity check returned "
                f"HTTP {exc.response.status_code}."
            ) from exc
        except httpx.HTTPError as exc:
            raise KeywordExtractionUnavailableError(
                "Keyword extraction server connectivity check failed."
            ) from exc

        self._server_checked = True
        logger.info(
            "Keyword extraction server connectivity check completed.",
            extra=build_log_extra(
                "keyword_extraction.server_check.completed",
                component="keyword_extraction",
                stage="server_check",
                outcome="succeeded",
                keyword_timeout_seconds=self.timeout_seconds,
                keyword_model_base_url=self.base_url,
                keyword_health_endpoint=self.health_endpoint,
                keyword_http_status=response.status_code,
            ),
        )

    @staticmethod
    def _is_response_format_unsupported(response: httpx.Response) -> bool:
        if response.status_code < 400:
            return False
        body = response.text.lower()
        return "response_format" in body or "json_object" in body or "schema" in body

    async def _generate_keywords(self, messages: list[dict[str, str]]) -> str:
        if not self.base_url:
            raise KeywordExtractionUnavailableError(
                "Keyword extraction server base URL is not configured."
            )

        generation_started_at = time.perf_counter()
        logger.info(
            "Calling llama-server chat completion for keyword extraction.",
            extra=build_log_extra(
                "keyword_extraction.generate.started",
                component="keyword_extraction",
                stage="generate",
                outcome="started",
                keyword_timeout_seconds=self.timeout_seconds,
                keyword_model_base_url=self.base_url,
                keyword_chat_endpoint=self.chat_endpoint,
                keyword_max_tokens=self.max_tokens,
                keyword_temperature=self.temperature,
                keyword_top_p=self.top_p,
            ),
        )

        response: httpx.Response | None = None
        try:
            response = await self._post_chat_completion(
                messages,
                include_response_format=True,
            )
            if self._is_response_format_unsupported(response):
                logger.info(
                    "llama-server response_format is unsupported; retrying without schema enforcement.",
                    extra=build_log_extra(
                        "keyword_extraction.generate.response_format_retry",
                        component="keyword_extraction",
                        stage="generate",
                        outcome="retrying",
                        keyword_timeout_seconds=self.timeout_seconds,
                        keyword_model_base_url=self.base_url,
                        keyword_chat_endpoint=self.chat_endpoint,
                    ),
                )
                response = await self._post_chat_completion(
                    messages,
                    include_response_format=False,
                )
            response.raise_for_status()
        except httpx.TimeoutException as exc:
            raise TimeoutError from exc
        except httpx.HTTPStatusError as exc:
            raise KeywordExtractionUnavailableError(
                f"Keyword extraction server returned HTTP {exc.response.status_code}."
            ) from exc
        except httpx.HTTPError as exc:
            raise KeywordExtractionUnavailableError(
                "Keyword extraction server request failed."
            ) from exc

        self._server_checked = True
        raw_output = self._remote_client.extract_message_content(response)
        logger.info(
            "llama-server keyword completion returned.",
            extra=build_log_extra(
                "keyword_extraction.generate.completed",
                component="keyword_extraction",
                stage="generate",
                outcome="succeeded",
                elapsed_ms=int((time.perf_counter() - generation_started_at) * 1000),
                keyword_model_base_url=self.base_url,
                keyword_chat_endpoint=self.chat_endpoint,
                keyword_http_status=response.status_code if response is not None else None,
                raw_output_preview=preview_text(
                    raw_output,
                    settings.LOG_EVENT_PREVIEW_MAX_LEN,
                ),
            ),
        )
        return raw_output

    # Legacy local llama-cpp flow is preserved as comments for rollback reference.
    # def _ensure_model(self):
    #     self.model_path = ensure_keyword_model_available(self.model_path)
    #     from llama_cpp import Llama
    #     self._model = Llama(
    #         model_path=str(self.model_path),
    #         n_ctx=self.n_ctx,
    #         n_threads=self.n_threads,
    #         n_gpu_layers=self.n_gpu_layers,
    #         verbose=False,
    #     )
    #     return self._model

    def _parse_extraction_result(self, raw_output: str) -> KeywordExtractionResult:
        parsed = self._load_extraction_payload(raw_output)
        if not isinstance(parsed, dict):
            raise KeywordExtractionUnavailableError(
                "Keyword extraction returned a non-object payload."
            )
        keyword_payload = parsed.get("keywords")
        if not isinstance(keyword_payload, list):
            raise KeywordExtractionUnavailableError(
                "Keyword extraction returned a non-list keywords payload."
            )
        draft_keywords = self._normalize_keywords_payload(keyword_payload)

        if not draft_keywords:
            raise KeywordExtractionUnavailableError(
                "Keyword extraction returned no usable draft_keywords."
            )

        logger.info(
            "Keyword extraction parsed keywords.",
            extra=build_log_extra(
                "keyword_extraction.parse.completed",
                component="keyword_extraction",
                stage="parse",
                outcome="succeeded",
                draft_keywords_preview=", ".join(draft_keywords),
                raw_output_preview=preview_text(
                    raw_output,
                    settings.LOG_EVENT_PREVIEW_MAX_LEN,
                ),
            ),
        )
        return KeywordExtractionResult(
            draft_keywords=draft_keywords,
            final_keywords=[],
        )

    def _parse_keywords(self, raw_output: str) -> list[str]:
        payload = self._load_extraction_payload(raw_output)
        if isinstance(payload, dict):
            return self._parse_extraction_result(raw_output).draft_keywords
        if not isinstance(payload, list):
            raise KeywordExtractionUnavailableError(
                "Keyword extraction returned a non-list payload."
            )
        draft_keywords = self._normalize_keywords_payload(payload)
        if not draft_keywords:
            raise KeywordExtractionUnavailableError(
                "Keyword extraction returned no usable draft_keywords."
            )
        return draft_keywords

    @staticmethod
    def _extract_json_payload(raw_output: str) -> str:
        object_match = re.search(r"\{[\s\S]*\}", raw_output)
        if object_match is not None:
            return object_match.group(0)

        match = re.search(r"\[[\s\S]*\]", raw_output)
        if match is None:
            raise KeywordExtractionUnavailableError(
                "Keyword extraction did not return JSON."
            )
        return match.group(0)

    def _load_extraction_payload(self, raw_output: str) -> Any:
        payload = self._extract_json_payload(raw_output)
        try:
            return json.loads(payload)
        except json.JSONDecodeError as exc:
            raise KeywordExtractionUnavailableError(
                "Keyword extraction returned invalid JSON."
            ) from exc

    @staticmethod
    def _split_extraction_payload(
        payload: Any,
    ) -> tuple[Any, Any]:
        if isinstance(payload, dict):
            return None, payload.get("keywords")
        return None, payload

    def _normalize_keywords_payload(self, payload: list[Any]) -> list[str]:
        return self._normalize_unique_values(
            payload,
            normalizer=self._normalize_keyword,
        )

    @staticmethod
    def _normalize_unique_values(
        payload: list[Any],
        normalizer,
        limit: int = 3,
    ) -> list[str]:
        normalized_values: list[str] = []
        seen_values: set[str] = set()

        for item in payload:
            if not isinstance(item, str):
                continue

            normalized = normalizer(item)
            if not normalized or normalized in seen_values:
                continue

            seen_values.add(normalized)
            normalized_values.append(normalized)
            if len(normalized_values) == limit:
                break

        return normalized_values

    def _get_morph_analyzer(self) -> Any | None:
        if self._morph_analyzer_initialized:
            return self._morph_analyzer

        with self._lock:
            if self._morph_analyzer_initialized:
                return self._morph_analyzer

            try:
                from kiwipiepy import Kiwi

                self._morph_analyzer = Kiwi()
                logger.info(
                    "Keyword morphology analyzer initialized.",
                    extra=build_log_extra(
                        "keyword_extraction.morph_analyzer_init.completed",
                        component="keyword_extraction",
                        stage="morph_analyzer_init",
                        outcome="succeeded",
                        morph_analyzer="kiwi",
                    ),
                )
            except Exception as exc:
                self._morph_analyzer = None
                logger.warning(
                    "Keyword morphology analyzer is unavailable; falling back to regex normalization.",
                    extra=build_log_extra(
                        "keyword_extraction.morph_analyzer_init.completed",
                        component="keyword_extraction",
                        stage="morph_analyzer_init",
                        outcome="failed",
                        error_type=exc.__class__.__name__,
                        morph_analyzer="kiwi",
                    ),
                    exc_info=True,
                )
            finally:
                self._morph_analyzer_initialized = True

        return self._morph_analyzer

    @staticmethod
    def _basic_normalize_text(text: str) -> str:
        normalized = text.strip().strip("\"'`")
        normalized = re.sub(r"^[\s\W_]+|[\s\W_]+$", "", normalized)
        normalized = re.sub(r"\s+", " ", normalized).strip()
        return normalized

    @staticmethod
    def _fallback_strip_trailing_particles(text: str) -> str:
        stripped = re.sub(
            r"(이랑|랑|하고|에서|으로|로|은|는|이|가|을|를|와|과|도|만|엔|에|의)$",
            "",
            text,
        )
        return stripped.strip()

    def _normalize_keyword(self, keyword: str) -> str:
        normalized = self._basic_normalize_text(keyword)
        if not normalized:
            return ""

        morphology_normalized = self._normalize_phrase_with_morphology(normalized)
        if morphology_normalized is None:
            return self._fallback_normalize_keyword(normalized)
        if morphology_normalized:
            if " " not in normalized and " " in morphology_normalized:
                return self._fallback_normalize_keyword(
                    morphology_normalized.replace(" ", "")
                )
            return morphology_normalized
        return ""

    def _normalize_phrase_with_morphology(
        self,
        text: str,
    ) -> str | None:
        analyzer = self._get_morph_analyzer()
        if analyzer is None:
            return None

        try:
            tokens = analyzer.tokenize(text)
        except Exception as exc:
            logger.warning(
                "Keyword morphology analysis failed; falling back to regex normalization.",
                extra=build_log_extra(
                    "keyword_extraction.morph_analysis.completed",
                    component="keyword_extraction",
                    stage="morph_analysis",
                    outcome="failed",
                    error_type=exc.__class__.__name__,
                    morph_target="keyword",
                ),
                exc_info=True,
            )
            return None

        return self._normalize_keyword_tokens(tokens)

    def _normalize_keyword_tokens(self, tokens: list[Any]) -> str:
        pieces: list[str] = []

        for token in tokens:
            form = getattr(token, "form", "").strip()
            tag = getattr(token, "tag", "")
            if not form or not tag:
                continue
            if tag in _MORPH_DISALLOWED_TAIL_TAGS or tag in {"MAG", "MAJ", "IC"}:
                continue
            if tag.startswith("V") or tag in {"XSV", "XSA", "VCP", "VCN"}:
                if pieces:
                    break
                continue
            if tag.startswith(_MORPH_ALLOWED_KEYWORD_TAG_PREFIXES):
                pieces.append(form)
                continue
            if pieces:
                break

        normalized = " ".join(pieces)
        return self._fallback_normalize_keyword(normalized)

    def _fallback_normalize_keyword(self, keyword: str) -> str:
        normalized = self._basic_normalize_text(keyword)
        normalized = self._fallback_strip_trailing_particles(normalized)

        if len(normalized) < 2:
            return ""
        if len(normalized) > 20:
            return ""
        if normalized in _PURE_WEATHER_SIGNALS:
            return ""
        if normalized in _GENERIC_KEYWORDS:
            return ""
        if normalized.endswith(("하고 싶어", "하고 싶다", "해보고 싶어", "추천해", "추천하고 싶어")):
            return ""
        if any(term in normalized for term in ("분위기", "홍보", "추천", "게시물", "포스팅")):
            return ""
        if re.search(r"(다|요|네|죠|군)$", normalized):
            return ""
        if not re.search(r"[가-힣A-Za-z0-9]", normalized):
            return ""
        return normalized

    def _fallback_normalize_weather_signal(self, signal: str) -> str:
        normalized = self._basic_normalize_text(signal)
        normalized = self._fallback_strip_trailing_particles(normalized)
        return self._canonicalize_weather_signal(normalized)

    @staticmethod
    def _canonicalize_weather_signal(signal: str) -> str:
        normalized = KeywordExtractionService._basic_normalize_text(signal)
        if len(normalized) < 1:
            return ""
        if len(normalized) > 20:
            return ""
        if not re.search(r"[가-힣A-Za-z]", normalized):
            return ""

        for hint, canonical in _WEATHER_CANONICAL_BY_HINT:
            if hint in normalized:
                return canonical

        if not any(hint in normalized for hint in _WEATHER_SIGNAL_HINTS):
            return ""
        return normalized


def build_keyword_extraction_service() -> KeywordExtractionService:
    client = settings.keyword_model_client
    return KeywordExtractionService(
        enabled=client.enabled,
        max_tokens=client.max_tokens,
        temperature=client.temperature,
        top_p=client.top_p,
        timeout_seconds=client.timeout_seconds,
        base_url=client.base_url,
        model_name=client.model_name,
        chat_endpoint=client.chat_endpoint,
        health_endpoint=client.health_endpoint,
        api_key=client.api_key,
    )


def get_keyword_extraction_service(request: Request) -> KeywordExtractionService:
    service = getattr(request.app.state, "keyword_extraction_service", None)
    if service is None:
        raise RuntimeError("Keyword extraction service is not initialized.")
    return service
