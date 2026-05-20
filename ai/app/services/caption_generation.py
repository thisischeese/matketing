from __future__ import annotations

from dataclasses import dataclass, field
import json
import logging
import re
import time
from typing import Any

from fastapi import Request
import httpx

from app.core.config import settings
from app.logging import build_log_extra, preview_text
from app.services.content_purpose import MENU_PROMOTION_PURPOSE
from app.services.remote_model_client import RemoteModelClient
from app.services.weather_tags import (
    PRECIP_CLEAR,
    PRECIP_CLOUDY,
    PRECIP_HEAVY_RAIN,
    PRECIP_RAIN,
    SPECIAL_FINE_DUST,
    SPECIAL_SEASONAL_CHANGE,
    SPECIAL_TYPHOON,
    TEMP_COLD,
    TEMP_FREEZING,
    TEMP_HOT,
    TEMP_MILD,
    TEMP_SCORCHING,
)

logger = logging.getLogger(__name__)

DEFAULT_FALLBACK_GUIDE_TEXT = (
    "사장님, 가게의 분위기와 메뉴가 잘 보이도록 화면을 가까이 담아보세요."
)
DEFAULT_FALLBACK_SUBJECT = "오늘의 대표 메뉴"

MAX_PROMPT_CHARS = 3200
MAX_UTTERANCE_CHARS = 280
MAX_MENU_DESCRIPTION_CHARS = 320
MAX_MENU_NAME_CHARS = 80
MAX_KEYWORD_CHARS = 24
MAX_MATCHED_KEYWORD_CHARS = 24

_DEFAULT_SYSTEM_INSTRUCTION = (
    "guide_text와 caption만 포함한 JSON 객체 하나만 반환하세요."
)
_STRICT_KOREAN_SYSTEM_INSTRUCTION = (
    "guide_text와 caption만 포함한 JSON 객체 하나만 반환하세요. "
    "guide_text와 caption은 자연스러운 한국어 문장으로만 작성하고 영어 문장은 사용하지 마세요."
)
_STRICT_KOREAN_USER_SUFFIX = """

중요:
- guide_text와 caption은 자연스러운 한국어 문장으로만 작성하세요.
- 영어 문장과 영어 설명은 사용하지 마세요.
- 브랜드명, 메뉴명, 고유명사는 필요한 경우에만 최소한으로 사용하세요.
"""

_MENU_PROMOTION_PROMPT_TEMPLATE = """당신은 50-60대 자영업자의 메뉴 홍보 게시물을 작성하는 한국어 마케팅 도우미입니다.

규칙:
- 출력은 JSON 객체 텍스트만 반환합니다.
- 반드시 다음 스키마를 사용합니다: {{"guide_text": "...", "caption": "..."}}.
- 게시물 목적은 "메뉴 홍보"입니다.
- guide_text는 사진이나 영상에서 무엇을 더 잘 보이게 촬영하면 좋은지 알려주는 한국어 한 문장입니다.
- caption은 메뉴 홍보용 한국어 인스타그램 캡션입니다.
- 입력에 없는 메뉴나 재료, 가게 이름을 지어내지 마세요.
- 메뉴명과 메뉴 설명을 우선으로 활용하고, 사용자 발화와 draft_keywords를 자연스럽게 반영하세요.
- 날씨는 문맥에 맞을 때만 자연스럽게 녹여 쓰고 날짜는 공휴일, 계절 등에 어긋나지 않는지 확인하는 용도로 사용하세요.

입력:
- owner_persona: {owner_persona}
- today: {today}
- weather_context: {weather_context}
- utterance: {utterance}
- draft_keywords: {draft_keywords}
- matched_keyword: {matched_keyword}
- menu_name: {menu_name}
- menu_description: {menu_description}

예시:
{{
    "guided_text" : "바삭한 에그타르트의 결이 자연스럽게 드러나도록 촬영해보세요.",
    "caption" : "직접 끓여낸 바닐라빈의 깊은 향을 담아 만든 에그타르트. 한 입 베어 물면 바삭한 파이지 사이로 부드러운 커스터드가 천천히 퍼집니다. 오늘도 마음에 잠깐의 휴식을 #에그타르트 #비오는날"
}}
"""

_PRIMARY_WEATHER_TAG_PRIORITY = (
    PRECIP_HEAVY_RAIN,
    PRECIP_RAIN,
    PRECIP_CLEAR,
    PRECIP_CLOUDY,
)

_SECONDARY_WEATHER_TAG_PRIORITY = (
    SPECIAL_TYPHOON,
    SPECIAL_FINE_DUST,
    SPECIAL_SEASONAL_CHANGE,
    TEMP_SCORCHING,
    TEMP_HOT,
    TEMP_COLD,
    TEMP_FREEZING,
    TEMP_MILD,
)

_WEATHER_CONTEXT_BY_TAG = {
    PRECIP_HEAVY_RAIN: "폭우가 잦아지는 날",
    PRECIP_RAIN: "비가 내리는 날",
    PRECIP_CLEAR: "맑은 날",
    PRECIP_CLOUDY: "흐린 날",
    SPECIAL_TYPHOON: "강풍을 조심해야 하는 날",
    SPECIAL_FINE_DUST: "미세먼지를 신경 써야 하는 날",
    SPECIAL_SEASONAL_CHANGE: "계절이 바뀌는 시기",
    TEMP_SCORCHING: "무더운 날",
    TEMP_HOT: "더운 날",
    TEMP_COLD: "쌀쌀한 날",
    TEMP_FREEZING: "매우 추운 날",
    TEMP_MILD: "포근한 날",
}


def _select_weather_copy_tags(weather_tags: list[str]) -> list[str]:
    selected: list[str] = []

    for tag in _PRIMARY_WEATHER_TAG_PRIORITY:
        if tag in weather_tags:
            selected.append(tag)
            break

    for tag in _SECONDARY_WEATHER_TAG_PRIORITY:
        if tag in weather_tags and tag not in selected:
            selected.append(tag)
            break

    if not selected:
        for tag in weather_tags:
            if tag in _WEATHER_CONTEXT_BY_TAG and tag not in selected:
                selected.append(tag)
            if len(selected) == 2:
                break

    return selected


def _build_weather_context(weather_tags: list[str]) -> str:
    selected_tags = _select_weather_copy_tags(weather_tags)
    if not selected_tags:
        return "날씨 정보 없음"
    return ", ".join(_WEATHER_CONTEXT_BY_TAG[tag] for tag in selected_tags)


def _truncate_text(value: str, limit: int) -> str:
    text = value.strip()
    if len(text) <= limit:
        return text
    if limit <= 3:
        return text[:limit]
    return f"{text[: limit - 3].rstrip()}..."


class CaptionGenerationUnavailableError(RuntimeError):
    """Raised when caption generation is unavailable."""


class CaptionGenerationLanguageError(CaptionGenerationUnavailableError):
    """Raised when caption generation returns text that is not sufficiently Korean."""


@dataclass
class CaptionGenerationResult:
    guide_text: str
    draft_caption: str

    @property
    def stored_caption(self) -> str:
        return self.draft_caption


@dataclass
class CaptionGenerationRequest:
    draft_keywords: list[str]
    owner_persona: str
    today: str = ""
    utterance: str = ""
    weather_tags: list[str] = field(default_factory=list)
    menu_name: str | None = None
    menu_description: str | None = None
    matched_keyword: str | None = None


@dataclass
class CaptionFallbackResult:
    result: CaptionGenerationResult
    fallback_source: str | None


class CaptionPipeline:
    def __init__(self, prompt_template: str) -> None:
        self.prompt_template = prompt_template

    def build_prompt(self, request: CaptionGenerationRequest) -> str:
        owner_persona = request.owner_persona.strip() or "사장님 감성"
        today = request.today.strip() or "오늘"
        weather_context = _build_weather_context(request.weather_tags)
        utterance = _truncate_text(request.utterance.strip() or "(없음)", MAX_UTTERANCE_CHARS)
        draft_keywords = self._format_keywords(request.draft_keywords)
        matched_keyword = _truncate_text(
            request.matched_keyword or self._default_matched_keyword(request),
            MAX_MATCHED_KEYWORD_CHARS,
        ) or "(없음)"
        menu_name = _truncate_text(
            request.menu_name or DEFAULT_FALLBACK_SUBJECT,
            MAX_MENU_NAME_CHARS,
        )
        menu_description = _truncate_text(
            request.menu_description or "(없음)",
            MAX_MENU_DESCRIPTION_CHARS,
        )

        prompt = self.prompt_template.format(
            owner_persona=owner_persona,
            today=today,
            weather_context=weather_context,
            utterance=utterance,
            draft_keywords=draft_keywords,
            matched_keyword=matched_keyword,
            menu_name=menu_name,
            menu_description=menu_description,
        )
        return self._shrink_prompt_if_needed(
            request=request,
            prompt=prompt,
            owner_persona=owner_persona,
            today=today,
            weather_context=weather_context,
            draft_keywords=draft_keywords,
            matched_keyword=matched_keyword,
            menu_name=menu_name,
            menu_description=menu_description,
            utterance=utterance,
        )

    def build_fallback(
        self,
        request: CaptionGenerationRequest,
        fallback_source: str | None,
    ) -> CaptionFallbackResult:
        subject = self._fallback_subject(request)
        weather_context = _build_weather_context(request.weather_tags)
        caption_prefix = weather_context if weather_context != "날씨 정보 없음" else "오늘"
        guide_text = (
            f"사장님, {subject}이 잘 보이도록 화면을 가까이 담아보세요."
            if subject
            else DEFAULT_FALLBACK_GUIDE_TEXT
        )
        caption = f"{caption_prefix} {subject}를 자연스럽게 소개해보세요."
        effective_fallback_source = fallback_source or "rule_based_fallback"
        return CaptionFallbackResult(
            result=CaptionGenerationResult(
                guide_text=guide_text,
                draft_caption=caption,
            ),
            fallback_source=effective_fallback_source,
        )

    def _shrink_prompt_if_needed(
        self,
        *,
        request: CaptionGenerationRequest,
        prompt: str,
        owner_persona: str,
        today: str,
        weather_context: str,
        draft_keywords: str,
        matched_keyword: str,
        menu_name: str,
        menu_description: str,
        utterance: str,
    ) -> str:
        current_prompt = prompt
        current_description = menu_description
        current_utterance = utterance

        if len(current_prompt) <= MAX_PROMPT_CHARS:
            return current_prompt

        current_description = _truncate_text(
            current_description,
            min(160, len(current_description)),
        )
        current_prompt = self.prompt_template.format(
            owner_persona=owner_persona,
            today=today,
            weather_context=weather_context,
            utterance=current_utterance,
            draft_keywords=draft_keywords,
            matched_keyword=matched_keyword,
            menu_name=menu_name,
            menu_description=current_description,
        )
        if len(current_prompt) <= MAX_PROMPT_CHARS:
            return current_prompt

        current_utterance = _truncate_text(
            current_utterance,
            min(160, len(current_utterance)),
        )
        current_prompt = self.prompt_template.format(
            owner_persona=owner_persona,
            today=today,
            weather_context=weather_context,
            utterance=current_utterance,
            draft_keywords=draft_keywords,
            matched_keyword=matched_keyword,
            menu_name=menu_name,
            menu_description=current_description,
        )
        if len(current_prompt) <= MAX_PROMPT_CHARS:
            return current_prompt

        compact_keywords = self._format_keywords(request.draft_keywords[:2])
        return self.prompt_template.format(
            owner_persona=owner_persona,
            today=today,
            weather_context=weather_context,
            utterance=current_utterance,
            draft_keywords=compact_keywords,
            matched_keyword=matched_keyword,
            menu_name=menu_name,
            menu_description=_truncate_text(current_description, 120),
        )

    @staticmethod
    def _format_keywords(keywords: list[str]) -> str:
        normalized = [
            _truncate_text(keyword.strip(), MAX_KEYWORD_CHARS)
            for keyword in keywords
            if keyword.strip()
        ]
        return ", ".join(normalized[:3]) if normalized else "(없음)"

    @staticmethod
    def _default_matched_keyword(request: CaptionGenerationRequest) -> str:
        return next((keyword for keyword in request.draft_keywords if keyword.strip()), "")

    def _fallback_subject(self, request: CaptionGenerationRequest) -> str:
        if request.menu_name:
            related_keywords = [
                keyword for keyword in request.draft_keywords if keyword != request.menu_name
            ]
            if related_keywords:
                return f"{request.menu_name}와 {', '.join(related_keywords[:2])}"
            return request.menu_name
        if request.draft_keywords:
            return ", ".join(request.draft_keywords[:2])
        return ""


def _build_menu_promotion_pipeline() -> CaptionPipeline:
    return CaptionPipeline(_MENU_PROMOTION_PROMPT_TEMPLATE)


class CaptionGenerationService:
    def __init__(
        self,
        enabled: bool = True,
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        timeout_seconds: float = 30.0,
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
        self._response_format_supported = True
        self._pipeline = _build_menu_promotion_pipeline()

    async def preload(self) -> None:
        await self._check_server_connection()

    async def is_healthy(self) -> bool:
        try:
            await self._check_server_connection()
        except CaptionGenerationUnavailableError:
            return False
        return True

    async def generate_text(
        self,
        request: CaptionGenerationRequest,
    ) -> CaptionGenerationResult:
        if not self.enabled:
            raise CaptionGenerationUnavailableError(
                "Caption generation model is disabled."
            )

        prompt = self._pipeline.build_prompt(request)
        started_at = time.perf_counter()

        logger.info(
            "Caption generation inference started.",
            extra=build_log_extra(
                "caption_generation.inference.started",
                component="caption_generation",
                stage="inference",
                outcome="started",
                caption_timeout_seconds=self.timeout_seconds,
                caption_model_base_url=self.base_url,
                caption_chat_endpoint=self.chat_endpoint,
                caption_health_endpoint=self.health_endpoint,
                caption_keyword_count=len(request.draft_keywords),
                caption_keywords_preview=", ".join(request.draft_keywords[:3]),
                caption_prompt_length=len(prompt),
                menu_name=request.menu_name,
                purpose=MENU_PROMOTION_PURPOSE,
            ),
        )

        try:
            result = await self._generate_korean_result(prompt)
        except TimeoutError as exc:
            raise CaptionGenerationUnavailableError(
                "Caption generation timed out."
            ) from exc
        except CaptionGenerationUnavailableError:
            raise
        except Exception as exc:
            raise CaptionGenerationUnavailableError(
                "Caption generation inference failed."
            ) from exc

        logger.info(
            "Caption generation inference finished.",
            extra=build_log_extra(
                "caption_generation.inference.completed",
                component="caption_generation",
                stage="inference",
                outcome="succeeded",
                elapsed_ms=int((time.perf_counter() - started_at) * 1000),
                guide_text_length=len(result.guide_text),
                draft_caption_length=len(result.draft_caption),
                purpose=MENU_PROMOTION_PURPOSE,
            ),
        )
        return result

    def build_fallback_result(
        self,
        request: CaptionGenerationRequest,
        fallback_source: str | None,
    ) -> CaptionFallbackResult:
        return self._pipeline.build_fallback(
            request,
            fallback_source,
        )

    def _build_request_payload(
        self,
        prompt: str,
        include_response_format: bool,
        *,
        strict_language: bool,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "messages": [
                {
                    "role": "system",
                    "content": self._build_system_instruction(
                        strict_language=strict_language
                    ),
                },
                {
                    "role": "user",
                    "content": self._build_user_prompt(
                        prompt,
                        strict_language=strict_language,
                    ),
                },
            ],
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
                        "guide_text": {"type": "string"},
                        "caption": {"type": "string"},
                    },
                    "required": ["guide_text", "caption"],
                },
            }
        return payload

    async def _post_chat_completion(
        self,
        prompt: str,
        *,
        include_response_format: bool,
        strict_language: bool,
    ) -> httpx.Response:
        payload = self._build_request_payload(
            prompt,
            include_response_format,
            strict_language=strict_language,
        )
        return await self._remote_client.post_chat_completion(payload)

    async def _check_server_connection(self) -> None:
        if not self.base_url:
            raise CaptionGenerationUnavailableError(
                "Caption generation server base URL is not configured."
            )

        try:
            response = await self._remote_client.check_health()
        except httpx.TimeoutException as exc:
            raise CaptionGenerationUnavailableError(
                "Caption generation server connectivity check timed out."
            ) from exc
        except httpx.HTTPStatusError as exc:
            raise CaptionGenerationUnavailableError(
                "Caption generation server connectivity check returned "
                f"HTTP {exc.response.status_code}."
            ) from exc
        except httpx.HTTPError as exc:
            raise CaptionGenerationUnavailableError(
                "Caption generation server connectivity check failed."
            ) from exc

        self._server_checked = True
        logger.info(
            "Caption generation server connectivity check completed.",
            extra=build_log_extra(
                "caption_generation.server_check.completed",
                component="caption_generation",
                stage="server_check",
                outcome="succeeded",
                caption_model_base_url=self.base_url,
                caption_health_endpoint=self.health_endpoint,
                caption_http_status=response.status_code,
            ),
        )

    @staticmethod
    def _is_response_format_unsupported(response: httpx.Response) -> bool:
        if response.status_code != 400:
            return False
        body = response.text.lower()
        if "response_format" not in body:
            return False
        return any(
            marker in body
            for marker in ("unsupported", "not supported", "unknown", "invalid")
        )

    async def _generate(
        self,
        prompt: str,
        *,
        strict_language: bool,
    ) -> str:
        if not self.base_url:
            raise CaptionGenerationUnavailableError(
                "Caption generation server base URL is not configured."
            )

        response: httpx.Response | None = None
        first_unsupported_status: int | None = None
        try:
            response = await self._post_chat_completion(
                prompt,
                include_response_format=self._response_format_supported,
                strict_language=strict_language,
            )
            if (
                self._response_format_supported
                and self._is_response_format_unsupported(response)
            ):
                first_unsupported_status = response.status_code
                self._response_format_supported = False
                response = await self._post_chat_completion(
                    prompt,
                    include_response_format=False,
                    strict_language=strict_language,
                )
            response.raise_for_status()
        except httpx.TimeoutException as exc:
            raise TimeoutError from exc
        except httpx.HTTPStatusError as exc:
            detail = (
                f"Caption generation server returned HTTP {exc.response.status_code}."
            )
            if first_unsupported_status is not None:
                detail += (
                    f" Initial response_format request returned HTTP "
                    f"{first_unsupported_status}."
                )
            raise CaptionGenerationUnavailableError(detail) from exc
        except httpx.HTTPError as exc:
            raise CaptionGenerationUnavailableError(
                "Caption generation server request failed."
            ) from exc

        self._server_checked = True
        raw_output = self._remote_client.extract_message_content(response)
        logger.info(
            "llama-server caption completion returned.",
            extra=build_log_extra(
                "caption_generation.generate.completed",
                component="caption_generation",
                stage="generate",
                outcome="succeeded",
                purpose=MENU_PROMOTION_PURPOSE,
                caption_language_mode=(
                    "strict_korean_retry" if strict_language else "default"
                ),
                caption_model_base_url=self.base_url,
                caption_chat_endpoint=self.chat_endpoint,
                caption_http_status=(
                    response.status_code if response is not None else None
                ),
                raw_output_preview=preview_text(
                    raw_output,
                    settings.LOG_EVENT_PREVIEW_MAX_LEN,
                ),
            ),
        )
        return raw_output

    async def _generate_korean_result(
        self,
        prompt: str,
    ) -> CaptionGenerationResult:
        raw_output = await self._generate(
            prompt,
            strict_language=False,
        )
        try:
            return self._parse_generation_result(raw_output)
        except CaptionGenerationLanguageError:
            logger.warning(
                "Caption generation returned non-Korean text; retrying with stricter language guidance.",
                extra=build_log_extra(
                    "caption_generation.language_retry.attempted",
                    component="caption_generation",
                    stage="language_validation",
                    outcome="retrying",
                    purpose=MENU_PROMOTION_PURPOSE,
                    non_korean_detected=True,
                    language_retry_attempted=True,
                ),
            )

        raw_output = await self._generate(
            prompt,
            strict_language=True,
        )
        try:
            return self._parse_generation_result(raw_output)
        except CaptionGenerationLanguageError as exc:
            logger.warning(
                "Caption generation retry still returned non-Korean text.",
                extra=build_log_extra(
                    "caption_generation.language_retry.failed",
                    component="caption_generation",
                    stage="language_validation",
                    outcome="failed",
                    purpose=MENU_PROMOTION_PURPOSE,
                    non_korean_detected=True,
                    language_retry_attempted=True,
                    language_retry_failed=True,
                ),
            )
            raise exc

    @staticmethod
    def _extract_json_payload(raw_output: str) -> str:
        match = re.search(r"\{[\s\S]*\}", raw_output)
        if match is None:
            raise CaptionGenerationUnavailableError(
                "Caption generation did not return JSON."
            )
        return match.group(0)

    def _parse_generation_result(
        self,
        raw_output: str,
    ) -> CaptionGenerationResult:
        try:
            payload = json.loads(self._extract_json_payload(raw_output))
        except json.JSONDecodeError as exc:
            raise CaptionGenerationUnavailableError(
                "Caption generation returned invalid JSON."
            ) from exc

        if not isinstance(payload, dict):
            raise CaptionGenerationUnavailableError(
                "Caption generation returned a non-object payload."
            )

        guide_text = self._normalize_text(payload.get("guide_text"))
        draft_caption = self._normalize_text(payload.get("caption"))

        if not guide_text or not draft_caption:
            raise CaptionGenerationUnavailableError(
                "Caption generation returned incomplete text fields."
            )

        self._validate_korean_output(guide_text, draft_caption)

        return CaptionGenerationResult(
            guide_text=guide_text,
            draft_caption=draft_caption,
        )

    @staticmethod
    def _normalize_text(value: Any) -> str:
        if not isinstance(value, str):
            return ""
        return value.strip()

    @staticmethod
    def _build_system_instruction(*, strict_language: bool) -> str:
        if strict_language:
            return _STRICT_KOREAN_SYSTEM_INSTRUCTION
        return _DEFAULT_SYSTEM_INSTRUCTION

    @staticmethod
    def _build_user_prompt(prompt: str, *, strict_language: bool) -> str:
        if strict_language:
            return f"{prompt}{_STRICT_KOREAN_USER_SUFFIX}"
        return prompt

    def _validate_korean_output(self, guide_text: str, draft_caption: str) -> None:
        if self._looks_non_korean(guide_text) or self._looks_non_korean(draft_caption):
            raise CaptionGenerationLanguageError(
                "Caption generation returned text that is not sufficiently Korean."
            )

    @staticmethod
    def _looks_non_korean(text: str) -> bool:
        hangul_count = len(re.findall(r"[가-힣]", text))
        ascii_count = len(re.findall(r"[A-Za-z]", text))
        if hangul_count == 0:
            return True
        if ascii_count == 0:
            return False
        return ascii_count > hangul_count * 1.5


def build_caption_generation_service() -> CaptionGenerationService:
    client = settings.caption_model_client
    return CaptionGenerationService(
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


def get_caption_generation_service(request: Request) -> CaptionGenerationService:
    service = getattr(request.app.state, "caption_generation_service", None)
    if service is None:
        raise RuntimeError("Caption generation service is not initialized.")
    return service
