from __future__ import annotations

from dataclasses import dataclass
import re


INTENT_CAPTION_GENERATION = "caption_generation"
INTENT_SHOOTING_GUIDE = "shooting_guide"
INTENT_IMAGE_POSTPROCESS = "image_postprocess"

_IMAGE_POSTPROCESS_PATTERNS = (
    r"\bedit\b",
    r"\bretouch\b",
    r"\bfilter\b",
    r"\bupscale\b",
    r"\bcolor grading\b",
    r"\bdenoise\b",
    r"\bsharpen\b",
    r"보정",
    r"후보정",
    r"편집",
    r"필터",
    r"업스케일",
)

_SHOOTING_GUIDE_PATTERNS = (
    r"\bshoot\b",
    r"\bangle\b",
    r"\blighting\b",
    r"\bcomposition\b",
    r"\bframing\b",
    r"\bguide\b",
    r"촬영",
    r"구도",
    r"앵글",
    r"조명",
    r"배치",
    r"가이드",
)

_CAPTION_HINT_PATTERNS = (
    r"\bcaption\b",
    r"\bpost\b",
    r"\bpromo\b",
    r"\bmenu\b",
    r"\bcopy\b",
    r"캡션",
    r"문구",
    r"홍보",
    r"소개",
    r"메뉴",
)


@dataclass(frozen=True)
class IntentClassificationResult:
    intent: str
    rule: str
    confidence: float


class IntentClassifierService:
    def classify(self, utterance: str) -> IntentClassificationResult:
        normalized = utterance.strip().lower()
        if self._matches_any(normalized, _IMAGE_POSTPROCESS_PATTERNS):
            return IntentClassificationResult(
                intent=INTENT_IMAGE_POSTPROCESS,
                rule="image_postprocess_keywords",
                confidence=0.95,
            )
        if self._matches_any(normalized, _CAPTION_HINT_PATTERNS):
            return IntentClassificationResult(
                intent=INTENT_CAPTION_GENERATION,
                rule="caption_generation_keywords",
                confidence=0.90,
            )
        if self._matches_any(normalized, _SHOOTING_GUIDE_PATTERNS):
            return IntentClassificationResult(
                intent=INTENT_SHOOTING_GUIDE,
                rule="shooting_guide_keywords",
                confidence=0.80,
            )
        return IntentClassificationResult(
            intent=INTENT_CAPTION_GENERATION,
            rule="default_caption_generation",
            confidence=0.60,
        )

    @staticmethod
    def _matches_any(text: str, patterns: tuple[str, ...]) -> bool:
        return any(re.search(pattern, text) is not None for pattern in patterns)


def build_intent_classifier_service() -> IntentClassifierService:
    return IntentClassifierService()
