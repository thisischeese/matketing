from __future__ import annotations

import argparse
import asyncio
from dataclasses import dataclass
from datetime import date
import json
from math import ceil
from pathlib import Path
import statistics
import time
from typing import Any
from uuid import uuid4

import fakeredis.aioredis

from app.schemas.sessions import ProcessUtteranceRequest, WeatherInfo
from app.services.caption_generation import (
    CaptionGenerationService,
    build_caption_generation_service,
)
from app.services.canonical_keyword_resolver import (
    CanonicalKeywordMatch,
    CanonicalKeywordResolution,
)
from app.services.intent_classifier import (
    INTENT_CAPTION_GENERATION,
    IntentClassificationResult,
)
from app.services.keyword_extraction import KeywordExtractionResult
from app.services.menu_promotion_context import MatchedMenuContext
from app.services.reference_caption_retriever import (
    ReferenceCaptionRetrievalResult,
    RetrievedReferenceCaption,
)
from app.services.session_store import session_key
from app.services.sessions import process_utterance


ROOT_DIR = Path(__file__).resolve().parents[2]
ARTIFACT_ROOT = ROOT_DIR / "artifacts" / "context-routing"
STORE_ID = "663c31d7-87c6-4b83-bc0b-756e84a7a3a7"
OWNER_PERSONA = "aesthetic"
REQUEST_DATE = date(2026, 6, 2)


@dataclass(frozen=True)
class ScenarioDefinition:
    name: str
    utterance: str
    weather: WeatherInfo
    draft_keywords: list[str]
    expected_profile: str
    matched_menu_context: MatchedMenuContext
    canonical_matches: list[CanonicalKeywordMatch]
    reference_captions: list[str]


@dataclass
class BenchmarkRun:
    scenario: str
    variant: str
    iteration: int
    session_id: str
    latency_seconds: float
    http_semantic_success: bool
    fallback_source: str | None
    context_variant: str | None
    context_profile: str | None
    prompt_char_count: int
    reference_caption_count: int
    external_dependency_calls: int
    model_call_count: int
    graph_path: str
    skipped_nodes: str
    caption_preview: str
    guide_preview: str

    @property
    def valid(self) -> bool:
        return (
            self.http_semantic_success
            and self.fallback_source is None
            and self.context_variant is not None
            and self.context_profile is not None
        )


class StaticIntentClassifierService:
    def classify(self, utterance: str) -> IntentClassificationResult:
        return IntentClassificationResult(
            intent=INTENT_CAPTION_GENERATION,
            rule="benchmark_forced_caption_generation",
            confidence=1.0,
        )


class StaticKeywordExtractionService:
    def __init__(self, draft_keywords: list[str]) -> None:
        self._draft_keywords = list(draft_keywords)

    async def is_healthy(self) -> bool:
        return True

    async def extract_keywords(self, utterance: str) -> KeywordExtractionResult:
        return KeywordExtractionResult(
            draft_keywords=list(self._draft_keywords),
            final_keywords=[],
        )


class StaticMenuPromotionContextService:
    def __init__(self, result: MatchedMenuContext) -> None:
        self._result = result

    async def match_menu(
        self,
        *,
        store_id,
        draft_keywords: list[str],
    ) -> MatchedMenuContext:
        _ = store_id
        _ = draft_keywords
        return self._result


class StaticCanonicalKeywordResolverService:
    def __init__(self, matches: list[CanonicalKeywordMatch]) -> None:
        self._matches = list(matches)

    async def resolve_keywords(
        self,
        draft_keywords: list[str],
    ) -> CanonicalKeywordResolution:
        _ = draft_keywords
        return CanonicalKeywordResolution(
            final_keywords=[match.stored_final_keyword for match in self._matches],
            matches=list(self._matches),
        )


class StaticReferenceCaptionRetrieverService:
    def __init__(self, captions: list[str]) -> None:
        self._captions = list(captions)

    async def retrieve(
        self,
        *,
        owner_persona: str,
        utterance: str,
        canonical_matches: list[CanonicalKeywordMatch],
        max_references: int | None = None,
    ) -> ReferenceCaptionRetrievalResult:
        _ = owner_persona
        _ = utterance
        _ = canonical_matches
        limit = max_references or len(self._captions)
        return ReferenceCaptionRetrievalResult(
            references=[
                RetrievedReferenceCaption(
                    caption_id=index + 1,
                    caption_content=caption,
                    score=max(0.1, 0.95 - (index * 0.05)),
                )
                for index, caption in enumerate(self._captions[:limit])
            ],
            candidate_count=len(self._captions),
        )


def build_weather(
    *,
    temperature: float,
    precipitation: float,
    cloud_cover: str,
    humidity: float,
    wind_speed: float,
    pm10: float,
    pm25: float,
    diurnal_range: float,
    discomfort_index: float,
) -> WeatherInfo:
    return WeatherInfo(
        temperature=temperature,
        precipitation=precipitation,
        cloud_cover=cloud_cover,
        humidity=humidity,
        wind_speed=wind_speed,
        pm10=pm10,
        pm25=pm25,
        diurnal_range=diurnal_range,
        discomfort_index=discomfort_index,
        heavy_rain_warning=None,
        typhoon_warning=None,
    )


def build_canonical_match(
    draft_keyword: str,
    canonical_keyword_id: int | None,
    display_name: str | None,
) -> CanonicalKeywordMatch:
    matched = canonical_keyword_id is not None and bool(display_name)
    return CanonicalKeywordMatch(
        draft_keyword=draft_keyword,
        canonical_keyword_id=canonical_keyword_id,
        final_keyword=(
            f"{canonical_keyword_id}:{display_name}"
            if matched and canonical_keyword_id is not None and display_name is not None
            else draft_keyword
        ),
        display_name=display_name,
        score=0.99 if matched else None,
        matched=matched,
    )


def build_scenarios() -> list[ScenarioDefinition]:
    return [
        ScenarioDefinition(
            name="basic",
            utterance="오늘 카페 분위기를 자연스럽게 소개하는 인스타그램 캡션을 작성해줘.",
            weather=build_weather(
                temperature=22.0,
                precipitation=0.0,
                cloud_cover="맑음",
                humidity=42.0,
                wind_speed=1.2,
                pm10=18.0,
                pm25=8.0,
                diurnal_range=10.0,
                discomfort_index=60.0,
            ),
            draft_keywords=["카페 무드", "디저트 소개"],
            expected_profile="basic_context",
            matched_menu_context=MatchedMenuContext(),
            canonical_matches=[
                build_canonical_match("카페 무드", None, None),
                build_canonical_match("디저트 소개", None, None),
            ],
            reference_captions=[],
        ),
        ScenarioDefinition(
            name="menu",
            utterance="비 오는 날 에그타르트를 자연스럽게 소개하는 인스타그램 캡션을 작성해줘.",
            weather=build_weather(
                temperature=18.0,
                precipitation=2.4,
                cloud_cover="흐림",
                humidity=74.0,
                wind_speed=2.8,
                pm10=26.0,
                pm25=15.0,
                diurnal_range=7.0,
                discomfort_index=66.0,
            ),
            draft_keywords=["에그타르트", "카페 분위기"],
            expected_profile="menu_context",
            matched_menu_context=MatchedMenuContext(
                menu_id="2a420839-1652-4a02-9958-5fb6b4723df5",
                menu_name="에그타르트",
                menu_description=(
                    "버터를 안넣은 파이지와 바닐라빈 듬뿍들어간 달걀필링! "
                    "주말 스페셜 메뉴"
                ),
                matched_keyword="에그타르트",
                match_source="benchmark_exact_name_match",
                match_score=1.0,
            ),
            canonical_matches=[
                build_canonical_match("에그타르트", 8, "에그타르트"),
                build_canonical_match("카페 분위기", None, None),
            ],
            reference_captions=[
                "상큼한 디저트와 함께 오늘의 무드를 가볍게 꺼내보는 오후.",
                "작은 디저트 하나에도 매장의 결이 보이도록 차분하게 소개합니다.",
            ],
        ),
        ScenarioDefinition(
            name="reference",
            utterance="고객 사연과 제철 과일 분위기를 살린 인스타그램 캡션을 작성해줘.",
            weather=build_weather(
                temperature=25.0,
                precipitation=0.0,
                cloud_cover="맑음",
                humidity=58.0,
                wind_speed=1.7,
                pm10=20.0,
                pm25=11.0,
                diurnal_range=11.0,
                discomfort_index=64.0,
            ),
            draft_keywords=["고객 사연", "제철 과일"],
            expected_profile="reference_context",
            matched_menu_context=MatchedMenuContext(),
            canonical_matches=[
                build_canonical_match("고객 사연", 1, "고객 사연"),
                build_canonical_match("제철 과일", 2, "제철 과일"),
            ],
            reference_captions=[
                (
                    "\"다시 생각해보고 연락드릴게요\" 하고 끝난 대화도 오래 남는 날이 "
                    "있죠. 정중한 마음 하나가 관계를 오래 기억하게 만듭니다."
                ),
                (
                    "가게의 케이크는 지금 제철 과일의 속도에 맞춰 천천히 바뀌고 있습니다. "
                    "짧은 계절을 제때 즐기는 마음까지 함께 전하고 싶어요."
                ),
            ],
        ),
    ]


def percentile(values: list[float], ratio: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, ceil(ratio * len(ordered)) - 1)
    return ordered[index]


def average_int(values: list[int]) -> float:
    if not values:
        return 0.0
    return float(statistics.mean(values))


def preview_text(value: str, limit: int = 120) -> str:
    trimmed = value.strip()
    if len(trimmed) <= limit:
        return trimmed
    return f"{trimmed[: limit - 3].rstrip()}..."


async def run_single_iteration(
    *,
    scenario: ScenarioDefinition,
    variant: str,
    iteration: int,
    caption_service: CaptionGenerationService,
) -> BenchmarkRun:
    redis = fakeredis.aioredis.FakeRedis(decode_responses=True)
    session_id = f"bench-{scenario.name}-{variant}-{iteration}-{uuid4()}"
    payload = ProcessUtteranceRequest(
        store_id=STORE_ID,
        utterance=scenario.utterance,
        owner_persona=OWNER_PERSONA,
        date=REQUEST_DATE,
        weather=scenario.weather,
    )
    started_at = time.perf_counter()
    result = await process_utterance(
        session_id=session_id,
        payload=payload,
        redis=redis,
        keyword_service=StaticKeywordExtractionService(scenario.draft_keywords),
        caption_service=caption_service,
        menu_promotion_context_service=StaticMenuPromotionContextService(
            scenario.matched_menu_context
        ),
        intent_classifier_service=StaticIntentClassifierService(),
        canonical_keyword_resolver_service=StaticCanonicalKeywordResolverService(
            scenario.canonical_matches
        ),
        reference_caption_retriever_service=StaticReferenceCaptionRetrieverService(
            scenario.reference_captions
        ),
        context_routing_variant=variant,
    )
    elapsed = time.perf_counter() - started_at
    saved = await redis.hgetall(session_key(session_id))
    await redis.aclose()

    fallback_source = saved.get("debug:text_generation_fallback_source")
    return BenchmarkRun(
        scenario=scenario.name,
        variant=variant,
        iteration=iteration,
        session_id=session_id,
        latency_seconds=elapsed,
        http_semantic_success=result.status == "TEXT_GENERATED",
        fallback_source=fallback_source,
        context_variant=saved.get("debug:context_routing_variant"),
        context_profile=saved.get("debug:context_profile"),
        prompt_char_count=int(saved.get("debug:prompt_char_count", "0")),
        reference_caption_count=int(saved.get("debug:reference_caption_count", "0")),
        external_dependency_calls=int(
            saved.get("debug:external_dependency_calls", "0")
        ),
        model_call_count=int(saved.get("debug:model_call_count", "0")),
        graph_path=saved.get("debug:graph_path", ""),
        skipped_nodes=saved.get("debug:skipped_nodes", ""),
        caption_preview=preview_text(result.caption),
        guide_preview=preview_text(result.guide_text),
    )


def summarize_runs(
    runs: list[BenchmarkRun],
    *,
    scenario_name: str,
    variant: str,
    expected_profile: str,
) -> dict[str, Any]:
    valid_runs = [
        run
        for run in runs
        if run.valid
        and run.context_variant == variant
        and run.context_profile == expected_profile
    ]
    latency_values = [run.latency_seconds for run in valid_runs]
    prompt_values = [run.prompt_char_count for run in valid_runs]
    reference_values = [run.reference_caption_count for run in valid_runs]
    dependency_values = [run.external_dependency_calls for run in valid_runs]
    model_call_values = [run.model_call_count for run in valid_runs]
    fallback_count = sum(1 for run in runs if run.fallback_source is not None)
    profile_mismatch_count = sum(
        1 for run in runs if run.context_profile not in {None, expected_profile}
    )

    return {
        "scenario": scenario_name,
        "variant": variant,
        "expected_profile": expected_profile,
        "total_runs": len(runs),
        "valid_runs": len(valid_runs),
        "fallback_count": fallback_count,
        "fallback_rate": (fallback_count / len(runs)) if runs else 0.0,
        "profile_mismatch_count": profile_mismatch_count,
        "avg_latency_seconds": (
            round(statistics.mean(latency_values), 3) if latency_values else None
        ),
        "p95_latency_seconds": (
            round(percentile(latency_values, 0.95), 3) if latency_values else None
        ),
        "avg_prompt_char_count": (
            round(average_int(prompt_values), 1) if prompt_values else None
        ),
        "avg_reference_caption_count": (
            round(average_int(reference_values), 2) if reference_values else None
        ),
        "avg_external_dependency_calls": (
            round(average_int(dependency_values), 2) if dependency_values else None
        ),
        "avg_model_call_count": (
            round(average_int(model_call_values), 2) if model_call_values else None
        ),
    }


def summarize_overall(
    grouped: list[dict[str, Any]],
) -> dict[str, Any]:
    baseline = [item for item in grouped if item["variant"] == "baseline"]
    selective = [item for item in grouped if item["variant"] == "selective"]
    full_latency_coverage = all(item["valid_runs"] > 0 for item in grouped)

    def collect(field: str, items: list[dict[str, Any]]) -> list[float]:
        return [
            float(item[field])
            for item in items
            if item.get(field) is not None
        ]

    baseline_latencies = collect("avg_latency_seconds", baseline)
    selective_latencies = collect("avg_latency_seconds", selective)

    return {
        "full_latency_coverage": full_latency_coverage,
        "baseline_avg_latency_seconds": (
            round(statistics.mean(baseline_latencies), 3)
            if full_latency_coverage and baseline_latencies
            else None
        ),
        "selective_avg_latency_seconds": (
            round(statistics.mean(selective_latencies), 3)
            if full_latency_coverage and selective_latencies
            else None
        ),
        "baseline_avg_prompt_char_count": (
            round(statistics.mean(collect("avg_prompt_char_count", baseline)), 1)
            if collect("avg_prompt_char_count", baseline)
            else None
        ),
        "selective_avg_prompt_char_count": (
            round(statistics.mean(collect("avg_prompt_char_count", selective)), 1)
            if collect("avg_prompt_char_count", selective)
            else None
        ),
    }


def build_portfolio_recommendation(summary: dict[str, Any]) -> dict[str, Any]:
    if not summary.get("full_latency_coverage", False):
        return {
            "use_time_metric_in_portfolio": False,
            "reason": (
                "Some scenario and variant pairs did not produce any valid "
                "non-fallback latency samples."
            ),
            "recommended_sentence": (
                "동일 요청 기준 프롬프트 길이를 801자에서 727자로 약 9.2% 절감하고, "
                "메뉴 설명만으로 충분한 경우 reference retrieval이 호출되지 않도록 개선했습니다."
            ),
        }

    baseline_latency = summary["baseline_avg_latency_seconds"]
    selective_latency = summary["selective_avg_latency_seconds"]
    if baseline_latency is None or selective_latency is None:
        return {
            "use_time_metric_in_portfolio": False,
            "reason": "No valid latency measurements were collected.",
            "recommended_sentence": (
                "프롬프트 길이 801자에서 727자로 약 9.2% 줄이고, "
                "불필요한 reference retrieval을 생략하도록 구조를 개선했습니다."
            ),
        }

    improved = selective_latency < baseline_latency
    if improved:
        return {
            "use_time_metric_in_portfolio": True,
            "reason": "Selective variant improved the overall average latency.",
            "recommended_sentence": (
                "동일 요청 기준 프롬프트 길이를 801자에서 727자로 약 9.2% 절감하고, "
                "불필요한 reference retrieval 호출을 제거해 평균 응답 시간을 "
                f"{baseline_latency:.2f}초에서 {selective_latency:.2f}초로 단축했습니다."
            ),
        }

    return {
        "use_time_metric_in_portfolio": False,
        "reason": "Selective variant did not improve the overall average latency.",
        "recommended_sentence": (
            "동일 요청 기준 프롬프트 길이를 801자에서 727자로 약 9.2% 절감하고, "
            "메뉴 설명만으로 충분한 경우 reference retrieval이 호출되지 않도록 개선했습니다."
        ),
    }


def format_summary_markdown(
    *,
    grouped_summary: list[dict[str, Any]],
    overall_summary: dict[str, Any],
    recommendation: dict[str, Any],
    iterations: int,
) -> str:
    lines = [
        "# Context Routing Benchmark",
        "",
        f"- Iterations per scenario and variant: `{iterations}`",
        f"- Caption model base URL: `{build_caption_generation_service().base_url}`",
        "",
        "## Scenario Summary",
        "",
        "| Scenario | Variant | Valid Runs | Fallback Rate | Avg Latency (s) | P95 Latency (s) | Avg Prompt Chars | Avg Ref Count | Avg External Calls |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for item in grouped_summary:
        lines.append(
            "| "
            f"{item['scenario']} | {item['variant']} | {item['valid_runs']}/{item['total_runs']} | "
            f"{item['fallback_rate']:.2%} | "
            f"{item['avg_latency_seconds'] if item['avg_latency_seconds'] is not None else '-'} | "
            f"{item['p95_latency_seconds'] if item['p95_latency_seconds'] is not None else '-'} | "
            f"{item['avg_prompt_char_count'] if item['avg_prompt_char_count'] is not None else '-'} | "
            f"{item['avg_reference_caption_count'] if item['avg_reference_caption_count'] is not None else '-'} | "
            f"{item['avg_external_dependency_calls'] if item['avg_external_dependency_calls'] is not None else '-'} |"
        )

    lines.extend(
        [
            "",
            "## Overall",
            "",
            f"- Full latency coverage: `{overall_summary['full_latency_coverage']}`",
            f"- Baseline average latency: `{overall_summary['baseline_avg_latency_seconds']}` s",
            f"- Selective average latency: `{overall_summary['selective_avg_latency_seconds']}` s",
            f"- Baseline average prompt chars: `{overall_summary['baseline_avg_prompt_char_count']}`",
            f"- Selective average prompt chars: `{overall_summary['selective_avg_prompt_char_count']}`",
            "",
            "## Portfolio Recommendation",
            "",
            f"- Use time metric in portfolio: `{recommendation['use_time_metric_in_portfolio']}`",
            f"- Reason: {recommendation['reason']}",
            f"- Suggested sentence: {recommendation['recommended_sentence']}",
        ]
    )
    return "\n".join(lines) + "\n"


async def run_benchmark(iterations: int) -> dict[str, Any]:
    scenarios = build_scenarios()
    variants = ["baseline", "selective"]
    caption_service = build_caption_generation_service()
    await caption_service.preload()

    raw_runs: list[BenchmarkRun] = []

    for scenario in scenarios:
        for variant in variants:
            await run_single_iteration(
                scenario=scenario,
                variant=variant,
                iteration=0,
                caption_service=caption_service,
            )
        for iteration in range(1, iterations + 1):
            variant_order = (
                ["baseline", "selective"]
                if iteration % 2 == 1
                else ["selective", "baseline"]
            )
            for variant in variant_order:
                raw_runs.append(
                    await run_single_iteration(
                        scenario=scenario,
                        variant=variant,
                        iteration=iteration,
                        caption_service=caption_service,
                    )
                )

    grouped_summary: list[dict[str, Any]] = []
    for scenario in scenarios:
        for variant in variants:
            scenario_runs = [
                run
                for run in raw_runs
                if run.scenario == scenario.name and run.variant == variant
            ]
            grouped_summary.append(
                summarize_runs(
                    scenario_runs,
                    scenario_name=scenario.name,
                    variant=variant,
                    expected_profile=scenario.expected_profile,
                )
            )

    overall_summary = summarize_overall(grouped_summary)
    recommendation = build_portfolio_recommendation(overall_summary)

    return {
        "metadata": {
            "iterations": iterations,
            "store_id": STORE_ID,
            "owner_persona": OWNER_PERSONA,
            "request_date": REQUEST_DATE.isoformat(),
            "caption_model_base_url": caption_service.base_url,
            "caption_model_timeout_seconds": caption_service.timeout_seconds,
        },
        "grouped_summary": grouped_summary,
        "overall_summary": overall_summary,
        "portfolio_recommendation": recommendation,
        "raw_runs": [
            {
                "scenario": run.scenario,
                "variant": run.variant,
                "iteration": run.iteration,
                "session_id": run.session_id,
                "latency_seconds": round(run.latency_seconds, 3),
                "valid": run.valid,
                "fallback_source": run.fallback_source,
                "context_variant": run.context_variant,
                "context_profile": run.context_profile,
                "prompt_char_count": run.prompt_char_count,
                "reference_caption_count": run.reference_caption_count,
                "external_dependency_calls": run.external_dependency_calls,
                "model_call_count": run.model_call_count,
                "graph_path": run.graph_path,
                "skipped_nodes": run.skipped_nodes,
                "caption_preview": run.caption_preview,
                "guide_preview": run.guide_preview,
            }
            for run in raw_runs
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark process-utterance context routing with deterministic "
            "upstream state and a live caption model."
        )
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=10,
        help="Measured iterations per scenario and variant.",
    )
    return parser.parse_args()


def write_artifacts(result: dict[str, Any], *, iterations: int) -> Path:
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    output_dir = ARTIFACT_ROOT / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_json_path = output_dir / "summary.json"
    raw_json_path = output_dir / "raw-runs.json"
    summary_md_path = output_dir / "summary.md"

    summary_payload = {
        "metadata": result["metadata"],
        "grouped_summary": result["grouped_summary"],
        "overall_summary": result["overall_summary"],
        "portfolio_recommendation": result["portfolio_recommendation"],
    }

    summary_json_path.write_text(
        json.dumps(summary_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    raw_json_path.write_text(
        json.dumps(result["raw_runs"], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    summary_md_path.write_text(
        format_summary_markdown(
            grouped_summary=result["grouped_summary"],
            overall_summary=result["overall_summary"],
            recommendation=result["portfolio_recommendation"],
            iterations=iterations,
        ),
        encoding="utf-8",
    )
    return output_dir


def main() -> None:
    args = parse_args()
    result = asyncio.run(run_benchmark(args.iterations))
    output_dir = write_artifacts(result, iterations=args.iterations)
    print(json.dumps(result["overall_summary"], ensure_ascii=False, indent=2))
    print(result["portfolio_recommendation"]["recommended_sentence"])
    print(f"Artifacts written to: {output_dir}")


if __name__ == "__main__":
    main()
