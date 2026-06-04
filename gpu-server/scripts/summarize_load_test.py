from __future__ import annotations

import csv
import json
import math
import statistics
import sys
from pathlib import Path
import re


FALLBACK_PATTERN = re.compile(
    r'event="session\.process_utterance\.text_generation\.completed".*outcome="fallback"'
)


def percentile(values: list[float], ratio: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]

    ordered = sorted(values)
    index = (len(ordered) - 1) * ratio
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return ordered[int(index)]
    weight = index - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_gpu_samples(path: Path) -> list[dict[str, float]]:
    if not path.exists():
        return []

    rows: list[dict[str, float]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            try:
                rows.append(
                    {
                        "gpu": float(row["utilization_gpu"]),
                        "memory": float(row["memory_used_mib"]),
                        "memory_total": float(row["memory_total_mib"]),
                    }
                )
            except (KeyError, ValueError):
                continue
    return rows


def count_fallbacks(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(
        1
        for line in path.read_text(encoding="utf-8", errors="ignore").splitlines()
        if FALLBACK_PATTERN.search(line)
    )


def metric_value(summary: dict, metric_name: str, value_name: str, default: float = 0.0) -> float:
    metrics = summary.get("metrics", {})
    metric = metrics.get(metric_name, {})
    values = metric.get("values", {})
    try:
        return float(values.get(value_name, default))
    except (TypeError, ValueError):
        return default


def summarize_stage(stage_dir: Path) -> dict[str, float | int | str]:
    summary_path = stage_dir / "k6-summary.json"
    gpu_path = stage_dir / "gpu-metrics.csv"
    fastapi_log = stage_dir / "fastapi.log"

    summary = load_json(summary_path)
    gpu_samples = read_gpu_samples(gpu_path)
    gpu_utils = [sample["gpu"] for sample in gpu_samples]
    vram_usage = [
        sample["memory"] / sample["memory_total"] * 100
        for sample in gpu_samples
        if sample["memory_total"] > 0
    ]

    return {
        "stage": stage_dir.name,
        "http_avg_ms": metric_value(summary, "http_req_duration", "avg"),
        "http_p50_ms": metric_value(summary, "http_req_duration", "med"),
        "http_p95_ms": metric_value(summary, "http_req_duration", "p(95)"),
        "http_max_ms": metric_value(summary, "http_req_duration", "max"),
        "http_failed_rate": metric_value(summary, "http_req_failed", "rate"),
        "timeout_rate": metric_value(summary, "timeout_rate", "rate"),
        "http_5xx_rate": metric_value(summary, "http_5xx_rate", "rate"),
        "final_edit_503_rate": metric_value(summary, "final_edit_503_rate", "rate"),
        "fallback_suspected_count": int(metric_value(summary, "fallback_suspected_count", "count")),
        "fallback_log_count": count_fallbacks(fastapi_log),
        "final_edit_result_count": int(metric_value(summary, "final_edit_result_count", "count")),
        "pipeline_completed_count": int(metric_value(summary, "pipeline_completed_count", "count")),
        "gpu_util_avg": statistics.mean(gpu_utils) if gpu_utils else 0.0,
        "gpu_util_p95": percentile(gpu_utils, 0.95) if gpu_utils else 0.0,
        "vram_max_pct": max(vram_usage) if vram_usage else 0.0,
    }


def render_markdown(stage_rows: list[dict[str, float | int | str]]) -> str:
    lines = [
        "# Load Test Summary",
        "",
        "| stage | avg ms | p50 ms | p95 ms | max ms | failed rate | timeout rate | 5xx rate | fallback log count | final-edit 503 rate | final result count | gpu avg | gpu p95 | vram max % |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    for row in stage_rows:
        lines.append(
            "| {stage} | {http_avg_ms:.1f} | {http_p50_ms:.1f} | {http_p95_ms:.1f} | {http_max_ms:.1f} | "
            "{http_failed_rate:.3f} | {timeout_rate:.3f} | {http_5xx_rate:.3f} | {fallback_log_count} | "
            "{final_edit_503_rate:.3f} | {final_edit_result_count} | {gpu_util_avg:.1f} | {gpu_util_p95:.1f} | {vram_max_pct:.1f} |".format(
                **row
            )
        )

    return "\n".join(lines) + "\n"


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: python summarize_load_test.py <experiment_root>")
        return 1

    experiment_root = Path(sys.argv[1]).resolve()
    stage_dirs = sorted(experiment_root.glob("repeat-*/vu-*"))
    if not stage_dirs:
        print(f"no stage directories found under {experiment_root}")
        return 1

    rows = [summarize_stage(stage_dir) for stage_dir in stage_dirs]
    markdown = render_markdown(rows)

    (experiment_root / "summary.md").write_text(markdown, encoding="utf-8")
    (experiment_root / "summary.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(markdown)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
