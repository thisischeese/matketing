"""CLI entry point for the best-shot benchmark."""

from __future__ import annotations

import argparse
from pathlib import Path

from bestshot_benchmark.contact_sheet import render_contact_sheets
from bestshot_benchmark.io import read_frames_csv, read_videos_csv, write_frames_csv
from bestshot_benchmark.runner import run_benchmark
from bestshot_benchmark.sampling import sample_videos


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="benchmark_aesthetic_models",
        description="Prepare and run local best-shot aesthetic benchmarks.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    sample_parser = subparsers.add_parser(
        "sample-frames",
        help="Sample frames from videos.csv using the service-aligned 5fps policy.",
    )
    sample_parser.add_argument("--videos-csv", type=Path, required=True)
    sample_parser.add_argument("--workspace", type=Path, required=True)

    contact_parser = subparsers.add_parser(
        "render-contact-sheets",
        help="Render manual labeling contact sheets from frames.csv.",
    )
    contact_parser.add_argument("--frames-csv", type=Path, required=True)
    contact_parser.add_argument("--workspace", type=Path, required=True)
    contact_parser.add_argument("--columns", type=int, default=4)
    contact_parser.add_argument("--shuffle-seed", type=int, default=None)

    run_parser = subparsers.add_parser(
        "run",
        help="Run scorer benchmark and write report artifacts.",
    )
    run_parser.add_argument("--frames-csv", type=Path, required=True)
    run_parser.add_argument("--labels-csv", type=Path, required=True)
    run_parser.add_argument("--workspace", type=Path, required=True)
    run_parser.add_argument("--device", type=str, default=None)
    run_parser.add_argument(
        "--models",
        nargs="+",
        default=["nima", "clipiqa", "qalign"],
    )
    run_parser.add_argument("--total-budget-ms", type=float, default=3000.0)

    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    if args.command == "sample-frames":
        videos = read_videos_csv(args.videos_csv)
        sampled_frames = sample_videos(videos, workspace=args.workspace)
        write_frames_csv(args.workspace / "frames.csv", sampled_frames)
        print(f"Sampled {len(sampled_frames)} frames into {args.workspace}")
        return

    if args.command == "render-contact-sheets":
        frames = read_frames_csv(args.frames_csv)
        output_paths = render_contact_sheets(
            frames,
            output_dir=args.workspace / "contact_sheets",
            columns=args.columns,
            shuffle_seed=args.shuffle_seed,
        )
        print(f"Rendered {len(output_paths)} contact sheets into {args.workspace / 'contact_sheets'}")
        return

    if args.command == "run":
        summary = run_benchmark(
            frames_csv_path=args.frames_csv,
            labels_csv_path=args.labels_csv,
            workspace=args.workspace,
            model_names=args.models,
            device=args.device,
            total_budget_ms=args.total_budget_ms,
        )
        winner = summary.get("winner")
        if winner is None:
            print("Benchmark completed. No model satisfied the latency cutoff.")
        else:
            print(f"Benchmark completed. Winner: {winner['model_name']}")
        return

    parser.error(f"Unknown command: {args.command}")

