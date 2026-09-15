"""Repeatable local benchmark for every configured ABSA engine.

Usage: python benchmarks/run_batch_benchmark.py reviews.csv --batch-sizes 1 8 32 --iterations 3
"""
import argparse
import csv
import json
import platform
import statistics
import sys
from pathlib import Path
from time import perf_counter

import psutil

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.inference.engine import inference_engine_manager  # noqa: E402
from app.services.enrichment import analyze_enriched  # noqa: E402


def percentile(values, p):
    ordered = sorted(values)
    index = min(len(ordered) - 1, round((len(ordered) - 1) * p))
    return ordered[index] if ordered else 0.0


def load_reviews(path):
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        column = next((name for name in ("review", "review_text", "text", "comment", "komentar", "ulasan") if name in (reader.fieldnames or [])), None)
        if not column:
            raise SystemExit("Kolom review tidak ditemukan.")
        return [row[column] for row in reader if row.get(column)]


def run(dataset, reviews, version, batch_size, iterations):
    # Warm-up is deliberately separate from measured iterations.
    warm_start = perf_counter()
    warm_result = analyze_enriched(inference_engine_manager, reviews[0], version)
    warmup_ms = (perf_counter() - warm_start) * 1000
    iteration_results = []
    process = psutil.Process()
    for _ in range(iterations):
        latencies = []
        before_memory = process.memory_info().rss
        started = perf_counter()
        success = failure = 0
        for offset in range(0, len(reviews), batch_size):
            for review in reviews[offset:offset + batch_size]:
                row_started = perf_counter()
                try:
                    analyze_enriched(inference_engine_manager, review, version)
                    success += 1
                except Exception:
                    failure += 1
                latencies.append((perf_counter() - row_started) * 1000)
        total_ms = (perf_counter() - started) * 1000
        iteration_results.append({
            "total_time_ms": total_ms, "average_latency_ms": statistics.fmean(latencies),
            "median_latency_ms": statistics.median(latencies), "p95_latency_ms": percentile(latencies, .95),
            "p99_latency_ms": percentile(latencies, .99), "throughput_rows_per_second": success / (total_ms / 1000),
            "success_count": success, "failure_count": failure,
            "memory_delta_mb": (process.memory_info().rss - before_memory) / 1024 / 1024,
        })
    keys = ("total_time_ms", "average_latency_ms", "median_latency_ms", "p95_latency_ms", "p99_latency_ms", "throughput_rows_per_second", "memory_delta_mb")
    return {
        "dataset_name": dataset.name, "dataset_size": len(reviews), "model_version": version,
        "batch_size": batch_size, "device": platform.platform(), "warmup_status": True,
        "warmup_time_ms": round(warmup_ms, 3), "iterations": iterations,
        "inference_mode": warm_result.get("inference_mode"),
        "model_bundle_loaded": warm_result.get("model_bundle_loaded"),
        "model_load_error": warm_result.get("model_load_error"),
        **{key: round(statistics.fmean(item[key] for item in iteration_results), 3) for key in keys},
        "success_count": iteration_results[0]["success_count"], "failure_count": iteration_results[0]["failure_count"],
        "runs": iteration_results,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[32])
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--models", nargs="+", default=inference_engine_manager.supported_versions)
    args = parser.parse_args()
    reviews = load_reviews(args.dataset)
    results = [run(args.dataset, reviews, version, size, args.iterations) for version in args.models for size in args.batch_sizes]
    print(json.dumps({"benchmark": results}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
