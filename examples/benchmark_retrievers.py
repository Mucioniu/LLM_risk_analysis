from __future__ import annotations

import argparse
import hashlib
import json
import platform
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import scipy
import sklearn


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from credit_assistant.rag import (  # noqa: E402
    DEFAULT_BM25_B,
    DEFAULT_BM25_K1,
    RagIndex,
)
from credit_assistant.service import default_corpus_paths  # noqa: E402


DEFAULT_CASES = PROJECT_ROOT / "examples" / "evaluation_cases.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "examples" / "benchmark_retrievers_tfidf_vs_bm25.json"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare TF-IDF and BM25 over the frozen policy questions without "
            "calling an LLM. Expected-keyword coverage is a diagnostic, not a "
            "substitute for passage-level relevance judgments."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--retrievers",
        nargs="+",
        choices=("tfidf", "bm25"),
        default=("tfidf", "bm25"),
    )
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--visible-chars", type=int, default=900)
    parser.add_argument("--repetitions", type=int, default=500)
    parser.add_argument("--bm25-k1", type=float, default=DEFAULT_BM25_K1)
    parser.add_argument("--bm25-b", type=float, default=DEFAULT_BM25_B)
    args = parser.parse_args(argv)
    if args.top_k <= 0 or args.visible_chars <= 0 or args.repetitions <= 0:
        parser.error("--top-k, --visible-chars, and --repetitions must be positive.")
    return args


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def keyword_coverage(text: str, keywords: list[str]) -> tuple[int, int, float]:
    if not keywords:
        return 0, 0, 1.0
    normalized = text.casefold()
    found = sum(keyword.casefold() in normalized for keyword in keywords)
    return found, len(keywords), found / len(keywords)


def visible_excerpt(text: str, max_chars: int) -> str:
    """Mirror the production formatter's excerpt normalization and limit."""
    normalized = text.replace("\n", " ")
    if len(normalized) > max_chars:
        return normalized[: max_chars - 3].rstrip() + "..."
    return normalized


def expected_source_coverage(
    retrieved_sources: list[str], expected: list[str]
) -> float | None:
    if not expected:
        return None
    hits = sum(
        any(name.casefold() in source.casefold() for source in retrieved_sources)
        for name in expected
    )
    return hits / len(expected)


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(round((len(ordered) - 1) * fraction)))
    return ordered[index]


def benchmark_latency(
    index: RagIndex,
    questions: list[str],
    *,
    top_k: int,
    repetitions: int,
) -> dict[str, float]:
    for question in questions:
        index.search(question, top_k=top_k)

    timings_ms: list[float] = []
    for _ in range(repetitions):
        for question in questions:
            started = time.perf_counter_ns()
            index.search(question, top_k=top_k)
            timings_ms.append((time.perf_counter_ns() - started) / 1_000_000)
    return {
        "samples": len(timings_ms),
        "median_ms": statistics.median(timings_ms),
        "p95_ms": percentile(timings_ms, 0.95),
    }


def evaluate_retriever(
    name: str,
    cases: list[dict[str, Any]],
    *,
    top_k: int,
    visible_chars: int,
    repetitions: int,
    bm25_k1: float,
    bm25_b: float,
) -> dict[str, Any]:
    started = time.perf_counter()
    index = RagIndex.from_paths(
        default_corpus_paths(),
        retriever=name,
        bm25_k1=bm25_k1,
        bm25_b=bm25_b,
    )
    build_ms = (time.perf_counter() - started) * 1000

    answerable: list[dict[str, Any]] = []
    case_results: list[dict[str, Any]] = []
    for case in cases:
        results = index.search(str(case["question"]), top_k=top_k)
        expected_keywords = [str(value) for value in case.get("expected_keywords", [])]
        full_text = "\n".join(result.chunk.text for result in results)
        # Mirror production's per-chunk text normalization and truncation while
        # excluding source metadata from the answer-keyword diagnostic.
        visible_text = "\n".join(
            visible_excerpt(result.chunk.text, visible_chars) for result in results
        )
        top_one_text = results[0].chunk.text if results else ""
        top_one = keyword_coverage(top_one_text, expected_keywords)
        full = keyword_coverage(full_text, expected_keywords)
        visible = keyword_coverage(visible_text, expected_keywords)
        retrieved_sources = [result.chunk.source for result in results]
        result_document = {
            "case_id": case["id"],
            "expect_missing": bool(case.get("expect_missing", False)),
            "expected_source_coverage_at_k": expected_source_coverage(
                retrieved_sources,
                [str(value) for value in case.get("expected_source_contains", [])],
            ),
            "expected_keyword_coverage_top_1": {
                "found": top_one[0],
                "total": top_one[1],
                "score": top_one[2],
            },
            "expected_keyword_coverage_top_k_full_chunks": {
                "found": full[0],
                "total": full[1],
                "score": full[2],
            },
            "expected_keyword_coverage_top_k_visible_prefixes": {
                "found": visible[0],
                "total": visible[1],
                "score": visible[2],
            },
            "results": [
                {
                    "rank": rank,
                    "source": result.chunk.source,
                    "location": result.chunk.location,
                    "score": result.score,
                    "visible_prefix": result.chunk.text[:visible_chars],
                }
                for rank, result in enumerate(results, start=1)
            ],
        }
        case_results.append(result_document)
        if not result_document["expect_missing"]:
            answerable.append(result_document)

    def mean_metric(key: str) -> float:
        return statistics.fmean(item[key]["score"] for item in answerable)

    aggregate = {
        "answerable_cases": len(answerable),
        "expected_source_coverage_at_k": statistics.fmean(
            item["expected_source_coverage_at_k"] for item in answerable
        ),
        "mean_expected_keyword_coverage_top_1": mean_metric(
            "expected_keyword_coverage_top_1"
        ),
        "mean_expected_keyword_coverage_top_k_full_chunks": mean_metric(
            "expected_keyword_coverage_top_k_full_chunks"
        ),
        "mean_expected_keyword_coverage_top_k_visible_prefixes": mean_metric(
            "expected_keyword_coverage_top_k_visible_prefixes"
        ),
        "all_expected_keywords_visible_at_k_cases": sum(
            item["expected_keyword_coverage_top_k_visible_prefixes"]["score"] == 1.0
            for item in answerable
        ),
    }
    return {
        "retriever": name,
        "configuration": {
            "lowercase": True,
            "strip_accents": "unicode",
            "ngram_range": [1, 2],
            "max_features": 20_000,
            "bm25_k1": bm25_k1 if name == "bm25" else None,
            "bm25_b": bm25_b if name == "bm25" else None,
            "top_k": top_k,
            "visible_chars_per_chunk": visible_chars,
        },
        "index": {
            "chunks": len(index.chunks),
            "vocabulary": len(index.vectorizer.vocabulary_),
            "average_analyzer_features_per_chunk": index.average_document_length,
            "build_ms": build_ms,
        },
        "latency": benchmark_latency(
            index,
            [str(case["question"]) for case in cases],
            top_k=top_k,
            repetitions=repetitions,
        ),
        "aggregate": aggregate,
        "cases": case_results,
    }


def markdown_report(document: dict[str, Any]) -> str:
    top_k = document["configuration"]["top_k"]
    lines = [
        "# Paired sparse-retriever benchmark",
        "",
        "Both retrievers use the same frozen chunks, query set, lowercase/accent handling, "
        "and unigram/bigram vocabulary. Raw TF-IDF and BM25 scores are not compared because "
        "their scales differ. Expected-keyword coverage is a provisional visible-evidence "
        "diagnostic, not a passage-relevance judgment.",
        "",
        f"| Retriever | Expected-source coverage@{top_k} | Top-1 keyword coverage | Top-{top_k} full-chunk coverage | Top-{top_k} visible-prefix coverage | Median search | p95 search |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for result in document["results"]:
        aggregate = result["aggregate"]
        latency = result["latency"]
        lines.append(
            f"| {result['retriever'].upper()} | {aggregate['expected_source_coverage_at_k']:.2%} | "
            f"{aggregate['mean_expected_keyword_coverage_top_1']:.2%} | "
            f"{aggregate['mean_expected_keyword_coverage_top_k_full_chunks']:.2%} | "
            f"{aggregate['mean_expected_keyword_coverage_top_k_visible_prefixes']:.2%} | "
            f"{latency['median_ms']:.3f} ms | {latency['p95_ms']:.3f} ms |"
        )

    lines.extend(
        [
            "",
            "## Per-question visible-prefix coverage",
            "",
            "The missing-policy question is shown but excluded from the aggregate keyword metrics.",
            "",
            "| Case | "
            + " | ".join(result["retriever"].upper() for result in document["results"])
            + " |",
            "|---|" + "---:|" * len(document["results"]),
        ]
    )
    cases_by_method = {
        result["retriever"]: {case["case_id"]: case for case in result["cases"]}
        for result in document["results"]
    }
    case_ids = [case["case_id"] for case in document["results"][0]["cases"]]
    for case_id in case_ids:
        values = []
        for result in document["results"]:
            case = cases_by_method[result["retriever"]][case_id]
            score = case["expected_keyword_coverage_top_k_visible_prefixes"]["score"]
            values.append("excluded" if case["expect_missing"] else f"{score:.2%}")
        lines.append(f"| {case_id} | " + " | ".join(values) + " |")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    args.cases = args.cases.resolve()
    args.output = args.output.resolve()
    loaded = json.loads(args.cases.read_text(encoding="utf-8"))
    cases = loaded.get("policy_questions", [])
    if not isinstance(cases, list) or not cases:
        raise SystemExit("No policy questions found.")

    corpus_paths = default_corpus_paths()
    document = {
        "schema_version": 1,
        "benchmark": "paired_sparse_retrievers",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "configuration": {
            "top_k": args.top_k,
            "visible_chars_per_chunk": args.visible_chars,
            "latency_repetitions_per_question": args.repetitions,
            "effectiveness_questions": sum(
                not bool(case.get("expect_missing", False)) for case in cases
            ),
            "latency_questions": len(cases),
        },
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "limitations": [
            "Expected document names are too coarse to establish passage relevance.",
            "Expected answer keywords are a diagnostic and include generic tokens.",
            "The missing-policy case is excluded from keyword aggregates.",
            "Raw score magnitudes are not comparable across retrievers.",
        ],
        "input_sha256": {
            "cases": sha256_file(args.cases),
            "benchmark_implementation": sha256_file(Path(__file__).resolve()),
            "document_loader_implementation": sha256_file(
                PROJECT_ROOT / "credit_assistant" / "document_loader.py"
            ),
            "rag_implementation": sha256_file(
                PROJECT_ROOT / "credit_assistant" / "rag.py"
            ),
            "service_implementation": sha256_file(
                PROJECT_ROOT / "credit_assistant" / "service.py"
            ),
            **{path.name: sha256_file(path) for path in corpus_paths},
        },
        "results": [
            evaluate_retriever(
                name,
                cases,
                top_k=args.top_k,
                visible_chars=args.visible_chars,
                repetitions=args.repetitions,
                bm25_k1=args.bm25_k1,
                bm25_b=args.bm25_b,
            )
            for name in args.retrievers
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(document, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    markdown_path = args.output.with_suffix(".md")
    markdown_path.write_text(markdown_report(document), encoding="utf-8")
    print(markdown_report(document), end="")
    print(f"JSON: {args.output}")
    print(f"Markdown: {markdown_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
