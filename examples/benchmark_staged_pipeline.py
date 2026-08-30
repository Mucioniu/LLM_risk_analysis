from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import statistics
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from credit_assistant.credit_engine import (  # noqa: E402
    MAX_AMOUNT_RON,
    ClientProfile,
    evaluate_client,
)
from credit_assistant.evaluation import (  # noqa: E402
    MetricResult,
    format_score,
    load_evaluation_cases,
    numeric_agreement_details,
    required_sections_score,
    source_presence,
)
from credit_assistant.service import (  # noqa: E402
    LlmStageError,
    build_default_index,
    compare_staged_llm_to_deterministic,
    extract_llm_decision,
    format_staged_credit_markdown,
    llm_json_to_extracted,
    run_staged_llm_generation,
)


DEFAULT_CASES_PATH = PROJECT_ROOT / "examples" / "evaluation_cases.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "examples" / "benchmark_staged_pipeline_full_seed42.json"
NUMERIC_FIELDS = (
    ("stressed_monthly_payment", "Stressed monthly payment", 1.0, "RON"),
    ("dti_pct", "DTI", 0.05, "pp"),
    ("maximum_amount_by_dti", "Maximum recommended amount", 1.0, "RON"),
)


class InfrastructureFailure(RuntimeError):
    """A transient local-server failure that should be retried, not scored as model output."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark every client case through the current three-stage LLM+RAG pipeline, "
            "including stressed payment, DTI, and maximum recommended amount."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--cases-path", type=Path, default=DEFAULT_CASES_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--rag-model", default="mistral-small3.2:latest")
    parser.add_argument(
        "--retriever",
        choices=("tfidf", "bm25"),
        default="tfidf",
        help="Sparse retriever used to construct the Stage-1 evidence context.",
    )
    parser.add_argument("--bm25-k1", type=float, default=1.2)
    parser.add_argument("--bm25-b", type=float, default=0.75)
    parser.add_argument("--calculation-model", default="qwen3:14b")
    parser.add_argument("--synthesis-model", default="mistral-small3.2:latest")
    parser.add_argument(
        "--calculation-reasoning",
        choices=("off", "on", "low", "medium", "high"),
        default="on",
    )
    parser.add_argument("--num-ctx", type=int, default=8192)
    parser.add_argument("--calculation-num-predict", type=int, default=6000)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--timeout-seconds", type=float, default=900.0)
    parser.add_argument("--max-cases", type=int, default=None)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--resume", action="store_true")
    mode.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)

    if args.num_ctx <= 0 or args.calculation_num_predict <= 0:
        parser.error("Context and prediction budgets must be positive.")
    if args.timeout_seconds <= 0:
        parser.error("--timeout-seconds must be positive.")
    if args.temperature < 0:
        parser.error("--temperature cannot be negative.")
    if args.bm25_k1 <= 0 or not 0 <= args.bm25_b <= 1:
        parser.error("--bm25-k1 must be positive and --bm25-b must be in [0, 1].")
    if args.max_cases is not None and args.max_cases <= 0:
        parser.error("--max-cases must be positive.")
    return args


def configure_environment(args: argparse.Namespace) -> None:
    reasoning = "true" if args.calculation_reasoning == "on" else args.calculation_reasoning
    os.environ.update(
        {
            "OPENAI_API_KEY": "ollama",
            "OPENAI_BASE_URL": "http://127.0.0.1:11434/v1",
            "OLLAMA_BASE_URL": "http://127.0.0.1:11434",
            "OLLAMA_NATIVE_CHAT": "true",
            "OPENAI_RAG_MODEL": args.rag_model,
            "RAG_RETRIEVER": args.retriever,
            "RAG_BM25_K1": str(args.bm25_k1),
            "RAG_BM25_B": str(args.bm25_b),
            "OPENAI_CALCULATION_MODEL": args.calculation_model,
            "OPENAI_SYNTHESIS_MODEL": args.synthesis_model,
            "OLLAMA_RAG_THINK": "false",
            "OLLAMA_CALCULATION_THINK": reasoning,
            "OLLAMA_SYNTHESIS_THINK": "false",
            "OPENAI_RAG_TEMPERATURE": str(args.temperature),
            "OPENAI_CALCULATION_TEMPERATURE": str(args.temperature),
            "OPENAI_SYNTHESIS_TEMPERATURE": str(args.temperature),
            "OLLAMA_RAG_NUM_CTX": str(args.num_ctx),
            "OLLAMA_CALCULATION_NUM_CTX": str(args.num_ctx),
            "OLLAMA_SYNTHESIS_NUM_CTX": str(args.num_ctx),
            "OLLAMA_RAG_NUM_PREDICT": "1800",
            "OLLAMA_CALCULATION_NUM_PREDICT": str(args.calculation_num_predict),
            "OLLAMA_SYNTHESIS_NUM_PREDICT": "1400",
            "OPENAI_TIMEOUT_SECONDS": str(args.timeout_seconds),
            "OPENAI_SEED": str(args.seed),
        }
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def config_for(args: argparse.Namespace, case_ids: list[str]) -> dict[str, Any]:
    tracked_inputs = {
        "evaluation_cases": args.cases_path.resolve(),
        "service": PROJECT_ROOT / "credit_assistant" / "service.py",
        "evaluation": PROJECT_ROOT / "credit_assistant" / "evaluation.py",
        "llm": PROJECT_ROOT / "credit_assistant" / "llm.py",
        "credit_engine": PROJECT_ROOT / "credit_assistant" / "credit_engine.py",
        "rag": PROJECT_ROOT / "credit_assistant" / "rag.py",
        "benchmark_runner": Path(__file__).resolve(),
    }
    return {
        "pipeline": "rag_policy__isolated_calculation__final_synthesis",
        "retriever": args.retriever,
        "bm25_k1": args.bm25_k1 if args.retriever == "bm25" else None,
        "bm25_b": args.bm25_b if args.retriever == "bm25" else None,
        "rag_model": args.rag_model,
        "calculation_model": args.calculation_model,
        "synthesis_model": args.synthesis_model,
        "rag_reasoning": "off",
        "calculation_reasoning": args.calculation_reasoning,
        "synthesis_reasoning": "off",
        "num_ctx": args.num_ctx,
        "calculation_num_predict": args.calculation_num_predict,
        "temperature": args.temperature,
        "seed": args.seed,
        "timeout_seconds": args.timeout_seconds,
        "cases_path": str(args.cases_path.resolve()),
        "case_ids": case_ids,
        "numeric_fields": [field for field, _, _, _ in NUMERIC_FIELDS],
        "numeric_tolerances": {
            field: tolerance for field, _, tolerance, _ in NUMERIC_FIELDS
        },
        "maximum_amount_included": True,
        "reference_engine_use": "post_hoc_only_after_llm_generation",
        "case_score_metric_count": 8,
        "input_sha256": {
            name: sha256_file(path) for name, path in tracked_inputs.items()
        },
    }


def reference_dict(profile: ClientProfile, deterministic: Any) -> dict[str, Any]:
    return {
        "decision": deterministic.decision.value,
        "expected_case_decision": deterministic.decision.value,
        "stressed_monthly_payment": deterministic.stressed_monthly_payment,
        "dti_pct": deterministic.dti * 100,
        "maximum_amount_by_dti": deterministic.maximum_amount_by_dti,
        "weighted_income": deterministic.weighted_income,
        "maximum_total_payment_capacity": deterministic.max_monthly_payment,
        "available_payment_capacity": deterministic.available_payment_capacity,
        "profile_requested_amount": profile.requested_amount,
    }


def stage_error_markdown(exc: LlmStageError) -> str:
    raw = exc.raw_response or "The LLM returned no content."
    return (
        "## LLM staged pipeline error\n\n"
        f"The {exc.stage} stage failed its strict output gate.\n\n"
        f"Error: {exc}\n\n"
        f"```text\n{raw}\n```"
    )


def metric_rows(
    *,
    expected_decision: str,
    deterministic: Any,
    extracted: Any,
    scores: dict[str, float],
    answer: str,
) -> list[MetricResult]:
    return [
        MetricResult(
            "llm_decision_vs_expected",
            1.0 if extracted.decision == expected_decision else 0.0,
            f"LLM: {extracted.decision or 'not found'} / expected: {expected_decision}",
        ),
        MetricResult(
            "llm_decision_vs_formulas",
            scores.get("Decision", 0.0),
            f"LLM: {extracted.decision or 'not found'} / formulas: {deterministic.decision.value}",
        ),
        MetricResult(
            "overall_llm_vs_formulas_score",
            scores.get("overall_llm_vs_formulas_score", 0.0),
            "Decision and all three locked LLM calculation fields.",
        ),
        MetricResult(
            "isolated_numeric_agreement",
            scores.get("isolated_numeric_agreement", 0.0),
            numeric_agreement_details(extracted, deterministic),
        ),
        MetricResult(
            "all_three_numeric_fields_correct",
            scores.get("all_three_numeric_fields_correct", 0.0),
            "One only when stressed payment, DTI, and maximum amount all agree.",
        ),
        required_sections_score(answer),
        source_presence(answer),
        format_score(answer),
    ]


def numeric_comparisons(extracted: Any, deterministic: Any) -> dict[str, dict[str, Any]]:
    expected_values = {
        "stressed_monthly_payment": deterministic.stressed_monthly_payment,
        "dti_pct": deterministic.dti * 100,
        "maximum_amount_by_dti": deterministic.maximum_amount_by_dti,
    }
    comparisons: dict[str, dict[str, Any]] = {}
    for field, _, tolerance, unit in NUMERIC_FIELDS:
        actual = getattr(extracted, field)
        expected = expected_values[field]
        absolute_error = abs(actual - expected) if actual is not None else None
        comparisons[field] = {
            "llm": actual,
            "reference": expected,
            "absolute_error": absolute_error,
            "tolerance": tolerance,
            "error_unit": unit,
            "agreement": bool(absolute_error is not None and absolute_error <= tolerance),
        }
    return comparisons


def run_case(case: dict[str, Any], index: Any) -> dict[str, Any]:
    profile = ClientProfile(**case["profile"])
    started = time.perf_counter()
    generation = None
    stage_error = None
    try:
        generation = run_staged_llm_generation(profile, index)
    except LlmStageError as exc:
        if "The LLM is unavailable or incorrectly configured" in (exc.raw_response or ""):
            raise InfrastructureFailure(f"{exc.stage}: {exc.raw_response}") from exc
        stage_error = exc
    generation_seconds = time.perf_counter() - started

    # The benchmark reference is deliberately evaluated only after LLM generation has ended.
    deterministic = evaluate_client(profile)
    expected_decision = str(case.get("expected_decision", deterministic.decision.value))
    if generation is not None:
        answer = format_staged_credit_markdown(generation)
        extracted = llm_json_to_extracted(generation.credit_json)
        raw_stages = {
            "policy": generation.raw_policy,
            "calculation": generation.raw_calculation,
            "synthesis": generation.raw_synthesis,
        }
        pipeline_status = "ok"
        failed_stage = None
        stage_error_text = None
    else:
        assert stage_error is not None
        answer = stage_error_markdown(stage_error)
        extracted = extract_llm_decision(answer)
        raw_stages = {
            "policy": stage_error.raw_response if stage_error.stage == "RAG/policy" else None,
            "calculation": stage_error.raw_response if stage_error.stage == "calculation" else None,
            "synthesis": stage_error.raw_response if stage_error.stage == "final synthesis" else None,
        }
        pipeline_status = "stage_error"
        failed_stage = stage_error.stage
        stage_error_text = str(stage_error)

    comparison, scores = compare_staged_llm_to_deterministic(deterministic, extracted)
    metrics = metric_rows(
        expected_decision=expected_decision,
        deterministic=deterministic,
        extracted=extracted,
        scores=scores,
        answer=answer,
    )
    case_score = statistics.fmean(metric.score for metric in metrics)

    return {
        "case_id": case["id"],
        "pipeline_status": pipeline_status,
        "failed_stage": failed_stage,
        "stage_error": stage_error_text,
        "expected_decision": expected_decision,
        "llm_decision": extracted.decision,
        "reference_decision": deterministic.decision.value,
        "generation_seconds": generation_seconds,
        "case_score": case_score,
        "metrics": [asdict(metric) for metric in metrics],
        "numeric_comparisons": numeric_comparisons(extracted, deterministic),
        "all_three_numeric_fields_correct": bool(
            scores.get("all_three_numeric_fields_correct", 0.0) == 1.0
        ),
        "profile": asdict(profile),
        "extracted": asdict(extracted),
        "post_hoc_reference": reference_dict(profile, deterministic),
        "answer_markdown": answer,
        "comparison_markdown": comparison,
        "raw_stages": raw_stages,
    }


def metric_score(case: dict[str, Any], name: str) -> float:
    for metric in case.get("metrics", []):
        if metric.get("name") == name:
            return float(metric.get("score", 0.0))
    return 0.0


def validate_case_result(case: dict[str, Any]) -> None:
    comparisons = case["numeric_comparisons"]
    payment = float(comparisons["stressed_monthly_payment"]["agreement"])
    dti = float(comparisons["dti_pct"]["agreement"])
    maximum = float(comparisons["maximum_amount_by_dti"]["agreement"])
    decision = metric_score(case, "llm_decision_vs_formulas")
    expected_overall = (decision + payment + dti + maximum) / 4
    expected_isolated = (payment + dti + maximum) / 3
    expected_all_three = float(payment == dti == maximum == 1.0)
    checks = {
        "overall_llm_vs_formulas_score": expected_overall,
        "isolated_numeric_agreement": expected_isolated,
        "all_three_numeric_fields_correct": expected_all_three,
    }
    for metric_name, expected in checks.items():
        actual = metric_score(case, metric_name)
        if not abs(actual - expected) <= 1e-12:
            raise ValueError(
                f"{case['case_id']} has inconsistent {metric_name}: {actual} != {expected}."
            )
    metrics = case.get("metrics", [])
    if len(metrics) != 8:
        raise ValueError(f"{case['case_id']} must have exactly eight scoring metrics.")
    expected_case_score = statistics.fmean(float(metric["score"]) for metric in metrics)
    if not abs(float(case["case_score"]) - expected_case_score) <= 1e-12:
        raise ValueError(f"{case['case_id']} has an inconsistent case score.")


def error_statistics(values: list[float]) -> dict[str, float | None]:
    return {
        "mean_absolute_error": statistics.fmean(values) if values else None,
        "median_absolute_error": statistics.median(values) if values else None,
        "maximum_absolute_error": max(values) if values else None,
    }


def summarize(cases: list[dict[str, Any]]) -> dict[str, Any]:
    count = len(cases)
    latencies = [float(case["generation_seconds"]) for case in cases]
    metric_names = [
        "llm_decision_vs_expected",
        "llm_decision_vs_formulas",
        "overall_llm_vs_formulas_score",
        "isolated_numeric_agreement",
        "all_three_numeric_fields_correct",
        "required_sections",
        "rag_source_presence",
        "markdown_format",
    ]
    field_counts = {
        field: sum(
            1
            for case in cases
            if case.get("numeric_comparisons", {}).get(field, {}).get("agreement")
        )
        for field, _, _, _ in NUMERIC_FIELDS
    }
    combined_correct = sum(field_counts.values())
    combined_total = count * len(NUMERIC_FIELDS)
    error_by_field = {
        field: [
            float(error)
            for case in cases
            if isinstance(
                error := case.get("numeric_comparisons", {}).get(field, {}).get("absolute_error"),
                (int, float),
            )
        ]
        for field, _, _, _ in NUMERIC_FIELDS
    }
    maximum_slices: dict[str, dict[str, Any]] = {}
    for slice_name, predicate in (
        ("product_cap", lambda value: abs(value - MAX_AMOUNT_RON) <= 1e-9),
        ("zero_capacity", lambda value: abs(value) <= 1e-9),
        (
            "interior_annuity",
            lambda value: abs(value) > 1e-9 and abs(value - MAX_AMOUNT_RON) > 1e-9,
        ),
    ):
        slice_cases = [
            case
            for case in cases
            if predicate(
                float(case["numeric_comparisons"]["maximum_amount_by_dti"]["reference"])
            )
        ]
        slice_errors = [
            float(comparison["absolute_error"])
            for case in slice_cases
            if isinstance(
                comparison := case["numeric_comparisons"]["maximum_amount_by_dti"],
                dict,
            )
            and isinstance(comparison.get("absolute_error"), (int, float))
        ]
        slice_correct = sum(
            1
            for case in slice_cases
            if case["numeric_comparisons"]["maximum_amount_by_dti"]["agreement"]
        )
        maximum_slices[slice_name] = {
            "case_count": len(slice_cases),
            "correct_count": slice_correct,
            "agreement": slice_correct / len(slice_cases) if slice_cases else None,
            **error_statistics(slice_errors),
        }
    return {
        "completed_case_count": count,
        "pipeline_success_count": sum(1 for case in cases if case.get("pipeline_status") == "ok"),
        "pipeline_failure_count": sum(
            1 for case in cases if case.get("pipeline_status") != "ok"
        ),
        "average_case_score": (
            statistics.fmean(float(case["case_score"]) for case in cases) if cases else None
        ),
        "metric_averages": {
            name: statistics.fmean(metric_score(case, name) for case in cases) if cases else None
            for name in metric_names
        },
        "numeric_correct_counts": field_counts,
        "numeric_agreement_by_field": {
            field: field_counts[field] / count if count else None for field in field_counts
        },
        "combined_numeric_correct": combined_correct,
        "combined_numeric_comparisons": combined_total,
        "combined_numeric_agreement": combined_correct / combined_total if combined_total else None,
        "all_three_correct_count": sum(
            1 for case in cases if case.get("all_three_numeric_fields_correct")
        ),
        "all_three_correct_rate": (
            sum(1 for case in cases if case.get("all_three_numeric_fields_correct")) / count
            if count
            else None
        ),
        "absolute_error_statistics": {
            field: error_statistics(errors) for field, errors in error_by_field.items()
        },
        "maximum_amount_slices": maximum_slices,
        "latency_seconds": {
            "total": sum(latencies),
            "mean": statistics.fmean(latencies) if latencies else None,
            "median": statistics.median(latencies) if latencies else None,
            "minimum": min(latencies) if latencies else None,
            "maximum": max(latencies) if latencies else None,
        },
    }


def percent(value: object) -> str:
    return "n/a" if not isinstance(value, (int, float)) else f"{float(value):.2%}"


def yes_no(value: object) -> str:
    return "YES" if value else "NO"


def markdown_report(document: dict[str, Any]) -> str:
    config = document["config"]
    summary = document["summary"]
    lines = [
        "# Full staged-pipeline client benchmark",
        "",
        f"Status: `{document['status']}`  ",
        f"Updated: `{document['updated_at']}`  ",
        f"Cases completed: `{summary['completed_case_count']}/{len(config['case_ids'])}`  ",
        f"Route: `{config['rag_model']}` → `{config['calculation_model']}` → "
        f"`{config['synthesis_model']}`  ",
        f"Calculation reasoning: `{config['calculation_reasoning']}`; seed: `{config['seed']}`  ",
        "Maximum recommended amount: **included**, with RON 1.00 tolerance.",
        "",
        "## Aggregate results",
        "",
        f"- Average eight-metric case score: {percent(summary['average_case_score'])}",
        f"- Decision agreement vs expected: "
        f"{percent(summary['metric_averages']['llm_decision_vs_expected'])}",
        f"- Stressed-payment agreement: "
        f"{percent(summary['numeric_agreement_by_field']['stressed_monthly_payment'])}",
        f"- DTI agreement: {percent(summary['numeric_agreement_by_field']['dti_pct'])}",
        f"- Maximum-amount agreement: "
        f"{percent(summary['numeric_agreement_by_field']['maximum_amount_by_dti'])}",
        f"- Combined numeric agreement: {percent(summary['combined_numeric_agreement'])}",
        f"- All-three-correct case rate: {percent(summary['all_three_correct_rate'])}",
        f"- Pipeline success: `{summary['pipeline_success_count']}/"
        f"{summary['completed_case_count']}`",
        "",
        "Maximum-amount agreement by reference-value class:",
        "",
        "| Class | Cases | Correct | Agreement | Mean absolute error |",
        "|---|---:|---:|---:|---:|",
    ]
    for slice_name, values in summary["maximum_amount_slices"].items():
        mean_error = values["mean_absolute_error"]
        mean_error_text = "n/a" if mean_error is None else f"{mean_error:,.2f} RON"
        lines.append(
            f"| {slice_name} | {values['case_count']} | {values['correct_count']} | "
            f"{percent(values['agreement'])} | {mean_error_text} |"
        )
    lines.extend(
        [
        "",
        "## Per-case results",
        "",
        "| Case | Status | Expected | LLM | Payment | DTI | Maximum | All 3 | Case score | Seconds |",
        "|---|---|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for case in document.get("cases", []):
        numeric = case["numeric_comparisons"]
        lines.append(
            f"| {case['case_id']} | {case['pipeline_status']} | {case['expected_decision']} | "
            f"{case['llm_decision'] or 'not found'} | "
            f"{yes_no(numeric['stressed_monthly_payment']['agreement'])} | "
            f"{yes_no(numeric['dti_pct']['agreement'])} | "
            f"{yes_no(numeric['maximum_amount_by_dti']['agreement'])} | "
            f"{yes_no(case['all_three_numeric_fields_correct'])} | "
            f"{percent(case['case_score'])} | {case['generation_seconds']:.2f} |"
        )
    return "\n".join(lines) + "\n"


def checkpoint(output: Path, document: dict[str, Any]) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    document["updated_at"] = utc_now()
    document["summary"] = summarize(document.get("cases", []))
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(document, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    temporary.replace(output)
    markdown_path = output.with_suffix(".md")
    markdown_path.write_text(markdown_report(document), encoding="utf-8")


def initialize(args: argparse.Namespace, config: dict[str, Any]) -> dict[str, Any]:
    output = args.output.resolve()
    if args.resume:
        if not output.exists():
            raise SystemExit(f"Cannot resume missing checkpoint: {output}")
        document = json.loads(output.read_text(encoding="utf-8"))
        if document.get("config") != config:
            raise SystemExit("Checkpoint configuration differs from this invocation.")
        document["status"] = "in_progress"
        return document
    if output.exists() and not args.overwrite:
        raise SystemExit(f"Output already exists: {output}; use --resume or --overwrite.")
    return {
        "schema_version": 1,
        "benchmark": "full_staged_pipeline_client_cases",
        "status": "in_progress",
        "created_at": utc_now(),
        "updated_at": utc_now(),
        "config": config,
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "command": [sys.executable, *sys.argv],
        },
        "cases": [],
        "summary": summarize([]),
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    args.cases_path = args.cases_path.resolve()
    args.output = args.output.resolve()
    configure_environment(args)

    loaded = load_evaluation_cases(args.cases_path)
    cases = loaded.get("client_cases", [])
    if not isinstance(cases, list) or not cases:
        raise SystemExit("No client cases found.")
    if args.max_cases is not None:
        cases = cases[: args.max_cases]
    case_ids = [str(case.get("id", "")) for case in cases]
    if any(not case_id for case_id in case_ids) or len(case_ids) != len(set(case_ids)):
        raise SystemExit("Every client case must have a unique non-empty ID.")

    config = config_for(args, case_ids)
    document = initialize(args, config)
    for saved_case in document.get("cases", []):
        validate_case_result(saved_case)
    checkpoint(args.output, document)
    completed_ids = {
        str(case.get("case_id"))
        for case in document.get("cases", [])
        if isinstance(case, dict) and case.get("case_id")
    }
    index = build_default_index(
        args.retriever,
        bm25_k1=args.bm25_k1,
        bm25_b=args.bm25_b,
    )

    try:
        for position, case in enumerate(cases, start=1):
            case_id = str(case["id"])
            if case_id in completed_ids:
                continue
            print(f"[{position}/{len(cases)}] START {case_id}", flush=True)
            try:
                result = run_case(case, index)
            except InfrastructureFailure as exc:
                document["status"] = "infrastructure_error"
                document["last_error"] = str(exc)
                checkpoint(args.output, document)
                print(
                    f"Infrastructure failure; resume with --resume. Error: {exc}",
                    file=sys.stderr,
                    flush=True,
                )
                return 2
            validate_case_result(result)
            document["cases"].append(result)
            checkpoint(args.output, document)
            print(
                f"[{position}/{len(cases)}] DONE {case_id}: "
                f"score={result['case_score']:.2%}, "
                f"all_three={result['all_three_numeric_fields_correct']}, "
                f"seconds={result['generation_seconds']:.2f}",
                flush=True,
            )
    except KeyboardInterrupt:
        document["status"] = "interrupted"
        checkpoint(args.output, document)
        print(f"Interrupted; resume with --resume. Checkpoint: {args.output}", file=sys.stderr)
        return 130

    document["status"] = "complete"
    checkpoint(args.output, document)
    print(json.dumps(document["summary"], ensure_ascii=False, indent=2), flush=True)
    print(f"JSON: {args.output}", flush=True)
    print(f"Markdown: {args.output.with_suffix('.md')}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
