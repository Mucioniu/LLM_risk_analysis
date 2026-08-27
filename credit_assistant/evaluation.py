from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .credit_engine import ClientProfile, CreditEvaluation, evaluate_client
from .service import (
    LlmExtractedDecision,
    answer_policy_question,
    build_llm_credit_analysis,
)
from .rag import RagIndex


DEFAULT_EVALUATION_CASES = Path("examples/evaluation_cases.json")
REQUIRED_ANALYSIS_SECTIONS = [
    "Decision",
    "Financial calculation",
    "Rejection reasons",
    "Manual review reasons",
    "Notes",
    "RAG sources used",
]


@dataclass(frozen=True)
class MetricResult:
    name: str
    score: float
    details: str


@dataclass(frozen=True)
class CaseResult:
    case_id: str
    case_type: str
    latency_seconds: float
    metrics: list[MetricResult]

    @property
    def score(self) -> float:
        if not self.metrics:
            return 0.0
        return sum(metric.score for metric in self.metrics) / len(self.metrics)


def load_evaluation_cases(path: Path = DEFAULT_EVALUATION_CASES) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def ratio(found: int, total: int) -> float:
    if total <= 0:
        return 1.0
    return found / total


def contains_any(text: str, values: list[str]) -> bool:
    normalized = text.lower()
    return any(value.lower() in normalized for value in values)


def keyword_coverage(text: str, expected_keywords: list[str]) -> MetricResult:
    normalized = text.lower()
    found = [keyword for keyword in expected_keywords if keyword.lower() in normalized]
    return MetricResult(
        "keyword_coverage",
        ratio(len(found), len(expected_keywords)),
        f"{len(found)}/{len(expected_keywords)} keywords found: {', '.join(found) or 'none'}",
    )


def format_score(text: str) -> MetricResult:
    checks = {
        "has_newlines": "\n" in text.strip(),
        "has_markdown_heading": bool(re.search(r"^#{2,3}\s+", text, flags=re.MULTILINE)),
        "has_no_decorative_asterisks": "***" not in text,
        "has_no_think_tag": "<think>" not in text.lower(),
    }
    passed = sum(1 for value in checks.values() if value)
    failed = [name for name, value in checks.items() if not value]
    return MetricResult(
        "markdown_format",
        ratio(passed, len(checks)),
        "OK" if not failed else "Issues: " + ", ".join(failed),
    )


def source_presence(text: str) -> MetricResult:
    has_sources = "Relevant excerpts" in text or "RAG sources used" in text
    has_numbered_source = bool(re.search(r"\[\d+\]|\n\d+\.", text))
    score = ratio(int(has_sources) + int(has_numbered_source), 2)
    return MetricResult(
        "rag_source_presence",
        score,
        "Sources are displayed" if score == 1.0 else "Sources are missing or not clearly numbered",
    )


def retrieval_hit(index: RagIndex, query: str, expected_sources: list[str]) -> MetricResult:
    if not expected_sources:
        return MetricResult("retrieval_hit_at_5", 1.0, "Case with no explicit expected source")

    retrieved = index.search(query, top_k=5)
    retrieved_names = [result.chunk.source for result in retrieved]
    hits = [
        expected
        for expected in expected_sources
        if any(expected.lower() in source.lower() for source in retrieved_names)
    ]
    return MetricResult(
        "retrieval_hit_at_5",
        ratio(len(hits), len(expected_sources)),
        f"Sources found: {', '.join(hits) or 'none'}",
    )


def missing_answer_score(text: str, expect_missing: bool) -> MetricResult:
    missing_markers = [
        "is missing",
        "is not mentioned",
        "not found",
        "does not appear",
        "is not specified",
    ]
    says_missing = contains_any(text, missing_markers)
    if not expect_missing:
        return MetricResult("missing_information_response", 1.0, "This is not a missing-information case")

    return MetricResult(
        "missing_information_response",
        1.0 if says_missing else 0.0,
        "The model flags the missing information" if says_missing else "The model does not flag the missing information",
    )


def evaluate_policy_question_case(case: dict[str, Any], index: RagIndex) -> CaseResult:
    started = time.perf_counter()
    answer = answer_policy_question(case["question"], index, use_llm=True)
    latency = time.perf_counter() - started

    metrics = [
        retrieval_hit(index, case["question"], case.get("expected_source_contains", [])),
        keyword_coverage(answer, case.get("expected_keywords", [])),
        missing_answer_score(answer, bool(case.get("expect_missing", False))),
        source_presence(answer),
        format_score(answer),
    ]
    return CaseResult(case["id"], "policy_questions", latency, metrics)


def expected_numeric_values(profile: ClientProfile, evaluation: CreditEvaluation) -> list[str]:
    return [
        f"{profile.monthly_income:,.2f}",
        f"{evaluation.weighted_income:,.2f}",
        f"{evaluation.max_monthly_payment:,.2f}",
        f"{evaluation.stressed_monthly_payment:,.2f}",
        f"{evaluation.dti * 100:.2f}%",
        f"{evaluation.maximum_amount_by_dti:,.2f}",
    ]


def required_sections_score(text: str) -> MetricResult:
    found = [section for section in REQUIRED_ANALYSIS_SECTIONS if section.lower() in text.lower()]
    return MetricResult(
        "required_sections",
        ratio(len(found), len(REQUIRED_ANALYSIS_SECTIONS)),
        f"{len(found)}/{len(REQUIRED_ANALYSIS_SECTIONS)} sections found",
    )


def decision_consistency(text: str, expected_decision: str) -> MetricResult:
    return MetricResult(
        "decision_consistency",
        1.0 if expected_decision.lower() in text.lower() else 0.0,
        f"Expected decision: {expected_decision}",
    )


def numeric_consistency(text: str, values: list[str]) -> MetricResult:
    normalized = text.replace(" ", "")
    found = [value for value in values if value.replace(" ", "") in normalized]
    return MetricResult(
        "numeric_consistency",
        ratio(len(found), len(values)),
        f"{len(found)}/{len(values)} values found",
    )


def numeric_agreement_details(
    extracted: LlmExtractedDecision,
    deterministic: CreditEvaluation,
) -> str:
    """Describe each locked numeric comparison without changing metric weighting."""
    targets = (
        (
            "Stressed payment",
            extracted.stressed_monthly_payment,
            deterministic.stressed_monthly_payment,
            1.0,
            "RON",
            "RON",
            2,
        ),
        (
            "DTI",
            extracted.dti_pct,
            deterministic.dti * 100,
            0.05,
            "%",
            "pp",
            4,
        ),
        (
            "Maximum amount",
            extracted.maximum_amount_by_dti,
            deterministic.maximum_amount_by_dti,
            1.0,
            "RON",
            "RON",
            2,
        ),
    )
    details: list[str] = []
    for label, actual, reference, tolerance, value_unit, error_unit, decimals in targets:
        if actual is None:
            details.append(
                f"{label}: NO (LLM value not found; "
                f"reference {reference:,.{decimals}f} {value_unit})"
            )
            continue
        absolute_error = abs(actual - reference)
        agrees = absolute_error <= tolerance
        details.append(
            f"{label}: {'YES' if agrees else 'NO'} "
            f"(LLM {actual:,.{decimals}f} {value_unit}; "
            f"reference {reference:,.{decimals}f} {value_unit}; "
            f"absolute error {absolute_error:,.{decimals}f} {error_unit}; "
            f"tolerance {tolerance:,.{decimals}f} {error_unit})"
        )
    return "; ".join(details)


def evaluate_client_case(case: dict[str, Any], index: RagIndex) -> CaseResult:
    profile = ClientProfile(**case["profile"])
    deterministic = evaluate_client(profile)
    expected_decision = case.get("expected_decision", deterministic.decision.value)

    started = time.perf_counter()
    analysis = build_llm_credit_analysis(profile, index)
    answer = analysis.answer_markdown
    latency = time.perf_counter() - started

    metrics = [
        MetricResult(
            "llm_decision_vs_expected",
            1.0 if analysis.extracted.decision == expected_decision else 0.0,
            f"LLM: {analysis.extracted.decision or 'not found'} / expected: {expected_decision}",
        ),
        MetricResult(
            "llm_decision_vs_formulas",
            analysis.metric_scores.get("Decision", 0.0),
            f"LLM: {analysis.extracted.decision or 'not found'} / formulas: {deterministic.decision.value}",
        ),
        MetricResult(
            "overall_llm_vs_formulas_score",
            analysis.metric_scores.get("overall_llm_vs_formulas_score", 0.0),
            "Comparison of the decision and the three locked LLM calculation fields.",
        ),
        MetricResult(
            "isolated_numeric_agreement",
            analysis.metric_scores.get("isolated_numeric_agreement", 0.0),
            numeric_agreement_details(analysis.extracted, deterministic),
        ),
        MetricResult(
            "all_three_numeric_fields_correct",
            analysis.metric_scores.get("all_three_numeric_fields_correct", 0.0),
            "One only when all three isolated numerical fields agree in this case.",
        ),
        required_sections_score(answer),
        source_presence(answer),
        format_score(answer),
    ]
    return CaseResult(case["id"], "client_analysis", latency, metrics)


def run_evaluation_suite(
    index: RagIndex,
    *,
    max_policy_cases: int | None = None,
    max_client_cases: int | None = None,
) -> list[CaseResult]:
    cases = load_evaluation_cases()
    policy_cases = cases.get("policy_questions", [])
    client_cases = cases.get("client_cases", [])
    if max_policy_cases is not None:
        policy_cases = policy_cases[:max_policy_cases]
    if max_client_cases is not None:
        client_cases = client_cases[:max_client_cases]

    results: list[CaseResult] = []
    for case in policy_cases:
        results.append(evaluate_policy_question_case(case, index))
    for case in client_cases:
        results.append(evaluate_client_case(case, index))
    return results


def summarize_evaluation_markdown(results: list[CaseResult]) -> str:
    if not results:
        return "There are no evaluation cases."

    overall = sum(result.score for result in results) / len(results)
    total_latency = sum(result.latency_seconds for result in results)
    by_type: dict[str, list[CaseResult]] = {}
    for result in results:
        by_type.setdefault(result.case_type, []).append(result)

    lines = [
        "## Metrics report",
        "",
        f"Overall average score: {overall:.2%}",
        f"Cases evaluated: {len(results)}",
        f"Total time: {total_latency:.2f}s",
        "",
        "### Score by section",
        "",
        "| Section | Cases | Average score | Average latency |",
        "|---|---:|---:|---:|",
    ]
    for case_type, case_results in by_type.items():
        section_score = sum(result.score for result in case_results) / len(case_results)
        section_latency = sum(result.latency_seconds for result in case_results) / len(case_results)
        lines.append(
            f"| {case_type} | {len(case_results)} | {section_score:.2%} | {section_latency:.2f}s |"
        )

    lines.extend(["", "### Case details", ""])
    for result in results:
        lines.extend(
            [
                f"#### {result.case_id} ({result.case_type})",
                "",
                f"Case score: {result.score:.2%}",
                f"Latency: {result.latency_seconds:.2f}s",
                "",
                "| Metric | Score | Details |",
                "|---|---:|---|",
            ]
        )
        for metric in result.metrics:
            details = metric.details.replace("|", "/")
            lines.append(f"| {metric.name} | {metric.score:.2%} | {details} |")
        lines.append("")

    return "\n".join(lines)
