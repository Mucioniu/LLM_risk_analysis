from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .credit_engine import ClientProfile, CreditEvaluation, evaluate_client
from .service import answer_policy_question, build_llm_credit_analysis
from .rag import RagIndex


DEFAULT_EVALUATION_CASES = Path("examples/evaluation_cases.json")
REQUIRED_ANALYSIS_SECTIONS = [
    "Decizie",
    "Calcul financiar",
    "Motive de respingere",
    "Motive de analiza manuala",
    "Observatii",
    "Surse RAG folosite",
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
        "acoperire_cuvinte_cheie",
        ratio(len(found), len(expected_keywords)),
        f"{len(found)}/{len(expected_keywords)} cuvinte-cheie gasite: {', '.join(found) or 'niciunul'}",
    )


def format_score(text: str) -> MetricResult:
    checks = {
        "are_linii_noi": "\n" in text.strip(),
        "are_heading_markdown": bool(re.search(r"^#{2,3}\s+", text, flags=re.MULTILINE)),
        "nu_are_asteriscuri_decorative": "***" not in text,
        "nu_are_think": "<think>" not in text.lower(),
    }
    passed = sum(1 for value in checks.values() if value)
    failed = [name for name, value in checks.items() if not value]
    return MetricResult(
        "format_markdown",
        ratio(passed, len(checks)),
        "OK" if not failed else "Probleme: " + ", ".join(failed),
    )


def source_presence(text: str) -> MetricResult:
    has_sources = "Fragmente relevante" in text or "Surse RAG folosite" in text
    has_numbered_source = bool(re.search(r"\[\d+\]|\n\d+\.", text))
    score = ratio(int(has_sources) + int(has_numbered_source), 2)
    return MetricResult(
        "prezenta_surse_rag",
        score,
        "Sursele sunt afisate" if score == 1.0 else "Sursele lipsesc sau nu sunt numerotate clar",
    )


def retrieval_hit(index: RagIndex, query: str, expected_sources: list[str]) -> MetricResult:
    if not expected_sources:
        return MetricResult("retrieval_hit_at_5", 1.0, "Caz fara sursa asteptata explicita")

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
        f"Surse gasite: {', '.join(hits) or 'niciuna'}",
    )


def missing_answer_score(text: str, expect_missing: bool) -> MetricResult:
    missing_markers = [
        "lipseste",
        "nu este mentionat",
        "nu am gasit",
        "nu apare",
        "nu este specificat",
    ]
    says_missing = contains_any(text, missing_markers)
    if not expect_missing:
        return MetricResult("raspuns_lipsa_info", 1.0, "Nu este caz de informatie lipsa")

    return MetricResult(
        "raspuns_lipsa_info",
        1.0 if says_missing else 0.0,
        "Modelul semnaleaza lipsa informatiei" if says_missing else "Modelul nu semnaleaza lipsa informatiei",
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
    return CaseResult(case["id"], "intrebari_manual", latency, metrics)


def expected_numeric_values(profile: ClientProfile, evaluation: CreditEvaluation) -> list[str]:
    return [
        f"{profile.monthly_income:,.2f}",
        f"{evaluation.weighted_income:,.2f}",
        f"{evaluation.max_monthly_payment:,.2f}",
        f"{evaluation.stressed_monthly_payment:,.2f}",
        f"{evaluation.gmi * 100:.2f}%",
        f"{evaluation.max_credit_amount:,.2f}",
    ]


def required_sections_score(text: str) -> MetricResult:
    found = [section for section in REQUIRED_ANALYSIS_SECTIONS if section.lower() in text.lower()]
    return MetricResult(
        "sectiuni_obligatorii",
        ratio(len(found), len(REQUIRED_ANALYSIS_SECTIONS)),
        f"{len(found)}/{len(REQUIRED_ANALYSIS_SECTIONS)} sectiuni gasite",
    )


def decision_consistency(text: str, expected_decision: str) -> MetricResult:
    return MetricResult(
        "consistenta_decizie",
        1.0 if expected_decision.lower() in text.lower() else 0.0,
        f"Decizie asteptata: {expected_decision}",
    )


def numeric_consistency(text: str, values: list[str]) -> MetricResult:
    normalized = text.replace(" ", "")
    found = [value for value in values if value.replace(" ", "") in normalized]
    return MetricResult(
        "consistenta_valori_numerice",
        ratio(len(found), len(values)),
        f"{len(found)}/{len(values)} valori gasite",
    )


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
            "decizie_llm_vs_asteptat",
            1.0 if analysis.extracted.decision == expected_decision else 0.0,
            f"LLM: {analysis.extracted.decision or 'negasit'} / asteptat: {expected_decision}",
        ),
        MetricResult(
            "decizie_llm_vs_formule",
            analysis.metric_scores.get("Decizie", 0.0),
            f"LLM: {analysis.extracted.decision or 'negasit'} / formule: {deterministic.decision.value}",
        ),
        MetricResult(
            "scor_total_llm_vs_formule",
            analysis.metric_scores.get("scor_total_llm_vs_formule", 0.0),
            "Comparatie pe decizie si valori financiare extrase din raspunsul LLM.",
        ),
        required_sections_score(answer),
        source_presence(answer),
        format_score(answer),
    ]
    return CaseResult(case["id"], "analiza_client", latency, metrics)


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
        return "Nu exista cazuri de evaluare."

    overall = sum(result.score for result in results) / len(results)
    total_latency = sum(result.latency_seconds for result in results)
    by_type: dict[str, list[CaseResult]] = {}
    for result in results:
        by_type.setdefault(result.case_type, []).append(result)

    lines = [
        "## Raport metrici",
        "",
        f"Scor mediu total: {overall:.2%}",
        f"Cazuri evaluate: {len(results)}",
        f"Timp total: {total_latency:.2f}s",
        "",
        "### Scor pe sectiuni",
        "",
        "| Sectiune | Cazuri | Scor mediu | Latenta medie |",
        "|---|---:|---:|---:|",
    ]
    for case_type, case_results in by_type.items():
        section_score = sum(result.score for result in case_results) / len(case_results)
        section_latency = sum(result.latency_seconds for result in case_results) / len(case_results)
        lines.append(
            f"| {case_type} | {len(case_results)} | {section_score:.2%} | {section_latency:.2f}s |"
        )

    lines.extend(["", "### Detaliu cazuri", ""])
    for result in results:
        lines.extend(
            [
                f"#### {result.case_id} ({result.case_type})",
                "",
                f"Scor caz: {result.score:.2%}",
                f"Latenta: {result.latency_seconds:.2f}s",
                "",
                "| Metrica | Scor | Detalii |",
                "|---|---:|---|",
            ]
        )
        for metric in result.metrics:
            details = metric.details.replace("|", "/")
            lines.append(f"| {metric.name} | {metric.score:.2%} | {details} |")
        lines.append("")

    return "\n".join(lines)
