from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from credit_assistant.credit_engine import ClientProfile, evaluate_client
from credit_assistant.evaluation import load_evaluation_cases
from credit_assistant.service import build_default_index, build_llm_credit_analysis


def configure_ollama(model: str, *, think: bool = False) -> None:
    os.environ.setdefault("OPENAI_BASE_URL", "http://localhost:11434/v1")
    os.environ.setdefault("OPENAI_API_KEY", "ollama")
    os.environ["OPENAI_MODEL"] = model
    os.environ.setdefault("OPENAI_TIMEOUT_SECONDS", "300")
    os.environ.setdefault("OPENAI_MAX_TOKENS", "3000")
    if think:
        os.environ["OLLAMA_THINK"] = "true"
        os.environ.setdefault("OLLAMA_NUM_CTX", "8192")
        os.environ.setdefault("OLLAMA_NUM_PREDICT", "6000")
    else:
        os.environ.pop("OLLAMA_THINK", None)


def run_model(model: str, max_cases: int | None = None, *, think: bool = False) -> dict[str, Any]:
    configure_ollama(model, think=think)
    cases = load_evaluation_cases().get("client_cases", [])
    if max_cases is not None:
        cases = cases[:max_cases]

    index = build_default_index()
    results: list[dict[str, Any]] = []
    started_model = time.perf_counter()

    for case in cases:
        profile = ClientProfile(**case["profile"])
        deterministic = evaluate_client(profile)
        started_case = time.perf_counter()
        analysis = build_llm_credit_analysis(profile, index)
        latency = time.perf_counter() - started_case
        extracted = asdict(analysis.extracted)

        results.append(
            {
                "case_id": case["id"],
                "expected_decision": case.get("expected_decision", deterministic.decision.value),
                "deterministic_decision": deterministic.decision.value,
                "llm_decision": analysis.extracted.decision,
                "score_total": analysis.metric_scores.get(
                    "overall_llm_vs_formulas_score",
                    0.0,
                ),
                "decision_score": analysis.metric_scores.get("Decision", 0.0),
                "metrics": analysis.metric_scores,
                "latency_seconds": latency,
                "extracted": extracted,
                "deterministic": {
                    "weighted_income": deterministic.weighted_income,
                    "max_monthly_payment": deterministic.max_monthly_payment,
                    "available_payment_capacity": deterministic.available_payment_capacity,
                    "stressed_monthly_payment": deterministic.stressed_monthly_payment,
                    "dti_pct": deterministic.dti * 100,
                    "maturity_age": deterministic.maturity_age,
                    "maximum_amount_by_dti": deterministic.maximum_amount_by_dti,
                },
            }
        )

    case_count = len(results)
    average_score = (
        sum(result["score_total"] for result in results) / case_count if case_count else 0.0
    )
    decision_accuracy = (
        sum(result["decision_score"] for result in results) / case_count if case_count else 0.0
    )
    return {
        "model": model,
        "think": think,
        "case_count": case_count,
        "average_score_total": average_score,
        "decision_accuracy": decision_accuracy,
        "latency_seconds": time.perf_counter() - started_model,
        "cases": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark local Ollama models on client cases.")
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument("--think", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    summary = [run_model(model, args.max_cases, think=args.think) for model in args.models]
    text = json.dumps(summary, ensure_ascii=False, indent=2)
    if args.output:
        args.output.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
