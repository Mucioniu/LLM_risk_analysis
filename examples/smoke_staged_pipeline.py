from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from credit_assistant.credit_engine import ClientProfile, evaluate_client  # noqa: E402
from credit_assistant.service import (  # noqa: E402
    LlmStageError,
    build_default_index,
    run_staged_llm_generation,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one reproducible smoke case through the three-stage LLM-only pipeline."
    )
    parser.add_argument("--case-id", default="c_eur_variable_stress_rejected")
    parser.add_argument(
        "--cases-path",
        type=Path,
        default=PROJECT_ROOT / "examples" / "evaluation_cases.json",
    )
    parser.add_argument("--rag-model", default="mistral-small3.2:latest")
    parser.add_argument("--calculation-model", default="qwen3:14b")
    parser.add_argument("--synthesis-model", default="mistral-small3.2:latest")
    parser.add_argument(
        "--calculation-reasoning",
        choices=("auto", "off", "on", "low", "medium", "high"),
        default="on",
    )
    parser.add_argument("--num-ctx", type=int, default=8192)
    parser.add_argument("--num-predict", type=int, default=6000)
    parser.add_argument("--timeout-seconds", type=float, default=900)
    return parser.parse_args()


def load_profile(path: Path, case_id: str) -> ClientProfile:
    document = json.loads(path.read_text(encoding="utf-8"))
    for case in document.get("client_cases", []):
        if case.get("id") == case_id:
            return ClientProfile(**case["profile"])
    raise SystemExit(f"Unknown client case {case_id!r} in {path}.")


def configure(args: argparse.Namespace) -> None:
    os.environ.update(
        {
            "OPENAI_API_KEY": "ollama",
            "OPENAI_BASE_URL": "http://127.0.0.1:11434/v1",
            "OLLAMA_NATIVE_CHAT": "true",
            "OPENAI_RAG_MODEL": args.rag_model,
            "OPENAI_CALCULATION_MODEL": args.calculation_model,
            "OPENAI_SYNTHESIS_MODEL": args.synthesis_model,
            "OPENAI_TIMEOUT_SECONDS": str(args.timeout_seconds),
            "OLLAMA_NUM_CTX": str(args.num_ctx),
            "OLLAMA_CALCULATION_NUM_PREDICT": str(args.num_predict),
        }
    )
    if args.calculation_reasoning == "auto":
        os.environ.pop("OLLAMA_CALCULATION_THINK", None)
    elif args.calculation_reasoning == "off":
        os.environ["OLLAMA_CALCULATION_THINK"] = "false"
    elif args.calculation_reasoning == "on":
        os.environ["OLLAMA_CALCULATION_THINK"] = "true"
    else:
        os.environ["OLLAMA_CALCULATION_THINK"] = args.calculation_reasoning


def main() -> int:
    args = parse_args()
    configure(args)
    profile = load_profile(args.cases_path, args.case_id)
    index = build_default_index()

    started = time.perf_counter()
    try:
        generation = run_staged_llm_generation(profile, index)
    except LlmStageError as exc:
        print(
            json.dumps(
                {
                    "case_id": args.case_id,
                    "status": "stage_error",
                    "stage": exc.stage,
                    "error": str(exc),
                    "raw_response": exc.raw_response,
                    "elapsed_seconds": time.perf_counter() - started,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 1

    generation_finished = time.perf_counter()
    reference = evaluate_client(profile)
    print(
        json.dumps(
            {
                "case_id": args.case_id,
                "status": "ok",
                "models": {
                    "rag": args.rag_model,
                    "calculation": args.calculation_model,
                    "synthesis": args.synthesis_model,
                    "calculation_reasoning": args.calculation_reasoning,
                },
                "llm_generation_seconds": generation_finished - started,
                "profile": asdict(profile),
                "policy": asdict(generation.policy),
                "locked_calculation": asdict(generation.calculation),
                "calculation_audit": json.loads(generation.raw_calculation),
                "synthesis": asdict(generation.synthesis),
                "post_hoc_reference": {
                    "stressed_monthly_payment": reference.stressed_monthly_payment,
                    "dti_pct": reference.dti * 100,
                    "maximum_amount_by_dti": reference.maximum_amount_by_dti,
                    "decision": reference.decision.value,
                },
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
