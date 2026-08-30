from __future__ import annotations

import argparse
import json
import math
import platform
import statistics
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from credit_assistant.credit_engine import (  # noqa: E402
    DTI_LIMIT,
    INCOME_WEIGHTS,
    MAX_AMOUNT_RON,
    ClientProfile,
    evaluate_client,
)
from credit_assistant.evaluation import load_evaluation_cases  # noqa: E402


DEFAULT_CASES_PATH = PROJECT_ROOT / "examples" / "evaluation_cases.json"
TARGET_FIELDS = (
    "stressed_monthly_payment",
    "dti_pct",
    "maximum_amount_by_dti",
)
TOLERANCES = {
    "stressed_monthly_payment": 1.0,
    "dti_pct": 0.05,
    "maximum_amount_by_dti": 1.0,
}
JSON_FORMAT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "stressed_monthly_payment": {"type": "number"},
        "dti_pct": {"type": "number"},
        "maximum_amount_by_dti": {"type": "number"},
    },
    "required": list(TARGET_FIELDS),
    "additionalProperties": False,
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a controlled, numeric-only Ollama benchmark over every client profile. "
            "Use separate output files for reasoning=off and reasoning=high."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog=(
            "Example: python examples/benchmark_numeric_reasoning.py "
            "--model mistral-small4-iq1:none --reasoning template-none "
            "--output examples/benchmark_numeric_small4_none.json"
        ),
    )
    parser.add_argument("--model", required=True, help="Exact Ollama model tag to benchmark.")
    parser.add_argument(
        "--reasoning",
        required=True,
        choices=(
            "off",
            "on",
            "low",
            "medium",
            "high",
            "max",
            "template-none",
            "template-high",
        ),
        help=(
            "Value sent in the native Ollama `think` field. `off` sends JSON false "
            "explicitly; `on` sends true; named levels are sent as strings. "
            "The template modes omit `think` because the imported model alias embeds "
            "Mistral's MODEL_SETTINGS reasoning_effort directly."
        ),
    )
    parser.add_argument("--output", type=Path, required=True, help="Checkpoint JSON path.")
    parser.add_argument(
        "--cases-path",
        type=Path,
        default=DEFAULT_CASES_PATH,
        help="Evaluation case JSON containing client_cases.",
    )
    parser.add_argument(
        "--ollama-url",
        default="http://127.0.0.1:11434",
        help="Base URL for the local Ollama server.",
    )
    parser.add_argument("--num-ctx", type=int, default=16_384)
    parser.add_argument("--num-predict", type=int, default=6_000)
    parser.add_argument(
        "--max-cases",
        type=int,
        default=None,
        help="Run only the first N cases for a capability or performance smoke test.",
    )
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--timeout-seconds", type=float, default=1_800.0)
    output_mode = parser.add_mutually_exclusive_group()
    output_mode.add_argument(
        "--resume",
        action="store_true",
        help="Resume a compatible checkpoint and skip case IDs already present.",
    )
    output_mode.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing output file with a new run.",
    )
    args = parser.parse_args(argv)

    if args.num_ctx <= 0:
        parser.error("--num-ctx must be positive")
    if args.num_predict <= 0:
        parser.error("--num-predict must be positive")
    if args.max_cases is not None and args.max_cases <= 0:
        parser.error("--max-cases must be positive")
    if args.timeout_seconds <= 0:
        parser.error("--timeout-seconds must be positive")
    if args.temperature < 0:
        parser.error("--temperature cannot be negative")
    return args


def reasoning_payload_value(reasoning: str) -> bool | str | None:
    if reasoning.startswith("template-"):
        return None
    if reasoning == "off":
        return False
    if reasoning == "on":
        return True
    return reasoning


def api_chat_url(base_url: str) -> str:
    normalized = base_url.rstrip("/")
    if normalized.endswith("/api"):
        return f"{normalized}/chat"
    return f"{normalized}/api/chat"


def operating_rules_prompt() -> str:
    weights = "\n".join(
        f"- {income_type}: {weight:.10g}"
        for income_type, weight in INCOME_WEIGHTS.items()
    )
    return f"""Calculate exactly three numeric fields from the client profile.

Use these rules exactly. Do not infer alternative banking rules.

Income weights (multipliers, not percentages):
{weights}
- Any income type absent from the mapping has weight 0.

Definitions and order of operations:
1. weighted_income = monthly_income * income_weight.
2. maximum_total_payment_capacity = weighted_income * {DTI_LIMIT:.10g}.
3. available_payment_capacity = maximum_total_payment_capacity - existing_monthly_debts.
4. stressed_annual_interest_pct = annual_interest_pct + 2.0 only when variable_rate is true; otherwise use annual_interest_pct.
5. currency_stress_factor = 1.15 only when currency is exactly EUR and income_currency is exactly RON; otherwise use 1.0.
6. If requested_monthly_payment > 0, analyzed_monthly_payment is exactly requested_monthly_payment. The interest-rate shock does not alter an already supplied payment. Otherwise calculate the annuity payment from requested_amount using:
   r = stressed_annual_interest_pct / 100 / 12
   if r == 0: analyzed_monthly_payment = requested_amount / term_months
   otherwise: analyzed_monthly_payment = requested_amount * r / (1 - (1 + r)^(-term_months))
   If requested_amount <= 0 or term_months <= 0, the annuity payment is 0.
7. stressed_monthly_payment = analyzed_monthly_payment * currency_stress_factor.
8. If weighted_income > 0:
   dti_pct = (existing_monthly_debts + stressed_monthly_payment) / weighted_income * 100
   Otherwise dti_pct = 99900.0. This sentinel matches the reference engine.
9. payment_before_currency_stress = max(0, available_payment_capacity / currency_stress_factor).
10. Invert the annuity using the stressed interest rate:
    if payment_before_currency_stress <= 0 or term_months <= 0: principal_by_dti = 0
    else if r == 0: principal_by_dti = payment_before_currency_stress * term_months
    otherwise: principal_by_dti = payment_before_currency_stress * (1 - (1 + r)^(-term_months)) / r
11. maximum_amount_by_dti = min({MAX_AMOUNT_RON:.10g}, principal_by_dti).

Do not round intermediate calculations. Return final values with enough precision to be checked within RON 1.00 for money and 0.05 percentage points for DTI.
Return only this JSON object with JSON numbers and no additional keys or prose:
{{
  "stressed_monthly_payment": number,
  "dti_pct": number,
  "maximum_amount_by_dti": number
}}"""


def user_prompt(profile: ClientProfile) -> str:
    profile_json = json.dumps(asdict(profile), ensure_ascii=False, indent=2)
    return f"{operating_rules_prompt()}\n\nClient profile:\n{profile_json}"


def request_payload(args: argparse.Namespace, profile: ClientProfile) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": args.model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a precise numerical calculator. Follow the supplied formulas "
                    "and return only schema-conforming JSON."
                ),
            },
            {"role": "user", "content": user_prompt(profile)},
        ],
        "stream": False,
        "format": JSON_FORMAT_SCHEMA,
        "options": {
            "num_ctx": args.num_ctx,
            "num_predict": args.num_predict,
            "temperature": args.temperature,
            "seed": args.seed,
        },
    }
    think_value = reasoning_payload_value(args.reasoning)
    if think_value is not None:
        payload["think"] = think_value
    return payload


def finite_json_number(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def parse_numeric_content(content: str) -> tuple[dict[str, float] | None, str | None]:
    try:
        loaded = json.loads(content)
    except json.JSONDecodeError as exc:
        return None, f"Invalid JSON content: {exc}"

    if not isinstance(loaded, dict):
        return None, "The response content must be a JSON object."
    actual_keys = set(loaded)
    required_keys = set(TARGET_FIELDS)
    if actual_keys != required_keys:
        missing = sorted(required_keys - actual_keys)
        extra = sorted(actual_keys - required_keys)
        return None, f"Schema mismatch; missing={missing}, extra={extra}."

    parsed: dict[str, float] = {}
    for field in TARGET_FIELDS:
        number = finite_json_number(loaded.get(field))
        if number is None:
            return None, f"{field} must be a finite JSON number."
        parsed[field] = number
    return parsed, None


def integer_or_none(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return None


def nanoseconds_and_seconds(value: object) -> tuple[int | None, float | None]:
    nanoseconds = integer_or_none(value)
    if nanoseconds is None:
        return None, None
    return nanoseconds, nanoseconds / 1_000_000_000


def text_value(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def score_values(
    parsed: dict[str, float] | None,
    expected: dict[str, float],
) -> tuple[dict[str, float], dict[str, float | None]]:
    scores: dict[str, float] = {}
    absolute_errors: dict[str, float | None] = {}
    for field in TARGET_FIELDS:
        actual = parsed.get(field) if parsed is not None else None
        if actual is None:
            scores[field] = 0.0
            absolute_errors[field] = None
            continue
        error = abs(actual - expected[field])
        absolute_errors[field] = error
        scores[field] = 1.0 if error <= TOLERANCES[field] else 0.0
    return scores, absolute_errors


def evaluate_one_case(
    client: httpx.Client,
    endpoint: str,
    args: argparse.Namespace,
    case: dict[str, Any],
) -> dict[str, Any]:
    profile = ClientProfile(**case["profile"])
    deterministic = evaluate_client(profile)
    expected = {
        "stressed_monthly_payment": deterministic.stressed_monthly_payment,
        "dti_pct": deterministic.dti * 100,
        "maximum_amount_by_dti": deterministic.maximum_amount_by_dti,
    }
    payload = request_payload(args, profile)
    prompt = payload["messages"][1]["content"]

    result: dict[str, Any] = {
        "case_id": case["id"],
        "profile": asdict(profile),
        "expected": expected,
        "tolerances": dict(TOLERANCES),
        "prompt_length_chars": len(prompt),
        "http_status": None,
        "error": None,
        "content": "",
        "thinking_present": False,
        "thinking_length_chars": 0,
        "parsed": None,
        "parse_error": None,
        "scores": {field: 0.0 for field in TARGET_FIELDS},
        "absolute_errors": {field: None for field in TARGET_FIELDS},
        "all_three_correct": False,
        "latency_seconds": None,
        "response_model": None,
        "response_created_at": None,
        "done_reason": None,
        "prompt_eval_count": None,
        "eval_count": None,
        "total_duration_ns": None,
        "total_duration_seconds": None,
        "load_duration_ns": None,
        "load_duration_seconds": None,
        "prompt_eval_duration_ns": None,
        "prompt_eval_duration_seconds": None,
        "eval_duration_ns": None,
        "eval_duration_seconds": None,
    }

    started = time.perf_counter()
    try:
        response = client.post(endpoint, json=payload)
        result["http_status"] = response.status_code
        try:
            response_data = response.json()
        except ValueError as exc:
            result["error"] = f"Response body was not JSON: {exc}"
            result["content"] = response.text
            return result

        if not isinstance(response_data, dict):
            result["error"] = "Ollama response was not a JSON object."
            result["content"] = text_value(response_data)
            return result

        result["response_model"] = response_data.get("model")
        result["response_created_at"] = response_data.get("created_at")
        result["done_reason"] = response_data.get("done_reason")
        result["prompt_eval_count"] = integer_or_none(response_data.get("prompt_eval_count"))
        result["eval_count"] = integer_or_none(response_data.get("eval_count"))

        for response_key, output_prefix in (
            ("total_duration", "total_duration"),
            ("load_duration", "load_duration"),
            ("prompt_eval_duration", "prompt_eval_duration"),
            ("eval_duration", "eval_duration"),
        ):
            nanoseconds, seconds = nanoseconds_and_seconds(response_data.get(response_key))
            result[f"{output_prefix}_ns"] = nanoseconds
            result[f"{output_prefix}_seconds"] = seconds

        message = response_data.get("message")
        if not isinstance(message, dict):
            message = {}
        content = text_value(message.get("content"))
        thinking = text_value(message.get("thinking") or message.get("reasoning"))
        result["content"] = content
        result["thinking_present"] = bool(thinking.strip())
        result["thinking_length_chars"] = len(thinking)

        api_error = response_data.get("error")
        if api_error:
            result["error"] = str(api_error)
        elif not 200 <= response.status_code < 300:
            result["error"] = f"HTTP {response.status_code}"
        elif not content.strip():
            result["error"] = "Ollama returned no final content."
        else:
            parsed, parse_error = parse_numeric_content(content)
            scores, absolute_errors = score_values(parsed, expected)
            result["parsed"] = parsed
            result["parse_error"] = parse_error
            result["scores"] = scores
            result["absolute_errors"] = absolute_errors
            result["all_three_correct"] = all(scores[field] == 1.0 for field in TARGET_FIELDS)
    except httpx.HTTPError as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        result["latency_seconds"] = time.perf_counter() - started
    return result


def summarize(case_results: list[dict[str, Any]]) -> dict[str, Any]:
    count = len(case_results)
    correct_counts = {
        field: sum(float(case.get("scores", {}).get(field, 0.0)) for case in case_results)
        for field in TARGET_FIELDS
    }
    latencies = [
        float(case["latency_seconds"])
        for case in case_results
        if isinstance(case.get("latency_seconds"), (int, float))
    ]
    valid_json_count = sum(
        1 for case in case_results if case.get("parsed") is not None and not case.get("parse_error")
    )
    http_success_count = sum(
        1
        for case in case_results
        if isinstance(case.get("http_status"), int)
        and 200 <= int(case["http_status"]) < 300
        and not case.get("error")
    )
    all_three_count = sum(1 for case in case_results if case.get("all_three_correct"))
    total_comparisons = count * len(TARGET_FIELDS)
    total_correct = sum(correct_counts.values())

    return {
        "completed_case_count": count,
        "http_success_count": http_success_count,
        "valid_json_count": valid_json_count,
        "thinking_present_count": sum(
            1 for case in case_results if case.get("thinking_present")
        ),
        "correct_counts": correct_counts,
        "agreement_by_field": {
            field: (correct_counts[field] / count if count else None)
            for field in TARGET_FIELDS
        },
        "combined_target_correct": total_correct,
        "combined_target_comparisons": total_comparisons,
        "combined_target_agreement": (
            total_correct / total_comparisons if total_comparisons else None
        ),
        "all_three_correct_count": all_three_count,
        "all_three_correct_rate": all_three_count / count if count else None,
        "latency_seconds": {
            "total": sum(latencies),
            "mean": statistics.fmean(latencies) if latencies else None,
            "median": statistics.median(latencies) if latencies else None,
            "minimum": min(latencies) if latencies else None,
            "maximum": max(latencies) if latencies else None,
        },
    }


def checkpoint(path: Path, document: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    document["updated_at"] = utc_now()
    document["summary"] = summarize(document.get("cases", []))
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps(document, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    temporary_path.replace(path)


def build_config(args: argparse.Namespace, case_ids: list[str]) -> dict[str, Any]:
    return {
        "model": args.model,
        "reasoning_cli": args.reasoning,
        "think_payload": reasoning_payload_value(args.reasoning),
        "ollama_endpoint": api_chat_url(args.ollama_url),
        "num_ctx": args.num_ctx,
        "num_predict": args.num_predict,
        "temperature": args.temperature,
        "seed": args.seed,
        "timeout_seconds": args.timeout_seconds,
        "cases_path": str(args.cases_path.resolve()),
        "case_ids": case_ids,
        "target_fields": list(TARGET_FIELDS),
        "tolerances": dict(TOLERANCES),
        "one_call_per_case": True,
    }


def new_document(config: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "benchmark": "numeric_reasoning",
        "status": "in_progress",
        "created_at": utc_now(),
        "updated_at": utc_now(),
        "config": config,
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "command": [sys.executable, *sys.argv],
        },
        "reference_engine": {
            "dti_limit": DTI_LIMIT,
            "maximum_amount_ron": MAX_AMOUNT_RON,
            "income_weights": INCOME_WEIGHTS,
        },
        "cases": [],
        "summary": summarize([]),
    }


def load_or_initialize(
    args: argparse.Namespace,
    config: dict[str, Any],
) -> dict[str, Any]:
    output_path = args.output.resolve()
    if args.resume:
        if not output_path.exists():
            raise SystemExit(f"Cannot resume because the checkpoint does not exist: {output_path}")
        try:
            document = json.loads(output_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise SystemExit(f"Cannot read checkpoint {output_path}: {exc}") from exc
        if not isinstance(document, dict) or document.get("config") != config:
            raise SystemExit(
                "Checkpoint configuration does not match this invocation; use a new output "
                "path or explicitly --overwrite it."
            )
        if not isinstance(document.get("cases"), list):
            raise SystemExit("Checkpoint cases field is not a list.")
        document["status"] = "in_progress"
        return document

    if output_path.exists() and not args.overwrite:
        raise SystemExit(
            f"Output already exists: {output_path}. Use --resume or --overwrite explicitly."
        )
    return new_document(config)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    args.cases_path = args.cases_path.resolve()
    args.output = args.output.resolve()

    loaded_cases = load_evaluation_cases(args.cases_path)
    client_cases = loaded_cases.get("client_cases", [])
    if not isinstance(client_cases, list) or not client_cases:
        raise SystemExit("No client_cases were found in the evaluation file.")
    if args.max_cases is not None:
        client_cases = client_cases[: args.max_cases]
    case_ids = [str(case.get("id", "")) for case in client_cases]
    if any(not case_id for case_id in case_ids) or len(case_ids) != len(set(case_ids)):
        raise SystemExit("Every client case must have a unique, non-empty id.")

    config = build_config(args, case_ids)
    document = load_or_initialize(args, config)
    checkpoint(args.output, document)

    completed_ids = {
        str(case.get("case_id"))
        for case in document["cases"]
        if isinstance(case, dict) and case.get("case_id")
    }
    endpoint = api_chat_url(args.ollama_url)
    interrupted = False
    try:
        with httpx.Client(timeout=args.timeout_seconds) as client:
            for index, case in enumerate(client_cases, start=1):
                case_id = str(case["id"])
                if case_id in completed_ids:
                    continue
                print(f"[{index}/{len(client_cases)}] {case_id}", flush=True)
                case_result = evaluate_one_case(client, endpoint, args, case)
                document["cases"].append(case_result)
                checkpoint(args.output, document)
    except KeyboardInterrupt:
        interrupted = True
        document["status"] = "interrupted"
        checkpoint(args.output, document)

    if interrupted:
        print(f"Interrupted; checkpoint saved to {args.output}", file=sys.stderr)
        return 130

    document["status"] = "complete"
    checkpoint(args.output, document)
    print(json.dumps(document["summary"], ensure_ascii=False, indent=2))
    print(f"Checkpoint written to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
