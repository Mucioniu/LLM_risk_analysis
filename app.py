from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import traceback
import time
from uuid import uuid4
from pathlib import Path

import gradio as gr
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, PlainTextResponse

from credit_assistant.credit_engine import INCOME_WEIGHTS, ClientProfile
from credit_assistant.service import (
    answer_policy_question,
    build_analysis_markdown,
    build_default_index,
    build_llm_credit_analysis,
)
from credit_assistant.evaluation import run_evaluation_suite, summarize_evaluation_markdown


ERROR_LOG = Path("runtime_errors.log")
logging.basicConfig(
    filename=ERROR_LOG,
    level=logging.ERROR,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    force=True,
)
LOGGER = logging.getLogger("credit_assistant_app")


APP_THEME = gr.themes.Soft(
    primary_hue="teal",
    secondary_hue="blue",
    neutral_hue="slate",
).set(
    body_background_fill="#eef7f6",
    body_background_fill_dark="#eef7f6",
    body_text_color="#243447",
    body_text_color_dark="#243447",
    body_text_color_subdued="#5b6b7c",
    body_text_color_subdued_dark="#5b6b7c",
    background_fill_primary="#ffffff",
    background_fill_primary_dark="#ffffff",
    background_fill_secondary="#f7fbfb",
    background_fill_secondary_dark="#f7fbfb",
    block_background_fill="rgba(255, 255, 255, 0.82)",
    block_background_fill_dark="rgba(255, 255, 255, 0.82)",
    block_border_color="rgba(105, 145, 151, 0.22)",
    block_border_color_dark="rgba(105, 145, 151, 0.22)",
    block_shadow="0 14px 35px rgba(37, 78, 83, 0.10)",
    block_shadow_dark="0 14px 35px rgba(37, 78, 83, 0.10)",
    input_background_fill="rgba(255, 255, 255, 0.94)",
    input_background_fill_dark="rgba(255, 255, 255, 0.94)",
    input_border_color="#cbdedd",
    input_border_color_dark="#cbdedd",
    input_border_color_focus="#168b86",
    input_border_color_focus_dark="#168b86",
    button_primary_background_fill="linear-gradient(135deg, #147d78, #2769a8)",
    button_primary_background_fill_dark="linear-gradient(135deg, #147d78, #2769a8)",
    button_primary_background_fill_hover="linear-gradient(135deg, #106b67, #20598f)",
    button_primary_background_fill_hover_dark="linear-gradient(135deg, #106b67, #20598f)",
    button_primary_border_color="#147d78",
    button_primary_border_color_dark="#147d78",
    button_primary_text_color="#ffffff",
    button_primary_text_color_dark="#ffffff",
)


APP_CSS = """
:root,
.dark {
    color-scheme: light !important;
}

html,
body {
    min-height: 100%;
}

body {
    background:
        radial-gradient(circle at 8% 4%, rgba(71, 201, 184, 0.25), transparent 30rem),
        radial-gradient(circle at 92% 8%, rgba(104, 164, 227, 0.24), transparent 32rem),
        radial-gradient(circle at 55% 92%, rgba(246, 192, 126, 0.20), transparent 34rem),
        linear-gradient(145deg, #f8fbf7 0%, #edf7f5 48%, #f7f1e8 100%) !important;
    background-attachment: fixed !important;
}

.gradio-container {
    max-width: 1240px !important;
    margin: 0 auto !important;
    padding: 30px 24px 48px !important;
    background: transparent !important;
    color: #243447 !important;
}

#novatech-hero {
    position: relative;
    overflow: hidden;
    margin-bottom: 18px;
    padding: 28px 32px 24px;
    border: 1px solid rgba(255, 255, 255, 0.78);
    border-radius: 24px;
    background:
        linear-gradient(120deg, rgba(255, 255, 255, 0.94), rgba(240, 250, 248, 0.84));
    box-shadow: 0 18px 50px rgba(42, 82, 87, 0.13);
    backdrop-filter: blur(16px);
}

#novatech-hero::after {
    content: "";
    position: absolute;
    width: 190px;
    height: 190px;
    right: -58px;
    top: -92px;
    border-radius: 50%;
    background: linear-gradient(135deg, rgba(43, 169, 157, 0.25), rgba(67, 117, 190, 0.19));
    pointer-events: none;
}

#novatech-hero h1 {
    margin-bottom: 10px;
    font-size: clamp(2rem, 4vw, 3.15rem);
    line-height: 1.06;
    letter-spacing: -0.035em;
    background: linear-gradient(100deg, #164e63, #147d78 52%, #315d9d);
    -webkit-background-clip: text;
    background-clip: text;
    color: transparent !important;
}

#novatech-hero p {
    max-width: 800px;
    color: #526475;
    font-size: 1.02rem;
    line-height: 1.65;
}

.tabs {
    overflow: hidden;
    padding: 7px;
    border: 1px solid rgba(113, 151, 155, 0.25) !important;
    border-radius: 22px !important;
    background: rgba(255, 255, 255, 0.73) !important;
    box-shadow: 0 18px 48px rgba(44, 79, 84, 0.11) !important;
    backdrop-filter: blur(18px);
}

.tab-nav {
    gap: 6px;
    padding: 5px 6px 9px;
    border-bottom-color: rgba(101, 139, 144, 0.18) !important;
}

.tab-nav button {
    border-radius: 11px 11px 4px 4px !important;
    color: #536476 !important;
    transition: background-color 160ms ease, color 160ms ease, transform 160ms ease;
}

.tab-nav button:hover {
    background: rgba(20, 125, 120, 0.08) !important;
    color: #116b67 !important;
}

.tab-nav button.selected {
    color: #116b67 !important;
}

button.primary {
    box-shadow: 0 10px 22px rgba(20, 125, 120, 0.22) !important;
    transition: transform 160ms ease, box-shadow 160ms ease !important;
}

button.primary:hover {
    transform: translateY(-1px);
    box-shadow: 0 13px 26px rgba(20, 125, 120, 0.28) !important;
}

@media (max-width: 700px) {
    .gradio-container {
        padding: 16px 12px 30px !important;
    }

    #novatech-hero {
        padding: 22px 20px 18px;
        border-radius: 19px;
    }

    .tabs {
        border-radius: 17px !important;
    }
}
"""


def env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default


SERVER_HOST = os.getenv("SERVER_HOST", "127.0.0.1")
SERVER_PORT = env_int("SERVER_PORT", 7860)


def close_existing_server_processes(port: int = SERVER_PORT) -> None:
    """Close stale Windows listeners on the app port before Uvicorn binds it."""
    if os.name != "nt":
        return

    current_pid = os.getpid()
    try:
        result = subprocess.run(
            ["netstat", "-ano", "-p", "tcp"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
    except Exception:
        LOGGER.error("Could not check active processes on port %s.\n%s", port, traceback.format_exc())
        return

    if result.returncode != 0:
        LOGGER.error("netstat failed while checking port %s: %s", port, result.stderr.strip())
        return

    listener_pids: set[int] = set()
    for line in result.stdout.splitlines():
        parts = line.split()
        if len(parts) < 5 or parts[0].upper() != "TCP":
            continue

        local_address = parts[1]
        state = parts[-2].upper()
        pid_text = parts[-1]
        if state != "LISTENING":
            continue
        if not local_address.endswith(f":{port}"):
            continue
        if not pid_text.isdigit():
            continue

        pid = int(pid_text)
        if pid != current_pid:
            listener_pids.add(pid)

    for pid in sorted(listener_pids):
        LOGGER.warning("Closing existing process on port %s: PID %s", port, pid)
        kill_result = subprocess.run(
            ["taskkill", "/PID", str(pid), "/F"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
        if kill_result.returncode != 0:
            LOGGER.error(
                "Could not close PID %s on port %s: %s",
                pid,
                port,
                kill_result.stderr.strip() or kill_result.stdout.strip(),
            )


class TeeStream:
    def __init__(self, *streams) -> None:
        self.streams = streams

    def write(self, data: str) -> int:
        for stream in self.streams:
            stream.write(data)
            stream.flush()
        return len(data)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()


_stderr_log = ERROR_LOG.open("a", encoding="utf-8")
sys.stderr = TeeStream(sys.stderr, _stderr_log)
INDEX = build_default_index()


def format_exception(title: str) -> str:
    traceback_text = traceback.format_exc()
    LOGGER.error("%s\n%s", title, traceback_text)
    return f"""## Application error

**Context:** {title}

```text
{traceback_text}
```
"""


def analyze_client(
    age: int,
    term_months: int,
    fico: int,
    monthly_income: float,
    income_type: str,
    existing_monthly_debts: float,
    requested_amount: float,
    requested_monthly_payment: float,
    annual_interest_pct: float,
    currency: str,
    income_currency: str,
    variable_rate: bool,
    active_delay_days: int,
    historical_90_delay_last_year: bool,
    historical_90_debt_settled: bool,
    income_increase_after_delay_pct: float,
    is_pep: bool,
    aml_risk: str,
    *optional_values,
) -> str:
    try:
        defaults = [
            False,  # is_non_eu
            False,  # married_to_ro_citizen
            False,  # owns_property_in_ro
            0,  # local_contract_months
            "Other",  # sector
            12,  # current_job_tenure_months
            0,  # previous_job_tenure_months
            0,  # gap_days_between_jobs
        ]
        values = list(optional_values) + defaults[len(optional_values) :]
        (
            is_non_eu,
            married_to_ro_citizen,
            owns_property_in_ro,
            local_contract_months,
            sector,
            current_job_tenure_months,
            previous_job_tenure_months,
            gap_days_between_jobs,
        ) = values[:8]

        profile = ClientProfile(
            age=int(age),
            term_months=int(term_months),
            fico=int(fico),
            monthly_income=float(monthly_income),
            income_type=income_type,
            existing_monthly_debts=float(existing_monthly_debts),
            requested_amount=float(requested_amount),
            requested_monthly_payment=float(requested_monthly_payment),
            annual_interest_pct=float(annual_interest_pct),
            currency=currency,
            income_currency=income_currency,
            variable_rate=bool(variable_rate),
            active_delay_days=int(active_delay_days),
            historical_90_delay_last_year=bool(historical_90_delay_last_year),
            historical_90_debt_settled=bool(historical_90_debt_settled),
            income_increase_after_delay_pct=float(income_increase_after_delay_pct),
            is_pep=bool(is_pep),
            aml_risk=aml_risk,
            is_non_eu=bool(is_non_eu),
            married_to_ro_citizen=bool(married_to_ro_citizen),
            owns_property_in_ro=bool(owns_property_in_ro),
            local_contract_months=int(local_contract_months),
            sector=str(sector),
            current_job_tenure_months=int(current_job_tenure_months),
            previous_job_tenure_months=int(previous_job_tenure_months),
            gap_days_between_jobs=int(gap_days_between_jobs),
        )
        return build_llm_credit_analysis(profile, INDEX).answer_markdown
    except Exception:
        return format_exception("Client evaluation")


def analyze_client_with_comparison(
    age: int,
    term_months: int,
    fico: int,
    monthly_income: float,
    income_type: str,
    existing_monthly_debts: float,
    requested_amount: float,
    requested_monthly_payment: float,
    annual_interest_pct: float,
    currency: str,
    income_currency: str,
    variable_rate: bool,
    active_delay_days: int,
    historical_90_delay_last_year: bool,
    historical_90_debt_settled: bool,
    income_increase_after_delay_pct: float,
    is_pep: bool,
    aml_risk: str,
    *optional_values,
) -> tuple[str, str]:
    try:
        defaults = [
            False,
            False,
            False,
            0,
            "Other",
            12,
            0,
            0,
        ]
        values = list(optional_values) + defaults[len(optional_values) :]
        (
            is_non_eu,
            married_to_ro_citizen,
            owns_property_in_ro,
            local_contract_months,
            sector,
            current_job_tenure_months,
            previous_job_tenure_months,
            gap_days_between_jobs,
        ) = values[:8]

        profile = ClientProfile(
            age=int(age),
            term_months=int(term_months),
            fico=int(fico),
            monthly_income=float(monthly_income),
            income_type=income_type,
            existing_monthly_debts=float(existing_monthly_debts),
            requested_amount=float(requested_amount),
            requested_monthly_payment=float(requested_monthly_payment),
            annual_interest_pct=float(annual_interest_pct),
            currency=currency,
            income_currency=income_currency,
            variable_rate=bool(variable_rate),
            active_delay_days=int(active_delay_days),
            historical_90_delay_last_year=bool(historical_90_delay_last_year),
            historical_90_debt_settled=bool(historical_90_debt_settled),
            income_increase_after_delay_pct=float(income_increase_after_delay_pct),
            is_pep=bool(is_pep),
            aml_risk=aml_risk,
            is_non_eu=bool(is_non_eu),
            married_to_ro_citizen=bool(married_to_ro_citizen),
            owns_property_in_ro=bool(owns_property_in_ro),
            local_contract_months=int(local_contract_months),
            sector=str(sector),
            current_job_tenure_months=int(current_job_tenure_months),
            previous_job_tenure_months=int(previous_job_tenure_months),
            gap_days_between_jobs=int(gap_days_between_jobs),
        )
        result = build_llm_credit_analysis(profile, INDEX)
        return result.answer_markdown, result.comparison_markdown
    except Exception:
        error = format_exception("Client evaluation")
        return error, error


def show_analyze_loading() -> str:
    return (
        "## Processing the evaluation...\n\n"
        "Retrieving the relevant RAG fragments and asking the local LLM to calculate the credit decision. "
        "The response is then compared with the Python formulas in the comparison tab. "
        "The first evaluation after startup may take a little longer."
    )


def ask_policy(question: str) -> str:
    try:
        if not question.strip():
            return "Enter a question about the manual."
        return answer_policy_question(question, INDEX, use_llm=True)
    except Exception:
        return format_exception("Question about the manual")


def read_error_log() -> str:
    if not ERROR_LOG.exists():
        return "No errors have been recorded in runtime_errors.log."
    content = ERROR_LOG.read_text(encoding="utf-8", errors="replace").strip()
    if not content:
        return "The runtime_errors.log file is empty."
    return f"```text\n{content[-5000:]}\n```"


def nonnegative_int(value, default: int) -> int:
    try:
        return max(0, int(float(value)))
    except (TypeError, ValueError):
        return default


def run_metrics(max_policy_cases: int, max_client_cases: int) -> str:
    try:
        policy_limit = nonnegative_int(max_policy_cases, 2)
        client_limit = nonnegative_int(max_client_cases, 2)
        if policy_limit == 0 and client_limit == 0:
            return "Select at least one evaluation case."

        results = run_evaluation_suite(
            INDEX,
            max_policy_cases=policy_limit,
            max_client_cases=client_limit,
        )
        return summarize_evaluation_markdown(results)
    except Exception:
        return format_exception("Metric evaluation")


with gr.Blocks(title="NovaTech Credit Assistant") as demo:
    gr.Markdown(
        "# NovaTech RAG Credit Assistant\n"
        "Evaluate a fictional client using the NovaTech manual and display the fragments retrieved from the corpus.\n\n"
        "**Diagnostics enabled:** server errors can be viewed directly at `/runtime-errors`.",
        elem_id="novatech-hero",
    )
    last_comparison = gr.State(
        "## Comparison unavailable\n\nRun a client analysis first."
    )

    with gr.Tab("Client Analysis"):
        with gr.Row():
            with gr.Column():
                age = gr.Number(label="Age", value=35, precision=0)
                term_months = gr.Number(label="Loan term (months)", value=60, precision=0)
                fico = gr.Number(label="FICO score", value=720, precision=0)
                monthly_income = gr.Number(label="Declared monthly income (RON)", value=15000)
                income_type = gr.Dropdown(
                    label="Income type",
                    choices=list(INCOME_WEIGHTS.keys()),
                    value="Salary - permanent contract",
                )
                existing_monthly_debts = gr.Number(label="Existing monthly debt payments (RON)", value=0)
                requested_amount = gr.Number(label="Requested amount (RON)", value=100000)
                requested_monthly_payment = gr.Number(
                    label="Requested monthly payment (RON, optional; 0 = calculate from amount)",
                    value=0,
                )
                annual_interest_pct = gr.Number(label="Estimated annual interest rate (%)", value=10.0)
            with gr.Column():
                currency = gr.Radio(label="Loan currency", choices=["RON", "EUR"], value="RON")
                income_currency = gr.Radio(label="Income currency", choices=["RON", "EUR"], value="RON")
                variable_rate = gr.Checkbox(label="Variable interest rate", value=False)
                active_delay_days = gr.Number(label="Days currently past due", value=0, precision=0)
                historical_90_delay_last_year = gr.Checkbox(
                    label="Had a historical delay of >90 days in the past year", value=False
                )
                historical_90_debt_settled = gr.Checkbox(label="Historical debt has been settled", value=False)
                income_increase_after_delay_pct = gr.Number(
                    label="Income increase after delay (%)", value=0
                )
                is_pep = gr.Checkbox(label="Client PEP", value=False)
                aml_risk = gr.Radio(label="AML risk", choices=["Low", "Standard", "High"], value="Standard")

        analyze_button = gr.Button("Evaluate client", variant="primary")
        analysis_output = gr.Markdown()
        analyze_inputs = [
                age,
                term_months,
                fico,
                monthly_income,
                income_type,
                existing_monthly_debts,
                requested_amount,
                requested_monthly_payment,
                annual_interest_pct,
                currency,
                income_currency,
                variable_rate,
                active_delay_days,
                historical_90_delay_last_year,
                historical_90_debt_settled,
                income_increase_after_delay_pct,
                is_pep,
                aml_risk,
        ]
        analyze_event = analyze_button.click(
            show_analyze_loading,
            inputs=[],
            outputs=analysis_output,
            show_progress="full",
        )

    with gr.Tab("Manual Questions"):
        question = gr.Textbox(
            label="Question",
            value="What are the income weights and how is DTI calculated?",
            lines=3,
        )
        ask_button = gr.Button("Search the manual", variant="primary")
        answer_output = gr.Markdown()
        ask_button.click(ask_policy, inputs=[question], outputs=answer_output, show_progress="full")

    with gr.Tab("Metrics"):
        gr.Markdown(
            "Run a synthetic evaluation suite for RAG and the LLM. "
            "The cases are defined in `examples/evaluation_cases.json`."
        )
        with gr.Row():
            max_policy_cases = gr.Number(
                label="Manual Questions cases",
                value=2,
                precision=0,
            )
            max_client_cases = gr.Number(
                label="Client Analysis cases",
                value=2,
                precision=0,
            )
        metrics_button = gr.Button("Run metrics", variant="primary")
        metrics_output = gr.Markdown()
        metrics_button.click(
            run_metrics,
            inputs=[max_policy_cases, max_client_cases],
            outputs=metrics_output,
            show_progress="full",
        )

    with gr.Tab("LLM vs. Formulas Comparison"):
        gr.Markdown(
            "Compare the latest LLM-generated response with the values calculated independently using Python formulas."
        )
        comparison_output = gr.Markdown(
            "## Comparison unavailable\n\nRun a client analysis first."
        )
        refresh_comparison_button = gr.Button("Show the latest analysis comparison")
        refresh_comparison_button.click(
            lambda value: value,
            inputs=[last_comparison],
            outputs=[comparison_output],
            show_progress="full",
        )

    with gr.Tab("Server Errors"):
        gr.Markdown(
            "If only the red `Error` notification appears, click here to view the latest error saved by the server."
        )
        diagnostics_button = gr.Button("Show latest error")
        diagnostics_output = gr.Markdown()
        diagnostics_button.click(read_error_log, inputs=[], outputs=diagnostics_output, show_progress="full")

    analyze_event.then(
        analyze_client_with_comparison,
        inputs=analyze_inputs,
        outputs=[analysis_output, last_comparison],
        show_progress="full",
    ).then(
        lambda value: value,
        inputs=[last_comparison],
        outputs=[comparison_output],
        show_progress="hidden",
    )


demo.queue(default_concurrency_limit=1)


def get_runtime_errors_text() -> str:
    if not ERROR_LOG.exists():
        return "runtime_errors.log does not exist."
    content = ERROR_LOG.read_text(encoding="utf-8", errors="replace").strip()
    return content or "runtime_errors.log is empty."


def create_server() -> FastAPI:
    server = FastAPI()

    @server.middleware("http")
    async def gradio_predict_compatibility(request: Request, call_next):
        if request.url.path.rstrip("/") in {"/run/predict", "/api/predict"}:
            body = await request.body()
            try:
                payload = json.loads(body.decode("utf-8")) if body else {}
                if isinstance(payload, dict):
                    payload.setdefault("session_hash", f"server-{uuid4().hex}")
                    payload.setdefault("event_id", f"server-{uuid4().hex}")
                    data_len = len(payload.get("data", [])) if isinstance(payload.get("data"), list) else "non-list"
                    LOGGER.error(
                        "Predict request path=%s fn_index=%s data_len=%s",
                        request.url.path,
                        payload.get("fn_index"),
                        data_len,
                    )
                    body = json.dumps(payload).encode("utf-8")
            except Exception:
                LOGGER.error("Could not process the predict request body.\n%s", traceback.format_exc())

            async def receive():
                return {"type": "http.request", "body": body, "more_body": False}

            request = Request(request.scope, receive)

        return await call_next(request)

    @server.get("/runtime-errors", response_class=PlainTextResponse)
    def runtime_errors_endpoint() -> str:
        return get_runtime_errors_text()

    @server.post("/run/predict/")
    @server.post("/run/predict")
    async def run_predict_compatibility(request: Request):
        started = time.perf_counter()
        payload = await request.json()
        fn_index = int(payload.get("fn_index", 0))
        data = payload.get("data", [])
        if not isinstance(data, list):
            data = []

        LOGGER.error(
            "Compat /run/predict fn_index=%s data_len=%s",
            fn_index,
            len(data),
        )

        if fn_index == 0 and len(data) == 0:
            response_data = [show_analyze_loading()]
        elif fn_index == 5 and len(data) == 18:
            defaults = [
                35,
                60,
                720,
                15000,
                "Salary - permanent contract",
                0,
                100000,
                0,
                10,
                "RON",
                "RON",
                False,
                0,
                False,
                False,
                0,
                False,
                "Standard",
            ]
            args = data + defaults[len(data) :]
            answer, comparison = analyze_client_with_comparison(*args[:18])
            response_data = [answer, comparison]
        elif fn_index == 1 and len(data) == 1:
            response_data = [ask_policy(str(data[0]))]
        elif fn_index == 2 and len(data) == 2:
            defaults = [2, 2]
            args = data + defaults[len(data) :]
            response_data = [run_metrics(int(args[0]), int(args[1]))]
        elif fn_index in {3, 6} and len(data) == 1:
            response_data = [str(data[0])]
        elif fn_index == 4 and len(data) == 0:
            response_data = [read_error_log()]
        elif len(data) == 18:
            defaults = [
                35,
                60,
                720,
                15000,
                "Salary - permanent contract",
                0,
                100000,
                0,
                10,
                "RON",
                "RON",
                False,
                0,
                False,
                False,
                0,
                False,
                "Standard",
            ]
            args = data + defaults[len(data) :]
            answer, comparison = analyze_client_with_comparison(*args[:18])
            response_data = [answer, comparison]
        else:
            response_data = [f"## Error\nUnknown fn_index: {fn_index}, data_len={len(data)}"]

        duration = time.perf_counter() - started
        return JSONResponse(
            {
                "data": response_data,
                "is_generating": False,
                "duration": duration,
                "average_duration": duration,
            }
        )

    return gr.mount_gradio_app(
        server,
        demo,
        path="/",
        theme=APP_THEME,
        css=APP_CSS,
    )


if __name__ == "__main__":
    close_existing_server_processes()
    uvicorn.run(create_server(), host=SERVER_HOST, port=SERVER_PORT)
