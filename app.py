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
        LOGGER.error("Nu pot verifica procesele active pe portul %s.\n%s", port, traceback.format_exc())
        return

    if result.returncode != 0:
        LOGGER.error("netstat a esuat la verificarea portului %s: %s", port, result.stderr.strip())
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
        LOGGER.warning("Inchid proces existent pe portul %s: PID %s", port, pid)
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
                "Nu am putut inchide PID %s pe portul %s: %s",
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
    return f"""## Eroare in aplicatie

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
            "Altul",  # sector
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
        return format_exception("Evaluare client")


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
            "Altul",
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
        error = format_exception("Evaluare client")
        return error, error


def show_analyze_loading() -> str:
    return (
        "## Se proceseaza evaluarea...\n\n"
        "Recuperez fragmentele RAG relevante si cer LLM-ului local sa calculeze decizia de creditare. "
        "Apoi compar raspunsul cu formulele Python in tabul de comparatie. "
        "Prima evaluare dupa pornire poate dura putin mai mult."
    )


def ask_policy(question: str) -> str:
    try:
        if not question.strip():
            return "Scrie o intrebare despre manual."
        return answer_policy_question(question, INDEX, use_llm=True)
    except Exception:
        return format_exception("Intrebare despre manual")


def read_error_log() -> str:
    if not ERROR_LOG.exists():
        return "Nu exista erori inregistrate in runtime_errors.log."
    content = ERROR_LOG.read_text(encoding="utf-8", errors="replace").strip()
    if not content:
        return "Fisierul runtime_errors.log este gol."
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
            return "Alege cel putin un caz de evaluare."

        results = run_evaluation_suite(
            INDEX,
            max_policy_cases=policy_limit,
            max_client_cases=client_limit,
        )
        return summarize_evaluation_markdown(results)
    except Exception:
        return format_exception("Evaluare metrici")


with gr.Blocks(title="Asistent de Creditare NovaTech") as demo:
    gr.Markdown(
        "# Asistent de Creditare RAG NovaTech\n"
        "Evalueaza un client fictiv folosind manualul NovaTech si afiseaza fragmentele recuperate din corpus.\n\n"
        "**Diagnostic activ:** erorile serverului se pot vedea direct la `/runtime-errors`."
    )
    last_comparison = gr.State(
        "## Comparatie indisponibila\n\nRuleaza mai intai o analiza de client."
    )

    with gr.Tab("Analiza client"):
        with gr.Row():
            with gr.Column():
                age = gr.Number(label="Varsta", value=35, precision=0)
                term_months = gr.Number(label="Durata creditului (luni)", value=60, precision=0)
                fico = gr.Number(label="Scor FICO", value=720, precision=0)
                monthly_income = gr.Number(label="Venit lunar declarat (RON)", value=15000)
                income_type = gr.Dropdown(
                    label="Tip venit",
                    choices=list(INCOME_WEIGHTS.keys()),
                    value="Salariu - contract nedeterminat",
                )
                existing_monthly_debts = gr.Number(label="Rate existente lunare (RON)", value=0)
                requested_amount = gr.Number(label="Suma solicitata (RON)", value=100000)
                requested_monthly_payment = gr.Number(
                    label="Rata lunara dorita (RON, optional; 0 = calculeaza din suma)",
                    value=0,
                )
                annual_interest_pct = gr.Number(label="Dobanda anuala estimata (%)", value=10.0)
            with gr.Column():
                currency = gr.Radio(label="Moneda credit", choices=["RON", "EUR"], value="RON")
                income_currency = gr.Radio(label="Moneda venit", choices=["RON", "EUR"], value="RON")
                variable_rate = gr.Checkbox(label="Dobanda variabila", value=False)
                active_delay_days = gr.Number(label="Zile intarziere activa", value=0, precision=0)
                historical_90_delay_last_year = gr.Checkbox(
                    label="A avut intarziere istorica >90 zile in ultimul an", value=False
                )
                historical_90_debt_settled = gr.Checkbox(label="Datoria istorica a fost stinsa", value=False)
                income_increase_after_delay_pct = gr.Number(
                    label="Crestere venit dupa intarziere (%)", value=0
                )
                is_pep = gr.Checkbox(label="Client PEP", value=False)
                aml_risk = gr.Radio(label="Risc AML", choices=["Scazut", "Standard", "Ridicat"], value="Standard")

        analyze_button = gr.Button("Evalueaza clientul", variant="primary")
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

    with gr.Tab("Intrebari despre manual"):
        question = gr.Textbox(
            label="Intrebare",
            value="Care sunt ponderile veniturilor si cum se calculeaza GMI?",
            lines=3,
        )
        ask_button = gr.Button("Cauta in manual", variant="primary")
        answer_output = gr.Markdown()
        ask_button.click(ask_policy, inputs=[question], outputs=answer_output, show_progress="full")

    with gr.Tab("Metrici"):
        gr.Markdown(
            "Ruleaza un set sintetic de evaluare pentru RAG si LLM. "
            "Cazurile sunt definite in `examples/evaluation_cases.json`."
        )
        with gr.Row():
            max_policy_cases = gr.Number(
                label="Cazuri pentru Intrebari despre manual",
                value=2,
                precision=0,
            )
            max_client_cases = gr.Number(
                label="Cazuri pentru Analiza client",
                value=2,
                precision=0,
            )
        metrics_button = gr.Button("Ruleaza metricile", variant="primary")
        metrics_output = gr.Markdown()
        metrics_button.click(
            run_metrics,
            inputs=[max_policy_cases, max_client_cases],
            outputs=metrics_output,
            show_progress="full",
        )

    with gr.Tab("Comparatie LLM vs formule"):
        gr.Markdown(
            "Compara ultimul raspuns calculat de LLM cu valorile calculate independent prin formule Python."
        )
        comparison_output = gr.Markdown(
            "## Comparatie indisponibila\n\nRuleaza mai intai o analiza de client."
        )
        refresh_comparison_button = gr.Button("Afiseaza comparatia ultimei analize")
        refresh_comparison_button.click(
            lambda value: value,
            inputs=[last_comparison],
            outputs=[comparison_output],
            show_progress="full",
        )

    with gr.Tab("Erori server"):
        gr.Markdown(
            "Daca apare doar toast-ul rosu `Error`, apasa aici ca sa vezi ultima eroare salvata de server."
        )
        diagnostics_button = gr.Button("Afiseaza ultima eroare")
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
        return "Nu exista runtime_errors.log."
    content = ERROR_LOG.read_text(encoding="utf-8", errors="replace").strip()
    return content or "runtime_errors.log este gol."


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
                LOGGER.error("Nu pot procesa corpul requestului predict.\n%s", traceback.format_exc())

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
                "Salariu - contract nedeterminat",
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
                "Salariu - contract nedeterminat",
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
            response_data = [f"## Eroare\nfn_index necunoscut: {fn_index}, data_len={len(data)}"]

        duration = time.perf_counter() - started
        return JSONResponse(
            {
                "data": response_data,
                "is_generating": False,
                "duration": duration,
                "average_duration": duration,
            }
        )

    return gr.mount_gradio_app(server, demo, path="/")


if __name__ == "__main__":
    close_existing_server_processes()
    uvicorn.run(create_server(), host=SERVER_HOST, port=SERVER_PORT)
