from __future__ import annotations

import json
import logging
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
)


ERROR_LOG = Path("runtime_errors.log")
logging.basicConfig(
    filename=ERROR_LOG,
    level=logging.ERROR,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    force=True,
)
LOGGER = logging.getLogger("credit_assistant_app")


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
        return build_analysis_markdown(profile, INDEX, use_llm=True)
    except Exception:
        return format_exception("Evaluare client")


def ask_policy(question: str, use_llm: bool) -> str:
    try:
        if not question.strip():
            return "Scrie o intrebare despre manual."
        return answer_policy_question(question, INDEX, use_llm=bool(use_llm))
    except Exception:
        return format_exception("Intrebare despre manual")


def read_error_log() -> str:
    if not ERROR_LOG.exists():
        return "Nu exista erori inregistrate in runtime_errors.log."
    content = ERROR_LOG.read_text(encoding="utf-8", errors="replace").strip()
    if not content:
        return "Fisierul runtime_errors.log este gol."
    return f"```text\n{content[-5000:]}\n```"


with gr.Blocks(title="Asistent de Creditare RAG NovaTech") as demo:
    gr.Markdown(
        "# Asistent de Creditare RAG NovaTech\n"
        "Prototip educational: evalueaza un client fictiv folosind manualul NovaTech si afiseaza fragmentele recuperate din corpus.\n\n"
        "**Diagnostic activ:** erorile serverului se pot vedea direct la `/runtime-errors`."
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
        analyze_button.click(
            analyze_client,
            inputs=[
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
            ],
            outputs=analysis_output,
        )

    with gr.Tab("Intrebari despre manual"):
        question = gr.Textbox(
            label="Intrebare",
            value="Care sunt ponderile veniturilor si cum se calculeaza GMI?",
            lines=3,
        )
        use_llm_question = gr.Checkbox(label="Foloseste LLM daca este configurat", value=False)
        ask_button = gr.Button("Cauta in manual", variant="primary")
        answer_output = gr.Markdown()
        ask_button.click(ask_policy, inputs=[question, use_llm_question], outputs=answer_output)

    with gr.Tab("Diagnostic"):
        gr.Markdown(
            "Daca apare doar toast-ul rosu `Error`, apasa aici ca sa vezi ultima eroare salvata de server."
        )
        diagnostics_button = gr.Button("Afiseaza ultima eroare")
        diagnostics_output = gr.Markdown()
        diagnostics_button.click(read_error_log, inputs=[], outputs=diagnostics_output)

    gr.Markdown(
        "Nota: proiect demonstrativ pentru disertatie. Manualul NovaTech este fictiv si rezultatele nu sunt consultanta financiara."
    )


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

        if fn_index == 0:
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
            output = analyze_client(*args[:18])
        elif fn_index == 1:
            defaults = ["Care sunt ponderile veniturilor si cum se calculeaza GMI?", False]
            args = data + defaults[len(data) :]
            output = ask_policy(str(args[0]), bool(args[1]))
        elif fn_index == 2:
            output = read_error_log()
        else:
            output = f"## Eroare\nfn_index necunoscut: {fn_index}"

        duration = time.perf_counter() - started
        return JSONResponse(
            {
                "data": [output],
                "is_generating": False,
                "duration": duration,
                "average_duration": duration,
            }
        )

    return gr.mount_gradio_app(server, demo, path="/")


if __name__ == "__main__":
    uvicorn.run(create_server(), host="127.0.0.1", port=7860)
