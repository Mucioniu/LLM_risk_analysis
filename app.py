from __future__ import annotations

import gradio as gr

from credit_assistant.credit_engine import INCOME_WEIGHTS, ClientProfile
from credit_assistant.service import (
    answer_policy_question,
    build_analysis_markdown,
    build_default_index,
)


INDEX = build_default_index()


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
    is_non_eu: bool,
    married_to_ro_citizen: bool,
    owns_property_in_ro: bool,
    local_contract_months: int,
    sector: str,
    current_job_tenure_months: int,
    previous_job_tenure_months: int,
    gap_days_between_jobs: int,
    use_llm: bool,
) -> str:
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
        sector=sector,
        current_job_tenure_months=int(current_job_tenure_months),
        previous_job_tenure_months=int(previous_job_tenure_months),
        gap_days_between_jobs=int(gap_days_between_jobs),
    )
    return build_analysis_markdown(profile, INDEX, use_llm=bool(use_llm))


def ask_policy(question: str, use_llm: bool) -> str:
    if not question.strip():
        return "Scrie o intrebare despre manual."
    return answer_policy_question(question, INDEX, use_llm=bool(use_llm))


with gr.Blocks(title="Asistent de Creditare RAG NovaTech") as demo:
    gr.Markdown(
        "# Asistent de Creditare RAG NovaTech\n"
        "Prototip educational: evalueaza un client fictiv folosind manualul NovaTech si afiseaza fragmentele recuperate din corpus."
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
                use_llm_analysis = gr.Checkbox(label="Foloseste rezumat LLM daca este configurat", value=False)

        with gr.Accordion("Date suplimentare pentru exceptii si cetateni non-UE", open=False):
            with gr.Row():
                is_non_eu = gr.Checkbox(label="Cetatean non-UE", value=False)
                married_to_ro_citizen = gr.Checkbox(label="Casatorit cu cetatean roman", value=False)
                owns_property_in_ro = gr.Checkbox(label="Detine proprietate in Romania", value=False)
                local_contract_months = gr.Number(label="Vechime contract local (luni)", value=0, precision=0)
            with gr.Row():
                sector = gr.Radio(label="Sector activitate", choices=["IT", "Altul"], value="Altul")
                current_job_tenure_months = gr.Number(label="Vechime job curent (luni)", value=12, precision=0)
                previous_job_tenure_months = gr.Number(label="Vechime job anterior (luni)", value=0, precision=0)
                gap_days_between_jobs = gr.Number(label="Pauza intre joburi (zile)", value=0, precision=0)

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
                is_non_eu,
                married_to_ro_citizen,
                owns_property_in_ro,
                local_contract_months,
                sector,
                current_job_tenure_months,
                previous_job_tenure_months,
                gap_days_between_jobs,
                use_llm_analysis,
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

    gr.Markdown(
        "Nota: proiect demonstrativ pentru disertatie. Manualul NovaTech este fictiv si rezultatele nu sunt consultanta financiara."
    )


if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860)
