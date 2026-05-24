from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from math import pow


GMI_LIMIT = 0.40
MIN_AGE = 21
MAX_AGE_AT_MATURITY = 70
MAX_TERM_MONTHS = 60
MIN_AMOUNT_RON = 5_000
MAX_AMOUNT_RON = 150_000

INCOME_WEIGHTS = {
    "Salariu - contract nedeterminat": 1.00,
    "Salariu - contract determinat": 0.80,
    "Pensie permanenta": 1.00,
    "PFA/PFI": 0.75,
    "Dividende": 0.60,
    "Chirii": 0.50,
    "Drepturi de autor": 0.70,
    "Diurne navigatori/aeronaval": 0.60,
    "Contract management/mandat": 0.85,
    "Venit exclus de manual": 0.00,
}


class Decision(str, Enum):
    APPROVED = "APROBAT"
    MANUAL_REVIEW = "ANALIZA MANUALA"
    REJECTED = "RESPINS"


@dataclass(frozen=True)
class ClientProfile:
    age: int
    term_months: int
    fico: int
    monthly_income: float
    income_type: str
    existing_monthly_debts: float
    requested_amount: float
    annual_interest_pct: float
    requested_monthly_payment: float = 0.0
    currency: str = "RON"
    income_currency: str = "RON"
    variable_rate: bool = False
    active_delay_days: int = 0
    historical_90_delay_last_year: bool = False
    historical_90_debt_settled: bool = False
    income_increase_after_delay_pct: float = 0.0
    is_pep: bool = False
    aml_risk: str = "Standard"
    is_non_eu: bool = False
    married_to_ro_citizen: bool = False
    owns_property_in_ro: bool = False
    local_contract_months: int = 0
    sector: str = "Altul"
    current_job_tenure_months: int = 12
    previous_job_tenure_months: int = 0
    gap_days_between_jobs: int = 0


@dataclass
class CreditEvaluation:
    decision: Decision
    weighted_income: float
    income_weight: float
    max_monthly_payment: float
    available_payment_capacity: float
    stressed_monthly_payment: float
    gmi: float
    max_credit_amount: float
    maturity_age: float
    reject_reasons: list[str] = field(default_factory=list)
    manual_review_reasons: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def annuity_payment(principal: float, annual_interest_pct: float, months: int) -> float:
    if principal <= 0 or months <= 0:
        return 0.0
    monthly_rate = annual_interest_pct / 100 / 12
    if monthly_rate == 0:
        return principal / months
    return principal * monthly_rate / (1 - pow(1 + monthly_rate, -months))


def principal_from_payment(payment: float, annual_interest_pct: float, months: int) -> float:
    if payment <= 0 or months <= 0:
        return 0.0
    monthly_rate = annual_interest_pct / 100 / 12
    if monthly_rate == 0:
        return payment * months
    return payment * (1 - pow(1 + monthly_rate, -months)) / monthly_rate


def evaluate_client(profile: ClientProfile) -> CreditEvaluation:
    reject_reasons: list[str] = []
    manual_reasons: list[str] = []
    warnings: list[str] = []

    income_weight = INCOME_WEIGHTS.get(profile.income_type, 0.0)
    weighted_income = profile.monthly_income * income_weight
    max_monthly_payment = weighted_income * GMI_LIMIT
    available_capacity = max_monthly_payment - profile.existing_monthly_debts
    maturity_age = profile.age + profile.term_months / 12

    stressed_interest = profile.annual_interest_pct + (2.0 if profile.variable_rate else 0.0)
    currency_stress = 1.15 if profile.currency == "EUR" and profile.income_currency == "RON" else 1.0
    requested_payment = (
        profile.requested_monthly_payment
        if profile.requested_monthly_payment > 0
        else annuity_payment(profile.requested_amount, stressed_interest, profile.term_months)
    )
    stressed_payment = requested_payment * currency_stress
    gmi = (
        (profile.existing_monthly_debts + stressed_payment) / weighted_income
        if weighted_income > 0
        else 999.0
    )
    max_payment_before_stress = max(0.0, available_capacity / currency_stress)
    max_credit_amount = min(
        MAX_AMOUNT_RON,
        principal_from_payment(max_payment_before_stress, stressed_interest, profile.term_months),
    )

    if profile.age < MIN_AGE:
        reject_reasons.append("Varsta minima acceptata este 21 de ani.")
    if maturity_age > MAX_AGE_AT_MATURITY:
        reject_reasons.append("Varsta la maturitatea creditului depaseste 70 de ani.")
    if profile.age > 62:
        warnings.append("Este necesara polita de asigurare de viata pentru clienti peste 62 de ani.")
    if profile.term_months > MAX_TERM_MONTHS:
        reject_reasons.append("Perioada maxima pentru NovaFlex este 60 de luni.")
    if profile.requested_amount > 0 and profile.requested_amount < MIN_AMOUNT_RON:
        reject_reasons.append("Suma minima finantata este 5.000 RON.")
    if profile.requested_amount > MAX_AMOUNT_RON:
        reject_reasons.append("Suma maxima finantata este 150.000 RON.")

    if profile.fico < 620:
        reject_reasons.append("FICO sub 620 intra in risc inacceptabil.")
    elif profile.fico < 650:
        manual_reasons.append("FICO intre 620 si 649 intra in zona Gray Zone.")

    if profile.active_delay_days > 30:
        reject_reasons.append("Intarzierile active peste 30 de zile duc la respingere automata.")
    elif profile.active_delay_days >= 16:
        manual_reasons.append("Intarzierile active de 16-30 zile cer justificari si verificare manuala.")
    elif profile.active_delay_days > 0:
        warnings.append("Intarzierile sub 15 zile sunt tratate ca tehnice.")

    if profile.historical_90_delay_last_year:
        exception_ok = (
            profile.historical_90_debt_settled
            and profile.income_increase_after_delay_pct >= 50
        )
        if exception_ok:
            manual_reasons.append(
                "Exista intarziere istorica peste 90 de zile, dar se aplica exceptia cu datorie stinsa si venit crescut cu minimum 50%."
            )
        else:
            reject_reasons.append(
                "Intarziere istorica peste 90 de zile in ultimul an fara exceptie documentata."
            )

    if profile.is_non_eu:
        non_eu_ok = (
            profile.married_to_ro_citizen
            and profile.owns_property_in_ro
            and profile.local_contract_months >= 24
        )
        if not non_eu_ok:
            reject_reasons.append(
                "Clientul non-UE nu indeplineste cumulativ conditiile de casatorie, proprietate in Romania si contract local de minimum 24 luni."
            )

    if income_weight == 0:
        reject_reasons.append("Tipul de venit are pondere 0% si nu poate sustine creditul.")
    if available_capacity <= 0:
        reject_reasons.append("Ratele existente consuma deja capacitatea maxima de indatorare.")

    if profile.currency == "EUR" and profile.income_currency == "RON":
        warnings.append("S-a aplicat stres valutar de 15% pentru credit EUR cu venituri in RON.")
    if profile.variable_rate:
        warnings.append("S-a aplicat soc de dobanda de +2 puncte procentuale.")

    if profile.is_pep:
        manual_reasons.append("Client PEP: manualul interzice aprobarea automata.")
    if profile.aml_risk == "Ridicat":
        manual_reasons.append("Risc AML ridicat: este necesar aviz de conformitate.")

    if (
        profile.sector == "IT"
        and profile.current_job_tenure_months >= 3
        and profile.previous_job_tenure_months >= 24
        and profile.gap_days_between_jobs <= 30
    ):
        warnings.append("Se potriveste exceptia E-3.1 pentru sectorul IT din scenariile de test.")

    if profile.requested_amount > 0 and profile.requested_amount > max_credit_amount:
        reject_reasons.append(
            "Suma solicitata depaseste capacitatea maxima calculata prin GMI."
        )
    if gmi > GMI_LIMIT:
        reject_reasons.append("GMI depaseste limita operationala de 40%.")

    if reject_reasons:
        decision = Decision.REJECTED
    elif manual_reasons:
        decision = Decision.MANUAL_REVIEW
    else:
        decision = Decision.APPROVED

    return CreditEvaluation(
        decision=decision,
        weighted_income=weighted_income,
        income_weight=income_weight,
        max_monthly_payment=max_monthly_payment,
        available_payment_capacity=available_capacity,
        stressed_monthly_payment=stressed_payment,
        gmi=gmi,
        max_credit_amount=max_credit_amount,
        maturity_age=maturity_age,
        reject_reasons=reject_reasons,
        manual_review_reasons=manual_reasons,
        warnings=warnings,
    )
