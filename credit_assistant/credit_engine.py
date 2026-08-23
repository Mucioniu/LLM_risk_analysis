from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from math import pow


DTI_LIMIT = 0.40
MIN_AGE = 21
MAX_AGE_AT_MATURITY = 70
MAX_TERM_MONTHS = 60
MIN_AMOUNT_RON = 5_000
MAX_AMOUNT_RON = 150_000

INCOME_WEIGHTS = {
    "Salary - permanent contract": 1.00,
    "Salary - fixed-term contract": 0.80,
    "Pension": 1.00,
    "Self-employment/liberal professions": 0.75,
    "Dividends": 0.60,
    "Rental income": 0.50,
    "Copyright royalties": 0.70,
    "Seafarer/aviation per diem": 0.60,
    "Management/mandate contract": 0.85,
    "Income excluded by policy manual": 0.00,
}


class Decision(str, Enum):
    APPROVED = "APPROVED"
    MANUAL_REVIEW = "MANUAL REVIEW"
    REJECTED = "REJECTED"


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
    sector: str = "Other"
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
    dti: float
    maximum_amount_by_dti: float
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
    max_monthly_payment = weighted_income * DTI_LIMIT
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
    dti = (
        (profile.existing_monthly_debts + stressed_payment) / weighted_income
        if weighted_income > 0
        else 999.0
    )
    max_payment_before_stress = max(0.0, available_capacity / currency_stress)
    maximum_amount_by_dti = min(
        MAX_AMOUNT_RON,
        principal_from_payment(max_payment_before_stress, stressed_interest, profile.term_months),
    )

    if profile.age < MIN_AGE:
        reject_reasons.append("The minimum accepted age is 21.")
    if maturity_age > MAX_AGE_AT_MATURITY:
        reject_reasons.append("Age at loan maturity exceeds 70.")
    if profile.age > 62:
        warnings.append("Life insurance is required for clients over 62.")
    if profile.term_months > MAX_TERM_MONTHS:
        reject_reasons.append("The maximum NovaFlex term is 60 months.")
    if profile.requested_amount > 0 and profile.requested_amount < MIN_AMOUNT_RON:
        reject_reasons.append("The minimum financed amount is RON 5,000.")
    if profile.requested_amount > MAX_AMOUNT_RON:
        reject_reasons.append("The maximum financed amount is RON 150,000.")

    if profile.fico < 620:
        reject_reasons.append("A FICO score below 620 is an unacceptable risk.")
    elif profile.fico < 650:
        manual_reasons.append("A FICO score from 620 to 649 falls within the Gray Zone.")

    if profile.active_delay_days > 30:
        reject_reasons.append("Active delinquencies over 30 days result in automatic rejection.")
    elif profile.active_delay_days >= 16:
        manual_reasons.append("Active delinquencies of 16-30 days require justification and manual review.")
    elif profile.active_delay_days > 0:
        warnings.append("Delinquencies of up to 15 days are treated as technical delays.")

    if profile.historical_90_delay_last_year:
        exception_ok = (
            profile.historical_90_debt_settled
            and profile.income_increase_after_delay_pct >= 50
        )
        if exception_ok:
            manual_reasons.append(
                "A historical delinquency over 90 days exists, but the exception applies because the debt was settled and income increased by at least 50%."
            )
        else:
            reject_reasons.append(
                "A historical delinquency over 90 days occurred in the past year without a documented exception."
            )

    if profile.is_non_eu:
        non_eu_ok = (
            profile.married_to_ro_citizen
            and profile.owns_property_in_ro
            and profile.local_contract_months >= 24
        )
        if not non_eu_ok:
            reject_reasons.append(
                "The non-EU client does not meet all requirements for marriage to a Romanian citizen, property ownership in Romania, and a local contract of at least 24 months."
            )

    if income_weight == 0:
        reject_reasons.append("This income type has a 0% weight and cannot support the loan.")
    if available_capacity <= 0:
        reject_reasons.append("Existing payments already exhaust the maximum debt-service capacity.")

    if profile.currency == "EUR" and profile.income_currency == "RON":
        warnings.append("A 15% currency stress was applied to the EUR loan with income in RON.")
    if profile.variable_rate:
        warnings.append("An interest-rate shock of +2 percentage points was applied.")

    if profile.is_pep:
        manual_reasons.append("PEP client: the policy manual prohibits automatic approval.")
    if profile.aml_risk == "High":
        manual_reasons.append("High AML risk: compliance approval is required.")

    if (
        profile.sector == "IT"
        and profile.current_job_tenure_months >= 3
        and profile.previous_job_tenure_months >= 24
        and profile.gap_days_between_jobs <= 30
    ):
        warnings.append("The E-3.1 exception for the IT sector in the test scenarios applies.")

    if profile.requested_amount > 0 and profile.requested_amount > maximum_amount_by_dti:
        reject_reasons.append(
            "The requested amount exceeds the maximum capacity calculated using DTI."
        )
    if dti > DTI_LIMIT:
        reject_reasons.append("DTI exceeds the 40% operating limit.")

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
        dti=dti,
        maximum_amount_by_dti=maximum_amount_by_dti,
        maturity_age=maturity_age,
        reject_reasons=reject_reasons,
        manual_review_reasons=manual_reasons,
        warnings=warnings,
    )
