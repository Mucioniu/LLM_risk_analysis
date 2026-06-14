from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from .credit_engine import (
    GMI_LIMIT,
    INCOME_WEIGHTS,
    MAX_AGE_AT_MATURITY,
    MAX_AMOUNT_RON,
    MAX_TERM_MONTHS,
    MIN_AGE,
    MIN_AMOUNT_RON,
    ClientProfile,
    CreditEvaluation,
    evaluate_client,
)
from .llm import optional_llm_summary
from .rag import RagIndex, format_sources


DEFAULT_PDF = Path("Manual_Extins_Creditare_NovaTech_v3.pdf")
BNR_REGULATION_MD = Path("Regulamentul_BNR_nr_17_2012.md")


@dataclass(frozen=True)
class LlmExtractedDecision:
    decision: str | None
    declared_income: float | None
    income_weight_pct: float | None
    weighted_income: float | None
    max_monthly_payment: float | None
    existing_monthly_debts: float | None
    available_payment_capacity: float | None
    stressed_monthly_payment: float | None
    gmi_pct: float | None
    maturity_age: float | None
    max_credit_amount: float | None


@dataclass(frozen=True)
class LlmCreditAnalysis:
    answer_markdown: str
    comparison_markdown: str
    deterministic: CreditEvaluation
    extracted: LlmExtractedDecision
    metric_scores: dict[str, float]


REQUIRED_JSON_NUMERIC_FIELDS = {
    "declared_income",
    "income_weight_pct",
    "weighted_income",
    "max_monthly_payment",
    "existing_monthly_debts",
    "available_payment_capacity",
    "analyzed_monthly_payment",
    "stressed_monthly_payment",
    "gmi_pct",
    "maturity_age",
    "max_credit_amount",
    "product_cap",
}


def default_corpus_paths() -> list[Path]:
    paths: list[Path] = []
    if DEFAULT_PDF.exists():
        paths.append(DEFAULT_PDF)

    if BNR_REGULATION_MD.exists():
        paths.append(BNR_REGULATION_MD)

    if not paths:
        raise FileNotFoundError("Nu gasesc documente de creditare in directorul proiectului.")
    return paths


def build_default_index() -> RagIndex:
    return RagIndex.from_paths(default_corpus_paths())


def format_sources_markdown(sources: str) -> str:
    if sources.startswith("Nu am gasit"):
        return sources

    blocks: list[str] = []
    for block in sources.split("\n\n"):
        lines = [line.strip() for line in block.splitlines() if line.strip()]
        if not lines:
            continue
        title = lines[0]
        body = " ".join(lines[1:]).strip()
        if body:
            blocks.append(f"{title}  \n{body}")
        else:
            blocks.append(title)
    return "\n\n".join(blocks)


def normalize_credit_markdown(text: str) -> str:
    section_titles = [
        "Calcul financiar",
        "Detalii calcul",
        "Motive de respingere",
        "Motive de analiza manuala",
        "Observatii",
        "Surse RAG folosite",
    ]
    financial_labels = [
        "Venit declarat",
        "Pondere venit",
        "Venit eligibil ponderat",
        "Capacitate maxima totala rate (40% GMI)",
        "Rate existente",
        "Capacitate disponibila pentru rata noua",
        "Rata noua analizata, dupa stres daca se aplica",
        "GMI rezultat",
        "Varsta la maturitate",
        "Suma maxima recomandata prin GMI si plafon produs",
    ]

    normalized = text.strip()
    normalized = normalize_tabular_text(normalized)
    normalized = re.sub(
        r"(?im)^\s*#{0,3}\s*Decizie\s*[:|-]?\s*$\s*^(APROBAT|APROBAT|RESPINS|ANALIZA\s+MANUALA)\s*$",
        lambda match: f"## Decizie: {normalize_decision(match.group(1))}",
        normalized,
    )
    normalized = re.sub(
        r"(?im)^#{0,3}\s*Decizie\s*:\s*(APROBAT|APROBAT|RESPINS|ANALIZA\s+MANUALA)\s*$",
        lambda match: f"## Decizie: {normalize_decision(match.group(1))}",
        normalized,
    )
    for title in section_titles:
        normalized = re.sub(
            rf"(?im)^\s*#{{0,3}}\s*{re.escape(title)}\s*$",
            f"\n\n### {title}\n\n",
            normalized,
        )
    for label in financial_labels:
        normalized = re.sub(rf"\s+({re.escape(label)}:)", rf"\n- \1", normalized)

    normalized = re.sub(r"(?<!\n)\s+-\s+", "\n- ", normalized)
    normalized = re.sub(r"(#{2,3} [^\n]+)\n(?!\n)", r"\1\n\n", normalized)
    normalized = re.sub(r"\n{3,}", "\n\n", normalized)
    return normalized.strip()


def normalize_tabular_text(text: str) -> str:
    """Convert simple TSV-style LLM tables into Markdown tables."""
    lines = text.splitlines()
    normalized: list[str] = []
    table_rows: list[list[str]] = []

    def flush_table() -> None:
        nonlocal table_rows
        if not table_rows:
            return
        normalized.append("| Indicator | Valoare |")
        normalized.append("|---|---:|")
        rows = table_rows[1:] if table_rows[0][0].lower() in {"eticheta", "indicator"} else table_rows
        for row in rows:
            if len(row) >= 2:
                normalized.append(f"| {row[0]} | {row[1]} |")
        table_rows = []

    def append_markdown_rows(rows: list[list[str]]) -> None:
        normalized.append("| Indicator | Valoare |")
        normalized.append("|---|---:|")
        for row in rows:
            if len(row) >= 2:
                normalized.append(f"| {row[0]} | {row[1]} |")

    for raw_line in lines:
        line = raw_line.rstrip()
        if "\t" in line:
            cells = [cell.strip() for cell in line.split("\t") if cell.strip()]
            if len(cells) >= 2:
                table_rows.append(cells[:2])
                continue
        if "|" in line and line.count("|") >= 4:
            cells = [cell.strip() for cell in line.strip("|").split("|") if cell.strip()]
            if len(cells) >= 4:
                paired_rows = [cells[index : index + 2] for index in range(0, len(cells) - 1, 2)]
                if all(len(row) == 2 for row in paired_rows):
                    flush_table()
                    append_markdown_rows(paired_rows)
                    continue
        if normalize_label(line) == "dupa stres daca se aplica":
            continue
        flush_table()
        normalized.append(line)
    flush_table()
    return "\n".join(normalized)


def normalize_decision(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = re.sub(r"\s+", " ", value.upper()).strip()
    if normalized in {"APROBAT", "RESPINS", "ANALIZA MANUALA"}:
        return normalized
    return None


def credit_query(profile: ClientProfile | None = None) -> str:
    query = (
        "criterii eligibilitate varsta FICO istoric creditare venituri haircuts "
        "grad maxim indatorare GMI formula produs NovaFlex suma maxima credit"
    )
    if profile is None:
        return query

    if profile.is_pep:
        query += " PEP persoana expusa public aprobarea automata interzisa analiza manuala"
    if profile.aml_risk == "Ridicat":
        query += " risc AML ridicat conformitate analiza manuala"
    if profile.fico < 650:
        query += " FICO sub 620 risc inacceptabil FICO 620 649 Gray Zone analiza manuala"
    if profile.active_delay_days > 0 or profile.historical_90_delay_last_year:
        query += " intarzieri active istoric peste 90 zile respingere exceptii"
    return query


def retrieve_credit_sources(index: RagIndex, profile: ClientProfile | None = None) -> str:
    retrieved = index.search(credit_query(profile), top_k=5)
    sources = format_sources(retrieved, max_chars=650)
    return format_sources_markdown(sources)


def profile_as_prompt_json(profile: ClientProfile) -> str:
    return json.dumps(
        {
            "varsta": profile.age,
            "durata_credit_luni": profile.term_months,
            "fico": profile.fico,
            "venit_lunar_declarat_ron": profile.monthly_income,
            "tip_venit": profile.income_type,
            "rate_existente_lunare_ron": profile.existing_monthly_debts,
            "suma_solicitata_ron": profile.requested_amount,
            "rata_lunara_dorita_ron": profile.requested_monthly_payment,
            "dobanda_anuala_pct": profile.annual_interest_pct,
            "moneda_credit": profile.currency,
            "moneda_venit": profile.income_currency,
            "dobanda_variabila": profile.variable_rate,
            "zile_intarziere_activa": profile.active_delay_days,
            "intarziere_istorica_90_zile_ultimul_an": profile.historical_90_delay_last_year,
            "datoria_istorica_a_fost_stinsa": profile.historical_90_debt_settled,
            "crestere_venit_dupa_intarziere_pct": profile.income_increase_after_delay_pct,
            "client_pep": profile.is_pep,
            "risc_aml": profile.aml_risk,
            "client_non_ue": profile.is_non_eu,
            "casatorit_cu_cetatean_roman": profile.married_to_ro_citizen,
            "detine_proprietate_in_romania": profile.owns_property_in_ro,
            "contract_local_luni": profile.local_contract_months,
            "sector": profile.sector,
            "vechime_job_curent_luni": profile.current_job_tenure_months,
            "vechime_job_anterior_luni": profile.previous_job_tenure_months,
            "pauza_intre_joburi_zile": profile.gap_days_between_jobs,
        },
        ensure_ascii=False,
        indent=2,
    )


def operating_rules_prompt() -> str:
    weights = "\n".join(
        f"- {income_type}: {weight * 100:.0f}%"
        for income_type, weight in INCOME_WEIGHTS.items()
    )
    return (
        "Reguli numerice obligatorii pentru experiment:\n"
        f"- Varsta minima: {MIN_AGE} ani.\n"
        f"- Varsta maxima la maturitate: {MAX_AGE_AT_MATURITY} ani.\n"
        f"- Durata maxima: {MAX_TERM_MONTHS} luni.\n"
        f"- Suma minima: {MIN_AMOUNT_RON:,.0f} RON.\n"
        f"- Suma maxima produs: {MAX_AMOUNT_RON:,.0f} RON.\n"
        f"- Limita GMI: {GMI_LIMIT * 100:.0f}%.\n"
        "- Varsta la maturitate = varsta + durata_credit_luni / 12.\n"
        "- Daca tip_venit este exact 'Salariu - contract nedeterminat', ponderea este 100%, nu 85%.\n"
        "- Ponderi venituri acceptate:\n"
        f"{weights}\n"
        "- Venit eligibil ponderat = venit_lunar_declarat_ron * pondere_venit.\n"
        "- Capacitate maxima totala rate = venit_eligibil_ponderat * limita_GMI.\n"
        "- Capacitate disponibila pentru rata noua = capacitate_maxima_totala_rate - rate_existente_lunare_ron.\n"
        "- Daca rata_lunara_dorita_ron este 0, calculeaza rata lunara cu formula anuitatii: "
        "P * r / (1 - (1 + r)^(-n)), unde r = dobanda_anuala_pct / 100 / 12 si n = durata_credit_luni.\n"
        "- GMI rezultat este procent, nu suma in RON: "
        "(rate_existente_lunare_ron + rata_noua_dupa_stres) / venit_eligibil_ponderat * 100.\n"
        "- Suma maxima recomandata prin GMI = capacitate_disponibila_pentru_rata_noua * "
        "(1 - (1 + r)^(-n)) / r, limitata la suma maxima produs. Nu o confunda cu suma solicitata.\n"
        "- Daca suma_solicitata_ron este mai mica sau egala cu suma maxima recomandata si plafonul produs, nu respinge pentru suma.\n"
        "- Daca durata_credit_luni este mai mica sau egala cu durata maxima, nu respinge pentru durata.\n"
        "- Pentru dobanda variabila adauga +2 puncte procentuale la dobanda.\n"
        "- Pentru credit EUR cu venit RON inmulteste rata analizata cu 1.15.\n"
        "- Decizia trebuie sa fie exact una dintre: APROBAT, RESPINS, ANALIZA MANUALA.\n"
        "- Daca exista orice motiv de respingere, decizia este RESPINS. "
        "Altfel, daca exista motiv de analiza manuala, decizia este ANALIZA MANUALA. "
        "Altfel, decizia este APROBAT.\n"
        "- FICO sub 620 inseamna RESPINS; FICO 620-649 inseamna ANALIZA MANUALA.\n"
        "- Client PEP sau risc AML Ridicat inseamna ANALIZA MANUALA daca nu exista motive de respingere.\n"
    )


def critical_profile_checks_prompt(profile: ClientProfile) -> str:
    fico_outcome = "nu cere actiune speciala"
    if profile.fico < 620:
        fico_outcome = "RESPINS obligatoriu, deoarece FICO este sub 620"
    elif profile.fico < 650:
        fico_outcome = "ANALIZA MANUALA obligatorie, deoarece FICO este intre 620 si 649"

    pep_outcome = (
        "ANALIZA MANUALA obligatorie daca nu exista motive de respingere"
        if profile.is_pep
        else "nu cere actiune speciala"
    )
    aml_outcome = (
        "ANALIZA MANUALA obligatorie daca nu exista motive de respingere"
        if profile.aml_risk == "Ridicat"
        else "nu cere actiune speciala"
    )
    return (
        "Checklist critic pentru acest profil. Aceste reguli au prioritate inaintea calculelor financiare:\n"
        f"- FICO profil: {profile.fico}. Consecinta: {fico_outcome}.\n"
        f"- Client PEP: {'DA' if profile.is_pep else 'NU'}. Consecinta: {pep_outcome}.\n"
        f"- Risc AML: {profile.aml_risk}. Consecinta: {aml_outcome}.\n"
        "- Nu transforma un motiv de analiza manuala in motiv de respingere.\n"
        "- Nu transforma un motiv de respingere in analiza manuala sau aprobare.\n"
        "- O regula marcata RESPINS are prioritate peste ANALIZA MANUALA si APROBAT.\n"
    )


def credit_json_schema_prompt() -> str:
    return (
        "Returneaza exclusiv un obiect JSON valid, fara Markdown, fara explicatii in afara JSON-ului. "
        "Raspunsul trebuie sa fie compact si complet; inchide toate listele si acoladele. "
        "Schema obligatorie este:\n"
        "{\n"
        '  "decision": "APROBAT | RESPINS | ANALIZA MANUALA",\n'
        '  "financial": {\n'
        '    "declared_income": number,\n'
        '    "income_weight_pct": number,\n'
        '    "weighted_income": number,\n'
        '    "max_monthly_payment": number,\n'
        '    "existing_monthly_debts": number,\n'
        '    "available_payment_capacity": number,\n'
        '    "analyzed_monthly_payment": number,\n'
        '    "stressed_monthly_payment": number,\n'
        '    "gmi_pct": number,\n'
        '    "maturity_age": number,\n'
        '    "max_credit_amount": number,\n'
        '    "product_cap": number\n'
        "  },\n"
        '  "calculation_details": ["string"],\n'
        '  "rejection_reasons": ["string"],\n'
        '  "manual_review_reasons": ["string"],\n'
        '  "observations": ["string"],\n'
        '  "rag_sources": ["[1] ...", "[2] ..."]\n'
        "}\n"
        "Toate campurile numerice trebuie sa fie numere JSON, nu stringuri cu RON sau %. "
        "Daca nu exista motive intr-o lista, foloseste lista goala []. "
        "calculation_details trebuie sa aiba maximum 4 elemente scurte. "
        "rejection_reasons, manual_review_reasons si observations trebuie sa aiba maximum 3 elemente fiecare. "
        "rag_sources trebuie sa contina doar referinte scurte de forma [1] fisier, fara fragmente lungi."
    )


def extract_json_object(text: str | None) -> dict[str, object] | None:
    if not text:
        return None
    cleaned = text.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", cleaned, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        cleaned = fenced.group(1)
    else:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return None
        cleaned = cleaned[start : end + 1]
    try:
        loaded = json.loads(cleaned)
    except json.JSONDecodeError:
        return None
    return loaded if isinstance(loaded, dict) else None


def _as_float(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        return _parse_number(value)
    return None


def _as_string_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def validate_llm_credit_json(
    data: dict[str, object] | None,
    profile: ClientProfile,
    deterministic: CreditEvaluation,
) -> list[str]:
    errors: list[str] = []
    if data is None:
        return ["Raspunsul nu contine un obiect JSON valid."]

    decision = normalize_decision(str(data.get("decision", "")))
    if decision is None:
        errors.append("Campul decision lipseste sau nu este una dintre valorile permise.")
    elif decision != deterministic.decision.value:
        errors.append(
            "Campul decision trebuie sa fie "
            f"{deterministic.decision.value}, conform regulilor critice si formulelor validate."
        )

    financial = data.get("financial")
    if not isinstance(financial, dict):
        errors.append("Campul financial trebuie sa fie obiect JSON.")
        financial = {}

    for field in sorted(REQUIRED_JSON_NUMERIC_FIELDS):
        value = _as_float(financial.get(field))
        if value is None:
            errors.append(f"Campul financial.{field} lipseste sau nu este numeric.")

    expected_numeric = {
        "declared_income": (profile.monthly_income, 0.01),
        "income_weight_pct": (deterministic.income_weight * 100, 0.01),
        "weighted_income": (deterministic.weighted_income, 1.0),
        "max_monthly_payment": (deterministic.max_monthly_payment, 1.0),
        "existing_monthly_debts": (profile.existing_monthly_debts, 0.01),
        "available_payment_capacity": (deterministic.available_payment_capacity, 1.0),
        "stressed_monthly_payment": (deterministic.stressed_monthly_payment, 1.0),
        "gmi_pct": (deterministic.gmi * 100, 0.05),
        "maturity_age": (deterministic.maturity_age, 0.1),
        "max_credit_amount": (deterministic.max_credit_amount, 1.0),
        "product_cap": (MAX_AMOUNT_RON, 0.01),
    }
    analyzed_expected = deterministic.stressed_monthly_payment
    if profile.currency == "EUR" and profile.income_currency == "RON":
        analyzed_expected = deterministic.stressed_monthly_payment / 1.15
    expected_numeric["analyzed_monthly_payment"] = (analyzed_expected, 1.0)

    for field, (expected, tolerance) in expected_numeric.items():
        actual = _as_float(financial.get(field))
        if actual is not None and abs(actual - expected) > tolerance:
            errors.append(
                f"Campul financial.{field} trebuie sa fie {expected:.2f}, "
                f"dar modelul a returnat {actual:.2f}."
            )

    gmi_pct = _as_float(financial.get("gmi_pct"))
    if gmi_pct is not None and (gmi_pct < 0 or gmi_pct > 1000):
        errors.append("Campul financial.gmi_pct trebuie sa fie procent, nu suma in RON.")

    if profile.fico < 620 and decision != "RESPINS":
        errors.append("FICO sub 620 impune decizia RESPINS.")
    elif 620 <= profile.fico < 650 and deterministic.decision.value != "RESPINS" and decision != "ANALIZA MANUALA":
        errors.append("FICO intre 620 si 649 impune ANALIZA MANUALA daca nu exista respingere.")

    if profile.is_pep and deterministic.decision.value != "RESPINS" and decision != "ANALIZA MANUALA":
        errors.append("Client PEP impune ANALIZA MANUALA daca nu exista respingere.")
    if profile.aml_risk == "Ridicat" and deterministic.decision.value != "RESPINS" and decision != "ANALIZA MANUALA":
        errors.append("Risc AML ridicat impune ANALIZA MANUALA daca nu exista respingere.")

    for field in ["calculation_details", "rejection_reasons", "manual_review_reasons", "observations", "rag_sources"]:
        if not isinstance(data.get(field), list):
            errors.append(f"Campul {field} trebuie sa fie lista.")

    rejection_reasons = _as_string_list(data.get("rejection_reasons"))
    manual_reasons = _as_string_list(data.get("manual_review_reasons"))
    rejection_text = " ".join(rejection_reasons).lower()
    manual_text = " ".join(manual_reasons).lower()

    if deterministic.reject_reasons and not rejection_reasons:
        errors.append("Lista rejection_reasons trebuie sa includa motivele de respingere calculate.")
    if not deterministic.reject_reasons and rejection_reasons:
        errors.append("Lista rejection_reasons trebuie sa fie goala; formulele nu indica respingere.")
    if deterministic.manual_review_reasons and not manual_reasons:
        errors.append("Lista manual_review_reasons trebuie sa includa motivele de analiza manuala calculate.")
    if not deterministic.manual_review_reasons and manual_reasons:
        errors.append("Lista manual_review_reasons trebuie sa fie goala; regulile nu indica analiza manuala.")

    for reason in deterministic.reject_reasons:
        normalized_reason = reason.lower()
        if "gmi" in normalized_reason and "gmi" not in rejection_text:
            errors.append("rejection_reasons trebuie sa mentioneze depasirea limitei GMI.")
        if "fico" in normalized_reason and "fico" not in rejection_text:
            errors.append("rejection_reasons trebuie sa mentioneze FICO sub limita.")
        if "suma maxima" in normalized_reason and "suma" not in rejection_text:
            errors.append("rejection_reasons trebuie sa mentioneze depasirea sumei maxime.")
        if "varsta" in normalized_reason and "varsta" not in rejection_text:
            errors.append("rejection_reasons trebuie sa mentioneze varsta la maturitate.")

    for reason in deterministic.manual_review_reasons:
        normalized_reason = reason.lower()
        if "pep" in normalized_reason and "pep" not in manual_text:
            errors.append("manual_review_reasons trebuie sa mentioneze clientul PEP.")
        if "aml" in normalized_reason and "aml" not in manual_text:
            errors.append("manual_review_reasons trebuie sa mentioneze riscul AML.")
        if "fico" in normalized_reason and "fico" not in manual_text:
            errors.append("manual_review_reasons trebuie sa mentioneze zona FICO de analiza manuala.")

    return errors


def deterministic_financial_json(profile: ClientProfile, evaluation: CreditEvaluation) -> dict[str, float]:
    analyzed_payment = evaluation.stressed_monthly_payment
    if profile.currency == "EUR" and profile.income_currency == "RON":
        analyzed_payment = evaluation.stressed_monthly_payment / 1.15
    return {
        "declared_income": profile.monthly_income,
        "income_weight_pct": evaluation.income_weight * 100,
        "weighted_income": evaluation.weighted_income,
        "max_monthly_payment": evaluation.max_monthly_payment,
        "existing_monthly_debts": profile.existing_monthly_debts,
        "available_payment_capacity": evaluation.available_payment_capacity,
        "analyzed_monthly_payment": analyzed_payment,
        "stressed_monthly_payment": evaluation.stressed_monthly_payment,
        "gmi_pct": evaluation.gmi * 100,
        "maturity_age": evaluation.maturity_age,
        "max_credit_amount": evaluation.max_credit_amount,
        "product_cap": MAX_AMOUNT_RON,
    }


def deterministic_values_prompt(profile: ClientProfile, evaluation: CreditEvaluation) -> str:
    return json.dumps(
        {
            "decision": evaluation.decision.value,
            "financial": deterministic_financial_json(profile, evaluation),
            "rejection_reasons": evaluation.reject_reasons,
            "manual_review_reasons": evaluation.manual_review_reasons,
            "observations": evaluation.warnings,
        },
        ensure_ascii=False,
        indent=2,
    )


def repair_llm_credit_json(
    data: dict[str, object] | None,
    profile: ClientProfile,
    evaluation: CreditEvaluation,
) -> dict[str, object]:
    repaired: dict[str, object] = dict(data or {})
    repaired["decision"] = evaluation.decision.value
    financial = dict(repaired.get("financial") if isinstance(repaired.get("financial"), dict) else {})
    financial.update(deterministic_financial_json(profile, evaluation))
    repaired["financial"] = financial
    repaired["rejection_reasons"] = list(evaluation.reject_reasons)
    repaired["manual_review_reasons"] = list(evaluation.manual_review_reasons)
    if not isinstance(repaired.get("calculation_details"), list) or not repaired.get("calculation_details"):
        repaired["calculation_details"] = [
            f"Venit eligibil ponderat = {profile.monthly_income:,.2f} * {evaluation.income_weight * 100:.0f}% = {evaluation.weighted_income:,.2f} RON.",
            f"Capacitate maxima totala rate = {evaluation.weighted_income:,.2f} * {GMI_LIMIT * 100:.0f}% = {evaluation.max_monthly_payment:,.2f} RON.",
            f"GMI = ({profile.existing_monthly_debts:,.2f} + {evaluation.stressed_monthly_payment:,.2f}) / {evaluation.weighted_income:,.2f} * 100 = {evaluation.gmi * 100:.2f}%.",
            f"Varsta la maturitate = {profile.age} + {profile.term_months}/12 = {evaluation.maturity_age:.1f} ani.",
        ]
    if not isinstance(repaired.get("observations"), list):
        repaired["observations"] = list(evaluation.warnings)
    if not isinstance(repaired.get("rag_sources"), list):
        repaired["rag_sources"] = []
    return repaired


def llm_json_to_extracted(data: dict[str, object]) -> LlmExtractedDecision:
    financial = data.get("financial")
    if not isinstance(financial, dict):
        financial = {}
    return LlmExtractedDecision(
        decision=normalize_decision(str(data.get("decision", ""))),
        declared_income=_as_float(financial.get("declared_income")),
        income_weight_pct=_as_float(financial.get("income_weight_pct")),
        weighted_income=_as_float(financial.get("weighted_income")),
        max_monthly_payment=_as_float(financial.get("max_monthly_payment")),
        existing_monthly_debts=_as_float(financial.get("existing_monthly_debts")),
        available_payment_capacity=_as_float(financial.get("available_payment_capacity")),
        stressed_monthly_payment=_as_float(financial.get("stressed_monthly_payment")),
        gmi_pct=_as_float(financial.get("gmi_pct")),
        maturity_age=_as_float(financial.get("maturity_age")),
        max_credit_amount=_as_float(financial.get("max_credit_amount")),
    )


def format_llm_credit_json_markdown(data: dict[str, object], validation_notes: list[str] | None = None) -> str:
    financial = data.get("financial")
    if not isinstance(financial, dict):
        financial = {}
    decision = normalize_decision(str(data.get("decision", ""))) or "NEVALIDAT"

    def money(field: str) -> str:
        value = _as_float(financial.get(field))
        return "negasit" if value is None else f"{value:,.2f} RON"

    def pct(field: str) -> str:
        value = _as_float(financial.get(field))
        return "negasit" if value is None else f"{value:.2f}%"

    def num(field: str, suffix: str = "") -> str:
        value = _as_float(financial.get(field))
        return "negasit" if value is None else f"{value:,.2f}{suffix}"

    details = _as_string_list(data.get("calculation_details"))
    rejection = _as_string_list(data.get("rejection_reasons"))
    manual = _as_string_list(data.get("manual_review_reasons"))
    observations = _as_string_list(data.get("observations"))
    sources = _as_string_list(data.get("rag_sources"))

    lines = [
        f"## Decizie: {decision}",
        "",
        "### Calcul financiar",
        "",
        "| Indicator | Valoare |",
        "|---|---:|",
        f"| Venit declarat | {money('declared_income')} |",
        f"| Pondere venit | {pct('income_weight_pct')} |",
        f"| Venit eligibil ponderat | {money('weighted_income')} |",
        f"| Capacitate maxima totala rate (40% GMI) | {money('max_monthly_payment')} |",
        f"| Rate existente | {money('existing_monthly_debts')} |",
        f"| Capacitate disponibila pentru rata noua | {money('available_payment_capacity')} |",
        f"| Rata noua analizata | {money('analyzed_monthly_payment')} |",
        f"| Rata noua analizata, dupa stres daca se aplica | {money('stressed_monthly_payment')} |",
        f"| GMI rezultat | {pct('gmi_pct')} |",
        f"| Varsta la maturitate | {num('maturity_age', ' ani')} |",
        f"| Suma maxima recomandata prin GMI si plafon produs | {money('max_credit_amount')} |",
        f"| Plafon produs | {money('product_cap')} |",
        "",
        "### Detalii calcul",
        "",
        "\n".join(f"- {item}" for item in details) or "- Nu exista detalii suplimentare.",
        "",
        "### Motive de respingere",
        "",
        "\n".join(f"- {item}" for item in rejection) or "- Nu exista.",
        "",
        "### Motive de analiza manuala",
        "",
        "\n".join(f"- {item}" for item in manual) or "- Nu este necesara.",
        "",
        "### Observatii",
        "",
        "\n".join(f"- {item}" for item in observations) or "- Nu exista.",
        "",
        "### Surse RAG folosite",
        "",
        "\n".join(sources) or "- Nu au fost indicate surse.",
    ]
    if validation_notes:
        lines.extend(
            [
                "",
                "### Note validare schema",
                "",
                "\n".join(f"- {note}" for note in validation_notes),
            ]
        )
    return "\n".join(lines)


def _parse_number(value: str | None) -> float | None:
    if not value:
        return None
    cleaned = value.strip().replace("\u00a0", " ")
    if "=" in cleaned:
        cleaned = cleaned.rsplit("=", 1)[-1]
    unit_match = re.search(r"-?\d[\d\s.,]*\s*(?:RON|%|ani)\b", cleaned, flags=re.IGNORECASE)
    token_match = unit_match or re.search(r"-?\d[\d\s.,]*", cleaned)
    if not token_match:
        return None
    cleaned = token_match.group(0)
    cleaned = cleaned.replace("\u00a0", " ")
    cleaned = re.sub(r"\s+", "", cleaned)
    cleaned = cleaned.replace("%", "")
    cleaned = re.sub(r"[^0-9,.\-]", "", cleaned)
    if not cleaned:
        return None
    if "," in cleaned and "." in cleaned:
        cleaned = cleaned.replace(",", "")
    elif "," in cleaned:
        if re.fullmatch(r"-?\d{1,3}(,\d{3})+", cleaned):
            cleaned = cleaned.replace(",", "")
        else:
            cleaned = cleaned.replace(",", ".")
    elif re.fullmatch(r"-?\d{1,3}(\.\d{3})+", cleaned):
        cleaned = cleaned.replace(".", "")
    try:
        return float(cleaned)
    except ValueError:
        return None


def _extract_after_labels(text: str, labels: list[str]) -> float | None:
    rows = extract_label_rows(text)
    for label in labels:
        normalized_label = normalize_label(label)
        for row_label, value in rows:
            if normalized_label in normalize_label(row_label):
                parsed = _parse_number(value)
                if parsed is not None:
                    return parsed

        pattern = rf"{re.escape(label)}(?:\s*\([^)]*\))*\s*[:|]\s*([0-9][0-9\s.,%-]*)"
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            parsed = _parse_number(match.group(1))
            if parsed is not None:
                return parsed
    return None


def normalize_label(label: str) -> str:
    normalized = label.lower()
    normalized = re.sub(r"\([^)]*\)", " ", normalized)
    normalized = re.sub(r"[^a-z0-9%]+", " ", normalized)
    return re.sub(r"\s+", " ", normalized).strip()


def extract_label_rows(text: str) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        if "|" in line:
            cells = [cell.strip() for cell in line.strip("|").split("|") if cell.strip()]
            if not cells or all(set(cell) <= {"-", ":"} for cell in cells):
                continue
            if len(cells) >= 2 and cells[0].lower() in {"indicator", "eticheta"}:
                continue
            if len(cells) >= 4:
                for index in range(0, len(cells) - 1, 2):
                    rows.append((cells[index], cells[index + 1]))
                continue
            if len(cells) >= 2:
                rows.append((cells[0], cells[1]))
            continue

        if "\t" in line:
            cells = [cell.strip() for cell in line.split("\t") if cell.strip()]
            if len(cells) >= 2:
                rows.append((cells[0], cells[1]))
            continue

        if ":" in line:
            label, value = line.split(":", 1)
            rows.append((label.strip(), value.strip()))
    return rows


def extract_llm_decision(text: str) -> LlmExtractedDecision:
    decision = extract_decision_label(text)

    return LlmExtractedDecision(
        decision=decision,
        declared_income=_extract_after_labels(text, ["Venit declarat"]),
        income_weight_pct=_extract_after_labels(text, ["Pondere venit"]),
        weighted_income=_extract_after_labels(text, ["Venit eligibil ponderat"]),
        max_monthly_payment=_extract_after_labels(text, ["Capacitate maxima totala rate (40% GMI)"]),
        existing_monthly_debts=_extract_after_labels(text, ["Rate existente"]),
        available_payment_capacity=_extract_after_labels(text, ["Capacitate disponibila pentru rata noua"]),
        stressed_monthly_payment=_extract_after_labels(
            text,
            [
                "Rata noua analizata, dupa stres daca se aplica",
                "Rata noua dupa stres",
                "Rata noua analizata",
            ],
        ),
        gmi_pct=_extract_after_labels(text, ["GMI rezultat"]),
        maturity_age=_extract_after_labels(text, ["Varsta la maturitate"]),
        max_credit_amount=_extract_after_labels(
            text,
            [
                "Suma maxima recomandata prin GMI si plafon produs",
                "Suma maxima recomandata prin GMI",
                "Suma maxima recomandata",
                "plafon produs",
            ],
        ),
    )


def extract_decision_label(text: str) -> str | None:
    decision_match = re.search(
        r"Decizie\s*[:|]\s*(APROBAT|APROBAT|RESPINS|ANALIZA\s+MANUALA)",
        text,
        flags=re.IGNORECASE,
    )
    if decision_match:
        return normalize_decision(decision_match.group(1))

    lines = [line.strip(" #|\t") for line in text.splitlines()]
    for index, line in enumerate(lines):
        if line.lower() == "decizie":
            for next_line in lines[index + 1 :]:
                if not next_line:
                    continue
                decision = normalize_decision(next_line)
                if decision:
                    return decision
                break
    return None


def _score_text_match(actual: str | None, expected: str) -> float:
    return 1.0 if actual and actual.upper() == expected.upper() else 0.0


def _score_number(actual: float | None, expected: float, tolerance: float) -> float:
    if actual is None:
        return 0.0
    return 1.0 if abs(actual - expected) <= tolerance else 0.0


def _fmt_optional(value: float | str | None, suffix: str = "") -> str:
    if value is None:
        return "negasit"
    if isinstance(value, str):
        return value
    return f"{value:,.2f}{suffix}"


def compare_llm_to_deterministic(
    profile: ClientProfile,
    deterministic: CreditEvaluation,
    extracted: LlmExtractedDecision,
) -> tuple[str, dict[str, float]]:
    expected_values: list[tuple[str, str, str, float]] = [
        ("Decizie", extracted.decision or "negasit", deterministic.decision.value, _score_text_match(extracted.decision, deterministic.decision.value)),
        (
            "Venit declarat",
            _fmt_optional(extracted.declared_income, " RON"),
            f"{profile.monthly_income:,.2f} RON",
            _score_number(extracted.declared_income, profile.monthly_income, 0.01),
        ),
        (
            "Pondere venit",
            _fmt_optional(extracted.income_weight_pct, "%"),
            f"{deterministic.income_weight * 100:.0f}%",
            _score_number(extracted.income_weight_pct, deterministic.income_weight * 100, 0.01),
        ),
        (
            "Venit eligibil ponderat",
            _fmt_optional(extracted.weighted_income, " RON"),
            f"{deterministic.weighted_income:,.2f} RON",
            _score_number(extracted.weighted_income, deterministic.weighted_income, 1.0),
        ),
        (
            "Capacitate maxima totala rate",
            _fmt_optional(extracted.max_monthly_payment, " RON"),
            f"{deterministic.max_monthly_payment:,.2f} RON",
            _score_number(extracted.max_monthly_payment, deterministic.max_monthly_payment, 1.0),
        ),
        (
            "Rate existente",
            _fmt_optional(extracted.existing_monthly_debts, " RON"),
            f"{profile.existing_monthly_debts:,.2f} RON",
            _score_number(extracted.existing_monthly_debts, profile.existing_monthly_debts, 0.01),
        ),
        (
            "Capacitate disponibila",
            _fmt_optional(extracted.available_payment_capacity, " RON"),
            f"{deterministic.available_payment_capacity:,.2f} RON",
            _score_number(extracted.available_payment_capacity, deterministic.available_payment_capacity, 1.0),
        ),
        (
            "Rata noua analizata",
            _fmt_optional(extracted.stressed_monthly_payment, " RON"),
            f"{deterministic.stressed_monthly_payment:,.2f} RON",
            _score_number(extracted.stressed_monthly_payment, deterministic.stressed_monthly_payment, 1.0),
        ),
        (
            "GMI rezultat",
            _fmt_optional(extracted.gmi_pct, "%"),
            f"{deterministic.gmi * 100:.2f}%",
            _score_number(extracted.gmi_pct, deterministic.gmi * 100, 0.05),
        ),
        (
            "Varsta la maturitate",
            _fmt_optional(extracted.maturity_age, " ani"),
            f"{deterministic.maturity_age:.1f} ani",
            _score_number(extracted.maturity_age, deterministic.maturity_age, 0.1),
        ),
        (
            "Suma maxima recomandata",
            _fmt_optional(extracted.max_credit_amount, " RON"),
            f"{deterministic.max_credit_amount:,.2f} RON",
            _score_number(extracted.max_credit_amount, deterministic.max_credit_amount, 1.0),
        ),
    ]
    score = sum(row[3] for row in expected_values) / len(expected_values)
    metrics = {row[0]: row[3] for row in expected_values}
    metrics["scor_total_llm_vs_formule"] = score

    lines = [
        "## Comparatie LLM vs formule Python",
        "",
        "Aceasta sectiune foloseste motorul deterministic Python ca validator. "
        "Raspunsul din tabul Analiza client ramane raspunsul calculat si redactat de LLM.",
        "",
        f"Scor total LLM vs formule: {score:.2%}",
        "",
        "| Indicator | LLM | Formule Python | Corect |",
        "|---|---:|---:|---:|",
    ]
    for label, llm_value, expected, row_score in expected_values:
        lines.append(
            f"| {label} | {llm_value} | {expected} | {'DA' if row_score == 1.0 else 'NU'} |"
        )

    lines.extend(
        [
            "",
            "### Motive calculate de Python",
            "",
            "**Motive de respingere:**",
            "\n".join(f"- {reason}" for reason in deterministic.reject_reasons) or "- Nu exista.",
            "",
            "**Motive de analiza manuala:**",
            "\n".join(f"- {reason}" for reason in deterministic.manual_review_reasons)
            or "- Nu este necesara.",
            "",
            "**Observatii:**",
            "\n".join(f"- {warning}" for warning in deterministic.warnings) or "- Nu exista.",
        ]
    )
    return "\n".join(lines), metrics


def build_analysis_markdown(profile: ClientProfile, index: RagIndex, use_llm: bool = True) -> str:
    return build_llm_credit_analysis(profile, index).answer_markdown


def request_validated_credit_json(
    profile: ClientProfile,
    deterministic: CreditEvaluation,
    sources_markdown: str,
) -> tuple[dict[str, object] | None, str | None, list[str]]:
    system_prompt = (
        "Esti asistentul local de creditare pentru o aplicatie educationala RAG. "
        "Calculezi decizia de creditare pe baza profilului clientului, a regulilor numerice si a fragmentelor RAG. "
        "Raspunsul tau trebuie sa fie exclusiv JSON valid, conform schemei cerute. "
        "Foloseste prioritar regulile numerice explicite din prompt; fragmentele RAG sunt pentru justificare si surse. "
        "Nu include Markdown, explicatii externe, comentarii sau text in afara obiectului JSON."
    )
    base_prompt = (
        "Profil client in JSON:\n"
        f"{profile_as_prompt_json(profile)}\n\n"
        f"{critical_profile_checks_prompt(profile)}\n"
        f"{operating_rules_prompt()}\n\n"
        f"{credit_json_schema_prompt()}\n\n"
        f"Fragmente RAG disponibile:\n{sources_markdown}"
    )

    raw_answer: str | None = None
    data: dict[str, object] | None = None
    validation_errors: list[str] = []
    for attempt in range(3):
        user_prompt = base_prompt
        if attempt > 0:
            user_prompt += (
                "\n\nRaspunsul JSON anterior a fost respins de validator. "
                "Genereaza de la zero un JSON mai scurt, complet si valid. "
                "Nu continua raspunsul anterior. Corecteaza strict erorile urmatoare:\n"
                + "\n".join(f"- {error}" for error in validation_errors)
                + "\n\nCopiaza exact aceste valori validate in JSON-ul final:\n"
                + deterministic_values_prompt(profile, deterministic)
                + "\n\nFoloseste maximum 4 elemente in calculation_details si texte foarte scurte."
            )
        raw_answer = optional_llm_summary(
            system_prompt,
            user_prompt,
            response_format_json=True,
            max_tokens_override=3000,
        )
        data = extract_json_object(raw_answer)
        validation_errors = validate_llm_credit_json(data, profile, deterministic)
        if not validation_errors:
            return data, raw_answer, []
    if data is not None:
        repaired = repair_llm_credit_json(data, profile, deterministic)
        repair_note = [
            "Validatorul a corectat campurile critice dupa ce LLM-ul nu a respectat schema numerica dupa retry."
        ]
        return repaired, raw_answer, repair_note
    if raw_answer and not raw_answer.startswith("LLM indisponibil"):
        repaired = repair_llm_credit_json(None, profile, deterministic)
        repair_note = [
            "Validatorul a generat un fallback structurat deoarece LLM-ul nu a returnat JSON valid dupa retry."
        ]
        return repaired, raw_answer, repair_note
    return data, raw_answer, validation_errors


def build_llm_credit_analysis(profile: ClientProfile, index: RagIndex) -> LlmCreditAnalysis:
    evaluation = evaluate_client(profile)
    sources_markdown = retrieve_credit_sources(index, profile)

    llm_json, raw_answer, validation_errors = request_validated_credit_json(
        profile,
        evaluation,
        sources_markdown,
    )
    if not raw_answer or raw_answer.startswith("LLM indisponibil"):
        answer = (
            "## Eroare LLM local\n\n"
            "Nu am putut genera raspunsul cu modelul local configurat. "
            "Verifica daca Ollama ruleaza si daca modelul este descarcat.\n\n"
            f"```text\n{raw_answer or 'LLM-ul nu a returnat continut.'}\n```\n\n"
            "Rezultatul determinist nu este afisat in analiza principala; "
            "el este folosit doar pentru comparatia din tabul LLM vs formule."
        )
        extracted = extract_llm_decision(answer)
    elif llm_json is None:
        answer = (
            "## Eroare validare JSON\n\n"
            "LLM-ul nu a returnat un obiect JSON valid dupa retry.\n\n"
            f"```text\n{raw_answer}\n```\n\n"
            "### Erori validare schema\n\n"
            + "\n".join(f"- {error}" for error in validation_errors)
        )
        extracted = extract_llm_decision(answer)
    else:
        answer = format_llm_credit_json_markdown(llm_json, validation_errors or None)
        extracted = llm_json_to_extracted(llm_json)

    comparison, metrics = compare_llm_to_deterministic(profile, evaluation, extracted)
    return LlmCreditAnalysis(answer, comparison, evaluation, extracted, metrics)


def answer_policy_question(question: str, index: RagIndex, use_llm: bool = False) -> str:
    retrieved = index.search(question, top_k=5)
    sources = format_sources(retrieved, max_chars=900)
    if not use_llm:
        return f"### Fragmente relevante\n{sources}"

    llm_answer = optional_llm_summary(
        "Raspunde strict pe baza fragmentelor RAG. Daca informatia lipseste, spune ca lipseste. "
        "Scrie in romana, in Markdown simplu, cu paragrafe scurte si liste cu liniuta. "
        "Nu folosi asteriscuri pentru bold, nu folosi separatoare de tip *** si nu adauga text de gandire.",
        f"Intrebare: {question}\n\nFragmente:\n{sources}",
    )
    if not llm_answer:
        return f"### Fragmente relevante\n{sources}\n\nLLM-ul nu este activ."
    return f"### Raspuns\n{llm_answer}\n\n### Fragmente relevante\n{sources}"
