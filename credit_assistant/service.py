from __future__ import annotations

import json
import re
import unicodedata
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

TOP_LEVEL_ALIASES = {
    "decision": ["decision", "decizie", "decizia", "hotarare", "verdict", "status"],
    "financial": [
        "financial",
        "detalii_financiare",
        "calcul_financiar",
        "calcule_financiare",
        "financial_details",
        "calculation",
        "calcul",
    ],
    "calculation_details": [
        "calculation_details",
        "detalii_calcul",
        "explicatii_calcul",
        "pasii_calculului",
        "rationale",
    ],
    "rejection_reasons": [
        "rejection_reasons",
        "motive_respingere",
        "motive_de_respingere",
        "motiv",
        "motive",
    ],
    "manual_review_reasons": [
        "manual_review_reasons",
        "motive_analiza_manuala",
        "motive_de_analiza_manuala",
        "manual_review",
    ],
    "observations": ["observations", "observatii", "note", "comentarii", "warnings"],
    "rag_sources": ["rag_sources", "surse_rag", "surse", "sources", "citations"],
}

FINANCIAL_ALIASES = {
    "declared_income": [
        "declared_income",
        "venit_declarat",
        "venit_lunar_declarat",
        "venit_lunar_declarat_ron",
        "income",
    ],
    "income_weight_pct": [
        "income_weight_pct",
        "pondere_venit",
        "pondere_venit_pct",
        "procent_pondere",
        "income_weight",
    ],
    "weighted_income": [
        "weighted_income",
        "venit_ponderat",
        "venit_eligibil_ponderat",
        "venit_eligibil",
        "eligible_income",
    ],
    "max_monthly_payment": [
        "max_monthly_payment",
        "capacitate_maxima_rate",
        "capacitate_maxima_totala_rate",
        "rata_maxima_totala",
        "maximum_debt_sum",
    ],
    "existing_monthly_debts": [
        "existing_monthly_debts",
        "rate_existente",
        "datorii_existente",
        "rate_existente_lunare",
        "existing_debts",
    ],
    "available_payment_capacity": [
        "available_payment_capacity",
        "capacitate_disponibila",
        "capacitate_plata_disponibila",
        "capacitate_disponibila_pentru_rata_noua",
    ],
    "analyzed_monthly_payment": [
        "analyzed_monthly_payment",
        "rata_analizata",
        "rata_lunara_analizata",
        "rata_noua_analizata",
        "rata_ceruta",
        "monthly_payment",
    ],
    "stressed_monthly_payment": [
        "stressed_monthly_payment",
        "rata_dupa_stres",
        "rata_noua_dupa_stres",
        "rata_stresata",
        "rata_analizata_dupa_stres",
    ],
    "gmi_pct": ["gmi_pct", "gmi", "grad_indatorare", "grad_maxim_indatorare"],
    "maturity_age": [
        "maturity_age",
        "varsta_maturitate",
        "varsta_la_maturitate",
        "age_at_maturity",
    ],
    "max_credit_amount": [
        "max_credit_amount",
        "suma_maxima_credit",
        "suma_maxima_recomandata",
        "max_credite",
        "maximum_credit_amount",
    ],
    "product_cap": ["product_cap", "plafon_produs", "suma_maxima_produs", "product_limit"],
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

    normalized = re.sub(r"(?<![\n\d)])\s+-\s+(?=[^\d(])", "\n- ", normalized)
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


def _strip_accents(value: str) -> str:
    decomposed = unicodedata.normalize("NFD", value)
    return "".join(ch for ch in decomposed if unicodedata.category(ch) != "Mn")


def _normalize_key(value: str) -> str:
    normalized = _strip_accents(value).lower()
    normalized = re.sub(r"[^a-z0-9]+", "_", normalized)
    return normalized.strip("_")


def normalize_decision(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = _strip_accents(re.sub(r"\s+", " ", value.upper()).strip())
    if normalized in {"APROBAT", "RESPINS", "ANALIZA MANUALA"}:
        return normalized
    if normalized in {"APROBARE", "APROBATA", "APPROVED", "ACCEPTAT"}:
        return "APROBAT"
    if normalized in {"RESPINGERE", "RESPINSA", "REJECTED", "REFUZAT"}:
        return "RESPINS"
    if normalized in {"ANALIZA", "REVIZUIRE MANUALA", "MANUAL REVIEW", "MANUAL_REVIEW"}:
        return "ANALIZA MANUALA"
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
        "- Dobanda folosita in formule = dobanda_anuala_pct + 2 daca dobanda_variabila este true, "
        "altfel dobanda_anuala_pct.\n"
        "- Factor stres valutar = 1.15 doar pentru credit EUR cu venit RON, altfel 1.00.\n"
        "- Rata noua analizata = rata_lunara_dorita_ron daca aceasta este > 0; altfel calculeaza "
        "formula anuitatii: P * r / (1 - (1 + r)^(-n)), unde P = suma_solicitata_ron, "
        "r = dobanda_folosita_in_formule / 100 / 12 si n = durata_credit_luni.\n"
        "- Rata noua dupa stres = rata_noua_analizata * factor_stres_valutar.\n"
        "- Pentru anuitate, P este intotdeauna suma_solicitata_ron, nu plafonul produsului si nu suma maxima recomandata.\n"
        "- Pentru 10% dobanda anuala, rata lunara folosita in formula este 0.10 / 12 = 0.0083333333. "
        "Nu folosi 10 / 12 si nu folosi o dobanda implicita mai mica decat dobanda_anuala_pct.\n"
        "- GMI rezultat este procent, nu suma in RON: "
        "(rate_existente_lunare_ron + rata_noua_dupa_stres) / venit_eligibil_ponderat * 100.\n"
        "- Suma maxima recomandata prin GMI = max(0, capacitate_disponibila_pentru_rata_noua / factor_stres_valutar) * "
        "(1 - (1 + r)^(-n)) / r, limitata la suma maxima produs. Nu o confunda cu suma solicitata.\n"
        "- Daca suma_solicitata_ron este mai mica sau egala cu suma maxima recomandata si plafonul produs, nu respinge pentru suma.\n"
        "- Daca durata_credit_luni este mai mica sau egala cu durata maxima, nu respinge pentru durata.\n"
        "- Socul de dobanda si stresul valutar se aplica exact ca mai sus; nu le aplica de doua ori.\n"
        "- Decizia trebuie sa fie exact una dintre: APROBAT, RESPINS, ANALIZA MANUALA.\n"
        "- Daca exista orice motiv de respingere, decizia este RESPINS. "
        "Altfel, daca exista motiv de analiza manuala, decizia este ANALIZA MANUALA. "
        "Altfel, decizia este APROBAT.\n"
        "- FICO sub 620 inseamna RESPINS; FICO 620-649 inseamna ANALIZA MANUALA.\n"
        "- Client PEP sau risc AML Ridicat inseamna ANALIZA MANUALA daca nu exista motive de respingere.\n"
    )


def calculation_guardrails_prompt() -> str:
    return (
        "Erori de calcul pe care trebuie sa le eviti explicit:\n"
        "- Daca rata_lunara_dorita_ron > 0, rata_noua_analizata este exact acea rata dorita, "
        "nu se recalculeaza din suma solicitata.\n"
        "- Procentele se convertesc in factori doar in calcule: 100% inseamna 1.00, 75% inseamna 0.75, "
        "iar limita GMI 40% inseamna 0.40. Nu inmulti venitul cu 100 sau cu 40.\n"
        "- Capacitatea maxima totala rate = venit_eligibil_ponderat * 0.40; nu poate fi mai mare decat venitul ponderat.\n"
        "- Nu seta GMI rezultat la 40% doar pentru ca limita este 40%; GMI trebuie calculat din rate / venit ponderat.\n"
        "- Daca rata_lunara_dorita_ron = 0 si suma_solicitata_ron > 0, rata se calculeaza cu formula anuitatii; "
        "nu folosi impartirea simpla suma_solicitata_ron / durata_credit_luni.\n"
        "- Pentru credit in RON cu venit in RON si dobanda fixa, rata dupa stres este egala cu rata analizata.\n"
        "- suma_solicitata_ron = 0 inseamna caz bazat pe rata dorita; nu respinge pentru suma minima.\n"
        "- Varsta la maturitate este varsta + durata_credit_luni / 12, nu varsta + durata_credit_luni.\n"
        "- Daca GMI rezultat este peste 40%, decizia trebuie sa fie RESPINS.\n"
        "- Plafonul produsului NovaFlex este exact 150000 RON, nu 225000 RON.\n"
        "- Suma maxima recomandata nu poate depasi plafonul produsului de 150000 RON.\n"
        "- Suma maxima recomandata se obtine prin inversarea formulei de anuitate pe capacitatea disponibila, "
        "apoi se limiteaza la plafonul produsului; nu copia suma solicitata.\n"
        "- In calculation_details nu scrie doar concluzii; pentru fiecare valoare ceruta include formula, "
        "valorile inlocuite si rezultatul numeric.\n"
    )


def calculation_trace_prompt() -> str:
    return (
        "Trasabilitate obligatorie pentru debugging:\n"
        "- calculation_details trebuie sa contina exact 4 elemente, in ordinea de mai jos.\n"
        "- Fiecare element trebuie sa includa formula=..., valori=..., rezultat=... si sa foloseasca numerele profilului.\n"
        "1. Rata noua analizata: formula=rata_lunara_dorita_ron daca > 0, altfel P*r/(1-(1+r)^(-n)); "
        "valori=P=..., r=..., n=..., factor_stres_valutar=...; rezultat=rata_noua_analizata=... RON, "
        "rata_dupa_stres=... RON.\n"
        "2. GMI rezultat: formula=(rate_existente_lunare_ron + rata_dupa_stres) / venit_eligibil_ponderat * 100; "
        "valori=...; rezultat=...%.\n"
        "3. Varsta la maturitate: formula=varsta + durata_credit_luni / 12; valori=...; rezultat=... ani.\n"
        "4. Suma maxima recomandata: formula=min(150000, max(0, capacitate_disponibila / factor_stres_valutar) "
        "* (1-(1+r)^(-n))/r); valori=capacitate_disponibila=..., factor_stres_valutar=..., "
        "r=..., n=..., plafon_produs=150000; rezultat=... RON.\n"
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
        f"{calculation_trace_prompt()}\n"
        "Toate campurile numerice trebuie sa fie numere JSON, nu stringuri cu RON sau %. "
        "Daca nu exista motive intr-o lista, foloseste lista goala []. "
        "calculation_details trebuie sa aiba exact cele 4 elemente de trasabilitate de mai sus, "
        "scurte dar cu formula, valori si rezultat. "
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
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    if not isinstance(value, list):
        return []
    items: list[str] = []
    for item in value:
        if isinstance(item, dict):
            text = item.get("text") or item.get("source") or item.get("document") or item.get("id")
            if text is not None and str(text).strip():
                items.append(str(text).strip())
            continue
        if str(item).strip():
            items.append(str(item).strip())
    return items


def _get_alias(data: object, aliases: list[str]) -> object | None:
    if not isinstance(data, dict):
        return None
    for alias in aliases:
        if alias in data:
            return data[alias]

    normalized_aliases = {_normalize_key(alias) for alias in aliases}
    for key, value in data.items():
        if _normalize_key(str(key)) in normalized_aliases:
            return value
    return None


def _coerce_financial_value(value: object) -> float | None:
    number = _as_float(value)
    if number is None:
        return None
    return number


def _has_extracted_values(data: dict[str, object] | None) -> bool:
    if data is None:
        return False
    if normalize_decision(str(data.get("decision", ""))):
        return True
    financial = data.get("financial")
    if not isinstance(financial, dict):
        return False
    return any(_as_float(financial.get(field)) is not None for field in REQUIRED_JSON_NUMERIC_FIELDS)


def _numeric_field_count(data: dict[str, object] | None) -> int:
    if data is None:
        return 0
    financial = data.get("financial")
    if not isinstance(financial, dict):
        return 0
    return sum(1 for field in REQUIRED_JSON_NUMERIC_FIELDS if _as_float(financial.get(field)) is not None)


def extract_calculation_details_from_text(text: str | None, limit: int = 6) -> list[str]:
    if not text:
        return []

    match = re.search(
        r"(?is)(?:^|\n)\s*(?:#{1,6}\s*)?Detalii\s+calcul\s*:?\s*\n(?P<body>.*?)(?="
        r"\n\s*(?:#{1,6}\s*)?(?:Calcul\s+financiar|Motive\s+de\s+respingere|"
        r"Motive\s+de\s+analiza\s+manuala|Observatii|Surse\s+RAG\s+folosite|Decizie)\s*:?\s*(?:\n|$)|\Z)",
        text,
    )
    if not match:
        return []

    body = match.group("body")
    bullet_matches = re.findall(
        r"(?ms)^\s*(?:[-*]|\d+[\.)])\s+(.*?)(?="
        r"^\s*(?:[-*]|\d+[\.)])\s+|^\s*\|?\s*(?:Indicator|Eticheta)\s*\||\Z)",
        body,
    )
    candidates = bullet_matches if bullet_matches else body.splitlines()

    details: list[str] = []
    for raw_line in candidates:
        line = re.sub(r"\s+", " ", raw_line).strip()
        if not line:
            continue
        if line.startswith("|") or re.search(r"\bIndicator\b.*\bValoare\b", line, flags=re.IGNORECASE):
            continue
        line = re.sub(r"^[-*]\s+", "", line)
        line = re.sub(r"^\d+[\.)]\s+", "", line)
        if not line:
            continue
        details.append(line.rstrip(".") + ".")
        if len(details) >= limit:
            break
    return details


def _parse_marker_number(text: str, markers: list[str]) -> float | None:
    for marker in markers:
        pattern = rf"(?i)\b{re.escape(marker)}\s*[:=]\s*(-?\d[\d\s.,]*)"
        match = re.search(pattern, text)
        if not match:
            continue
        parsed = _parse_number(match.group(1).rstrip("., "))
        if parsed is not None:
            return parsed
    return None


def _parse_named_value(text: str, names: list[str]) -> float | None:
    for name in names:
        flexible_name = re.escape(name).replace(r"\_", r"[_\s-]*")
        pattern = rf"(?i)(?<![a-z0-9_]){flexible_name}\s*=\s*(-?\d[\d\s.,]*)"
        match = re.search(pattern, text)
        if not match:
            continue
        parsed = _parse_number(match.group(1).rstrip("., "))
        if parsed is not None:
            return parsed
    return None


def _set_financial_if_missing(financial: dict[str, object], field: str, value: float | None) -> None:
    if value is not None and _as_float(financial.get(field)) is None:
        financial[field] = value


def backfill_financial_from_calculation_details(data: dict[str, object]) -> None:
    financial = data.setdefault("financial", {})
    if not isinstance(financial, dict):
        return

    details = _as_string_list(data.get("calculation_details"))
    if not details:
        return

    for detail in details:
        normalized = normalize_label(detail)
        label_normalized = normalize_label(detail.split(":", 1)[0])

        _set_financial_if_missing(
            financial,
            "existing_monthly_debts",
            _parse_named_value(detail, ["rate_existente_lunare_ron", "rate_existente", "existing_monthly_debts"]),
        )
        _set_financial_if_missing(
            financial,
            "available_payment_capacity",
            _parse_named_value(
                detail,
                [
                    "capacitate_disponibila_pentru_rata_noua",
                    "capacitate_disponibila",
                    "available_payment_capacity",
                ],
            ),
        )
        _set_financial_if_missing(
            financial,
            "product_cap",
            _parse_named_value(detail, ["plafon_produs", "product_cap", "suma_maxima_produs"]),
        )
        if "plafon produs" in normalized:
            _set_financial_if_missing(financial, "product_cap", MAX_AMOUNT_RON)

        result = _parse_marker_number(detail, ["rezultat", "result"])
        if result is None:
            continue

        if "rata noua" in label_normalized:
            _set_financial_if_missing(financial, "analyzed_monthly_payment", result)
            stressed = _parse_named_value(detail, ["rata_dupa_stres", "rata_noua_dupa_stres"])
            _set_financial_if_missing(financial, "stressed_monthly_payment", stressed)
        elif "varsta la maturitate" in label_normalized or "varsta maturitate" in label_normalized:
            _set_financial_if_missing(financial, "maturity_age", result)
        elif "suma maxima" in label_normalized:
            _set_financial_if_missing(financial, "max_credit_amount", result)
        elif "gmi" in label_normalized:
            _set_financial_if_missing(financial, "gmi_pct", result)


def extracted_decision_to_credit_json(
    extracted: LlmExtractedDecision,
    raw_text: str | None = None,
) -> dict[str, object] | None:
    financial_values = {
        "declared_income": extracted.declared_income,
        "income_weight_pct": extracted.income_weight_pct,
        "weighted_income": extracted.weighted_income,
        "max_monthly_payment": extracted.max_monthly_payment,
        "existing_monthly_debts": extracted.existing_monthly_debts,
        "available_payment_capacity": extracted.available_payment_capacity,
        "analyzed_monthly_payment": extracted.stressed_monthly_payment,
        "stressed_monthly_payment": extracted.stressed_monthly_payment,
        "gmi_pct": extracted.gmi_pct,
        "maturity_age": extracted.maturity_age,
        "max_credit_amount": extracted.max_credit_amount,
    }
    if raw_text:
        product_cap = _extract_after_labels(
            financial_extraction_text(raw_text),
            ["Plafon produs", "Product cap", "Product limit"],
        )
        if product_cap is not None:
            financial_values["product_cap"] = product_cap

    financial = {key: value for key, value in financial_values.items() if value is not None}
    if not financial and extracted.decision is None:
        return None

    return {
        "decision": extracted.decision,
        "financial": financial,
        "calculation_details": extract_calculation_details_from_text(raw_text),
        "rejection_reasons": [],
        "manual_review_reasons": [],
        "observations": [
            "Raspunsul LLM a fost interpretat din text liber sau tabelar, nu din schema JSON canonica."
        ],
        "rag_sources": [],
    }


def canonicalize_llm_credit_json(
    data: dict[str, object] | None,
    raw_text: str | None = None,
) -> dict[str, object] | None:
    canonical: dict[str, object] = {
        "calculation_details": [],
        "rejection_reasons": [],
        "manual_review_reasons": [],
        "observations": [],
        "rag_sources": [],
    }

    if isinstance(data, dict):
        decision_value = _get_alias(data, TOP_LEVEL_ALIASES["decision"])
        decision = normalize_decision(str(decision_value)) if decision_value is not None else None
        if decision:
            canonical["decision"] = decision

        financial_source = _get_alias(data, TOP_LEVEL_ALIASES["financial"])
        if not isinstance(financial_source, dict):
            financial_source = data

        financial: dict[str, float] = {}
        for field, aliases in FINANCIAL_ALIASES.items():
            value = _get_alias(financial_source, aliases)
            if value is None and financial_source is not data:
                value = _get_alias(data, aliases)
            number = _coerce_financial_value(value)
            if number is not None:
                financial[field] = number
        if financial:
            canonical["financial"] = financial

        for field in [
            "calculation_details",
            "rejection_reasons",
            "manual_review_reasons",
            "observations",
            "rag_sources",
        ]:
            value = _get_alias(data, TOP_LEVEL_ALIASES[field])
            canonical[field] = _as_string_list(value)

    if raw_text:
        normalized_text = normalize_credit_markdown(raw_text)
        text_data = extracted_decision_to_credit_json(
            extract_llm_decision(normalized_text),
            normalized_text,
        )
        if text_data is not None:
            if "decision" not in canonical and text_data.get("decision"):
                canonical["decision"] = text_data["decision"]
            text_financial = text_data.get("financial")
            if isinstance(text_financial, dict):
                financial = canonical.setdefault("financial", {})
                if isinstance(financial, dict):
                    for field, value in text_financial.items():
                        financial.setdefault(field, value)
            if not _as_string_list(canonical.get("calculation_details")):
                text_details = _as_string_list(text_data.get("calculation_details"))
                if text_details:
                    canonical["calculation_details"] = text_details
            if not isinstance(data, dict):
                canonical["observations"] = _as_string_list(text_data.get("observations"))

    financial = canonical.get("financial")
    if isinstance(financial, dict):
        backfill_financial_from_calculation_details(canonical)
        analyzed = _as_float(financial.get("analyzed_monthly_payment"))
        stressed = _as_float(financial.get("stressed_monthly_payment"))
        if analyzed is None and stressed is not None:
            financial["analyzed_monthly_payment"] = stressed
        elif stressed is None and analyzed is not None:
            financial["stressed_monthly_payment"] = analyzed

    if _has_extracted_values(canonical):
        return canonical
    return None


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


def _financial_object(data: dict[str, object]) -> dict[str, object]:
    financial = data.get("financial")
    return financial if isinstance(financial, dict) else {}


def llm_self_review_findings(profile: ClientProfile, data: dict[str, object]) -> list[str]:
    """Rules that can be checked from the profile and from the LLM's own JSON."""
    financial = _financial_object(data)
    decision = normalize_decision(str(data.get("decision", "")))
    rejection_reasons = _as_string_list(data.get("rejection_reasons"))
    manual_reasons = _as_string_list(data.get("manual_review_reasons"))
    findings: list[str] = []

    maturity_age = profile.age + profile.term_months / 12
    gmi_pct = _as_float(financial.get("gmi_pct"))
    reported_maturity_age = _as_float(financial.get("maturity_age"))
    weighted_income = _as_float(financial.get("weighted_income"))
    max_monthly_payment = _as_float(financial.get("max_monthly_payment"))
    existing_debts = _as_float(financial.get("existing_monthly_debts"))
    available_capacity = _as_float(financial.get("available_payment_capacity"))
    analyzed_payment = _as_float(financial.get("analyzed_monthly_payment"))
    stressed_payment = _as_float(financial.get("stressed_monthly_payment"))
    max_credit_amount = _as_float(financial.get("max_credit_amount"))
    product_cap = _as_float(financial.get("product_cap"))

    if reported_maturity_age is not None and abs(reported_maturity_age - maturity_age) > 0.1:
        findings.append(
            "Varsta la maturitate este inconsistenta cu profilul: trebuie calculata ca "
            "varsta + durata_credit_luni / 12."
        )
    if (
        profile.requested_monthly_payment > 0
        and analyzed_payment is not None
        and abs(analyzed_payment - profile.requested_monthly_payment) > 0.01
    ):
        findings.append(
            "rata_lunara_dorita_ron este pozitiva, deci rata analizata trebuie sa fie exact rata dorita din profil."
        )
    if (
        profile.currency == "RON"
        and profile.income_currency == "RON"
        and not profile.variable_rate
        and analyzed_payment is not None
        and stressed_payment is not None
        and abs(analyzed_payment - stressed_payment) > 0.01
    ):
        findings.append(
            "Creditul este RON/RON cu dobanda fixa, deci rata dupa stres trebuie sa fie egala cu rata analizata."
        )
    if profile.requested_amount > 0 and analyzed_payment is not None and profile.term_months > 0:
        straight_line_payment = profile.requested_amount / profile.term_months
        if analyzed_payment <= straight_line_payment * 1.10:
            findings.append(
                "Rata analizata pare calculata prin impartire simpla sau cu dobanda prea mica; "
                "pentru suma solicitata trebuie folosita formula anuitatii."
            )
    if weighted_income and max_monthly_payment is not None:
        model_expected_capacity = weighted_income * GMI_LIMIT
        if abs(max_monthly_payment - model_expected_capacity) > 1.0:
            findings.append(
                "Capacitatea maxima totala rate nu este consistenta cu venitul ponderat si limita GMI de 40%."
            )
    if max_monthly_payment is not None and existing_debts is not None and available_capacity is not None:
        model_expected_available = max_monthly_payment - existing_debts
        if abs(available_capacity - model_expected_available) > 1.0:
            findings.append(
                "Capacitatea disponibila nu este consistenta cu capacitatea maxima minus ratele existente."
            )
    if weighted_income and stressed_payment is not None and existing_debts is not None and gmi_pct is not None:
        model_expected_gmi = (existing_debts + stressed_payment) / weighted_income * 100
        if abs(gmi_pct - model_expected_gmi) > 0.1:
            findings.append(
                "GMI-ul returnat nu este consistent cu rata dupa stres, ratele existente si venitul ponderat."
            )
    if product_cap is not None and abs(product_cap - MAX_AMOUNT_RON) > 0.01:
        findings.append("Plafonul produsului este inconsistent cu regula NovaFlex de 150000 RON.")
    if max_credit_amount is not None and available_capacity is not None and available_capacity > 0:
        if max_credit_amount > MAX_AMOUNT_RON + 1:
            findings.append(
                "Suma maxima recomandata depaseste plafonul produsului; trebuie limitata la 150000 RON."
            )
        if max_credit_amount <= 1:
            findings.append(
                "Suma maxima recomandata este zero desi exista capacitate disponibila pozitiva."
            )
        if (
            profile.requested_amount > 0
            and abs(max_credit_amount - profile.requested_amount) <= 1
            and max_credit_amount < MAX_AMOUNT_RON
        ):
            findings.append(
                "Suma maxima recomandata pare copiata din suma solicitata; trebuie calculata din capacitatea disponibila."
            )

    hard_rejections: list[str] = []
    if profile.age < MIN_AGE:
        hard_rejections.append(f"varsta clientului este sub limita minima de {MIN_AGE} ani")
    if maturity_age > MAX_AGE_AT_MATURITY:
        hard_rejections.append(f"varsta la maturitate este {maturity_age:.1f}, peste limita de {MAX_AGE_AT_MATURITY} ani")
    if profile.term_months > MAX_TERM_MONTHS:
        hard_rejections.append(f"durata creditului este peste limita de {MAX_TERM_MONTHS} luni")
    if profile.requested_amount > 0 and profile.requested_amount < MIN_AMOUNT_RON:
        hard_rejections.append(f"suma solicitata este sub minimul de {MIN_AMOUNT_RON:,.0f} RON")
    if profile.requested_amount > MAX_AMOUNT_RON:
        hard_rejections.append(f"suma solicitata depaseste plafonul produsului de {MAX_AMOUNT_RON:,.0f} RON")
    if profile.fico < 620:
        hard_rejections.append("FICO este sub 620")
    if gmi_pct is not None and gmi_pct > GMI_LIMIT * 100:
        hard_rejections.append(f"GMI-ul returnat de model este {gmi_pct:.2f}%, peste limita de {GMI_LIMIT * 100:.0f}%")
    if available_capacity is not None and stressed_payment is not None and stressed_payment > available_capacity:
        hard_rejections.append("rata noua dupa stres depaseste capacitatea disponibila returnata de model")
    if max_credit_amount is not None and profile.requested_amount > max_credit_amount:
        hard_rejections.append("suma solicitata depaseste suma maxima recomandata returnata de model")

    if hard_rejections and decision != "RESPINS":
        findings.append("Decizia trebuie revizuita la RESPINS deoarece: " + "; ".join(hard_rejections) + ".")
    if hard_rejections and not rejection_reasons:
        findings.append("Lista rejection_reasons este goala desi exista motive hard de respingere.")

    manual_flags: list[str] = []
    if 620 <= profile.fico < 650:
        manual_flags.append("FICO este intre 620 si 649")
    if profile.is_pep:
        manual_flags.append("clientul este PEP")
    if profile.aml_risk == "Ridicat":
        manual_flags.append("riscul AML este Ridicat")
    if manual_flags and not hard_rejections and decision == "APROBAT":
        findings.append("Decizia trebuie revizuita la ANALIZA MANUALA deoarece: " + "; ".join(manual_flags) + ".")
    if manual_flags and not hard_rejections and not manual_reasons:
        findings.append("Lista manual_review_reasons este goala desi exista motive de analiza manuala.")

    return findings


def llm_self_review_flags_prompt(profile: ClientProfile, data: dict[str, object]) -> str:
    financial = _financial_object(data)
    gmi_pct = _as_float(financial.get("gmi_pct"))
    available_capacity = _as_float(financial.get("available_payment_capacity"))
    stressed_payment = _as_float(financial.get("stressed_monthly_payment"))
    max_credit_amount = _as_float(financial.get("max_credit_amount"))
    maturity_age = profile.age + profile.term_months / 12

    def yes_no(condition: bool) -> str:
        return "DA" if condition else "NU"

    lines = [
        "Tabel de verificare hard. Daca ORICARE rezultat este DA, decizia finala trebuie sa fie RESPINS:",
        f"- varsta_sub_minim: {profile.age} < {MIN_AGE} => {yes_no(profile.age < MIN_AGE)}",
        f"- varsta_la_maturitate_peste_limita: {maturity_age:.1f} > {MAX_AGE_AT_MATURITY} => {yes_no(maturity_age > MAX_AGE_AT_MATURITY)}",
        f"- durata_peste_limita: {profile.term_months} > {MAX_TERM_MONTHS} => {yes_no(profile.term_months > MAX_TERM_MONTHS)}",
        f"- suma_sub_minim: {profile.requested_amount:.2f} < {MIN_AMOUNT_RON:.2f} si suma > 0 => {yes_no(profile.requested_amount > 0 and profile.requested_amount < MIN_AMOUNT_RON)}",
        f"- suma_peste_plafon_produs: {profile.requested_amount:.2f} > {MAX_AMOUNT_RON:.2f} => {yes_no(profile.requested_amount > MAX_AMOUNT_RON)}",
        f"- fico_sub_620: {profile.fico} < 620 => {yes_no(profile.fico < 620)}",
    ]
    if gmi_pct is not None:
        lines.append(
            f"- gmi_returnat_de_model_peste_limita: {gmi_pct:.2f}% > {GMI_LIMIT * 100:.0f}% => {yes_no(gmi_pct > GMI_LIMIT * 100)}"
        )
    if available_capacity is not None and stressed_payment is not None:
        lines.append(
            "- rata_returnata_de_model_peste_capacitate: "
            f"{stressed_payment:.2f} > {available_capacity:.2f} => {yes_no(stressed_payment > available_capacity)}"
        )
    if max_credit_amount is not None:
        lines.append(
            "- suma_solicitata_peste_maxim_returnat_de_model: "
            f"{profile.requested_amount:.2f} > {max_credit_amount:.2f} => {yes_no(profile.requested_amount > max_credit_amount)}"
        )

    lines.extend(
        [
            "",
            "Tabel de verificare analiza manuala. Se aplica doar daca toate verificarile hard de respingere sunt NU:",
            f"- fico_620_649: 620 <= {profile.fico} < 650 => {yes_no(620 <= profile.fico < 650)}",
            f"- client_pep: {yes_no(profile.is_pep)}",
            f"- aml_ridicat: {profile.aml_risk} == Ridicat => {yes_no(profile.aml_risk == 'Ridicat')}",
            "",
            "Regula de consistenta: este interzis sa scrii APROBAT daca o verificare hard este DA.",
            "Regula de observatii: este interzis sa scrii ca suma este sub plafon cand suma_peste_plafon_produs este DA.",
        ]
    )
    return "\n".join(lines)


def needs_llm_self_review(profile: ClientProfile, data: dict[str, object]) -> bool:
    return bool(llm_self_review_findings(profile, data))


def request_llm_credit_self_review(
    profile: ClientProfile,
    sources_markdown: str,
    first_data: dict[str, object],
) -> tuple[dict[str, object] | None, str | None]:
    findings = llm_self_review_findings(profile, first_data)
    if not findings:
        return None, None

    system_prompt = (
        "Esti un verificator local pentru un raspuns JSON de creditare generat de acelasi LLM. "
        "Nu primesti valori calculate de Python. Verifici doar profilul clientului, regulile explicite, "
        "fragmentele RAG si JSON-ul anterior. Returnezi exclusiv JSON valid in aceeasi schema."
    )
    findings_text = "\n".join(f"- {finding}" for finding in findings)
    user_prompt = (
        "JSON-ul anterior contine posibile inconsistente de decizie sau calcul. "
        "Recalculeaza singur anuitatea si regulile hard, apoi returneaza un JSON complet corectat.\n\n"
        "Nu copia automat valorile anterioare daca observi o eroare. "
        "Nu folosi valori dintr-un motor Python; foloseste doar regulile de mai jos.\n\n"
        "Profil client in JSON:\n"
        f"{profile_as_prompt_json(profile)}\n\n"
        f"{critical_profile_checks_prompt(profile)}\n"
        f"{operating_rules_prompt()}\n"
        f"{calculation_guardrails_prompt()}\n\n"
        "Verificari declansate pe baza profilului si a JSON-ului tau anterior:\n"
        f"{findings_text}\n\n"
        f"{llm_self_review_flags_prompt(profile, first_data)}\n\n"
        "Reguli de decizie obligatorii la revizuire:\n"
        "- Daca exista oricare motiv hard de respingere, decision trebuie sa fie RESPINS.\n"
        "- Daca decision este RESPINS, rejection_reasons trebuie sa explice concret motivele.\n"
        "- Daca nu exista respingere, dar exista PEP, AML Ridicat sau FICO 620-649, decision trebuie sa fie ANALIZA MANUALA.\n"
        "- Pentru rata noua, daca rata_lunara_dorita_ron este 0, foloseste formula anuitatii cu P = suma_solicitata_ron.\n"
        "- Verifica dupa calcul ca GMI peste 40% produce RESPINS.\n\n"
        f"{credit_json_schema_prompt()}\n\n"
        "JSON anterior:\n"
        f"{json.dumps(first_data, ensure_ascii=False, indent=2)}\n\n"
        f"Fragmente RAG disponibile:\n{sources_markdown}"
    )
    raw_answer = optional_llm_summary(
        system_prompt,
        user_prompt,
        response_format_json=True,
        max_tokens_override=3000,
    )
    return canonicalize_llm_credit_json(extract_json_object(raw_answer), raw_answer), raw_answer


def request_llm_decision_adjudication(
    profile: ClientProfile,
    data: dict[str, object],
) -> tuple[dict[str, object] | None, str | None]:
    findings = llm_self_review_findings(profile, data)
    if not findings:
        return None, None

    system_prompt = (
        "Esti un adjudicator de decizie pentru creditare. Nu calculezi valori financiare. "
        "Citesti doar verificarile DA/NU si decizi daca raspunsul LLM trebuie sa fie "
        "APROBAT, RESPINS sau ANALIZA MANUALA. Returnezi exclusiv JSON valid."
    )
    user_prompt = (
        "Stabileste doar decizia si listele de motive. Nu modifica valorile financiare.\n\n"
        f"{llm_self_review_flags_prompt(profile, data)}\n\n"
        "Reguli:\n"
        "- Daca oricare verificare hard are rezultat DA, decision = RESPINS.\n"
        "- Toate verificarile hard cu DA trebuie trecute in rejection_reasons.\n"
        "- Doar daca toate verificarile hard sunt NU, poti folosi ANALIZA MANUALA pentru PEP, AML Ridicat sau FICO 620-649.\n"
        "- Nu pune motive hard in observations; ele trebuie sa fie in rejection_reasons.\n\n"
        "Returneaza exact aceasta schema:\n"
        "{\n"
        '  "decision": "APROBAT | RESPINS | ANALIZA MANUALA",\n'
        '  "rejection_reasons": ["string"],\n'
        '  "manual_review_reasons": ["string"],\n'
        '  "observations": ["string"]\n'
        "}\n"
    )
    raw_answer = optional_llm_summary(
        system_prompt,
        user_prompt,
        response_format_json=True,
        max_tokens_override=1000,
    )
    return extract_json_object(raw_answer), raw_answer


def merge_llm_decision_adjudication(
    data: dict[str, object],
    adjudication: dict[str, object],
) -> dict[str, object]:
    decision = normalize_decision(str(adjudication.get("decision", "")))
    if decision is None:
        return data

    merged = dict(data)
    merged["decision"] = decision
    for field in ["rejection_reasons", "manual_review_reasons", "observations"]:
        value = adjudication.get(field)
        if isinstance(value, list):
            merged[field] = [humanize_rule_reason(str(item)) for item in value if str(item).strip()]
    return merged


def humanize_rule_reason(reason: str) -> str:
    normalized = reason.strip()
    mapping = {
        "varsta_sub_minim": f"Varsta clientului este sub limita minima de {MIN_AGE} ani.",
        "varsta_la_maturitate_peste_limita": f"Varsta la maturitate depaseste limita de {MAX_AGE_AT_MATURITY} ani.",
        "durata_peste_limita": f"Durata creditului depaseste limita de {MAX_TERM_MONTHS} luni.",
        "suma_sub_minim": f"Suma solicitata este sub minimul de {MIN_AMOUNT_RON:,.0f} RON.",
        "suma_peste_plafon_produs": f"Suma solicitata depaseste plafonul produsului de {MAX_AMOUNT_RON:,.0f} RON.",
        "fico_sub_620": "Scorul FICO este sub 620.",
        "gmi_returnat_de_model_peste_limita": f"GMI-ul returnat de model depaseste limita de {GMI_LIMIT * 100:.0f}%.",
        "rata_returnata_de_model_peste_capacitate": "Rata noua returnata de model depaseste capacitatea disponibila.",
        "suma_solicitata_peste_maxim_returnat_de_model": "Suma solicitata depaseste suma maxima recomandata returnata de model.",
        "fico_620_649": "Scorul FICO este in intervalul 620-649 si necesita analiza manuala.",
        "client_pep": "Clientul este PEP si necesita analiza manuala.",
        "aml_ridicat": "Riscul AML este ridicat si necesita analiza manuala.",
    }
    return mapping.get(normalized, normalized)


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
            row_normalized = normalize_label(row_label)
            if normalized_label == row_normalized or row_normalized.startswith(f"{normalized_label} "):
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
    normalized = _strip_accents(label).lower()
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


def strip_calculation_details_for_financial_extraction(text: str) -> str:
    """Remove arithmetic trace text so formulas are not parsed as output values."""
    return re.sub(
        r"(?is)(?:^|\n)\s*(?:#{1,6}\s*)?Detalii\s+calcul\s*:?\s*\n.*?(?="
        r"\n\s*(?:#{1,6}\s*)?(?:Calcul\s+financiar|Motive\s+de\s+respingere|"
        r"Motive\s+de\s+analiza\s+manuala|Observatii|Surse\s+RAG\s+folosite|"
        r"Note\s+validare\s+schema|Decizie)\s*:?\s*(?:\n|$)|"
        r"\n\s*\|?\s*(?:Indicator|Eticheta)\s*\||\Z)",
        "\n",
        text,
    )


def extract_named_section(text: str, title: str) -> str | None:
    match = re.search(
        rf"(?is)(?:^|\n)\s*(?:#{{1,6}}\s*)?{re.escape(title)}\s*:?\s*\n(?P<body>.*?)(?="
        r"\n\s*(?:#{1,6}\s*)?(?:Detalii\s+calcul|Motive\s+de\s+respingere|"
        r"Motive\s+de\s+analiza\s+manuala|Observatii|Surse\s+RAG\s+folosite|"
        r"Note\s+validare\s+schema|Decizie)\s*:?\s*(?:\n|$)|\Z)",
        text,
    )
    if not match:
        return None
    return match.group("body").strip()


def financial_extraction_text(text: str) -> str:
    without_details = strip_calculation_details_for_financial_extraction(text)
    financial_section = extract_named_section(without_details, "Calcul financiar")
    return financial_section or without_details


def extract_llm_decision(text: str) -> LlmExtractedDecision:
    decision = extract_decision_label(text)
    financial_text = financial_extraction_text(text)

    return LlmExtractedDecision(
        decision=decision,
        declared_income=_extract_after_labels(
            financial_text,
            ["Venit declarat", "Venit lunar declarat", "declared_income", "income"],
        ),
        income_weight_pct=_extract_after_labels(
            financial_text,
            ["Pondere venit", "Pondere", "income_weight_pct", "income weight"],
        ),
        weighted_income=_extract_after_labels(
            financial_text,
            ["Venit eligibil ponderat", "Venit ponderat", "Venit eligibil", "weighted_income"],
        ),
        max_monthly_payment=_extract_after_labels(
            financial_text,
            [
                "Capacitate maxima totala rate (40% GMI)",
                "Capacitate maxima totala rate",
                "Capacitate maxima",
                "Maximum debt sum",
                "max_monthly_payment",
            ],
        ),
        existing_monthly_debts=_extract_after_labels(
            financial_text,
            ["Rate existente", "Datorii existente", "existing_monthly_debts"],
        ),
        available_payment_capacity=_extract_after_labels(
            financial_text,
            [
                "Capacitate disponibila pentru rata noua",
                "Capacitate disponibila",
                "Capacitate plata disponibila",
                "available_payment_capacity",
            ],
        ),
        stressed_monthly_payment=_extract_after_labels(
            financial_text,
            [
                "Rata noua analizata, dupa stres daca se aplica",
                "Rata noua dupa stres",
                "Rata noua analizata",
                "Rata analizata",
                "Rata ceruta",
                "stressed_monthly_payment",
                "analyzed_monthly_payment",
            ],
        ),
        gmi_pct=_extract_after_labels(financial_text, ["GMI rezultat", "GMI", "gmi_pct"]),
        maturity_age=_extract_after_labels(
            financial_text,
            ["Varsta la maturitate", "Varsta maturitate", "maturity_age"],
        ),
        max_credit_amount=_extract_after_labels(
            financial_text,
            [
                "Suma maxima recomandata prin GMI si plafon produs",
                "Suma maxima recomandata prin GMI",
                "Suma maxima recomandata",
                "Suma maxima credit",
                "Maximum credit amount",
                "max_credit_amount",
            ],
        ),
    )


def extract_decision_label(text: str) -> str | None:
    decision_match = re.search(
        r"Decizi(?:e|a)\s*[:|]\s*(APROBAT(?:A)?|APROBARE|RESPINS(?:A)?|RESPINGERE|ANALIZA\s+MANUALA|MANUAL\s+REVIEW)",
        text,
        flags=re.IGNORECASE,
    )
    if decision_match:
        return normalize_decision(decision_match.group(1))

    lines = [line.strip(" #|\t") for line in text.splitlines()]
    for index, line in enumerate(lines):
        if normalize_label(line) in {"decizie", "decizia"}:
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
        "Aceasta sectiune foloseste metoda deterministica calculata in Python ca validator. "
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


def request_freeform_credit_calculation(
    profile: ClientProfile,
    sources_markdown: str,
) -> tuple[dict[str, object] | None, str | None]:
    system_prompt = (
        "Esti asistentul local de creditare pentru o aplicatie educationala RAG. "
        "Calculezi singur valorile financiare si returnezi un tabel Markdown compact. "
        "Nu folosi JSON in acest raspuns."
    )
    user_prompt = (
        "Modelul anterior nu a produs o schema JSON usor de interpretat. "
        "Recalculeaza profilul si raspunde in Markdown simplu, cu exact aceste randuri in tabelul final.\n\n"
        "Profil client in JSON:\n"
        f"{profile_as_prompt_json(profile)}\n\n"
        f"{critical_profile_checks_prompt(profile)}\n"
        f"{operating_rules_prompt()}\n"
        f"{calculation_guardrails_prompt()}\n"
        "Include o sectiune 'Detalii calcul' cu exact 4 bullets pentru debugging, inaintea tabelului final. "
        "Fiecare bullet trebuie sa contina formula=..., valori=..., rezultat=..., pentru aceste calcule:\n"
        "- Rata noua analizata si rata dupa stres\n"
        "- GMI rezultat\n"
        "- Varsta la maturitate\n"
        "- Suma maxima recomandata\n\n"
        "La final include acest tabel cu doua coloane, Indicator si Valoare:\n"
        "- Decizie\n"
        "- Venit declarat\n"
        "- Pondere venit\n"
        "- Venit eligibil ponderat\n"
        "- Capacitate maxima totala rate (40% GMI)\n"
        "- Rate existente\n"
        "- Capacitate disponibila pentru rata noua\n"
        "- Rata noua analizata\n"
        "- Rata noua analizata, dupa stres daca se aplica\n"
        "- GMI rezultat\n"
        "- Varsta la maturitate\n"
        "- Suma maxima recomandata prin GMI si plafon produs\n"
        "- Plafon produs\n\n"
        f"Fragmente RAG disponibile:\n{sources_markdown}"
    )
    raw_answer = optional_llm_summary(
        system_prompt,
        user_prompt,
        response_format_json=False,
        max_tokens_override=2500,
    )
    return canonicalize_llm_credit_json(None, raw_answer), raw_answer


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
        f"{operating_rules_prompt()}\n"
        f"{calculation_guardrails_prompt()}\n\n"
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
                "\n\nRaspunsul anterior nu a putut fi citit ca JSON valid. "
                "Genereaza de la zero un JSON mai scurt, complet si valid. "
                "Nu continua raspunsul anterior si nu adauga text in afara obiectului JSON. "
                "Foloseste exact cele 4 elemente cerute in calculation_details, cu formula, valori si rezultat."
            )
        raw_answer = optional_llm_summary(
            system_prompt,
            user_prompt,
            response_format_json=True,
            max_tokens_override=3000,
        )
        data = canonicalize_llm_credit_json(extract_json_object(raw_answer), raw_answer)
        validation_errors = validate_llm_credit_json(data, profile, deterministic)
        if data is not None:
            for _ in range(2):
                reviewed_data, reviewed_raw = request_llm_credit_self_review(
                    profile,
                    sources_markdown,
                    data,
                )
                if reviewed_data is None:
                    break
                data = reviewed_data
                raw_answer = reviewed_raw
                validation_errors = validate_llm_credit_json(data, profile, deterministic)
                if not needs_llm_self_review(profile, data):
                    break
            if needs_llm_self_review(profile, data):
                adjudication, adjudication_raw = request_llm_decision_adjudication(profile, data)
                if adjudication is not None:
                    data = merge_llm_decision_adjudication(data, adjudication)
                    raw_answer = adjudication_raw or raw_answer
                    validation_errors = validate_llm_credit_json(data, profile, deterministic)
            if _numeric_field_count(data) < 6:
                freeform_data, freeform_raw = request_freeform_credit_calculation(profile, sources_markdown)
                if _numeric_field_count(freeform_data) > _numeric_field_count(data):
                    data = freeform_data
                    raw_answer = freeform_raw or raw_answer
                    validation_errors = validate_llm_credit_json(data, profile, deterministic)
            return data, raw_answer, validation_errors
    freeform_data, freeform_raw = request_freeform_credit_calculation(profile, sources_markdown)
    if freeform_data is not None:
        return freeform_data, freeform_raw, validate_llm_credit_json(freeform_data, profile, deterministic)
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
