from __future__ import annotations

from pathlib import Path

from .credit_engine import ClientProfile, evaluate_client
from .llm import optional_llm_summary
from .rag import RagIndex, format_sources


DEFAULT_DOCX = Path("Manual_Extins_Creditare_NovaTech_v3.docx")
DEFAULT_PDF = Path("Manual_Extins_Creditare_NovaTech_v3.pdf")


def default_corpus_paths() -> list[Path]:
    if DEFAULT_DOCX.exists():
        return [DEFAULT_DOCX]
    if DEFAULT_PDF.exists():
        return [DEFAULT_PDF]
    raise FileNotFoundError("Nu gasesc manualul NovaTech in directorul proiectului.")


def build_default_index() -> RagIndex:
    return RagIndex.from_paths(default_corpus_paths())


def build_analysis_markdown(profile: ClientProfile, index: RagIndex, use_llm: bool = False) -> str:
    evaluation = evaluate_client(profile)
    query = (
        "criterii eligibilitate varsta FICO istoric creditare venituri haircuts "
        "grad maxim indatorare GMI formula produs NovaFlex suma maxima credit"
    )
    retrieved = index.search(query, top_k=5)
    sources = format_sources(retrieved, max_chars=650)

    rejection = "\n".join(f"- {reason}" for reason in evaluation.reject_reasons) or "- Nu exista."
    manual = (
        "\n".join(f"- {reason}" for reason in evaluation.manual_review_reasons)
        or "- Nu este necesara."
    )
    warnings = "\n".join(f"- {warning}" for warning in evaluation.warnings) or "- Nu exista."

    base = f"""## Decizie: {evaluation.decision.value}

### Calcul financiar
- Venit declarat: {profile.monthly_income:,.2f} RON/luna
- Pondere venit: {evaluation.income_weight * 100:.0f}%
- Venit eligibil ponderat: {evaluation.weighted_income:,.2f} RON/luna
- Capacitate maxima totala rate (40% GMI): {evaluation.max_monthly_payment:,.2f} RON/luna
- Rate existente: {profile.existing_monthly_debts:,.2f} RON/luna
- Capacitate disponibila pentru rata noua: {evaluation.available_payment_capacity:,.2f} RON/luna
- Rata noua analizata, dupa stres daca se aplica: {evaluation.stressed_monthly_payment:,.2f} RON/luna
- GMI rezultat: {evaluation.gmi * 100:.2f}%
- Varsta la maturitate: {evaluation.maturity_age:.1f} ani
- Suma maxima recomandata prin GMI si plafon produs: {evaluation.max_credit_amount:,.2f} RON

### Motive de respingere
{rejection}

### Motive de analiza manuala
{manual}

### Observatii
{warnings}

### Surse RAG folosite
{sources}
"""

    if not use_llm:
        return base

    llm_answer = optional_llm_summary(
        "Esti un asistent pentru o aplicatie educationala RAG de creditare fictiva. "
        "Raspunde in romana, citeaza doar informatiile din context si nu inventa reguli.",
        f"Profil client:\n{profile}\n\nRezultat calculator:\n{base}\n\nContext RAG:\n{sources}",
    )
    if not llm_answer:
        return base + "\n\n### Rezumat LLM\nLLM-ul nu este activ. Seteaza OPENAI_API_KEY pentru generare."
    return base + f"\n\n### Rezumat LLM\n{llm_answer}"


def answer_policy_question(question: str, index: RagIndex, use_llm: bool = False) -> str:
    retrieved = index.search(question, top_k=5)
    sources = format_sources(retrieved, max_chars=900)
    if not use_llm:
        return f"### Fragmente relevante\n{sources}"

    llm_answer = optional_llm_summary(
        "Raspunde strict pe baza fragmentelor RAG. Daca informatia lipseste, spune ca lipseste.",
        f"Intrebare: {question}\n\nFragmente:\n{sources}",
    )
    if not llm_answer:
        return f"### Fragmente relevante\n{sources}\n\nLLM-ul nu este activ."
    return f"### Raspuns\n{llm_answer}\n\n### Fragmente relevante\n{sources}"
