import unittest

from credit_assistant.evaluation import (
    decision_consistency,
    format_score,
    keyword_coverage,
    numeric_consistency,
    required_sections_score,
)
from credit_assistant.credit_engine import ClientProfile, evaluate_client
from credit_assistant.service import (
    compare_llm_to_deterministic,
    extract_json_object,
    extract_llm_decision,
    format_llm_credit_json_markdown,
    llm_json_to_extracted,
    llm_self_review_flags_prompt,
    llm_self_review_findings,
    merge_llm_decision_adjudication,
    normalize_credit_markdown,
    needs_llm_self_review,
    validate_llm_credit_json,
)


class EvaluationMetricTests(unittest.TestCase):
    def test_keyword_coverage_scores_partial_match(self) -> None:
        metric = keyword_coverage("PFA are pondere 75%.", ["PFA", "75", "chirii"])

        self.assertEqual(metric.name, "acoperire_cuvinte_cheie")
        self.assertAlmostEqual(metric.score, 2 / 3)

    def test_format_score_detects_clean_markdown(self) -> None:
        metric = format_score("## Raspuns\n\n- element")

        self.assertEqual(metric.score, 1.0)

    def test_numeric_consistency_ignores_spaces(self) -> None:
        metric = numeric_consistency("Venit: 15,000.00 RON", ["15,000.00"])

        self.assertEqual(metric.score, 1.0)

    def test_decision_consistency(self) -> None:
        metric = decision_consistency("## Decizie: APROBAT", "APROBAT")

        self.assertEqual(metric.score, 1.0)

    def test_required_sections_score(self) -> None:
        text = (
            "Decizie\nCalcul financiar\nMotive de respingere\n"
            "Motive de analiza manuala\nObservatii\nSurse RAG folosite"
        )
        metric = required_sections_score(text)

        self.assertEqual(metric.score, 1.0)

    def test_extract_llm_decision_from_markdown_table(self) -> None:
        text = """
## Decizie: APROBAT

### Calcul financiar

| Indicator | Valoare |
|---|---:|
| Venit declarat | 15,000.00 RON/luna |
| Pondere venit | 100% |
| Venit eligibil ponderat | 15,000.00 RON/luna |
| GMI rezultat | 14.16% |
"""

        extracted = extract_llm_decision(text)

        self.assertEqual(extracted.decision, "APROBAT")
        self.assertEqual(extracted.declared_income, 15000)
        self.assertEqual(extracted.income_weight_pct, 100)
        self.assertEqual(extracted.weighted_income, 15000)
        self.assertEqual(extracted.gmi_pct, 14.16)

    def test_extract_llm_decision_from_mistral_plain_table(self) -> None:
        text = """
Decizie
Aprobat

Calcul financiar
Eticheta	Valoare
Venit declarat (RON)	15000.0
Pondere venit (%)	100%
Venit eligibil ponderat (RON)	15000.0
Capacitate maxima totala rate (40% GMI) (RON)	6000.0
Rate existente (RON)	0.0
Capacitate disponibila pentru rata noua (RON)	6000.0
Rata noua analizata (RON)	2124.70
Rata noua dupa stres (RON)	2124.70
GMI rezultat (%)	14.16%
Varsta la maturitate	40
Suma maxima recomandata prin GMI si plafon produs (RON)	150000
"""

        normalized = normalize_credit_markdown(text)
        extracted = extract_llm_decision(normalized)

        self.assertIn("## Decizie: APROBAT", normalized)
        self.assertIn("| Venit declarat (RON) | 15000.0 |", normalized)
        self.assertEqual(extracted.decision, "APROBAT")
        self.assertEqual(extracted.declared_income, 15000)
        self.assertEqual(extracted.income_weight_pct, 100)
        self.assertEqual(extracted.max_monthly_payment, 6000)
        self.assertEqual(extracted.stressed_monthly_payment, 2124.70)
        self.assertEqual(extracted.maturity_age, 40)

    def test_extract_llm_decision_from_mistral_malformed_financial_rows(self) -> None:
        text = """
Decizie: RESPINS
Calcul financiar
Eticheta	Valoare
Venit declarat	15000.0 RON
Pondere venit	100%
Venit eligibil ponderat	15000.0 RON
Capacitate maxima totala rate (40% GMI)	6000.0 RON
Rate existente	0.0 RON
Capacitate disponibila pentru rata noua	6000.0 RON
Rata noua analizata	17,238.49 RON
dupa stres daca se aplica
| | GMI rezultat | 42.8% | | Varsta la maturitate | 80 ani | | Suma maxima recomandata prin GMI | 150,000 RON | | plafon produs | 150,000 RON |
Detalii calcul
Varsta la maturitate: 35 + 60/12 = 80 ani.
GMI rezultat: (0 + 17238.49) / 15000 * 100 = 114.9%.
"""

        normalized = normalize_credit_markdown(text)
        extracted = extract_llm_decision(normalized)

        self.assertNotIn("35601280", normalized)
        self.assertEqual(extracted.decision, "RESPINS")
        self.assertEqual(extracted.declared_income, 15000)
        self.assertEqual(extracted.income_weight_pct, 100)
        self.assertEqual(extracted.stressed_monthly_payment, 17238.49)
        self.assertEqual(extracted.gmi_pct, 42.8)
        self.assertEqual(extracted.maturity_age, 80)
        self.assertEqual(extracted.max_credit_amount, 150000)

    def test_compare_llm_to_deterministic_scores_matching_values(self) -> None:
        profile = ClientProfile(
            age=35,
            term_months=60,
            fico=720,
            monthly_income=15000,
            income_type="Salariu - contract nedeterminat",
            existing_monthly_debts=0,
            requested_amount=100000,
            requested_monthly_payment=0,
            annual_interest_pct=10,
        )
        deterministic = evaluate_client(profile)
        text = f"""
## Decizie: APROBAT

| Indicator | Valoare |
|---|---:|
| Venit declarat | 15,000.00 RON/luna |
| Pondere venit | 100% |
| Venit eligibil ponderat | 15,000.00 RON/luna |
| Capacitate maxima totala rate (40% GMI) | 6,000.00 RON/luna |
| Rate existente | 0.00 RON/luna |
| Capacitate disponibila pentru rata noua | 6,000.00 RON/luna |
| Rata noua analizata, dupa stres daca se aplica | {deterministic.stressed_monthly_payment:,.2f} RON/luna |
| GMI rezultat | {deterministic.gmi * 100:.2f}% |
| Varsta la maturitate | 40.0 ani |
| Suma maxima recomandata prin GMI si plafon produs | 150,000.00 RON |
"""
        extracted = extract_llm_decision(text)

        _, metrics = compare_llm_to_deterministic(profile, deterministic, extracted)

        self.assertEqual(metrics["scor_total_llm_vs_formule"], 1.0)

    def test_extract_json_object_from_fenced_response(self) -> None:
        data = extract_json_object('```json\n{"decision": "RESPINS", "financial": {}}\n```')

        self.assertIsNotNone(data)
        self.assertEqual(data["decision"], "RESPINS")

    def test_schema_validation_rejects_wrong_pfa_decision(self) -> None:
        profile = ClientProfile(
            age=35,
            term_months=60,
            fico=680,
            monthly_income=10000,
            income_type="PFA/PFI",
            existing_monthly_debts=0,
            requested_amount=0,
            requested_monthly_payment=3500,
            annual_interest_pct=10,
        )
        deterministic = evaluate_client(profile)
        data = {
            "decision": "APROBAT",
            "financial": {
                "declared_income": 10000,
                "income_weight_pct": 75,
                "weighted_income": 7500,
                "max_monthly_payment": 3000,
                "existing_monthly_debts": 0,
                "available_payment_capacity": 3000,
                "analyzed_monthly_payment": 2984.16,
                "stressed_monthly_payment": 2984.16,
                "gmi_pct": 39.79,
                "maturity_age": 40,
                "max_credit_amount": 0,
                "product_cap": 150000,
            },
            "calculation_details": [],
            "rejection_reasons": [],
            "manual_review_reasons": [],
            "observations": [],
            "rag_sources": [],
        }

        errors = validate_llm_credit_json(data, profile, deterministic)

        self.assertTrue(any("RESPINS" in error for error in errors))
        self.assertTrue(any("gmi_pct" in error for error in errors))

    def test_json_markdown_and_extraction(self) -> None:
        data = {
            "decision": "RESPINS",
            "financial": {
                "declared_income": 10000,
                "income_weight_pct": 75,
                "weighted_income": 7500,
                "max_monthly_payment": 3000,
                "existing_monthly_debts": 0,
                "available_payment_capacity": 3000,
                "analyzed_monthly_payment": 3500,
                "stressed_monthly_payment": 3500,
                "gmi_pct": 46.67,
                "maturity_age": 40,
                "max_credit_amount": 0,
                "product_cap": 150000,
            },
            "calculation_details": ["GMI = 3500 / 7500 * 100."],
            "rejection_reasons": ["GMI depaseste limita de 40%."],
            "manual_review_reasons": [],
            "observations": [],
            "rag_sources": ["[1] Manual_Extins_Creditare_NovaTech_v3.pdf"],
        }

        markdown = format_llm_credit_json_markdown(data)
        extracted = llm_json_to_extracted(data)

        self.assertIn("## Decizie: RESPINS", markdown)
        self.assertEqual(extracted.decision, "RESPINS")
        self.assertEqual(extracted.weighted_income, 7500)
        self.assertEqual(extracted.gmi_pct, 46.67)

    def test_self_review_detects_approved_case_with_hard_rejections(self) -> None:
        profile = ClientProfile(
            age=35,
            term_months=60,
            fico=720,
            monthly_income=15000,
            income_type="Salariu - contract nedeterminat",
            existing_monthly_debts=0,
            requested_amount=1000000,
            requested_monthly_payment=0,
            annual_interest_pct=10,
        )
        data = {
            "decision": "APROBAT",
            "financial": {
                "declared_income": 15000,
                "income_weight_pct": 100,
                "weighted_income": 15000,
                "max_monthly_payment": 6000,
                "existing_monthly_debts": 0,
                "available_payment_capacity": 6000,
                "analyzed_monthly_payment": 17948.32,
                "stressed_monthly_payment": 17948.32,
                "gmi_pct": 119.65,
                "maturity_age": 40,
                "max_credit_amount": 150000,
                "product_cap": 150000,
            },
            "calculation_details": [],
            "rejection_reasons": [],
            "manual_review_reasons": [],
            "observations": [],
            "rag_sources": [],
        }

        findings = llm_self_review_findings(profile, data)

        self.assertTrue(needs_llm_self_review(profile, data))
        self.assertTrue(any("RESPINS" in finding for finding in findings))
        self.assertTrue(any("GMI" in finding for finding in findings))
        self.assertTrue(any("plafon" in finding for finding in findings))
        flags = llm_self_review_flags_prompt(profile, data)
        self.assertIn("suma_peste_plafon_produs: 1000000.00 > 150000.00 => DA", flags)
        self.assertIn("gmi_returnat_de_model_peste_limita: 119.65% > 40% => DA", flags)

    def test_merge_llm_decision_adjudication_keeps_financial_values(self) -> None:
        data = {
            "decision": "APROBAT",
            "financial": {"stressed_monthly_payment": 17298.34, "gmi_pct": 115.33},
            "rejection_reasons": [],
            "manual_review_reasons": [],
            "observations": ["GMI peste limita de 40%"],
        }
        adjudication = {
            "decision": "RESPINS",
            "rejection_reasons": ["GMI peste limita de 40%."],
            "manual_review_reasons": [],
            "observations": [],
        }

        merged = merge_llm_decision_adjudication(data, adjudication)

        self.assertEqual(merged["decision"], "RESPINS")
        self.assertEqual(merged["financial"], data["financial"])
        self.assertEqual(merged["rejection_reasons"], ["GMI peste limita de 40%."])

    def test_merge_llm_decision_adjudication_humanizes_flag_reasons(self) -> None:
        data = {
            "decision": "APROBAT",
            "financial": {},
            "rejection_reasons": [],
            "manual_review_reasons": [],
            "observations": [],
        }
        adjudication = {
            "decision": "RESPINS",
            "rejection_reasons": ["suma_peste_plafon_produs"],
            "manual_review_reasons": [],
            "observations": [],
        }

        merged = merge_llm_decision_adjudication(data, adjudication)

        self.assertEqual(
            merged["rejection_reasons"],
            ["Suma solicitata depaseste plafonul produsului de 150,000 RON."],
        )


if __name__ == "__main__":
    unittest.main()
