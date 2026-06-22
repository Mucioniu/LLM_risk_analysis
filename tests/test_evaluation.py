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
    annuity_examples_prompt,
    canonicalize_llm_credit_json,
    compare_llm_to_deterministic,
    credit_json_schema_prompt,
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

    def test_canonicalize_translated_json_response(self) -> None:
        raw = {
            "decizia": "RESPINS",
            "detalii_financiare": {
                "venit_ponderat": 7500,
                "capacitate_plata_disponibila": 3000,
                "rata_ceruta": 3500,
                "gmi": "46.67%",
                "max_credite": 42000,
            },
            "motiv": "GMI depaseste limita de 40%.",
            "sursa": [{"text": "Manual_Extins_Creditare_NovaTech_v3.pdf"}],
        }

        data = canonicalize_llm_credit_json(raw)

        self.assertIsNotNone(data)
        assert data is not None
        self.assertEqual(data["decision"], "RESPINS")
        financial = data["financial"]
        self.assertEqual(financial["weighted_income"], 7500)
        self.assertEqual(financial["analyzed_monthly_payment"], 3500)
        self.assertEqual(financial["stressed_monthly_payment"], 3500)
        self.assertEqual(financial["gmi_pct"], 46.67)
        self.assertEqual(data["rejection_reasons"], ["GMI depaseste limita de 40%."])

    def test_canonicalize_markdown_response_without_json(self) -> None:
        text = """
| Indicator | Valoare |
|---|---:|
| Decizie | RESPINS |
| Venit declarat | 10000 RON |
| Pondere venit | 75% |
| Venit ponderat | 7500 RON |
| Capacitate maxima | 3000 RON |
| Rate existente | 0 RON |
| Capacitate plata disponibila | 3000 RON |
| Rata ceruta | 3500 RON |
| GMI | 46.67% |
| Varsta maturitate | 40 ani |
| Suma maxima credit | 141196.11 RON |
| Plafon produs | 150000 RON |
"""

        data = canonicalize_llm_credit_json(None, text)

        self.assertIsNotNone(data)
        assert data is not None
        self.assertEqual(data["decision"], "RESPINS")
        financial = data["financial"]
        self.assertEqual(financial["declared_income"], 10000)
        self.assertEqual(financial["weighted_income"], 7500)
        self.assertEqual(financial["stressed_monthly_payment"], 3500)
        self.assertEqual(financial["max_credit_amount"], 141196.11)
        self.assertEqual(financial["product_cap"], 150000)

    def test_canonicalize_markdown_preserves_calculation_details(self) -> None:
        text = """
Detalii calcul
- Rata noua analizata: formula=P*r/(1-(1+r)^(-n)); valori=P=100000, r=0.008333, n=60; rezultat=2124.70 RON.
- GMI rezultat: formula=(rate_existente + rata_dupa_stres) / venit_ponderat * 100; valori=(0+2124.70)/15000*100; rezultat=14.16%.
- Varsta la maturitate: formula=varsta + durata_credit_luni / 12; valori=35+60/12; rezultat=40 ani.
- Suma maxima recomandata: formula=min(150000, capacitate_disponibila*(1-(1+r)^(-n))/r); valori=6000; rezultat=150000 RON.

| Indicator | Valoare |
|---|---:|
| Decizie | APROBAT |
| Venit declarat | 15000 RON |
| Pondere venit | 100% |
| Venit ponderat | 15000 RON |
| Capacitate maxima | 6000 RON |
| Rate existente | 0 RON |
| Capacitate plata disponibila | 6000 RON |
| Rata ceruta | 2124.70 RON |
| GMI | 14.16% |
| Varsta maturitate | 40 ani |
| Suma maxima credit | 150000 RON |
| Plafon produs | 150000 RON |
"""

        data = canonicalize_llm_credit_json(None, text)

        self.assertIsNotNone(data)
        assert data is not None
        details = data["calculation_details"]
        self.assertEqual(len(details), 4)
        self.assertIn("formula=", details[0])
        self.assertIn("GMI rezultat", details[1])

    def test_calculation_details_do_not_pollute_financial_extraction(self) -> None:
        text = """
Detalii calcul
- Rata noua analizata si rata dupa stres: formula=(suma_solicitata_ron * (dobanda_anuala_pct / 100 / 12)) / (1 - (1 + dobanda_anuala_pct / 100 / 12)^(-durata_credit_luni)), valori=suma_solicitata_ron=100000, dobanda_anuala_pct=10, durata_credit_luni=60, rezultat=1754.83 RON.
- GMI rezultat: formula=(rate_existente_lunare_ron + rata_noua_analizata) / venit_eligibil_ponderat * 100, valori=rate_existente_lunare_ron=0, rata_noua_analizata=1754.83, venit_eligibil_ponderat=15000, rezultat=11.69%.
- Suma maxima recomandata prin GMI si plafon produs: formula=(capacitate_disponibila_pentru_rata_noua * (1 - (1 + dobanda_anuala_pct / 100 / 12)^(-durata_credit_luni))) / (dobanda_anuala_pct / 100 / 12), valori=capacitate_disponibila_pentru_rata_noua=6000, rezultat=53879.44 RON.

Calcul financiar
| Indicator | Valoare |
|---|---:|
| Decizie | APROBAT |
| Venit declarat | 15000 RON |
| Pondere venit | 100% |
| Venit eligibil ponderat | 15000 RON |
| Capacitate maxima totala rate (40% GMI) | 6000 RON |
| Rate existente | 0 RON |
| Capacitate disponibila pentru rata noua | 6000 RON |
| Rata noua analizata | 2124.70 RON |
| Rata noua analizata, dupa stres daca se aplica | 2124.70 RON |
| GMI rezultat | 14.16% |
| Varsta la maturitate | 40 ani |
| Suma maxima recomandata prin GMI si plafon produs | 150000 RON |
| Plafon produs | 150000 RON |
"""

        data = canonicalize_llm_credit_json(None, text)

        self.assertIsNotNone(data)
        assert data is not None
        financial = data["financial"]
        self.assertEqual(financial["existing_monthly_debts"], 0)
        self.assertEqual(financial["available_payment_capacity"], 6000)
        self.assertEqual(financial["stressed_monthly_payment"], 2124.70)
        self.assertEqual(financial["gmi_pct"], 14.16)
        self.assertEqual(financial["max_credit_amount"], 150000)
        self.assertEqual(financial["product_cap"], 150000)

    def test_missing_financial_fields_are_backfilled_from_trace_results(self) -> None:
        text = """
Decizie: APROBAT
Calcul financiar
Indicator	Valoare
Venit declarat	15,000.00 RON
Pondere venit	100.00%
Venit eligibil ponderat	15,000.00 RON
Capacitate maxima totala rate (40% GMI)	6,000.00 RON

Detalii calcul
- Rata noua analizata si rata dupa stres: formula=(suma_solicitata_ron * (dobanda_anuala_pct / 100 / 12)) / (1 - (1 + dobanda_anuala_pct / 100 / 12)^(-durata_credit_luni)), valori=suma_solicitata_ron=100000, dobanda_anuala_pct=10, durata_credit_luni=60, rezultat=1754.89.
- GMI rezultat: formula=(rate_existente_lunare_ron + rata_noua_analizata) / venit_eligibil_ponderat * 100, valori=rate_existente_lunare_ron=0, rata_noua_analizata=1754.89, venit_eligibil_ponderat=15000, rezultat=35.09.
- Varsta la maturitate: formula=varsta + durata_credit_luni / 12, valori=varsta=35, durata_credit_luni=60, rezultat=40.0.
- Suma maxima recomandata prin GMI si plafon produs: formula=(capacitate_disponibila_pentru_rata_noua * (1 - (1 + dobanda_anuala_pct / 100 / 12)^(-durata_credit_luni))) / (dobanda_anuala_pct / 100 / 12), valori=capacitate_disponibila_pentru_rata_noua=6000, dobanda_anuala_pct=10, durata_credit_luni=60, rezultat=53948.72.
"""

        data = canonicalize_llm_credit_json(None, text)

        self.assertIsNotNone(data)
        assert data is not None
        financial = data["financial"]
        self.assertEqual(financial["existing_monthly_debts"], 0)
        self.assertEqual(financial["available_payment_capacity"], 6000)
        self.assertEqual(financial["analyzed_monthly_payment"], 1754.89)
        self.assertEqual(financial["stressed_monthly_payment"], 1754.89)
        self.assertEqual(financial["gmi_pct"], 35.09)
        self.assertEqual(financial["maturity_age"], 40)
        self.assertEqual(financial["max_credit_amount"], 53948.72)
        self.assertEqual(financial["product_cap"], 150000)

    def test_schema_prompt_requires_four_calculation_trace_steps(self) -> None:
        prompt = credit_json_schema_prompt()

        self.assertIn("calculation_details trebuie sa contina exact 4 elemente", prompt)
        self.assertIn("Rata noua analizata", prompt)
        self.assertIn("GMI rezultat", prompt)
        self.assertIn("Varsta la maturitate", prompt)
        self.assertIn("Suma maxima recomandata", prompt)

    def test_annuity_examples_prompt_calibrates_base_formula(self) -> None:
        prompt = annuity_examples_prompt()

        self.assertIn("r = 10 / 100 / 12 = 0.0083333333", prompt)
        self.assertIn("numitor = 1 - 0.6077885915 = 0.3922114085", prompt)
        self.assertIn("2124.70 RON", prompt)
        self.assertIn("1375.49", prompt)
        self.assertIn("1754.89", prompt)

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

    def test_self_review_detects_phi_style_calculation_shortcuts(self) -> None:
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
        data = {
            "decision": "APROBAT",
            "financial": {
                "declared_income": 15000,
                "income_weight_pct": 100,
                "weighted_income": 15000,
                "max_monthly_payment": 6000,
                "existing_monthly_debts": 0,
                "available_payment_capacity": 6000,
                "analyzed_monthly_payment": 1666.67,
                "stressed_monthly_payment": 1600,
                "gmi_pct": 14.16,
                "maturity_age": 95,
                "max_credit_amount": 100000,
                "product_cap": 225000,
            },
            "calculation_details": [],
            "rejection_reasons": [],
            "manual_review_reasons": [],
            "observations": [],
            "rag_sources": [],
        }

        findings = llm_self_review_findings(profile, data)
        joined = " ".join(findings)

        self.assertTrue(needs_llm_self_review(profile, data))
        self.assertIn("Varsta la maturitate", joined)
        self.assertIn("formula anuitatii", joined)
        self.assertIn("rata dupa stres", joined)
        self.assertIn("Plafonul produsului", joined)
        self.assertIn("Suma maxima recomandata pare copiata", joined)

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
