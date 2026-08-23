import unittest

from credit_assistant.evaluation import (
    decision_consistency,
    format_score,
    keyword_coverage,
    load_evaluation_cases,
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
    def test_evaluation_cases_are_unique_and_match_reference_engine(self) -> None:
        cases = load_evaluation_cases()
        policy_cases = cases["policy_questions"]
        client_cases = cases["client_cases"]
        all_ids = [case["id"] for case in policy_cases + client_cases]

        self.assertGreaterEqual(len(policy_cases), 13)
        self.assertGreaterEqual(len(client_cases), 22)
        self.assertEqual(len(all_ids), len(set(all_ids)))

        for case in client_cases:
            with self.subTest(case_id=case["id"]):
                profile = ClientProfile(**case["profile"])
                evaluation = evaluate_client(profile)
                self.assertEqual(evaluation.decision.value, case["expected_decision"])

    def test_keyword_coverage_scores_partial_match(self) -> None:
        metric = keyword_coverage(
            "Self-employment income has a 75% weight.",
            ["self-employment", "75", "rental"],
        )

        self.assertEqual(metric.name, "keyword_coverage")
        self.assertAlmostEqual(metric.score, 2 / 3)

    def test_format_score_detects_clean_markdown(self) -> None:
        metric = format_score("## Answer\n\n- item")

        self.assertEqual(metric.score, 1.0)

    def test_numeric_consistency_ignores_spaces(self) -> None:
        metric = numeric_consistency("Income: 15,000.00 RON", ["15,000.00"])

        self.assertEqual(metric.score, 1.0)

    def test_decision_consistency(self) -> None:
        metric = decision_consistency("## Decision: APPROVED", "APPROVED")

        self.assertEqual(metric.score, 1.0)

    def test_required_sections_score(self) -> None:
        text = (
            "Decision\nFinancial calculation\nRejection reasons\n"
            "Manual review reasons\nNotes\nRAG sources used"
        )
        metric = required_sections_score(text)

        self.assertEqual(metric.score, 1.0)

    def test_extract_llm_decision_from_markdown_table(self) -> None:
        text = """
## Decision: APPROVED

### Financial calculation

| Indicator | Value |
|---|---:|
| Declared income | 15,000.00 RON/month |
| Income weight | 100% |
| Weighted eligible income | 15,000.00 RON/month |
| DTI | 14.16% |
"""

        extracted = extract_llm_decision(text)

        self.assertEqual(extracted.decision, "APPROVED")
        self.assertEqual(extracted.declared_income, 15000)
        self.assertEqual(extracted.income_weight_pct, 100)
        self.assertEqual(extracted.weighted_income, 15000)
        self.assertEqual(extracted.dti_pct, 14.16)

    def test_extract_llm_decision_from_mistral_plain_table(self) -> None:
        text = """
Decision
Approved

Financial calculation
Label	Value
Declared income (RON)	15000.0
Income weight (%)	100%
Weighted eligible income (RON)	15000.0
Maximum total payment capacity (40% DTI) (RON)	6000.0
Existing payments (RON)	0.0
Available capacity for the new payment (RON)	6000.0
Analyzed new payment (RON)	2124.70
Analyzed new payment after stress (RON)	2124.70
DTI (%)	14.16%
Age at maturity	40
Maximum recommended amount through DTI and product cap (RON)	150000
"""

        normalized = normalize_credit_markdown(text)
        extracted = extract_llm_decision(normalized)

        self.assertIn("## Decision: APPROVED", normalized)
        self.assertIn("| Declared income (RON) | 15000.0 |", normalized)
        self.assertEqual(extracted.decision, "APPROVED")
        self.assertEqual(extracted.declared_income, 15000)
        self.assertEqual(extracted.income_weight_pct, 100)
        self.assertEqual(extracted.max_monthly_payment, 6000)
        self.assertEqual(extracted.stressed_monthly_payment, 2124.70)
        self.assertEqual(extracted.maturity_age, 40)

    def test_extract_llm_decision_from_mistral_malformed_financial_rows(self) -> None:
        text = """
Decision: REJECTED
Financial calculation
Label	Value
Declared income	15000.0 RON
Income weight	100%
Weighted eligible income	15000.0 RON
Maximum total payment capacity (40% DTI)	6000.0 RON
Existing payments	0.0 RON
Available capacity for the new payment	6000.0 RON
Analyzed new payment	17,238.49 RON
after stress if applicable
| | DTI | 42.8% | | Age at maturity | 80 years | | Maximum recommended amount through DTI | 150,000 RON | | product cap | 150,000 RON |
Calculation details
Age at maturity: 35 + 60/12 = 80 years.
DTI: (0 + 17238.49) / 15000 * 100 = 114.9%.
"""

        normalized = normalize_credit_markdown(text)
        extracted = extract_llm_decision(normalized)

        self.assertNotIn("35601280", normalized)
        self.assertEqual(extracted.decision, "REJECTED")
        self.assertEqual(extracted.declared_income, 15000)
        self.assertEqual(extracted.income_weight_pct, 100)
        self.assertEqual(extracted.stressed_monthly_payment, 17238.49)
        self.assertEqual(extracted.dti_pct, 42.8)
        self.assertEqual(extracted.maturity_age, 80)
        self.assertEqual(extracted.maximum_amount_by_dti, 150000)

    def test_compare_llm_to_deterministic_scores_matching_values(self) -> None:
        profile = ClientProfile(
            age=35,
            term_months=60,
            fico=720,
            monthly_income=15000,
            income_type="Salary - permanent contract",
            existing_monthly_debts=0,
            requested_amount=100000,
            requested_monthly_payment=0,
            annual_interest_pct=10,
        )
        deterministic = evaluate_client(profile)
        text = f"""
## Decision: APPROVED

| Indicator | Value |
|---|---:|
| Declared income | 15,000.00 RON/month |
| Income weight | 100% |
| Weighted eligible income | 15,000.00 RON/month |
| Maximum total payment capacity (40% DTI) | 6,000.00 RON/month |
| Existing payments | 0.00 RON/month |
| Available capacity for the new payment | 6,000.00 RON/month |
| Analyzed new payment after stress | {deterministic.stressed_monthly_payment:,.2f} RON/month |
| DTI | {deterministic.dti * 100:.2f}% |
| Age at maturity | 40.0 years |
| Maximum recommended amount through DTI and product cap | 150,000.00 RON |
"""
        extracted = extract_llm_decision(text)

        _, metrics = compare_llm_to_deterministic(profile, deterministic, extracted)

        self.assertEqual(metrics["overall_llm_vs_formulas_score"], 1.0)

    def test_extract_json_object_from_fenced_response(self) -> None:
        data = extract_json_object('```json\n{"decision": "REJECTED", "financial": {}}\n```')

        self.assertIsNotNone(data)
        self.assertEqual(data["decision"], "REJECTED")

    def test_canonicalize_json_response(self) -> None:
        raw = {
            "decision": "REJECTED",
            "financial": {
                "weighted_income": 7500,
                "available_payment_capacity": 3000,
                "analyzed_monthly_payment": 3500,
                "dti_pct": "46.67%",
                "maximum_amount_by_dti": 42000,
            },
            "rejection_reasons": ["DTI exceeds the 40% limit."],
            "rag_sources": ["NovaTech_Extended_Credit_Manual_v3.pdf"],
        }

        data = canonicalize_llm_credit_json(raw)

        self.assertIsNotNone(data)
        assert data is not None
        self.assertEqual(data["decision"], "REJECTED")
        financial = data["financial"]
        self.assertEqual(financial["weighted_income"], 7500)
        self.assertEqual(financial["analyzed_monthly_payment"], 3500)
        self.assertEqual(financial["stressed_monthly_payment"], 3500)
        self.assertEqual(financial["dti_pct"], 46.67)
        self.assertEqual(data["rejection_reasons"], ["DTI exceeds the 40% limit."])

    def test_canonicalize_markdown_response_without_json(self) -> None:
        text = """
| Indicator | Value |
|---|---:|
| Decision | REJECTED |
| Declared income | 10000 RON |
| Income weight | 75% |
| Weighted eligible income | 7500 RON |
| Maximum total payment capacity | 3000 RON |
| Existing payments | 0 RON |
| Available capacity for the new payment | 3000 RON |
| Analyzed new payment | 3500 RON |
| DTI | 46.67% |
| Age at maturity | 40 years |
| Maximum recommended amount | 141196.11 RON |
| Product cap | 150000 RON |
"""

        data = canonicalize_llm_credit_json(None, text)

        self.assertIsNotNone(data)
        assert data is not None
        self.assertEqual(data["decision"], "REJECTED")
        financial = data["financial"]
        self.assertEqual(financial["declared_income"], 10000)
        self.assertEqual(financial["weighted_income"], 7500)
        self.assertEqual(financial["stressed_monthly_payment"], 3500)
        self.assertEqual(financial["maximum_amount_by_dti"], 141196.11)
        self.assertEqual(financial["product_cap"], 150000)

    def test_canonicalize_markdown_preserves_calculation_details(self) -> None:
        text = """
Calculation details
- Analyzed new payment: formula=P*r/(1-(1+r)^(-n)); values=P=100000, r=0.008333, n=60; result=2124.70 RON.
- DTI: formula=(existing_payments + payment_after_stress) / weighted_income * 100; values=(0+2124.70)/15000*100; result=14.16%.
- Age at maturity: formula=age + loan_term_months / 12; values=35+60/12; result=40 years.
- Maximum recommended amount: formula=min(150000, available_capacity*(1-(1+r)^(-n))/r); values=6000; result=150000 RON.

| Indicator | Value |
|---|---:|
| Decision | APPROVED |
| Declared income | 15000 RON |
| Income weight | 100% |
| Weighted eligible income | 15000 RON |
| Maximum total payment capacity | 6000 RON |
| Existing payments | 0 RON |
| Available capacity for the new payment | 6000 RON |
| Analyzed new payment | 2124.70 RON |
| DTI | 14.16% |
| Age at maturity | 40 years |
| Maximum recommended amount | 150000 RON |
| Product cap | 150000 RON |
"""

        data = canonicalize_llm_credit_json(None, text)

        self.assertIsNotNone(data)
        assert data is not None
        details = data["calculation_details"]
        self.assertEqual(len(details), 4)
        self.assertIn("formula=", details[0])
        self.assertIn("DTI", details[1])

    def test_calculation_details_do_not_pollute_financial_extraction(self) -> None:
        text = """
Calculation details
- Analyzed new payment and payment after stress: formula=(requested_amount_ron * (annual_interest_pct / 100 / 12)) / (1 - (1 + annual_interest_pct / 100 / 12)^(-loan_term_months)), values=requested_amount_ron=100000, annual_interest_pct=10, loan_term_months=60, result=1754.83 RON.
- DTI: formula=(existing_monthly_payments_ron + analyzed_new_payment) / weighted_eligible_income * 100, values=existing_monthly_payments_ron=0, analyzed_new_payment=1754.83, weighted_eligible_income=15000, result=11.69%.
- Maximum recommended amount through DTI and product cap: formula=(available_capacity_for_new_payment * (1 - (1 + annual_interest_pct / 100 / 12)^(-loan_term_months))) / (annual_interest_pct / 100 / 12), values=available_capacity_for_new_payment=6000, result=53879.44 RON.

Financial calculation
| Indicator | Value |
|---|---:|
| Decision | APPROVED |
| Declared income | 15000 RON |
| Income weight | 100% |
| Weighted eligible income | 15000 RON |
| Maximum total payment capacity (40% DTI) | 6000 RON |
| Existing payments | 0 RON |
| Available capacity for the new payment | 6000 RON |
| Analyzed new payment | 2124.70 RON |
| Analyzed new payment after stress | 2124.70 RON |
| DTI | 14.16% |
| Age at maturity | 40 years |
| Maximum recommended amount | 150000 RON |
| Product cap | 150000 RON |
"""

        data = canonicalize_llm_credit_json(None, text)

        self.assertIsNotNone(data)
        assert data is not None
        financial = data["financial"]
        self.assertEqual(financial["existing_monthly_debts"], 0)
        self.assertEqual(financial["available_payment_capacity"], 6000)
        self.assertEqual(financial["stressed_monthly_payment"], 2124.70)
        self.assertEqual(financial["dti_pct"], 14.16)
        self.assertEqual(financial["maximum_amount_by_dti"], 150000)
        self.assertEqual(financial["product_cap"], 150000)

    def test_missing_financial_fields_are_backfilled_from_trace_results(self) -> None:
        text = """
Decision: APPROVED
Financial calculation
Indicator	Value
Declared income	15,000.00 RON
Income weight	100.00%
Weighted eligible income	15,000.00 RON
Maximum total payment capacity (40% DTI)	6,000.00 RON

Calculation details
- Analyzed new payment and payment after stress: formula=(requested_amount_ron * (annual_interest_pct / 100 / 12)) / (1 - (1 + annual_interest_pct / 100 / 12)^(-loan_term_months)), values=requested_amount_ron=100000, annual_interest_pct=10, loan_term_months=60, result=1754.89.
- DTI: formula=(existing_monthly_payments_ron + analyzed_new_payment) / weighted_eligible_income * 100, values=existing_monthly_payments_ron=0, analyzed_new_payment=1754.89, weighted_eligible_income=15000, result=35.09.
- Age at maturity: formula=age + loan_term_months / 12, values=age=35, loan_term_months=60, result=40.0.
- Maximum recommended amount through DTI and product cap: formula=(available_capacity_for_new_payment * (1 - (1 + annual_interest_pct / 100 / 12)^(-loan_term_months))) / (annual_interest_pct / 100 / 12), values=available_capacity_for_new_payment=6000, annual_interest_pct=10, loan_term_months=60, result=53948.72.
"""

        data = canonicalize_llm_credit_json(None, text)

        self.assertIsNotNone(data)
        assert data is not None
        financial = data["financial"]
        self.assertEqual(financial["existing_monthly_debts"], 0)
        self.assertEqual(financial["available_payment_capacity"], 6000)
        self.assertEqual(financial["analyzed_monthly_payment"], 1754.89)
        self.assertEqual(financial["stressed_monthly_payment"], 1754.89)
        self.assertEqual(financial["dti_pct"], 35.09)
        self.assertEqual(financial["maturity_age"], 40)
        self.assertEqual(financial["maximum_amount_by_dti"], 53948.72)
        self.assertEqual(financial["product_cap"], 150000)

    def test_schema_prompt_requires_four_calculation_trace_steps(self) -> None:
        prompt = credit_json_schema_prompt()

        self.assertIn("calculation_details must contain exactly 4 items", prompt)
        self.assertIn("Analyzed new payment", prompt)
        self.assertIn("DTI", prompt)
        self.assertIn("Age at maturity", prompt)
        self.assertIn("Maximum recommended amount", prompt)

    def test_annuity_examples_prompt_calibrates_base_formula(self) -> None:
        prompt = annuity_examples_prompt()

        self.assertIn("r = 10 / 100 / 12 = 0.0083333333", prompt)
        self.assertIn("denominator = 1 - 0.6077885915 = 0.3922114085", prompt)
        self.assertIn("2124.70 RON", prompt)
        self.assertIn("1375.49", prompt)
        self.assertIn("1754.89", prompt)
        self.assertIn("(1+r)^(-36)=0.7417397035", prompt)
        self.assertIn("P=30000 means 3 * 322.671872 = 968.015616 RON", prompt)
        self.assertIn("valid only for capacity=3000 and n=60", prompt)

    def test_schema_validation_rejects_wrong_self_employment_decision(self) -> None:
        profile = ClientProfile(
            age=35,
            term_months=60,
            fico=680,
            monthly_income=10000,
            income_type="Self-employment/liberal professions",
            existing_monthly_debts=0,
            requested_amount=0,
            requested_monthly_payment=3500,
            annual_interest_pct=10,
        )
        deterministic = evaluate_client(profile)
        data = {
            "decision": "APPROVED",
            "financial": {
                "declared_income": 10000,
                "income_weight_pct": 75,
                "weighted_income": 7500,
                "max_monthly_payment": 3000,
                "existing_monthly_debts": 0,
                "available_payment_capacity": 3000,
                "analyzed_monthly_payment": 2984.16,
                "stressed_monthly_payment": 2984.16,
                "dti_pct": 39.79,
                "maturity_age": 40,
                "maximum_amount_by_dti": 0,
                "product_cap": 150000,
            },
            "calculation_details": [],
            "rejection_reasons": [],
            "manual_review_reasons": [],
            "observations": [],
            "rag_sources": [],
        }

        errors = validate_llm_credit_json(data, profile, deterministic)

        self.assertTrue(any("REJECTED" in error for error in errors))
        self.assertTrue(any("dti_pct" in error for error in errors))

    def test_json_markdown_and_extraction(self) -> None:
        data = {
            "decision": "REJECTED",
            "financial": {
                "declared_income": 10000,
                "income_weight_pct": 75,
                "weighted_income": 7500,
                "max_monthly_payment": 3000,
                "existing_monthly_debts": 0,
                "available_payment_capacity": 3000,
                "analyzed_monthly_payment": 3500,
                "stressed_monthly_payment": 3500,
                "dti_pct": 46.67,
                "maturity_age": 40,
                "maximum_amount_by_dti": 0,
                "product_cap": 150000,
            },
            "calculation_details": ["DTI = 3500 / 7500 * 100."],
            "rejection_reasons": ["DTI exceeds the 40% limit."],
            "manual_review_reasons": [],
            "observations": [],
            "rag_sources": ["[1] NovaTech_Extended_Credit_Manual_v3.pdf"],
        }

        markdown = format_llm_credit_json_markdown(data)
        extracted = llm_json_to_extracted(data)

        self.assertIn("## Decision: REJECTED", markdown)
        self.assertEqual(extracted.decision, "REJECTED")
        self.assertEqual(extracted.weighted_income, 7500)
        self.assertEqual(extracted.dti_pct, 46.67)

    def test_self_review_detects_approved_case_with_hard_rejections(self) -> None:
        profile = ClientProfile(
            age=35,
            term_months=60,
            fico=720,
            monthly_income=15000,
            income_type="Salary - permanent contract",
            existing_monthly_debts=0,
            requested_amount=1000000,
            requested_monthly_payment=0,
            annual_interest_pct=10,
        )
        data = {
            "decision": "APPROVED",
            "financial": {
                "declared_income": 15000,
                "income_weight_pct": 100,
                "weighted_income": 15000,
                "max_monthly_payment": 6000,
                "existing_monthly_debts": 0,
                "available_payment_capacity": 6000,
                "analyzed_monthly_payment": 17948.32,
                "stressed_monthly_payment": 17948.32,
                "dti_pct": 119.65,
                "maturity_age": 40,
                "maximum_amount_by_dti": 150000,
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
        self.assertTrue(any("REJECTED" in finding for finding in findings))
        self.assertTrue(any("DTI" in finding for finding in findings))
        self.assertTrue(any("product cap" in finding for finding in findings))
        flags = llm_self_review_flags_prompt(profile, data)
        self.assertIn("requested_amount_above_product_cap: 1000000.00 > 150000.00 => YES", flags)
        self.assertIn("model_returned_dti_above_limit: 119.65% > 40% => YES", flags)

    def test_self_review_detects_phi_style_calculation_shortcuts(self) -> None:
        profile = ClientProfile(
            age=35,
            term_months=60,
            fico=720,
            monthly_income=15000,
            income_type="Salary - permanent contract",
            existing_monthly_debts=0,
            requested_amount=100000,
            requested_monthly_payment=0,
            annual_interest_pct=10,
        )
        data = {
            "decision": "APPROVED",
            "financial": {
                "declared_income": 15000,
                "income_weight_pct": 100,
                "weighted_income": 15000,
                "max_monthly_payment": 6000,
                "existing_monthly_debts": 0,
                "available_payment_capacity": 6000,
                "analyzed_monthly_payment": 1666.67,
                "stressed_monthly_payment": 1600,
                "dti_pct": 14.16,
                "maturity_age": 95,
                "maximum_amount_by_dti": 100000,
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
        self.assertIn("Age at maturity", joined)
        self.assertIn("annuity formula", joined)
        self.assertIn("payment after stress", joined)
        self.assertIn("The product cap", joined)
        self.assertIn("Maximum recommended amount appears copied", joined)

    def test_merge_llm_decision_adjudication_keeps_financial_values(self) -> None:
        data = {
            "decision": "APPROVED",
            "financial": {"stressed_monthly_payment": 17298.34, "dti_pct": 115.33},
            "rejection_reasons": [],
            "manual_review_reasons": [],
            "observations": ["DTI exceeds the 40% limit"],
        }
        adjudication = {
            "decision": "REJECTED",
            "rejection_reasons": ["DTI exceeds the 40% limit."],
            "manual_review_reasons": [],
            "observations": [],
        }

        merged = merge_llm_decision_adjudication(data, adjudication)

        self.assertEqual(merged["decision"], "REJECTED")
        self.assertEqual(merged["financial"], data["financial"])
        self.assertEqual(merged["rejection_reasons"], ["DTI exceeds the 40% limit."])

    def test_merge_llm_decision_adjudication_humanizes_flag_reasons(self) -> None:
        data = {
            "decision": "APPROVED",
            "financial": {},
            "rejection_reasons": [],
            "manual_review_reasons": [],
            "observations": [],
        }
        adjudication = {
            "decision": "REJECTED",
            "rejection_reasons": ["requested_amount_above_product_cap"],
            "manual_review_reasons": [],
            "observations": [],
        }

        merged = merge_llm_decision_adjudication(data, adjudication)

        self.assertEqual(
            merged["rejection_reasons"],
            ["The requested amount exceeds the product cap of 150,000 RON."],
        )


if __name__ == "__main__":
    unittest.main()
