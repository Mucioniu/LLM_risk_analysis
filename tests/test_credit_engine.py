import unittest

from credit_assistant.credit_engine import ClientProfile, Decision, evaluate_client


class CreditEngineTests(unittest.TestCase):
    def test_annex_case_1_is_approved(self) -> None:
        profile = ClientProfile(
            age=30,
            term_months=60,
            fico=720,
            monthly_income=15000,
            income_type="Salariu - contract nedeterminat",
            existing_monthly_debts=0,
            requested_amount=0,
            requested_monthly_payment=4000,
            annual_interest_pct=10,
            sector="IT",
            current_job_tenure_months=4,
            previous_job_tenure_months=24,
            gap_days_between_jobs=5,
        )

        result = evaluate_client(profile)

        self.assertIs(result.decision, Decision.APPROVED)
        self.assertEqual(round(result.gmi, 3), 0.267)

    def test_annex_case_2_rejected_by_age(self) -> None:
        profile = ClientProfile(
            age=67,
            term_months=60,
            fico=690,
            monthly_income=4500,
            income_type="Pensie permanenta",
            existing_monthly_debts=0,
            requested_amount=50000,
            annual_interest_pct=10,
        )

        result = evaluate_client(profile)

        self.assertIs(result.decision, Decision.REJECTED)
        self.assertTrue(any("70" in reason for reason in result.reject_reasons))

    def test_annex_case_3_rejected_by_pfa_haircut(self) -> None:
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

        result = evaluate_client(profile)

        self.assertIs(result.decision, Decision.REJECTED)
        self.assertEqual(result.weighted_income, 7500)


if __name__ == "__main__":
    unittest.main()

