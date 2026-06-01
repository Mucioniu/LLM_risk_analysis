import unittest

from credit_assistant.evaluation import (
    decision_consistency,
    format_score,
    keyword_coverage,
    numeric_consistency,
    required_sections_score,
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


if __name__ == "__main__":
    unittest.main()
