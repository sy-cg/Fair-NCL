import unittest

from fairness.augmentation_diagnostics import AugmentationDiagnostics


class AugmentationDiagnosticsTests(unittest.TestCase):
    def test_summary_reports_replacement_and_skew_reduction(self):
        diagnostics = AugmentationDiagnostics()
        diagnostics.record(replaced=True, similarity=0.8, base_bias=0.6, candidate_bias=0.2)
        diagnostics.record(replaced=True, similarity=0.4, base_bias=0.3, candidate_bias=0.5)
        diagnostics.record(replaced=False)

        summary = diagnostics.summary()

        self.assertEqual(summary["attempted_positions"], 3)
        self.assertEqual(summary["replaced_positions"], 2)
        self.assertAlmostEqual(summary["replacement_rate"], 2 / 3)
        self.assertAlmostEqual(summary["lower_skew_fraction"], 0.5)
        self.assertAlmostEqual(summary["mean_skew_change"], 0.1)
        self.assertAlmostEqual(summary["mean_similarity"], 0.6)


if __name__ == "__main__":
    unittest.main()
