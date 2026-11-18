import math
import unittest

import numpy as np

from analysis.summary_ablation_uncertainty import difference_bounds, t4_tail, t4_critical


class SummaryUncertaintyTests(unittest.TestCase):
    def test_t4_matches_standard_critical_value(self):
        self.assertAlmostEqual(t4_critical(.05), 2.7764451051977987, places=11)
        self.assertAlmostEqual(t4_tail(0), 1)

    def test_envelope_covers_every_actual_paired_t_interval_in_simulation(self):
        rng = np.random.default_rng(71)
        critical = t4_critical(.05)
        for correlation in (-1, -.5, 0, .5, 1):
            for _ in range(30):
                x = rng.normal(.15, .005, 5)
                y = .14 + correlation * (x - .15) + math.sqrt(1-correlation**2) * rng.normal(0, .004, 5)
                difference = x-y
                actual_mean = float(difference.mean())
                actual_se = float(difference.std(ddof=1) / math.sqrt(5))
                bound = difference_bounds(f'{x.mean():.4f}', f'{x.std(ddof=1):.4f}',
                                          f'{y.mean():.4f}', f'{y.std(ddof=1):.4f}')
                self.assertLessEqual(bound['ci_low'], actual_mean-critical*actual_se+1e-14)
                self.assertGreaterEqual(bound['ci_high'], actual_mean+critical*actual_se-1e-14)
                if actual_se > 1e-14:
                    self.assertGreaterEqual(bound['p_upper']+1e-12, t4_tail(abs(actual_mean)/actual_se))

    def test_rounding_and_direction_for_current_beta0_example(self):
        result = difference_bounds('0.1469', '0.0012', '0.1461', '0.0018')
        self.assertAlmostEqual(result['difference'], .0008)
        self.assertAlmostEqual(result['se_upper'], .0031 / math.sqrt(5))
        self.assertLess(result['ci_low'], 0)
        self.assertGreater(result['ci_high'], 0)


if __name__ == '__main__':
    unittest.main()
