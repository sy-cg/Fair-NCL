import unittest

from analysis import prepare_reviewer_fairness_data as export


class ReviewerFairnessExportTests(unittest.TestCase):
    def test_full_s1_retains_every_fair_ncl_estimand_and_original_value(self):
        source = export.read_csv(export.INPUT_DIR / 'paired_comparisons.csv')
        expected = [row for row in source if row['method'] == 'fair_ncl']
        actual = export.prepare_full_paired_comparisons()
        self.assertEqual(len(actual), 672)
        self.assertEqual(actual, expected)
        self.assertEqual({row['estimand'] for row in actual},
                         {'mean', 'gap', 'signed_gap', 'relative_gap', 'worst_group', 'group_mean'})
        by_id = {row['comparison_id']: row for row in actual}
        for row in export.prepare_paired_comparisons():
            self.assertEqual(by_id[row['comparison_id']], row)

    def test_pareto_contains_the_two_reported_coserec_settings(self):
        points = export.prepare_pareto()
        self.assertEqual(len(points), 66)
        summaries = export.read_csv(export.COSEREC_SUMMARY)
        for row in summaries:
            if row['method'] != 'coserec':
                continue
            matches = [p for p in points if (p['dataset'], p['backbone'], p['method']) ==
                       (row['dataset'], row['backbone'], 'coserec')]
            self.assertEqual(len(matches), 1)
            point = matches[0]
            self.assertEqual(point['ndcg_10'], float(row['NDCG@10_mean']))
            expected_gap = (float(row['gender_NDCG@10_Gap_mean']) +
                            float(row['age_group_NDCG@10_Gap_mean'])) / 2
            self.assertEqual(point['mean_ndcg_gap_10'], expected_gap)
        for point in points:
            block = [p for p in points if (p['dataset'], p['backbone']) ==
                     (point['dataset'], point['backbone'])]
            self.assertEqual(len(block), 9 if point['backbone'] == 'sasrec' else 8)
            dominates = any(p['ndcg_10'] >= point['ndcg_10'] and
                            p['mean_ndcg_gap_10'] <= point['mean_ndcg_gap_10'] and
                            (p['ndcg_10'] > point['ndcg_10'] or
                             p['mean_ndcg_gap_10'] < point['mean_ndcg_gap_10']) for p in block)
            self.assertEqual(point['is_pareto'], not dominates)

    def test_common_utility_reports_coserec_coverage_and_actual_feasibility(self):
        detailed, summary = export.prepare_common_utility(export.prepare_pareto())
        self.assertEqual(len(detailed), 66)
        coserec = next(row for row in summary if row['method'] == 'coserec')
        self.assertEqual(coserec['evaluated_blocks'], 2)
        self.assertEqual(coserec['test_feasible_blocks'], 1)
        self.assertEqual(coserec['constrained_test_pareto_blocks'], 1)
        for row in summary:
            if row['method'] != 'coserec':
                self.assertEqual(row['evaluated_blocks'], 8)


if __name__ == '__main__':
    unittest.main()
