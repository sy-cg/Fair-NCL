import json
from pathlib import Path
import unittest

from analysis.prepare_mce_objective_comparison import (RESOURCE_KEYS, SEEDS, build_sasrec_jobs,
                                                     resource_differences, select_reporting_groups,
                                                     training_status)


class MatchedMCEPlanTests(unittest.TestCase):
    def test_sasrec_plan_preserves_fair_views_and_all_five_seeds(self):
        source = [json.loads(line) for line in Path(
            'experiments/revision_supplementary_final_extended_595.jsonl'
        ).read_text(encoding='utf-8').splitlines() if line.strip()]
        jobs = build_sasrec_jobs(source)
        refs = {(j['dataset'], j['seed']): j for j in source
                if j['method'] == 'fair_ncl' and j['backbone'] == 'sasrec'}
        self.assertEqual(len(jobs), 10)
        self.assertEqual(len({j['job_id'] for j in jobs}), 10)
        for j in jobs:
            ref = refs[j['dataset'], j['seed']]
            for key, value in ref['params'].items():
                self.assertEqual(j['params'][key], value, (j['job_id'], key))
            self.assertEqual(j['params']['ncl_objective'], 'mce')
            self.assertEqual(j['params']['mce_uniform_weight'], 0.5)
            self.assertEqual(j['params']['mce_align_weight'], 1.0)
            self.assertEqual(j['params']['batch_size'], 192 if j['dataset'] == 'ml-1m' else 64)

    def test_zero_update_history_is_failure_even_with_result_files(self):
        self.assertEqual(training_status({'skipped_nonfinite_updates': [12, 12]}),
                         'zero_finite_updates')

    def test_some_skipped_updates_do_not_discard_trained_run(self):
        self.assertEqual(training_status({'train_loss': [1.2],
                                         'skipped_nonfinite_updates': [1]}),
                         'trained_with_skips')

    def test_missing_history_is_not_accepted_as_trained(self):
        self.assertEqual(training_status({}), 'no_training_record')

    def test_nonfinite_loss_record_is_not_accepted_as_trained(self):
        for value in (float('nan'), float('inf')):
            self.assertEqual(training_status({'train_loss': [1.2, value]}),
                             'nonfinite_loss_record')

    def test_equal_job_parameters_cannot_mask_different_skew_resources(self):
        reference = dict.fromkeys(RESOURCE_KEYS, 1)
        method = dict(reference)
        reference['num_unique_train_events'] = 360207
        method['num_unique_train_events'] = 562308
        self.assertIn('num_unique_train_events', resource_differences(reference, method))

    def test_runtime_and_roundoff_do_not_change_resource_match(self):
        reference = dict.fromkeys(RESOURCE_KEYS, 1)
        method = dict(reference, resource_build_seconds=500, cache_bytes=100000)
        method['score_mean'] = 1.0 + 1e-14
        self.assertEqual(resource_differences(reference, method), {})

    def test_missing_resource_fields_are_not_treated_as_a_match(self):
        self.assertTrue(resource_differences({}, {}))

    def test_current_result_reporting_keeps_resource_differences_in_audit(self):
        members = [({'seed': s}, {'classification': 'skew_resource_mismatch'}) for s in SEEDS]
        groups = {('ml-1m', 'sasrec', 'revision_mce_matched'): members}
        selected = select_reporting_groups(groups)
        self.assertEqual(selected[('ml-1m', 'sasrec')], members)
        self.assertEqual(members[0][1]['classification'], 'skew_resource_mismatch')

    def test_changed_view_parameters_and_failed_training_are_not_objective_controls(self):
        for category in ('view_parameter_mismatch', 'training_failure', 'incomplete'):
            members = [({'seed': s}, {'classification': 'pair_qualified'}) for s in SEEDS]
            members[0][1]['classification'] = category
            self.assertEqual(select_reporting_groups({('ml-1m', 'sasrec', 'final'): members}), {})

    def test_reporting_requires_complete_five_seed_groups(self):
        members = [({'seed': s}, {'classification': 'pair_qualified'}) for s in SEEDS[:-1]]
        self.assertEqual(select_reporting_groups({('ml-1m', 'sasrec', 'final'): members}), {})


if __name__ == '__main__':
    unittest.main()
