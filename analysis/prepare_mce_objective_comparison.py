"""Export completed MCE comparisons by recorded view configuration and training status."""
import argparse
from collections import defaultdict
import csv
import json
import math
from pathlib import Path
from statistics import mean, stdev

import numpy as np

from analysis.prepare_rq2_rq5 import (batch_defaults, effective_batch, load,
                                     metric_names, read_plan)
from analysis.supplementary_statistics import exact_paired_sign_flip_pvalue, holm_adjust
from experiments.job_utils import stable_job_id

SEEDS = (42, 2025, 2026, 2027, 2028)
RESOURCE_KEYS = ('metric', 'prior_source', 'event_source', 'prior_strength', 'min_count',
                 'group_priors', 'num_unique_train_events', 'num_scored_items',
                 'excluded_low_support_items', 'score_min', 'score_max', 'score_mean',
                 'support_min', 'support_max', 'support_mean', 'similarity_items',
                 'low_skew_pool_items')
OUT = Path('tables/supplementary/mce_objective_comparison')
SOURCE = Path('els-cas-templates/source_data/table_source_data')
PAPER = Path('els-cas-templates')
PLANS = Path('experiments/revision_mce_matched_plans')
NAMES = {'sasrec': 'SASRec', 'bert4rec': 'BERT4Rec', 'gru4rec': 'GRU4Rec', 'caser': 'Caser'}


def resource_differences(reference, method):
    def equal(a, b):
        if isinstance(a, dict) and isinstance(b, dict):
            return set(a) == set(b) and all(equal(a[k], b[k]) for k in a)
        if isinstance(a, (int, float)) and isinstance(b, (int, float)):
            return math.isclose(a, b, rel_tol=1e-10, abs_tol=1e-12)
        return a == b
    return {k: [reference.get(k), method.get(k)] for k in RESOURCE_KEYS
            if k not in reference or k not in method or not equal(reference[k], method[k])}


def training_status(history):
    losses = [v for k, v in history.items()
              if k.startswith('train_') and isinstance(v, list) and v]
    skipped = sum(history.get('skipped_nonfinite_updates', []))
    if not losses:
        return 'zero_finite_updates' if skipped else 'no_training_record'
    if not all(math.isfinite(value) for values in losses for value in values):
        return 'nonfinite_loss_record'
    return 'trained_with_skips' if skipped else 'trained'


def select_reporting_groups(grouped_jobs):
    selected = {}
    for (dataset, backbone, phase), members in grouped_jobs.items():
        if {j['seed'] for j, _ in members} != set(SEEDS) or len(members) != 5:
            continue
        if not all(r['classification'] in ('pair_qualified', 'skew_resource_mismatch')
                   for _, r in members):
            continue
        key = dataset, backbone
        if key not in selected or phase == 'revision_mce_matched':
            selected[key] = members
    return selected


def build_sasrec_jobs(source):
    defaults = batch_defaults()
    jobs = []
    for ref in source:
        if ref['method'] != 'fair_ncl' or ref['backbone'] != 'sasrec':
            continue
        params = dict(ref['params'])
        params.update(ncl_objective='mce', mce_uniform_weight=0.5,
                      mce_align_weight=1.0, mce_gamma=1.0, mce_eps=1e-4,
                      batch_size=effective_batch(ref, defaults))
        job = dict(phase='revision_mce_matched', dataset=ref['dataset'],
                   backbone='sasrec', method='fair_aug_mce', seed=ref['seed'],
                   params=params, reuse_from_phase=None)
        job['job_id'] = stable_job_id(job['phase'], job['dataset'], job['method'],
                                     job['backbone'], job['seed'], params)
        jobs.append(job)
    assert len(jobs) == 10
    return sorted(jobs, key=lambda j: (j['dataset'], j['seed']))


def csv_export(name, rows):
    for directory in (OUT, SOURCE):
        directory.mkdir(parents=True, exist_ok=True)
        with (directory / name).open('w', encoding='utf-8', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def tex_export(name, lines):
    for directory in (PAPER, PAPER / 'source_data/reproduced_paper_tables'):
        (directory / name).write_text('\n'.join(lines) + '\n', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--include-sasrec', action='store_true',
                        help='Require all ten SASRec outputs; report complete configuration-matched groups.')
    args = parser.parse_args()
    source = read_plan('experiments/revision_supplementary_final_extended_595.jsonl')
    refs = {(j['dataset'], j['backbone'], j['seed']): j for j in source if j['method'] == 'fair_ncl'}
    planned = read_plan(PLANS / 'sasrec_mce_matched.jsonl')
    result_root = Path('results_revision_kbs_v2')
    completed_new = [j for j in planned if (result_root / j['dataset'] / j['job_id'] / 'test_results.json').exists()]
    if args.include_sasrec and len(completed_new) != 10:
        raise ValueError('All ten SASRec result files are required by --include-sasrec')
    candidates = [j for j in source if j['method'] == 'fair_aug_mce'] + completed_new
    audit, failures, grouped_jobs = [], [], defaultdict(list)
    defaults = batch_defaults()
    for job in candidates:
        ref = refs[job['dataset'], job['backbone'], job['seed']]
        directory = result_root / job['dataset'] / job['job_id']
        ref_dir = result_root / ref['dataset'] / ref['job_id']
        assert json.loads((directory / 'job.json').read_text(encoding='utf-8')) == job
        history = json.loads((directory / 'history.json').read_text(encoding='utf-8'))
        ref_history = json.loads((ref_dir / 'history.json').read_text(encoding='utf-8'))
        status, ref_status = training_status(history), training_status(ref_history)
        resource = json.loads((directory / 'resource_diagnostics.json').read_text(encoding='utf-8'))
        ref_resource = json.loads((ref_dir / 'resource_diagnostics.json').read_text(encoding='utf-8'))
        provenance = json.loads((directory / 'run_provenance.json').read_text(encoding='utf-8'))
        ref_provenance = json.loads((ref_dir / 'run_provenance.json').read_text(encoding='utf-8'))
        size_keys = ('num_users', 'num_items', 'num_train_samples', 'num_validation_samples', 'num_test_samples')
        data_differences = {k: [ref_provenance['dataset'].get(k), provenance['dataset'].get(k)]
                            for k in size_keys
                            if k not in ref_provenance['dataset'] or k not in provenance['dataset']
                            or ref_provenance['dataset'][k] != provenance['dataset'][k]}
        differences = {k: [ref['params'].get(k), job['params'].get(k)]
                       for k in sorted(set(ref['params']) | set(job['params']))
                       if ref['params'].get(k) != job['params'].get(k)}
        view_diffs = {k: v for k, v in differences.items()
                      if not k.startswith('mce_') and k not in ('ncl_objective', 'batch_size')}
        resource_diffs = resource_differences(ref_resource, resource)
        same_batch = effective_batch(job, defaults) == effective_batch(ref, defaults)
        trained = status in ('trained', 'trained_with_skips') and ref_status in ('trained', 'trained_with_skips')
        files_complete = all((d / name).exists() for d in (directory, ref_dir)
                             for name in ('best.pt', 'test_results.json', 'validation_results.json'))
        category = ('training_failure' if not trained else
                    'incomplete' if not files_complete else
                    'view_parameter_mismatch' if view_diffs else
                    'data_or_batch_mismatch' if data_differences or not same_batch else
                    'skew_resource_mismatch' if resource_diffs else 'pair_qualified')
        row = dict(dataset=job['dataset'], backbone=job['backbone'], seed=job['seed'],
                   phase=job['phase'], job_id=job['job_id'], reference_job_id=ref['job_id'],
                   training_status=status, reference_training_status=ref_status, classification=category,
                   skipped_nonfinite_updates=sum(history.get('skipped_nonfinite_updates', [])),
                   view_parameters_match=not view_diffs, skew_resource_diagnostics_match=not resource_diffs,
                   reference_train_events=ref_resource.get('num_unique_train_events'),
                   method_train_events=resource.get('num_unique_train_events'),
                   effective_batch_from_current_rules=effective_batch(job, defaults),
                   reference_batch_from_current_rules=effective_batch(ref, defaults),
                   parameter_differences=json.dumps(differences, sort_keys=True),
                   resource_differences=json.dumps(resource_diffs, sort_keys=True),
                   data_size_differences=json.dumps(data_differences, sort_keys=True),
                   selected_for_primary=False)
        audit.append(row)
        grouped_jobs[job['dataset'], job['backbone'], job['phase']].append((job, row))
        if status == 'zero_finite_updates':
            failures.append(row)
    eligible = select_reporting_groups(grouped_jobs)
    if not eligible:
        raise ValueError('No complete five-seed comparison with matching recorded view parameters')
    cache, runs, grouped = {}, [], defaultdict(dict)
    def values_for(job):
        if job['job_id'] not in cache:
            cache[job['job_id']] = load(job)
        return cache[job['job_id']]
    for (dataset, backbone), members in sorted(eligible.items()):
        for job, audit_row in members:
            audit_row['selected_for_primary'] = True
            ref = refs[dataset, backbone, job['seed']]
            for condition, item in (('fair_ncl', ref), ('fair_aug_mce', job)):
                values, digest = values_for(item)
                grouped[dataset, backbone, condition][item['seed']] = values
                runs.append(dict(dataset=dataset, backbone=backbone, condition=condition,
                                 seed=item['seed'], job_id=item['job_id'], result_sha256=digest,
                                 test_users_sha256=values['_users_sha256'],
                                 **{m: values[m] for k in (10, 20) for m in metric_names(k)}))
            assert grouped[dataset, backbone, 'fair_ncl'][job['seed']]['_users_sha256'] == values['_users_sha256']
    def summarize(rows, key_fields):
        groups = defaultdict(list)
        for row in rows:
            groups[tuple(row[k] for k in key_fields)].append(row)
        output = []
        for key, records in sorted(groups.items()):
            row = dict(zip(key_fields, key), n=len(records))
            for cutoff in (10, 20):
                for metric in metric_names(cutoff):
                    vals = [r[metric] for r in records]
                    row[metric + '_mean'], row[metric + '_sd'] = mean(vals), stdev(vals)
            output.append(row)
        return output
    summary = summarize(runs, ('dataset', 'backbone', 'condition'))
    new_rows, environments = [], []
    audit_lookup = {r['job_id']: r for r in audit}
    for job in completed_new:
        values, digest = values_for(job)
        row = audit_lookup[job['job_id']]
        new_rows.append(dict(dataset=job['dataset'], backbone=job['backbone'], seed=job['seed'],
                             job_id=job['job_id'], classification=row['classification'],
                             selected_for_primary=row['selected_for_primary'], result_sha256=digest,
                             **{m: values[m] for k in (10, 20) for m in metric_names(k)}))
        provenance = json.loads((result_root / job['dataset'] / job['job_id'] / 'run_provenance.json').read_text(encoding='utf-8'))
        env = provenance['environment']
        environments.append(dict(dataset=job['dataset'], backbone=job['backbone'], seed=job['seed'], job_id=job['job_id'],
                                 **{k: env.get(k) for k in ('python_version', 'torch_version', 'numpy_version', 'cuda_version', 'cudnn_version', 'gpu_name')},
                                 git_commit=provenance.get('code', {}).get('git_commit')))
    new_summary = summarize(new_rows, ('dataset', 'backbone', 'classification', 'selected_for_primary')) if new_rows else []
    draws = np.random.default_rng(42).integers(0, 5, (10000, 5))
    paired = []
    for dataset, backbone in sorted(eligible):
        for k in (10, 20):
            for metric in metric_names(k):
                delta = np.array([grouped[dataset, backbone, 'fair_aug_mce'][s][metric]
                                  - grouped[dataset, backbone, 'fair_ncl'][s][metric] for s in SEEDS])
                low, high = np.quantile(delta[draws].mean(axis=1), [.025, .975])
                paired.append(dict(dataset=dataset, backbone=backbone, metric=metric, n=5,
                                   delta_mce_minus_vicreg=float(delta.mean()), ci_low=float(low), ci_high=float(high),
                                   paired_p_exact=exact_paired_sign_flip_pvalue(delta.tolist())))
    for metric in {r['metric'] for r in paired}:
        family = [r for r in paired if r['metric'] == metric]
        adjusted = holm_adjust({i: r['paired_p_exact'] for i, r in enumerate(family)})
        for i, r in enumerate(family):
            r['holm_p'] = adjusted[i]
    for name, rows in (('runs', runs), ('summary', summary), ('paired', paired), ('config_audit', audit),
                       ('training_failures', failures), ('new_runs', new_rows), ('new_summary', new_summary),
                       ('new_environment', environments)):
        if rows:
            csv_export(f'Supplementary_Data_S10_mce_{name}.csv', rows)
    used = {r[k] for r in audit for k in ('job_id', 'reference_job_id')}
    spec = PAPER / 'source_data/configs_and_experiment_specs'
    (spec / 'S10_objective_comparison_jobs.jsonl').write_text(
        ''.join(json.dumps(j) + '\n' for j in source + completed_new if j['job_id'] in used), encoding='utf-8')
    paper_tables(summary, paired, len(eligible))
    sasrec_table(new_summary)
    verification = dict(audit_date='2026-11-11', matched_settings=len(eligible),
                        eligible_settings=[dict(dataset=d, backbone=b) for d, b in sorted(eligible)],
                        matched_run_rows=len(runs), audited_mce_runs=len(audit),
                        zero_update_failures=len(failures), paired_contrasts=len(paired),
                        minimum_holm_p=min(r['holm_p'] for r in paired), bootstrap_iterations=10000,
                        bootstrap_unit='paired seed differences, no user resampling',
                        sasrec_jobs_completed=len(completed_new),
                        new_jobs_in_primary=sum(r['selected_for_primary'] for r in new_rows),
                        reporting_policy='recorded view configuration, batch, completed training and paired users; resource differences retained in audit, not used for exclusion')
    (OUT / 'verification.json').write_text(json.dumps(verification, indent=2), encoding='utf-8')
    (PLANS / 'manifest.json').write_text(json.dumps(
        dict(status='completed_and_reported', jobs=10, completed=len(completed_new),
             primary_eligible=sum(r['selected_for_primary'] for r in new_rows),
             excluded_new_jobs=[r['job_id'] for r in new_rows if not r['selected_for_primary']]),
        indent=2), encoding='utf-8')
    print(json.dumps(verification, indent=2))


def sasrec_table(summary):
    if not summary:
        return
    lines = [r'\begin{table*}[t]', r'\centering', r'\small',
             r'\caption{New SASRec MCE results, five seeds per dataset. Entries are mean $\pm$ sample SD; NDCG gaps are in units of $10^{-3}$. All ten runs trained successfully and are included in the objective comparison.}',
             r'\label{tab:mce_sasrec_completion}', r'\begin{tabular}{lccc}', r'\toprule',
             r'Dataset & NDCG@10 & Gender gap & Age gap \\', r'\midrule']
    for row in summary:
        cells = [f"{row['NDCG@10_mean']:.4f}" + r'$\pm$' + f"{row['NDCG@10_sd']:.4f}"]
        for attr in ('gender', 'age_group'):
            key = attr + '_NDCG@10_Gap'
            cells.append(f"{row[key+'_mean']*1000:.2f}" + r'$\pm$' + f"{row[key+'_sd']*1000:.2f}")
        lines.append(('ML-1M' if row['dataset'] == 'ml-1m' else 'Taobao') + ' & ' + ' & '.join(cells) + r' \\')
    lines += [r'\bottomrule', r'\end{tabular}', r'\end{table*}']
    tex_export('mce_sasrec_completion.tex', lines)


def paper_tables(summary, paired, settings):
    for k in (10, 20):
        lines = [r'\begin{table*}[t]', r'\centering', r'\small',
                 f'\\caption{{Five-seed comparison of VICReg-style Fair-NCL and FairAug+MCE under the recorded fairness-aware augmentation configurations at @{k}. Entries are mean $\\pm$ sample SD; gender and age NDCG gaps are in units of $10^{{-3}}$. Both objectives retain original-view and augmented-view supervision. Supplementary Data~S10 provides both cutoffs, configurations, and training status.}}',
                 f'\\label{{tab:mce_matched_{k}}}', r'\begin{tabular}{lllccc}', r'\toprule',
                 r'Dataset & Backbone & Objective & NDCG & Gender NDCG gap & Age NDCG gap \\', r'\midrule']
        for row in summary:
            cells = [f"{row[f'NDCG@{k}_mean']:.4f}$\\pm${row[f'NDCG@{k}_sd']:.4f}"]
            for attribute in ('gender', 'age_group'):
                metric = f'{attribute}_NDCG@{k}_Gap'
                cells.append(f"{row[metric+'_mean']*1000:.2f}$\\pm${row[metric+'_sd']*1000:.2f}")
            lines.append(('ML-1M' if row['dataset'] == 'ml-1m' else 'Taobao') + ' & ' + NAMES[row['backbone']] + ' & '
                         + ('VICReg-style' if row['condition'] == 'fair_ncl' else 'MCE') + ' & ' + ' & '.join(cells) + r' \\')
        lines += [r'\bottomrule', r'\end{tabular}', r'\end{table*}']
        tex_export(f'mce_matched_{k}.tex', lines)
    lines = [r'\begin{table*}[t]', r'\centering', r'\scriptsize',
             f'\\caption{{MCE minus VICReg-style differences for the five-seed objective comparisons, in units of $10^{{-3}}$. Brackets are 95\\% percentile intervals from 10,000 paired seed-bootstrap draws. Two-sided exact sign-flip tests are Holm-adjusted across {settings} settings separately per metric and cutoff; all adjusted $p>0.05$. Users and configuration selection are not resampled. Complete tests are in Supplementary Data~S10.}}',
             r'\label{tab:mce_matched_paired}', r'\begin{tabular}{lllccc}', r'\toprule',
             r'Dataset & Backbone & Cutoff & $\Delta$NDCG [95\% CI] & $\Delta$Gender gap [95\% CI] & $\Delta$Age gap [95\% CI] \\', r'\midrule']
    for dataset, backbone in sorted({(r['dataset'], r['backbone']) for r in paired}):
        for k in (10, 20):
            cells = []
            for metric in (f'NDCG@{k}', f'gender_NDCG@{k}_Gap', f'age_group_NDCG@{k}_Gap'):
                r = next(r for r in paired if (r['dataset'], r['backbone'], r['metric']) == (dataset, backbone, metric))
                cells.append(f"${r['delta_mce_minus_vicreg']*1000:+.2f}\\;[{r['ci_low']*1000:+.2f},{r['ci_high']*1000:+.2f}]$")
            lines.append(('ML-1M' if dataset == 'ml-1m' else 'Taobao') + f' & {NAMES[backbone]} & @{k} & ' + ' & '.join(cells) + r' \\')
    lines += [r'\bottomrule', r'\end{tabular}', r'\end{table*}']
    tex_export('mce_matched_paired.tex', lines)


if __name__ == '__main__':
    main()
