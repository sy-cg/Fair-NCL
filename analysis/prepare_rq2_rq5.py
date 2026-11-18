"""Audit the new ablations and SASRec sensitivity runs and export paper data."""
import ast
import csv
import hashlib
import itertools
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

import numpy as np

from analysis.supplementary_statistics import exact_paired_sign_flip_pvalue, holm_adjust
from analysis.export_displayed_ablation_tables import ablation_tables
from analysis.summary_ablation_uncertainty import comparison_bounds, beta0_bounds_table

ROOT = Path('results_revision_kbs_v2')
OUT = Path('tables/supplementary/rq2_rq5')
SOURCE = Path('els-cas-templates/source_data/table_source_data')
SEEDS = (42, 2025, 2026, 2027, 2028)
TUNE_SEEDS = SEEDS[:3]
DATASETS = ('ml-1m', 'taobao')
BACKBONES = ('sasrec', 'bert4rec')
ABLATIONS = ('wo_fairness_sampling', 'wo_semantic_sampling', 'wo_alignment',
             'wo_variance', 'wo_covariance', 'wo_augmented_ce', 'random_low_skew', 'high_skew')
ORDER = ('baseline',) + ABLATIONS + ('beta0_fixed', 'fair_ncl')
LABELS = {'baseline': 'Backbone', 'fair_ncl': 'Fair-NCL',
          'wo_fairness_sampling': 'w/o skew (uniform neighbors)',
          'wo_semantic_sampling': 'w/o semantic sampling',
          'wo_alignment': 'w/o alignment', 'wo_variance': 'w/o variance',
          'wo_covariance': 'w/o covariance', 'wo_augmented_ce': 'w/o augmented CE',
          'random_low_skew': 'Random low-skew replacement', 'high_skew': 'High-skew replacement',
          'beta0_fixed': r'Fixed-$\alpha$ $\beta=0$'}
AXES = ('epsilon', 'augment_ratio', 'utility_beta')


def read_plan(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line.strip()]


def write_csv(name, rows):
    for directory in (OUT, SOURCE):
        directory.mkdir(parents=True, exist_ok=True)
        with (directory / name).open('w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def metric_names(k):
    return (f'HitRate@{k}', f'NDCG@{k}', f'gender_HitRate@{k}_Gap',
            f'gender_NDCG@{k}_Gap', f'age_group_HitRate@{k}_Gap',
            f'age_group_NDCG@{k}_Gap', f'mean_NDCG@{k}_Gap')


def load(job):
    directory = ROOT / job['dataset'] / job['job_id']
    saved = json.loads((directory / 'job.json').read_text(encoding='utf-8'))
    assert saved == job, f'Plan mismatch: {job["job_id"]}'
    path = directory / 'test_results.json'
    result = json.loads(path.read_text(encoding='utf-8'))
    assert 'selected' in result and 'utility' in result, path
    values = dict(result['selected'])
    user_keys = sorted((int(r['user_id']), r.get('gender'), r.get('age_group')) for r in result['per_user'])
    values['_users_sha256'] = hashlib.sha256(json.dumps(user_keys).encode()).hexdigest()
    for k in (10, 20):
        for metric in ('HitRate', 'NDCG'):
            values[f'{metric}@{k}'] = float(result['utility'][f'{metric}@{k}'])
            for attribute in ('gender', 'age_group'):
                values[f'{attribute}_{metric}@{k}_Gap'] = float(
                    result['fairness'][attribute][f'{metric.lower()}@{k}_gap'])
        values[f'mean_NDCG@{k}_Gap'] = mean(values[f'{a}_NDCG@{k}_Gap'] for a in ('gender', 'age_group'))
    assert all(np.isfinite(values[m]) for k in (10, 20) for m in metric_names(k))
    return values, hashlib.sha256(path.read_bytes()).hexdigest()


def batch_defaults():
    tree = ast.parse(Path('config.py').read_text(encoding='utf-8'))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'Config')
    return {n.targets[0].id: ast.literal_eval(n.value) for n in cls.body
            if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)
            and n.targets[0].id in ('DATASET_MEMORY_PROFILES', 'METHOD_BATCH_FACTORS', 'BACKBONE_BATCH_FACTORS')}


def effective_batch(job, defaults):
    if 'batch_size' in job['params']:
        return int(job['params']['batch_size'])
    base = defaults['DATASET_MEMORY_PROFILES'][job['dataset']]['batch_size']
    factor = defaults['METHOD_BATCH_FACTORS'].get(job['method'], 1) * defaults['BACKBONE_BATCH_FACTORS'].get(job['backbone'], 1)
    return max(16, int(round(base * factor / 16) * 16))


def main():
    main_plan = read_plan('experiments/revision_supplementary_final_extended_595.jsonl')
    refs = {(j['dataset'], j['backbone'], j['seed'], j['method']): j for j in main_plan
            if j['backbone'] in BACKBONES}
    plans = [read_plan(f'experiments/revision_completion_plans/missing_components_beta0_{b}.jsonl') for b in BACKBONES]
    jobs = {(j['dataset'], j['backbone'], j['seed'],
             'beta0_fixed' if j['method'] == 'fair_ncl_factorial' else j['method']): j
            for plan in plans for j in plan}
    for dataset, backbone, seed in itertools.product(DATASETS, BACKBONES, SEEDS):
        for method in ('baseline', 'fair_ncl'):
            jobs[dataset, backbone, seed, method] = refs[dataset, backbone, seed, method]
        if backbone == 'bert4rec':
            jobs[dataset, backbone, seed, 'wo_augmented_ce'] = refs[dataset, backbone, seed, 'fair_aug_no_augmented_ce']
    expected = set(itertools.product(DATASETS, BACKBONES, SEEDS, ORDER))
    assert set(jobs) == expected, (expected - set(jobs), set(jobs) - expected)
    defaults = batch_defaults()
    run_rows, audit, grouped = [], [], defaultdict(dict)
    for (dataset, backbone, seed, condition), job in sorted(jobs.items()):
        values, digest = load(job)
        reference = refs[dataset, backbone, seed, 'fair_ncl']
        differences = {k: [reference['params'].get(k), job['params'].get(k)]
                       for k in set(reference['params']) | set(job['params'])
                       if reference['params'].get(k) != job['params'].get(k)}
        if condition == 'beta0_fixed':
            assert set(differences) <= {'utility_beta', 'factorial_cell', 'export_mechanism_analysis'}
            for k in ('utility_alpha', 'epsilon', 'augment_ratio', 'similarity_source',
                      'fair_ncl_align_weight', 'fair_ncl_var_weight', 'fair_ncl_cov_weight'):
                assert job['params'][k] == reference['params'][k], (condition, k)
            assert job['params']['utility_beta'] == 0
            assert effective_batch(job, defaults) == effective_batch(reference, defaults)
        grouped[dataset, backbone, condition][seed] = values
        run_rows.append(dict(dataset=dataset, backbone=backbone, condition=condition,
                             seed=seed, job_id=job['job_id'], result_sha256=digest,
                             test_users_sha256=values['_users_sha256'],
                             **{m: values[m] for k in (10, 20) for m in metric_names(k)}))
        audit.append(dict(dataset=dataset, backbone=backbone, condition=condition, seed=seed,
                          job_id=job['job_id'], phase=job['phase'],
                          effective_batch_from_current_rules=effective_batch(job, defaults),
                          reference_batch_from_current_rules=effective_batch(reference, defaults),
                          parameter_differences=json.dumps(differences, sort_keys=True)))
    for (dataset, backbone, condition), seeds in grouped.items():
        for seed, values in seeds.items():
            assert values['_users_sha256'] == grouped[dataset, backbone, 'fair_ncl'][seed]['_users_sha256']
    write_csv('Supplementary_Data_S8_ablation_runs.csv', run_rows)
    write_csv('Supplementary_Data_S8_config_audit.csv', audit)
    summary = []
    for (dataset, backbone, condition), seeds in sorted(grouped.items()):
        assert set(seeds) == set(SEEDS)
        row = dict(dataset=dataset, backbone=backbone, condition=condition, n=5)
        for k in (10, 20):
            for metric in metric_names(k):
                vals = [seeds[s][metric] for s in SEEDS]
                row[metric + '_mean'], row[metric + '_sd'] = mean(vals), stdev(vals)
        summary.append(row)
    lookup = {(r['dataset'], r['backbone'], r['condition']): r for r in summary}
    for row in summary:
        competitors = [lookup[row['dataset'], row['backbone'], c] for c in ORDER if c != 'baseline']
        for k in (10, 20):
            coordinates = lambda r: (r[f'NDCG@{k}_mean'], -r[f'gender_NDCG@{k}_Gap_mean'], -r[f'age_group_NDCG@{k}_Gap_mean'])
            point = np.array(coordinates(row))
            dominators = [r['condition'] for r in competitors if r is not row
                          and np.all(np.array(coordinates(r)) >= point)
                          and np.any(np.array(coordinates(r)) > point)]
            row[f'pareto3@{k}'] = not dominators
            row[f'dominated_by@{k}'] = '|'.join(dominators)
    write_csv('Supplementary_Data_S8_ablation_summary.csv', summary)
    rng = np.random.default_rng(42)
    draws = rng.integers(0, 5, (10000, 5))
    paired = []
    for dataset, backbone, condition, k in itertools.product(DATASETS, BACKBONES, ORDER[1:-1], (10, 20)):
        for metric in metric_names(k):
            delta = np.array([grouped[dataset, backbone, condition][s][metric]
                              - grouped[dataset, backbone, 'fair_ncl'][s][metric] for s in SEEDS])
            low, high = np.quantile(delta[draws].mean(axis=1), [.025, .975])
            paired.append(dict(dataset=dataset, backbone=backbone, condition=condition, metric=metric,
                               n=5, delta_vs_full=float(delta.mean()), ci_low=float(low), ci_high=float(high),
                               paired_p_exact=exact_paired_sign_flip_pvalue(delta.tolist())))
    for metric in {r['metric'] for r in paired}:
        family = [r for r in paired if r['metric'] == metric]
        adjusted = holm_adjust({i: r['paired_p_exact'] for i, r in enumerate(family)})
        for i, r in enumerate(family):
            r['holm_p'] = adjusted[i]
    write_csv('Supplementary_Data_S8_ablation_paired.csv', paired)
    sens = [j for j in read_plan('experiments/revision_completion_plans/sensitivity.jsonl')
            if j['backbone'] == 'sasrec' and j['seed'] in TUNE_SEEDS
            and j['params']['sensitivity_axis'] in AXES]
    assert len(sens) == 36
    sens_groups = defaultdict(dict)
    sens_jobs = {}
    sens_runs = []
    for job in sens:
        axis = job['params']['sensitivity_axis']
        assert axis in AXES and job['backbone'] == 'sasrec' and job['seed'] in TUNE_SEEDS
        base = refs[job['dataset'], 'sasrec', job['seed'], 'fair_ncl']
        changed = {k for k in set(job['params']) | set(base['params'])
                   if job['params'].get(k) != base['params'].get(k)}
        allowed = {axis, 'sensitivity_axis', 'export_mechanism_analysis'}
        if axis == 'utility_beta':
            allowed.add('utility_alpha')
            assert np.isclose(job['params']['utility_alpha'], 1 - job['params']['utility_beta'])
        assert changed <= allowed, (job['job_id'], changed - allowed)
        value = float(job['params'][axis])
        values, digest = load(job)
        assert values['_users_sha256'] == grouped[job['dataset'], 'sasrec', 'fair_ncl'][job['seed']]['_users_sha256']
        key = (job['dataset'], axis, value)
        assert job['seed'] not in sens_groups[key]
        sens_groups[key][job['seed']] = values
        sens_jobs[job['job_id']] = job
        sens_runs.append(dict(dataset=job['dataset'], backbone='sasrec', axis=axis, value=value,
                              reference=False, seed=job['seed'], job_id=job['job_id'], result_sha256=digest,
                              **{m: values[m] for k in (10, 20) for m in metric_names(k)}))
    for dataset, axis, seed in itertools.product(DATASETS, AXES, TUNE_SEEDS):
        job = refs[dataset, 'sasrec', seed, 'fair_ncl']
        values = grouped[dataset, 'sasrec', 'fair_ncl'][seed]
        value = float(job['params'][axis])
        sens_groups[dataset, axis, value][seed] = values
        sens_jobs[job['job_id']] = job
        sens_runs.append(dict(dataset=dataset, backbone='sasrec', axis=axis, value=value,
                              reference=True, seed=seed, job_id=job['job_id'],
                              result_sha256=next(r['result_sha256'] for r in run_rows if r['job_id'] == job['job_id']),
                              **{m: values[m] for k in (10, 20) for m in metric_names(k)}))
    sens_summary = []
    assert len(sens_groups) == 18
    for (dataset, axis, value), seeds in sorted(sens_groups.items()):
        assert set(seeds) == set(TUNE_SEEDS)
        center = refs[dataset, 'sasrec', 42, 'fair_ncl']['params'][axis]
        row = dict(dataset=dataset, backbone='sasrec', axis=axis, value=value, n=3,
                   reference=bool(np.isclose(value, center)))
        for k in (10, 20):
            for metric in metric_names(k):
                vals = [seeds[s][metric] for s in TUNE_SEEDS]
                row[metric + '_mean'], row[metric + '_sd'] = mean(vals), stdev(vals)
        sens_summary.append(row)
    write_csv('Supplementary_Data_S9_sensitivity_runs.csv', sens_runs)
    write_csv('Supplementary_Data_S9_sensitivity_summary.csv', sens_summary)
    for name, selected_jobs in (('S8_ablation_jobs.jsonl', jobs.values()), ('S9_sensitivity_jobs.jsonl', sens_jobs.values())):
        destination = Path('els-cas-templates/source_data/configs_and_experiment_specs') / name
        destination.write_text(''.join(json.dumps(j, sort_keys=True) + '\n' for j in selected_jobs), encoding='utf-8')
    paper_tables(lookup)
    supplementary_tables(sens_summary)
    lines = ['# RQ2 消融与 RQ5 参数敏感性分析报告', '',
             '八种消融：两数据集 × 两主干 × 五种子。额外固定 α、ε 的 β=0 对照覆盖相同四个设置。',
             '表中使用五种子均值与样本标准差；差值为变体减完整 Fair-NCL。配对区间仅对五个种子差值重采样 10,000 次，不含用户层重采样。',
             '每个指标/截断分别对四设置 × 九变体的 36 个精确符号翻转检验作 Holm 校正。',
             '三目标支配定义：NDCG 更高且性别与年龄 NDCG 差距均不大，至少一个严格更好；前沿只使用完整模型和九个变体，不含纯主干。', '',
             '## 消融均值和前沿 @10', '', '| 数据集 | 主干 | 变体 | NDCG | G gap | A gap | 三目标前沿 |', '|---|---|---|---|---|---|---|']
    for dataset, backbone, condition in itertools.product(DATASETS, BACKBONES, ORDER):
        r = lookup[dataset, backbone, condition]
        lines.append('| ' + ' | '.join([dataset, backbone, condition] +
                     [f"{r[m+'_mean']:.6f} ± {r[m+'_sd']:.6f}" for m in ('NDCG@10','gender_NDCG@10_Gap','age_group_NDCG@10_Gap')]
                     + [str(r['pareto3@10'])]) + ' |')
    lines += ['', '## 消融结果解读', '',
              '完整模型在 @10 的四个详细设置中均不在三目标均值前沿：ML-1M/SASRec 的去 covariance、ML-1M/BERT4Rec 的 high-skew、Taobao/SASRec 的去 alignment、Taobao/BERT4Rec 的去 variance 分别支配完整模型。@20 则在三设置非支配，Taobao/BERT4Rec 仍被去 variance 支配。这些都是观察均值关系，不是显著胜出。',
              '几何项的影响依赖数据：去 variance 降低 ML-1M/SASRec 的 NDCG@10，却在 Taobao/BERT4Rec 同时提高 NDCG 并降低两项差距。去 alignment 在 Taobao/SASRec 也呈现三个均值同时更优。因此不能保留“完整模型始终最均衡”“每个组件均必要”的结论。',
              '固定 α、ε 的 β=0 对照显示：完整模型在 Taobao/SASRec 以轻微效用代价换得更小均值差距，但 ML-1M/SASRec 的性别差距反而更大；BERT4Rec 也出现属性之间的不同方向。该对照提高了机制解释的可比性，但没有建立普遍的公平收益。',
              '504 个种子级配对对照的 Holm p 均为 1.00。部分未校正种子 bootstrap 区间不含零，也不能写成校正后显著。', '',
              '## 固定参数 β=0 对照 @10', '']
    for r in paired:
        if r['condition'] == 'beta0_fixed' and r['metric'] in ('NDCG@10','gender_NDCG@10_Gap','age_group_NDCG@10_Gap'):
            lines.append(f"- {r['dataset']}/{r['backbone']} {r['metric']}: Δ={r['delta_vs_full']:+.6f}, 95% seed CI [{r['ci_low']:+.6f}, {r['ci_high']:+.6f}], p={r['paired_p_exact']:.4f}, Holm p={r['holm_p']:.4f}.")
    lines += ['', '## SASRec 参数敏感性 @10', '',
              '36 条新增实验，每点三个种子；六条中心配置复用相同三个种子，跨三个参数轴重复展示，不视为额外独立实验。全部为测试集事后敏感性描述，不据此重新选择配置。',
              'β 轴同步令 α=1−β，并非固定 α 下的纯公平权重实验。', '',
              '| 数据集 | 参数 | 取值 | NDCG | G gap | A gap |', '|---|---|---|---|---|---|']
    for r in sens_summary:
        lines.append('| ' + ' | '.join([r['dataset'],r['axis'],str(r['value'])] +
                     [f"{r[m+'_mean']:.6f} ± {r[m+'_sd']:.6f}" for m in ('NDCG@10','gender_NDCG@10_Gap','age_group_NDCG@10_Gap')]) + ' |')
    lines += ['', '## 参数趋势解读', '',
              'ML-1M 的 ε 从 1 降至 0.5，NDCG@10 从 0.1475 变为 0.1465，性别差距从 0.0136 降至 0.0064，但年龄差距从 0.0065 升至 0.0089。没有单调改善两种属性的采样锐度。',
              'Taobao 的 r 从 0.05 增至 0.2，性别差距从 0.0027 降至 0.0007，而年龄差距从 0.0012 升至 0.0024；效用均值从 0.0342 变为 0.0337。平均 gap 会掩盖这一冲突，因此图中分开展示两属性。',
              'β 轴同步改变 α。ML-1M 年龄 gap 随端点 β 增大降低，但性别 gap 在中点最大；Taobao 的 NDCG 均值范围 0.0339–0.0343 小于或接近种子波动，不能据此选择所谓最佳 β。图中误差线为样本标准差，非置信区间或显著性检验。', '',
              '## 对照解释限制', '',
              '- w/o skew 使用候选内均匀采样；它与固定 α 的 β=0 语义加权采样不同。',
              '- w/o semantic 同时关闭语义评分并切换为共现候选，不能只解释成删除 α。',
              '- 随机低 skew 来自全局低 skew 池；high-skew 在局部邻域内奖励正向 skew 增加，不能视为保持语义项不变的简单符号翻转。',
              '- 部分方法的默认有效 batch size 不同；审计表给出按当前配置规则解析值，历史代码 commit 未记录，不能据此证明所有历史实现完全一致。',
              '- 五种子的双侧精确检验最小 p=0.0625；区间排除零不等同于通过 Holm 校正。',
              '- 新增 β=0 对照解决了 SASRec/BERT4Rec 的部分隔离比较；没有完整四因素析因设计或覆盖四主干的交互因果证据。', '']
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'analysis_report.md').write_text('\n'.join(lines), encoding='utf-8')
    print('Audited 220 ablation/reference records and 36 new sensitivity runs; exported S8/S9.')
    print('Full-model dominators @10:', {f'{d}/{b}': lookup[d,b,'fair_ncl']['dominated_by@10'] for d,b in itertools.product(DATASETS,BACKBONES)})


def paper_tables(lookup):
    for backbone in BACKBONES:
        title = {'sasrec': 'SASRec', 'bert4rec': 'BERT4Rec'}[backbone]
        for k in (10, 20):
            lines = [r'\begin{table*}[t]', r'\centering', r'\small', r'\setlength{\tabcolsep}{2.2pt}',
                     f'\\caption{{Five-seed {title} ablations at @{k}. Values are mean $\\pm$ sample standard deviation. HR/NDCG are higher-is-better; absolute group gaps are lower-is-better. The fixed-$\\alpha$ $\\beta=0$ row changes only the demographic weight in the sampler. Supplementary Data~S8 provides paired seed intervals, tests, configuration audits, and three-objective dominance. No metric is highlighted as a significant winner.}}',
                     f'\\label{{tab:ablation_{backbone}_{k}}}', r'\resizebox{\textwidth}{!}{%', r'\begin{tabular}{llcccccc}',
                     r'\toprule', r'Dataset & Variant & HR & NDCG & G-HR Gap & G-NDCG Gap & A-HR Gap & A-NDCG Gap \\', r'\midrule']
            for dataset in DATASETS:
                for index, condition in enumerate(ORDER):
                    row = lookup[dataset, backbone, condition]
                    label = f"\\multirow{{11}}{{*}}{{{'ML-1M' if dataset=='ml-1m' else 'Taobao'}}}" if index == 0 else ''
                    cells = [f"{row[m+'_mean']:.4f}$\\pm${row[m+'_sd']:.4f}" for m in metric_names(k)[:6]]
                    lines.append(label + ' & ' + LABELS[condition] + ' & ' + ' & '.join(cells) + r' \\')
                if dataset == 'ml-1m': lines.append(r'\midrule')
            lines += [r'\bottomrule', r'\end{tabular}%', '}', r'\end{table*}']
            for directory in (Path('els-cas-templates'), Path('els-cas-templates/source_data/reproduced_paper_tables')):
                (directory / f'ablation_{backbone}_{k}.tex').write_text('\n'.join(lines)+'\n', encoding='utf-8')


def supplementary_tables(sensitivity):
    # Current F.3 follows the author-updated displayed tables, not archived pairing.
    _, displayed_rows = ablation_tables(Path('els-cas-templates/cas-dc-sample.tex').read_text(encoding='utf-8'))
    lines = beta0_bounds_table(comparison_bounds(displayed_rows)).splitlines()
    sens = [r'\begin{table*}[t]', r'\centering', r'\small',
            r'\caption{SASRec local parameter sensitivity at @10. Entries are three-seed mean $\pm$ sample standard deviation. The center uses only the same three reference seeds and is repeated across axes. The $\beta$ axis also sets $\alpha=1-\beta$. Complete @10/@20 results are in Supplementary Data~S9.}',
            r'\label{tab:sensitivity_values}', r'\begin{tabular}{llcccc}', r'\toprule',
            r'Dataset & Parameter & Value & NDCG & Gender NDCG gap & Age NDCG gap \\', r'\midrule']
    for row in sensitivity:
        cells = [f"{row[m+'_mean']:.4f}$\\pm${row[m+'_sd']:.4f}" for m in ('NDCG@10','gender_NDCG@10_Gap','age_group_NDCG@10_Gap')]
        axis = {'epsilon': r'$\epsilon$', 'augment_ratio': '$r$', 'utility_beta': r'$\beta$'}[row['axis']]
        sens.append(('ML-1M' if row['dataset'] == 'ml-1m' else 'Taobao') + ' & ' + axis + f" & {row['value']:g} & " + ' & '.join(cells) + r' \\')
    sens += [r'\bottomrule', r'\end{tabular}', r'\end{table*}']
    for filename, content in (('ablation_beta0_paired.tex', lines), ('sensitivity_values.tex', sens)):
        for directory in (Path('els-cas-templates'), Path('els-cas-templates/source_data/reproduced_paper_tables')):
            (directory / filename).write_text('\n'.join(content) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
