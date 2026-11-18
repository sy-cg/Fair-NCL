"""Conservative t-interval envelopes from the four current manuscript tables.

No seed data or covariance is reconstructed. Confidence interpretation is
conditional on independent, normally distributed paired seed differences.
"""
import csv
import hashlib
import json
import math
from decimal import Decimal
from pathlib import Path

from analysis.export_displayed_ablation_tables import MANUSCRIPT, METRICS, ablation_tables
from analysis.supplementary_statistics import holm_adjust

OUTPUT = Path('tables/supplementary/rq2_rq5/summary_uncertainty')


def t4_tail(value):
    """Two-sided Student t tail for four degrees of freedom, in closed form."""
    u = abs(value) / math.sqrt(value * value + 4)
    return (1-u)**2 * (1+u/2)


def t4_critical(alpha):
    low, high = 0., 100.
    for _ in range(70):
        middle = (low+high)/2
        if t4_tail(middle) > alpha:
            low = middle
        else:
            high = middle
    return (low+high)/2


def difference_bounds(mean_full, sd_full, mean_control, sd_control):
    # Each displayed mean and sample SD has four decimal places.
    rounding = .00005
    difference = float(Decimal(mean_full)-Decimal(mean_control))
    se_upper = (float(sd_full)+float(sd_control)+2*rounding)/math.sqrt(5)
    half_width = 2*rounding+t4_critical(.05)*se_upper
    t_lower = max(0., abs(difference)-2*rounding)/se_upper
    return dict(difference=difference, se_upper=se_upper,
                ci_low=difference-half_width, ci_high=difference+half_width,
                p_upper=t4_tail(t_lower))


def comparison_bounds(rows):
    full = {(r['dataset'], r['backbone'], r['cutoff']): r
            for r in rows if r['variant'] == 'Fair-NCL'}
    results = []
    for row in rows:
        if row['variant'] in ('Backbone', 'Fair-NCL'):
            continue
        reference = full[row['dataset'], row['backbone'], row['cutoff']]
        for metric in METRICS:
            result = dict(dataset=row['dataset'], backbone=row['backbone'],
                          cutoff=row['cutoff'], variant=row['variant'], metric=metric,
                          n=5, df=4, rounding_half_unit=.00005,
                          mean_full=reference[metric+'_mean'], sd_full=reference[metric+'_sd'],
                          mean_control=row[metric+'_mean'], sd_control=row[metric+'_sd'])
            result.update(difference_bounds(result['mean_full'], result['sd_full'],
                                             result['mean_control'], result['sd_control']))
            results.append(result)
    for metric in METRICS:
        for cutoff in (10, 20):
            indices = [i for i, r in enumerate(results)
                       if r['metric'] == metric and r['cutoff'] == cutoff]
            assert len(indices) == 36
            adjusted = holm_adjust({i: results[i]['p_upper'] for i in indices})
            for i in indices:
                results[i]['p_upper_holm'] = adjusted[i]
    return results


def beta0_bounds_table(results):
    selected = {(r['dataset'], r['backbone'], r['cutoff'], r['metric']): r
                for r in results if r['variant'] == r'Fixed-$\alpha$ $\beta=0$'}
    lines = [r'\begin{table*}[t]', r'\centering', r'\small',
             r'\caption{Full Fair-NCL minus fixed-$\alpha$, fixed-$\epsilon$ $\beta=0$, in units of $10^{-3}$. Brackets are conservative 95\% t-interval envelopes from the displayed means/SDs, five seeds, unknown paired covariance, and four-decimal rounding. They are conditional on a normal paired-difference model and are not empirical paired-bootstrap intervals. Positive NDCG differences and negative gap differences favor Fair-NCL. Bounds are rounded outward.}',
             r'\label{tab:ablation_beta0_paired}', r'\resizebox{\textwidth}{!}{%',
             r'\begin{tabular}{lllccc}', r'\toprule',
             r'Dataset & Backbone & Cutoff & $\Delta$NDCG [envelope] & $\Delta$Gender gap [envelope] & $\Delta$Age gap [envelope] \\',
             r'\midrule']
    for dataset in ('ML-1M', 'Taobao'):
        for backbone in ('SASRec', 'BERT4Rec'):
            for cutoff in (10, 20):
                cells = []
                for metric in ('NDCG', 'G_NDCG', 'A_NDCG'):
                    row = selected[dataset, backbone, cutoff, metric]
                    delta = row['difference']*1000
                    low = math.floor(row['ci_low']*10000)/10
                    high = math.ceil(row['ci_high']*10000)/10
                    point = '0.0' if abs(delta) < 1e-10 else f'{delta:+.1f}'
                    cells.append(fr'${point}\;[{low:+.1f},{high:+.1f}]$')
                lines.append(f'{dataset} & {backbone} & @{cutoff} & '
                             + ' & '.join(cells) + r' \\')
    return '\n'.join(lines+[r'\bottomrule', r'\end{tabular}%', '}', r'\end{table*}'])+'\n'


def main():
    tables, rows = ablation_tables(MANUSCRIPT.read_text(encoding='utf-8'))
    results = comparison_bounds(rows)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with (OUTPUT/'comparison_bounds.csv').open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    (OUTPUT/'ablation_beta0_paired.tex').write_text(beta0_bounds_table(results), encoding='utf-8')
    summary = dict(source='four current manuscript tables only', n=5, df=4,
                   method='conservative paired-t sensitivity bounds; normality unverified',
                   raw_seed_records_used=False, covariance_estimated=False,
                   input_tables={name: hashlib.sha256(text.encode()).hexdigest()
                                 for name, text in tables.items()},
                   comparisons=len(results), family_size=36, rounding_half_unit=.00005,
                   envelope_excludes_zero=sum(r['ci_low']>0 or r['ci_high']<0 for r in results),
                   gap_envelope_excludes_zero=sum(r['ci_low']>0 or r['ci_high']<0
                                                 for r in results if r['metric'] not in ('HR', 'NDCG')),
                   min_holm_p_upper=min(r['p_upper_holm'] for r in results),
                   certified_after_holm=sum(r['p_upper_holm']<.05 for r in results))
    (OUTPUT/'summary.json').write_text(json.dumps(summary, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
