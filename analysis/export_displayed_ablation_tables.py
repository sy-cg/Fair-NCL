"""Export author-updated tables, point differences and conditional t bounds.

This reads displayed means/SDs, not seed records; it does not recover paired data.
"""
import csv
import re
from decimal import Decimal
from pathlib import Path

METRICS = ('HR', 'NDCG', 'G_HR', 'G_NDCG', 'A_HR', 'A_NDCG')
MANUSCRIPT = Path('els-cas-templates/cas-dc-sample.tex')
OUTPUT = Path('tables/supplementary/rq2_rq5/manuscript_exports')


def ablation_tables(source):
    tables, rows = {}, []
    for backbone in ('SASRec', 'BERT4Rec'):
        for cutoff in (10, 20):
            label = f'ablation_{backbone.lower()}_{cutoff}'
            at = source.index('\\label{tab:' + label + '}')
            start = source.rfind(r'\begin{table*}', 0, at)
            end = source.index(r'\end{table*}', at) + len(r'\end{table*}')
            tables[label] = source[start:end]
            for line in tables[label].splitlines():
                if '&' not in line or r'$\pm$' not in line:
                    continue
                cells = line.split('&')
                if r'\multirow' in cells[0]:
                    dataset = 'ML-1M' if 'ML-1M' in cells[0] else 'Taobao'
                row = dict(dataset=dataset, backbone=backbone, cutoff=cutoff,
                           variant=re.sub(r'^\\rev\{(.*)\}$', r'\1', cells[1].strip()))
                for metric, cell in zip(METRICS, cells[2:]):
                    value, sd = re.search(r'(\d+\.\d+)\$\\pm\$(\d+\.\d+)', cell).groups()
                    row[metric + '_mean'], row[metric + '_sd'] = value, sd
                rows.append(row)
    return tables, rows


def beta0_differences(rows):
    lookup = {(r['dataset'], r['backbone'], r['cutoff'], r['variant']): r for r in rows}
    differences = []
    for dataset in ('ML-1M', 'Taobao'):
        for backbone in ('SASRec', 'BERT4Rec'):
            for cutoff in (10, 20):
                key = dataset, backbone, cutoff
                full = lookup[key + ('Fair-NCL',)]
                control = lookup[key + (r'Fixed-$\alpha$ $\beta=0$',)]
                row = dict(dataset=dataset, backbone=backbone, cutoff=cutoff)
                for metric in METRICS:
                    row[metric] = str((Decimal(full[metric + '_mean'])
                                       - Decimal(control[metric + '_mean'])) * 1000)
                differences.append(row)
    return differences


def main():
    tables, rows = ablation_tables(MANUSCRIPT.read_text(encoding='utf-8'))
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for label, table in tables.items():
        (OUTPUT / (label + '.tex')).write_text(table + '\n', encoding='utf-8')
    from analysis.summary_ablation_uncertainty import comparison_bounds, beta0_bounds_table
    (OUTPUT / 'ablation_beta0_paired.tex').write_text(
        beta0_bounds_table(comparison_bounds(rows)), encoding='utf-8')
    for filename, records in [('displayed_ablation_summary.csv', rows),
                              ('displayed_beta0_differences_x1000.csv', beta0_differences(rows))]:
        with (OUTPUT / filename).open('w', encoding='utf-8', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
    print(f'Exported {len(rows)} displayed rows, 8 point contrasts and F.3 t-interval envelopes to {OUTPUT}')


if __name__ == '__main__':
    main()
