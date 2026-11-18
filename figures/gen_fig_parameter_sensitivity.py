"""Plot the audited three-seed SASRec sensitivity data (run from repo root)."""
import csv
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


SOURCE = Path('els-cas-templates/source_data/table_source_data/Supplementary_Data_S9_sensitivity_summary.csv')
AXES = ('epsilon', 'augment_ratio', 'utility_beta')
TITLES = (r'Sampling sharpness $\epsilon$', r'Augmentation ratio $r$', r'Skew weight $\beta$ ($\alpha=1-\beta$)')
COLORS = {'NDCG': '#165A8A', 'Gender gap': '#C77B26', 'Age gap': '#76558E'}


def main():
    with SOURCE.open(encoding='utf-8', newline='') as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 18 and all(int(r['n']) == 3 for r in rows)
    plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'DejaVu Sans'],
                         'font.size': 7.5, 'axes.labelsize': 8, 'axes.titlesize': 8.5,
                         'xtick.labelsize': 7, 'ytick.labelsize': 7,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'axes.linewidth': .6, 'pdf.fonttype': 42, 'svg.fonttype': 'none'})
    for dataset, title in (('ml-1m', 'ML-1M'), ('taobao', 'Taobao')):
        data = [r for r in rows if r['dataset'] == dataset]
        ndcg_low = min(float(r['NDCG@10_mean']) - float(r['NDCG@10_sd']) for r in data)
        ndcg_high = max(float(r['NDCG@10_mean']) + float(r['NDCG@10_sd']) for r in data)
        pad = .12 * (ndcg_high - ndcg_low)
        gap_high = max(float(r[f'{a}_NDCG@10_Gap_mean']) + float(r[f'{a}_NDCG@10_Gap_sd'])
                       for r in data for a in ('gender', 'age_group')) * 1.12
        gap_low = min(0, min(float(r[f'{a}_NDCG@10_Gap_mean']) - float(r[f'{a}_NDCG@10_Gap_sd'])
                             for r in data for a in ('gender', 'age_group'))) * 1.15
        fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.15))
        fig.subplots_adjust(left=.10, right=.985, top=.84, bottom=.16, wspace=.28, hspace=.42)
        for column, (axis, heading) in enumerate(zip(AXES, TITLES)):
            points = sorted([r for r in data if r['axis'] == axis], key=lambda r: float(r['value']))
            x = np.array([float(r['value']) for r in points])
            reference = next(float(r['value']) for r in points if r['reference'] == 'True')
            axes[0, column].set_title(heading, pad=10)
            for row_index, metric, label, marker in (
                    (0, 'NDCG@10', 'NDCG', 'o'),
                    (1, 'gender_NDCG@10_Gap', 'Gender gap', 's'),
                    (1, 'age_group_NDCG@10_Gap', 'Age gap', '^')):
                ax = axes[row_index, column]
                y = np.array([float(r[metric + '_mean']) for r in points])
                error = np.array([float(r[metric + '_sd']) for r in points])
                ax.errorbar(x, y, yerr=error, color=COLORS[label], marker=marker, markersize=4,
                            linewidth=1.15, elinewidth=.7, capsize=2, zorder=3)
            for row_index in (0, 1):
                ax = axes[row_index, column]
                ax.axvline(reference, color='#9A9A9A', linestyle=':', linewidth=.8, zorder=0)
                ax.grid(axis='y', color='#E5E5E5', linewidth=.5, zorder=0)
                ax.set_xticks(x, [f'{v:g}' for v in x])
                ax.set_xlim(x[0] - .12 * np.ptp(x), x[-1] + .12 * np.ptp(x))
                ax.tick_params(width=.6, length=3)
                ax.text(-.14, 1.06, chr(97 + row_index * 3 + column), transform=ax.transAxes,
                        fontweight='bold', fontsize=9)
                if column:
                    ax.tick_params(labelleft=False)
            axes[0, column].set_ylim(ndcg_low - pad, ndcg_high + pad)
            axes[1, column].set_ylim(gap_low, gap_high)
            axes[1, column].axhline(0, color='#B5B5B5', linewidth=.6, zorder=0)
            axes[1, column].set_xlabel((r'$\epsilon$', r'$r$', r'$\beta$')[column])
        axes[0, 0].set_ylabel('NDCG@10 (higher is better)')
        axes[1, 0].set_ylabel('Absolute NDCG@10 gap')
        fig.suptitle(f'{title} / SASRec', fontsize=10, fontweight='bold', y=.985)
        handles = [Line2D([], [], color=COLORS[label], marker=marker, linewidth=1.1,
                          markersize=4, label=label)
                   for label, marker in (('NDCG', 'o'), ('Gender gap', 's'), ('Age gap', '^'))]
        handles.append(Line2D([], [], color='#9A9A9A', linestyle=':', linewidth=.8,
                              label='Reference setting'))
        fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(.53, .015),
                   ncol=4, frameon=False, fontsize=7, handlelength=1.8, columnspacing=1.7)
        for directory in (Path('els-cas-templates'), Path('figures/parameter_sensitivity')):
            directory.mkdir(parents=True, exist_ok=True)
            stem = directory / f'parameter_sensitivity_{dataset.replace("-", "")}'
            for extension in ('pdf', 'svg', 'png'):
                fig.savefig(stem.with_suffix('.' + extension), dpi=300)
        plt.close(fig)
    print('Wrote two six-panel figures as PDF, SVG, and 300-dpi PNG.')


if __name__ == '__main__':
    main()
