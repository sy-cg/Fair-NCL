import ast
import re
import unittest
from pathlib import Path

from analysis.export_displayed_ablation_tables import ablation_tables
from analysis.summary_ablation_uncertainty import comparison_bounds, beta0_bounds_table


class DisplayedAblationExportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        source = Path('els-cas-templates/cas-dc-sample.tex').read_text(encoding='utf-8')
        cls.tables, cls.rows = ablation_tables(source)

    def test_reads_all_displayed_rows_and_separates_sd(self):
        self.assertEqual(len(self.tables), 4)
        self.assertEqual(len(self.rows), 88)
        row = next(r for r in self.rows if r['dataset'] == 'ML-1M'
                   and r['backbone'] == 'SASRec' and r['cutoff'] == 10
                   and r['variant'] == 'Fair-NCL')
        self.assertEqual(row['G_NDCG_mean'], '0.0116')
        self.assertEqual(row['G_NDCG_sd'], '0.0032')

    def test_beta0_table_has_current_direction_units_and_both_cutoffs(self):
        table = beta0_bounds_table(comparison_bounds(self.rows))
        self.assertIn('Full Fair-NCL minus', table)
        self.assertIn('95', table)
        self.assertIn('normal paired-difference model', table)
        self.assertNotIn('Holm', table)
        actual = []
        for line in table.splitlines():
            if '@10' in line or '@20' in line:
                cells = [c.strip() for c in line.split('&')]
                if len(cells) == 6:
                    actual.append(tuple(re.search(r'^\$([+-]?\d+\.\d+)', cell).group(1)
                                        for cell in cells[3:]))
        self.assertEqual(actual, [
            ('+0.8', '-2.2', '-0.2'), ('+1.0', '-0.6', '-0.1'),
            ('-2.3', '-4.8', '-0.7'), ('-2.4', '-4.2', '-1.2'),
            ('-0.4', '0.0', '-0.9'), ('-0.7', '-0.4', '-0.8'),
            ('+0.8', '0.0', '-0.8'), ('+0.9', '-0.3', '-0.6'),
        ])

    def test_main_table_does_not_mark_mean_ranks_as_numeric_winners(self):
        # Load the real formatting functions without reading archived CSV inputs.
        source = Path('tables/rebuild_five_seed_paper_assets.py').read_text(encoding='utf-8')
        tree = ast.parse(source)
        functions = [n for n in tree.body if isinstance(n, ast.FunctionDef)
                     and n.name in ('format_cell', 'table', 'main_table')]
        namespace = dict(METHODS=('baseline', 'fair_ncl'), BACKBONES=('sasrec',),
                         DISPLAY={'sasrec': 'SASRec'},
                         METHOD_DISPLAY={'baseline': '', 'fair_ncl': 'Fair-NCL+'},
                         cell=lambda *args: (0.5, 0.1))
        exec(compile(ast.Module(body=functions, type_ignores=[]), '<formatters>', 'exec'), namespace)
        table = namespace['main_table'](10, 'ml-1m')
        self.assertNotIn(r'\textbf{0.', table)
        self.assertNotIn(r'\underline{0.', table)
        self.assertNotIn('Bold and underlined', table)


if __name__ == '__main__':
    unittest.main()
