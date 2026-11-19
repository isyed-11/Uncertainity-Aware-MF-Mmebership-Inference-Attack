"""Same-year notebook/plot checks with synthetic metrics; no experiments."""
import ast
import importlib.util
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.dp_same_year import plot_low_fpr_budget


ROOT = Path(__file__).resolve().parents[1]


def metric_fixture():
    return pd.DataFrame([
        dict(private=True, epsilon_requested=2, delta=1e-6, target_test_accuracy=.8,
             mia_accuracy=.53, mia_tpr_at_1pct_fpr_roc=.04,
             mia_tpr_at_shadow_threshold=.03, mia_fpr_at_shadow_threshold=.02),
        dict(private=False, epsilon_requested=None, delta=None, target_test_accuracy=.81,
             mia_accuracy=.57, mia_tpr_at_1pct_fpr_roc=.10,
             mia_tpr_at_shadow_threshold=.09, mia_fpr_at_shadow_threshold=.04),
        dict(private=True, epsilon_requested=.5, delta=1e-6, target_test_accuracy=.75,
             mia_accuracy=.51, mia_tpr_at_1pct_fpr_roc=.02,
             mia_tpr_at_shadow_threshold=.01, mia_fpr_at_shadow_threshold=.005),
    ])


class SameYearDPTests(unittest.TestCase):
    def test_plot_preserves_budget_order_and_separates_roc_and_fixed_cutoff(self):
        with TemporaryDirectory() as directory:
            figure, axes = plot_low_fpr_budget(metric_fixture(), (.5, 2), output_dir=directory, delta=1e-6)
            try:
                np.testing.assert_allclose(axes[0].lines[0].get_ydata(), [.02, .04])
                np.testing.assert_allclose(axes[1].lines[0].get_ydata(), [.01, .03])
                np.testing.assert_allclose(axes[1].lines[1].get_ydata(), [.005, .02])
                for extension in ['png', 'svg']:
                    self.assertGreater((Path(directory) / f'mia_low_fpr_vs_epsilon.{extension}').stat().st_size, 0)
            finally:
                plt.close(figure)

    def test_plot_rejects_invalid_low_fpr_measurements_or_delta(self):
        for field, invalid in [('delta', 1e-5), ('mia_fpr_at_shadow_threshold', np.nan),
                               ('mia_tpr_at_1pct_fpr_roc', 1.1)]:
            results = metric_fixture()
            results.loc[0, field] = invalid
            with self.subTest(field=field), TemporaryDirectory() as directory:
                with self.assertRaises(ValueError):
                    plot_low_fpr_budget(results, (.5, 2), output_dir=directory, delta=1e-6)

    def test_generator_preserves_dp_training_and_queries_and_validates_both_public_sources(self):
        """Parse generated cell sources only; do not execute notebook cells."""
        spec = importlib.util.spec_from_file_location('same_year_generator', ROOT / 'scripts/create_dp_same_year_notebook.py')
        generator = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(generator)
        original = json.loads((ROOT / 'exp_4_differential_privacy_accuracy.ipynb').read_text(encoding='utf-8'))
        for year in [2014, 2023]:
            notebook = generator.build_notebook(year)
            def source(nb, index):
                return ''.join(nb['cells'][index]['source'])
            def value(nb, index, name):
                for node in ast.parse(source(nb, index)).body:
                    if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets):
                        return node.value
                raise AssertionError(name)
            with self.subTest(year=year):
                self.assertEqual(len(notebook['cells']), 23)
                for index, cell in enumerate(notebook['cells']):
                    if cell['cell_type'] == 'code':
                        ast.parse(source(notebook, index))
                        self.assertEqual(cell['outputs'], [])
                        self.assertIsNone(cell['execution_count'])
                for name in ['TRAIN_CONFIG', 'SPLIT_CONFIG', 'TRAIN_SIZE', 'DELTA', 'WALK_CONFIG',
                             'EXPANDED_NUMERIC_WALK', 'SHADOW_ALGORITHMS', 'N_SHADOW_MODELS', 'SHADOW_RECORDS_PER_CLASS']:
                    self.assertEqual(ast.dump(value(original, 2, name)), ast.dump(value(notebook, 2, name)), name)
                for index in [8, 14, 16]:
                    self.assertEqual(source(original, index), source(notebook, index))
                self.assertEqual(ast.literal_eval(value(notebook, 2, 'EPSILON_BUDGETS')), (.5, 1, 2, 4, 8))
                self.assertEqual(ast.literal_eval(value(notebook, 2, 'AUXILIARY_YEAR')), 2023)
                self.assertEqual(ast.literal_eval(value(notebook, 2, 'PUBLIC_PREPROCESSING_YEAR')), year)
                self.assertIn('early_stopping=False', source(notebook, 6))
                self.assertIn('partitions=partitions', source(notebook, 6))
                self.assertIn('disjoint_same_year_pool(', source(notebook, 4))


if __name__ == '__main__':
    unittest.main()
