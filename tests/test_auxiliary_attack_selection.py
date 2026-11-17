"""Selection-boundary tests using stubs; no estimator training or notebook runs."""
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.auxiliary_attack_selection import (
    FEATURE_SETS, POOLS, LossThresholdScore, candidate_specs, evaluate_frozen_attack,
    fixed_pool_splits, pool_frame, select_auxiliary_attack, validation_feature_diagnostics,
)


def fixture():
    rows, roles = [], {}
    for role in ('train', 'validation', 'test'):
        roles[role] = {f'acs_2014_{role}_{i}' for i in range(12)}
        for family in POOLS[1:]:
            for i in range(12):
                member = i % 2
                rows.append(dict(record_id=f'acs_2014_{role}_{i}', membership=member,
                    shadow_model=f'{family}_01', class_scaled_logit=(2 * member - 1) * .4 + i * .001,
                    blind_walk=5 + member * 10, entropy=.2 if member else .8,
                    cross_entropy=.1 if member else 1.5, index=f'{role}/{family}/{i}'))
    return pd.DataFrame(rows).set_index('index'), roles


class StubEstimator:
    fitted = []

    def fit(self, frame, labels):
        self.feature_names_in_ = np.asarray(frame.columns)
        self.fitted_ids = frame.index.tolist()
        self.fitted.append(self)
        return self

    def predict_proba(self, frame):
        if any(str(index).startswith('test/') for index in frame.index):
            raise AssertionError('Search attempted to score auxiliary test data.')
        p = (frame['class_scaled_logit'].to_numpy() + 1) / 2
        return np.column_stack([1 - p, p])


class AuxiliarySelectionTests(unittest.TestCase):
    def test_predeclared_grid_contains_reference_and_exactly_84_unique_candidates(self):
        specs = candidate_specs()
        self.assertEqual(len(specs), 84)
        self.assertEqual(len({spec['candidate'] for spec in specs}), 84)
        counts = pd.Series([spec['algorithm'] for spec in specs]).value_counts().to_dict()
        self.assertEqual(counts, dict(random_forest=72, logistic_regression=8, loss_threshold=4))
        reference = next(spec for spec in specs if spec['candidate'] == 'pooled/paper/rf_d12_leaf20')
        self.assertEqual((reference['max_depth'], reference['min_samples_leaf']), (12, 20))

    def test_all_pools_keep_master_record_roles_and_reject_overlap(self):
        frame, roles = fixture()
        for pool in POOLS:
            splits = fixed_pool_splits(pool_frame(frame, pool), roles)
            for role, part in splits.items():
                self.assertEqual(set(part['record_id']), roles[role])
            self.assertEqual(sum(map(len, splits.values())), len(pool_frame(frame, pool)))
        bad = {role: set(ids) for role, ids in roles.items()}
        bad['test'].add(next(iter(roles['train'])))
        with self.assertRaisesRegex(ValueError, 'overlap'):
            fixed_pool_splits(frame, bad)
        with self.assertRaisesRegex(ValueError, 'master role'):
            fixed_pool_splits(frame.assign(record_id='unassigned'), roles)

    def test_selection_fits_only_training_and_never_scores_test(self):
        frame, roles = fixture()
        StubEstimator.fitted = []
        with patch('src.auxiliary_attack_selection.RandomForestClassifier', side_effect=lambda **kw: StubEstimator()) as forest, \
                patch('src.auxiliary_attack_selection.make_pipeline', side_effect=lambda *args: StubEstimator()):
            comparison = select_auxiliary_attack(frame, roles)
        self.assertEqual(forest.call_count, 72)
        self.assertEqual(len(StubEstimator.fitted), 80)
        self.assertEqual({call.kwargs['max_depth'] for call in forest.call_args_list}, {4, 8, 12})
        self.assertEqual({call.kwargs['min_samples_leaf'] for call in forest.call_args_list}, {20, 50, 100})
        for estimator in StubEstimator.fitted:
            self.assertTrue(all(index.startswith('train/') for index in estimator.fitted_ids))
            self.assertIn(list(estimator.feature_names_in_), list(FEATURE_SETS.values()))
        table = comparison['validation_table']
        self.assertEqual(len(table), 84)
        self.assertFalse(any(column.startswith('test_') or column.startswith('target_') for column in table))
        self.assertEqual(comparison['selected']['spec']['candidate'], table.iloc[0]['candidate'])
        self.assertEqual(comparison['reference']['spec']['candidate'], 'pooled/paper/rf_d12_leaf20')
        self.assertEqual(set(comparison), {'selected', 'reference', 'validation_table'})

    def test_test_labels_and_values_cannot_change_candidate_selection(self):
        frame, roles = fixture()
        changed = frame.copy()
        mask = changed['record_id'].isin(roles['test'])
        changed.loc[mask, 'membership'] = 1 - changed.loc[mask, 'membership']
        changed.loc[mask, ['class_scaled_logit', 'blind_walk', 'entropy', 'cross_entropy']] = 99.
        with patch('src.auxiliary_attack_selection.RandomForestClassifier', side_effect=lambda **kw: StubEstimator()), \
                patch('src.auxiliary_attack_selection.make_pipeline', side_effect=lambda *args: StubEstimator()):
            first = select_auxiliary_attack(frame, roles)
            second = select_auxiliary_attack(changed, roles)
        pd.testing.assert_frame_equal(first['validation_table'], second['validation_table'])
        self.assertEqual(first['selected']['spec'], second['selected']['spec'])

    def test_fixed_loss_score_and_frozen_cutoffs(self):
        scorer = LossThresholdScore()
        values = pd.DataFrame(dict(cross_entropy=[.1, .8, 1.5, .2], membership=[1, 0, 0, 1]))
        scores = scorer.predict_proba(values)[:, 1]
        np.testing.assert_allclose(scores, np.exp(-values['cross_entropy']))
        result = dict(classifier=scorer, accuracy_threshold=.7, low_fpr_threshold=.85)
        metrics, returned = evaluate_frozen_attack(result, values)
        np.testing.assert_array_equal(scores, returned)
        self.assertEqual(metrics['threshold'], .7)
        self.assertEqual(metrics['shadow_selected_threshold'], .85)
        self.assertEqual(metrics['accuracy'], 1.)

    def test_feature_diagnostics_use_validation_only(self):
        frame, roles = fixture()
        first = validation_feature_diagnostics(frame, roles)
        frame.loc[frame['record_id'].isin(roles['test']), 'cross_entropy'] = 100.
        second = validation_feature_diagnostics(frame, roles)
        for before, after in zip(first, second):
            pd.testing.assert_frame_equal(before, after)
        self.assertEqual(len(first[0]), 32)
        self.assertEqual(len(first[1]), 16)


if __name__ == '__main__':
    unittest.main()
