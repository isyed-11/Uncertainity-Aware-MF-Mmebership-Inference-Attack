"""Explicit shadow-split checks with synthetic rows and fit/prediction stubs."""
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.expanded_data import EXPANDED_FEATURES
from src.same_year_mia import split_same_year_shadow
from src.shadow_models import train_shadow_model


def fixture():
    frame = pd.DataFrame({column: np.ones(100, dtype=np.float32)
                          for column in EXPANDED_FEATURES},
                         index=[f'acs_2023_{i}' for i in range(100)])
    frame['AGEP'] = np.arange(100, dtype=np.float32) + 20
    frame['income_gt_50k'] = np.arange(100) % 2
    return frame


class EstimatorStub:
    classes_ = np.array([1, 0])

    def fit(self, x, y):
        self.training_rows = len(y)
        return self

    def predict_proba(self, x):
        return np.tile([.6, .4], (x.shape[0], 1))

    def get_params(self):
        return {'stub': True}


class ExplicitShadowSplitTests(unittest.TestCase):
    def test_80_20_is_seeded_and_train_only_preprocessing_excludes_test_categories(self):
        auxiliary = fixture()
        parts = split_same_year_shadow(auxiliary, seed=42, early_stopping=False)
        other = split_same_year_shadow(auxiliary, seed=52, early_stopping=False)
        self.assertNotEqual(set(parts['train'].index), set(other['train'].index))
        auxiliary.loc[parts['nonmembers'].index, 'COW'] = 999
        parts = {name: auxiliary.loc[part.index].copy() for name, part in parts.items()}
        for algorithm in ['random_forest', 'gradient_boosting', 'xgboost']:
            estimator = EstimatorStub()
            with self.subTest(algorithm=algorithm), patch(
                    'src.shadow_models.build_shadow_estimator', return_value=estimator):
                result = train_shadow_model(auxiliary, algorithm, schema='expanded_acs', partitions=parts)
            self.assertEqual(estimator.training_rows, 80)
            self.assertEqual({k: len(v) for k, v in result['partition_ids'].items()},
                             {'train': 80, 'nonmembers': 20})
            processor = result['preprocessor']
            self.assertEqual(len(processor.feature_names_in_), 48)
            self.assertAlmostEqual(processor.named_transformers_['numeric'].named_steps['scale'].mean_[0],
                                   parts['train']['AGEP'].to_numpy(dtype=np.float64).mean())
            _, categorical, columns = processor.transformers_[1]
            self.assertNotIn(999, categorical.named_steps['encode'].categories_[columns.index('COW')])
            np.testing.assert_allclose(result['attack_data']['p_income_le_50k'], .4)
            self.assertEqual(result['attack_data']['membership'].value_counts().to_dict(), {1: 80, 0: 20})

    def test_fixed_schedule_mlp_never_receives_out_records_for_validation(self):
        auxiliary = fixture()
        parts = split_same_year_shadow(auxiliary, seed=42, early_stopping=False)
        model = object()
        with patch('src.shadow_models.train_target_model', return_value=(model, [], {})) as fit, \
                patch('src.shadow_models.predict_probabilities', side_effect=lambda m, x:
                      np.tile([.4, .6], (x.shape[0], 1))):
            result = train_shadow_model(auxiliary, 'mlp', schema='expanded_acs', partitions=parts,
                                        parameters=dict(early_stopping=False, restore_best=False))
        self.assertEqual(fit.call_args.args[0].shape[0], 80)
        self.assertIsNone(fit.call_args.args[2])
        self.assertIsNone(fit.call_args.args[3])
        self.assertFalse(fit.call_args.kwargs['early_stopping'])
        self.assertFalse(fit.call_args.kwargs['restore_best'])
        self.assertEqual(fit.call_args.kwargs['epochs'], 50)
        self.assertNotIn('validation', result['metrics'])

    def test_no_validation_rejects_mlp_epoch_selection(self):
        auxiliary = fixture()
        parts = split_same_year_shadow(auxiliary, early_stopping=False)
        with self.assertRaisesRegex(ValueError, 'disable early_stopping'):
            train_shadow_model(auxiliary, 'mlp', schema='expanded_acs', partitions=parts)

    def test_invalid_or_modified_partitions_are_rejected_before_fitting(self):
        auxiliary = fixture()
        original = split_same_year_shadow(auxiliary, early_stopping=False)
        cases = []
        overlap = dict(original, nonmembers=pd.concat([original['nonmembers'], original['train'].iloc[:2]]))
        cases.append(overlap)
        cases.append(dict(original, train=original['train'].iloc[2:]))
        modified = {name: part.copy() for name, part in original.items()}
        modified['train'].iloc[0, 0] = -999
        cases.append(modified)
        for parts in cases:
            with self.subTest(), patch('src.shadow_models.build_shadow_estimator') as factory:
                with self.assertRaises((ValueError, AssertionError)):
                    train_shadow_model(auxiliary, 'random_forest', schema='expanded_acs', partitions=parts)
                factory.assert_not_called()


if __name__ == '__main__':
    unittest.main()
