"""Common-schema baseline checks with fixtures and estimator stubs only."""
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from scipy import sparse

from src.expanded_data import EXPANDED_FEATURES
from src.shadow_models import MLP_SHADOW_CONFIG, PAPER_SHADOW_ALGORITHMS, train_shadow_model
from src.multiyear_baseline import (
    BASELINE_FEATURES, baseline_raw_path, build_baseline_preprocessor,
    load_baseline_year, preprocess_baseline_partitions, train_baseline_shadow,
)


class StubPreprocessor:
    def fit(self, frame):
        self.fitted_index = frame.index.copy()
        self.feature_names_in_ = np.asarray(frame.columns)
        return self

    def transform(self, frame):
        return sparse.csr_matrix(frame[['AGEP', 'WKHP']].to_numpy(dtype=np.float32))


class StubEstimator:
    classes_ = np.array([1, 0])

    def fit(self, x, y):
        self.fitted_labels = np.asarray(y).copy()
        return self

    def predict_proba(self, x):
        return np.tile([.6, .4], (x.shape[0], 1))

    def get_params(self):
        return {'stub': True}


def frame_fixture(count=100):
    frame = pd.DataFrame({name: np.ones(count, dtype=np.float32) for name in BASELINE_FEATURES},
        index=pd.Index([f'acs_2018_{i}' for i in range(count)], name='record_id'))
    frame['AGEP'] = 30.
    frame['WKHP'] = 40.
    frame['income_gt_50k'] = np.arange(count) % 2
    return frame


class MultiyearBaselineTests(unittest.TestCase):
    def test_common_schema_preserves_historical_schema_and_transform_settings(self):
        self.assertEqual(len(BASELINE_FEATURES), 47)
        self.assertIn('VPS', EXPANDED_FEATURES)
        self.assertNotIn('VPS', BASELINE_FEATURES)
        processor = build_baseline_preprocessor()
        numeric, categorical = processor.transformers
        self.assertEqual(numeric[2], ['AGEP', 'WKHP', 'JWMNP', 'JWRIP'])
        self.assertNotIn('VPS', categorical[2])
        self.assertEqual(set(numeric[2] + categorical[2]), set(BASELINE_FEATURES))
        self.assertTrue(numeric[1].named_steps['impute'].add_indicator)
        self.assertEqual(categorical[1].named_steps['encode'].handle_unknown, 'ignore')

    def test_2024_loader_requests_no_vps_and_preserves_labels_and_source_ids(self):
        raw = pd.DataFrame({name: [1., 1., 1., 1.] for name in BASELINE_FEATURES})
        raw['AGEP'] = [16, 30, 40, 50]
        raw['WKHP'] = [40, 40, 40, 40]
        raw['PINCP'] = [70000, 80000, 20000, 90000]
        raw['PWGTP'] = 1
        with patch('src.multiyear_baseline.pd.read_csv', return_value=raw.copy()) as read:
            frame = load_baseline_year(2024)
        self.assertNotIn('VPS', read.call_args.kwargs['usecols'])
        self.assertEqual(list(frame.index), ['acs_2024_1', 'acs_2024_2', 'acs_2024_3'])
        self.assertEqual(frame['income_gt_50k'].tolist(), [1, 0, 1])
        self.assertEqual(list(frame.columns), BASELINE_FEATURES + ['income_gt_50k'])
        self.assertTrue(str(baseline_raw_path(2014)).endswith('ss14pca.csv'))

    def test_target_preprocessing_fits_training_only(self):
        source = frame_fixture()
        partitions = dict(train=source.iloc[:60], validation=source.iloc[60:80], test=source.iloc[80:])
        processor = StubPreprocessor()
        with patch('src.multiyear_baseline.build_baseline_preprocessor', return_value=processor):
            fitted, arrays = preprocess_baseline_partitions(partitions)
        self.assertIs(fitted, processor)
        self.assertEqual(list(fitted.fitted_index), list(partitions['train'].index))
        for name, values in arrays.items():
            self.assertEqual(values['X'].dtype, np.dtype('float32'))
            np.testing.assert_array_equal(values['record_ids'], partitions[name].index)

    def test_shadow_uses_common_inputs_original_splits_and_class_order(self):
        processor, estimator = StubPreprocessor(), StubEstimator()
        with patch('src.multiyear_baseline.build_baseline_preprocessor', return_value=processor), \
                patch('src.multiyear_baseline.build_shadow_estimator', return_value=estimator) as factory:
            result = train_baseline_shadow(frame_fixture(), 'random_forest', seed=52)
        factory.assert_called_once_with('random_forest', seed=52)
        self.assertEqual(list(processor.feature_names_in_), BASELINE_FEATURES)
        self.assertEqual({name: len(ids) for name, ids in result['partition_ids'].items()},
                         {'train': 40, 'validation': 20, 'nonmembers': 40})
        self.assertEqual(set(processor.fitted_index), set(result['partition_ids']['train']))
        self.assertEqual(len(result['attack_data']), 80)
        np.testing.assert_array_equal(result['attack_data']['p_income_le_50k'], .4)
        np.testing.assert_array_equal(result['attack_data']['p_income_gt_50k'], .6)
        self.assertEqual(result['schema'], 'expanded_acs')

    def test_mlp_shadow_preserves_both_schemas_and_uses_only_shadow_validation(self):
        """Fit/prediction stubs only: no neural-network training or queries."""
        for schema in ('common47', 'expanded48'):
            with self.subTest(schema=schema):
                auxiliary = frame_fixture()
                auxiliary['AGEP'] = np.arange(len(auxiliary), dtype=np.float32) + 20
                if schema == 'expanded48':
                    auxiliary['VPS'] = 1.
                    auxiliary = auxiliary[EXPANDED_FEATURES + ['income_gt_50k']]
                processor, model = StubPreprocessor(), object()
                expected_history, expected_metadata = [{'epoch': 3}], {'returned_epoch': 3}
                owner = 'src.multiyear_baseline' if schema == 'common47' else 'src.shadow_models'
                preprocessor_owner = ('src.multiyear_baseline.build_baseline_preprocessor'
                    if schema == 'common47' else 'src.expanded_data.build_expanded_preprocessor')

                def probabilities(unused_model, x):
                    return np.tile([.4, .6], (x.shape[0], 1))

                with patch(preprocessor_owner, return_value=processor), \
                        patch(owner + '.train_target_model', return_value=(
                            model, expected_history, expected_metadata)) as fit, \
                        patch(owner + '.predict_probabilities', side_effect=probabilities) as predict, \
                        patch(owner + '.build_shadow_estimator') as tree:
                    if schema == 'common47':
                        result = train_baseline_shadow(auxiliary, 'mlp', seed=72,
                            parameters=dict(MLP_SHADOW_CONFIG))
                    else:
                        result = train_shadow_model(auxiliary, 'mlp', seed=72, schema='expanded_acs',
                            parameters=dict(MLP_SHADOW_CONFIG))
                tree.assert_not_called()
                self.assertEqual(fit.call_count, 1)
                self.assertEqual(fit.call_args.kwargs, dict(MLP_SHADOW_CONFIG, seed=72))
                self.assertEqual(predict.call_count, 3)
                expected_columns = BASELINE_FEATURES if schema == 'common47' else EXPANDED_FEATURES
                self.assertEqual(list(processor.feature_names_in_), expected_columns)
                ids = result['partition_ids']
                self.assertEqual({name: len(values) for name, values in ids.items()},
                    {'train': 40, 'validation': 20, 'nonmembers': 40})
                self.assertTrue(set(ids['train']).isdisjoint(ids['validation']))
                self.assertTrue(set(ids['nonmembers']).isdisjoint(set(ids['train']) | set(ids['validation'])))
                for x_arg, y_arg, name in [(0, 1, 'train'), (2, 3, 'validation')]:
                    np.testing.assert_array_equal(fit.call_args.args[x_arg].toarray(),
                        auxiliary.loc[ids[name], ['AGEP', 'WKHP']].to_numpy(dtype=np.float32))
                    np.testing.assert_array_equal(fit.call_args.args[y_arg],
                        auxiliary.loc[ids[name], 'income_gt_50k'].to_numpy())
                self.assertEqual(set(processor.fitted_index), set(ids['train']))
                self.assertIs(result['model'], model)
                self.assertIs(result['history'], expected_history)
                self.assertIs(result['metadata'], expected_metadata)
                self.assertEqual(len(result['attack_data']), 80)
                self.assertEqual(set(result['attack_data']['record_id']), set(ids['train']) | set(ids['nonmembers']))
                np.testing.assert_allclose(result['attack_data']['p_income_le_50k'], .4)
                np.testing.assert_allclose(result['attack_data']['p_income_gt_50k'], .6)

    def test_mlp_addition_keeps_historical_tree_pool_and_generic_defaults(self):
        self.assertEqual(PAPER_SHADOW_ALGORITHMS,
            ('random_forest', 'gradient_boosting', 'xgboost'))
        self.assertEqual(MLP_SHADOW_CONFIG['hidden_dims'], (128, 64))
        self.assertEqual(MLP_SHADOW_CONFIG['epochs'], 50)
        self.assertTrue(MLP_SHADOW_CONFIG['early_stopping'])
        self.assertTrue(MLP_SHADOW_CONFIG['restore_best'])

    def test_explicit_same_year_splits_fit_training_only_and_never_pass_test_for_validation(self):
        from src.same_year_mia import split_same_year_shadow

        auxiliary = frame_fixture()
        for early_stopping in (True, False):
            with self.subTest(early_stopping=early_stopping):
                parts = split_same_year_shadow(auxiliary, early_stopping=early_stopping)
                processor, model = StubPreprocessor(), object()

                def probabilities(unused, values):
                    return np.tile([.4, .6], (values.shape[0], 1))

                with patch('src.multiyear_baseline.build_baseline_preprocessor', return_value=processor), \
                        patch('src.multiyear_baseline.train_target_model', return_value=(model, [], {})) as fit, \
                        patch('src.multiyear_baseline.predict_probabilities', side_effect=probabilities):
                    result = train_baseline_shadow(auxiliary, 'mlp', partitions=parts,
                        parameters=dict(early_stopping=early_stopping, restore_best=early_stopping))
                self.assertEqual(set(processor.fitted_index), set(parts['train'].index))
                self.assertTrue(set(processor.fitted_index).isdisjoint(parts['nonmembers'].index))
                self.assertEqual(fit.call_args.args[0].shape[0], 75 if early_stopping else 80)
                if early_stopping:
                    self.assertEqual(fit.call_args.args[2].shape[0], 10)
                    self.assertEqual(result['metrics']['validation']['records'], 10)
                else:
                    self.assertIsNone(fit.call_args.args[2])
                    self.assertIsNone(fit.call_args.args[3])
                    self.assertNotIn('validation', result['metrics'])


if __name__ == '__main__':
    unittest.main()
