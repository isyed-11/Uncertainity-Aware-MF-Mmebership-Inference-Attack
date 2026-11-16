"""Common-schema baseline checks with fixtures and estimator stubs only."""
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from scipy import sparse

from src.expanded_data import EXPANDED_FEATURES
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


if __name__ == '__main__':
    unittest.main()
