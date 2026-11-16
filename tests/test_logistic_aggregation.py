"""Aggregation routing/calibration checks; estimators are stubs, never trained."""
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.uncertainty_mf_mia import fit_aggregation, evaluate_aggregation


class StubEstimator:
    def __init__(self, steps=None):
        self.steps = steps

    def fit(self, frame, labels):
        self.training_frame = frame.copy()
        self.training_labels = labels.copy()
        self.feature_names_in_ = np.asarray(frame.columns)
        return self

    def predict_proba(self, frame):
        p = frame['mean'].to_numpy().copy()
        if 'std' in frame:
            p += frame['std'].to_numpy() * .1
        p = np.clip(p, .01, .99)
        return np.column_stack([1 - p, p])


def fixture(prefix):
    mean = np.array([.1, .3, .6, .9, .2, .8, .4, .7])
    return pd.DataFrame(dict(single=mean - .01, mean=mean,
        std=[.01, .2, .03, .1, .02, .15, .04, .1],
        membership=[0, 0, 1, 1, 0, 1, 0, 1]),
        index=[f'{prefix}_{i}' for i in range(len(mean))])


class LogisticAggregationTests(unittest.TestCase):
    def test_optional_pair_uses_same_meta_records_and_separate_calibration(self):
        meta, calibration, test = fixture('meta'), fixture('cal'), fixture('test')
        created = []

        def pipeline(steps):
            estimator = StubEstimator(steps)
            created.append(estimator)
            return estimator

        with patch('src.uncertainty_mf_mia.RandomForestClassifier', side_effect=lambda **kw: StubEstimator()), \
                patch('src.uncertainty_mf_mia.Pipeline', side_effect=pipeline):
            learned, thresholds = fit_aggregation(meta, calibration, include_logistic=True)
        expected = {'single', 'average', 'learned_mean', 'uncertainty',
                    'logistic_mean', 'logistic_uncertainty'}
        self.assertEqual(set(thresholds), expected)
        self.assertEqual(list(learned['logistic_mean'].feature_names_in_), ['mean'])
        self.assertEqual(list(learned['logistic_uncertainty'].feature_names_in_), ['mean', 'std'])
        for estimator in learned.values():
            self.assertEqual(list(estimator.training_frame.index), list(meta.index))
            pd.testing.assert_series_equal(estimator.training_labels, meta['membership'])
        for estimator in created:
            scaler, classifier = estimator.steps
            self.assertEqual(scaler[0], 'scale')
            self.assertEqual(type(scaler[1]).__name__, 'StandardScaler')
            self.assertEqual(classifier[1].C, 1.0)
            self.assertEqual(classifier[1].max_iter, 1000)
        metrics, predictions = evaluate_aggregation(test, learned, thresholds)
        self.assertEqual(set(metrics['method']), expected)
        self.assertEqual(set(predictions), expected)

    def test_existing_call_keeps_original_four_methods(self):
        with patch('src.uncertainty_mf_mia.RandomForestClassifier', side_effect=lambda **kw: StubEstimator()), \
                patch('src.uncertainty_mf_mia.Pipeline', side_effect=AssertionError('not requested')):
            learned, thresholds = fit_aggregation(fixture('meta'), fixture('cal'))
        self.assertEqual(set(learned), {'learned_mean', 'uncertainty'})
        self.assertEqual(len(thresholds), 4)


if __name__ == '__main__':
    unittest.main()
