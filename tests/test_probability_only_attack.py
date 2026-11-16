"""Query-budget and routing checks using deterministic stubs, never fitted models."""
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.expanded_data import EXPANDED_FEATURES
from src.paper_attack_features import extract_paper_features
from src.uncertainty_mf_mia import completion_scores


class ProbabilityOnlyTests(unittest.TestCase):
    def test_no_walk_uses_one_oracle_pass_and_keeps_probability_features(self):
        frame = pd.DataFrame({'AGEP': [20., 30., 40., 50., 60.]})
        labels = np.array([0, 1, 0, 1, 1])
        calls = []

        def oracle(rows):
            calls.append(len(rows))
            p = rows['AGEP'].to_numpy() / 100
            return np.column_stack([1 - p, p])

        fast, audit = extract_paper_features(oracle, frame, labels,
            config={'batch_size': 2, 'max_steps': 2, 'repeats': 1},
            include_blind_walk=False)
        self.assertEqual(calls, [len(frame)])
        self.assertEqual(audit['record_queries'], len(frame))
        self.assertEqual(audit['max_record_queries'], len(frame))
        self.assertFalse(audit['blind_walk_enabled'])
        self.assertIsNone(audit['walk_censored_fraction'])
        self.assertNotIn('blind_walk', fast)
        full, _ = extract_paper_features(oracle, frame, labels,
            config={'batch_size': 2, 'max_steps': 2, 'repeats': 1})
        pd.testing.assert_frame_equal(fast, full.drop(columns='blind_walk'))

    def test_completion_scoring_forwards_walk_mode_and_numeric_protocol(self):
        frame = pd.DataFrame({column: [1., 2.] for column in EXPANDED_FEATURES})
        frame['income_gt_50k'] = [0, 1]

        class Reconstructor:
            def iter_samples(self, candidates, *, seeds):
                for _ in seeds:
                    yield candidates.copy()

        class Classifier:
            feature_names_in_ = ['class_scaled_logit', 'entropy', 'cross_entropy']

            def predict_proba(self, features):
                return np.tile([.4, .6], (len(features), 1))

        protocol = {'JWMNP': (2., 0, 200)}
        for walk_enabled in [False, True]:
            def features(query, candidates, labels, **settings):
                self.assertEqual(settings['include_blind_walk'], walk_enabled)
                self.assertEqual(settings['numeric_protocol'], protocol)
                return pd.DataFrame({name: [.1, .2] for name in Classifier.feature_names_in_}), {
                    'record_queries': len(candidates)}

            with patch('src.uncertainty_mf_mia.extract_paper_features', side_effect=features) as extract:
                scores, audits = completion_scores(None, frame, Reconstructor(), Classifier(),
                    k=3, seed=42, walk_config={}, numeric_protocol=protocol,
                    include_blind_walk=walk_enabled)
            self.assertEqual(extract.call_count, 3)
            np.testing.assert_array_equal(scores, np.full((2, 3), .6))
            self.assertEqual(len(audits), 3)


if __name__ == '__main__':
    unittest.main()
