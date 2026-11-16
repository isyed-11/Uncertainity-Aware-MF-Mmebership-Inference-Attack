"""Utility regression tests with stub trees; no model fitting or experiments."""
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.expanded_data import EXPANDED_FEATURES
from src.uncertainty_mf_mia import ConditionalDonorChain, completion_scores
from src.uncertainty_checkpoints import (
    atomic_save, load_verified, cached_reconstructor, cached_audit_draws, frame_signature,
)


class StubTransform:
    def __init__(self):
        self.calls = 0

    def transform(self, frame):
        self.calls += 1
        return frame.to_numpy(dtype=np.float32)


class StubTree:
    def apply(self, x):
        return (x.toarray()[:, 0] > 1).astype(int)


def candidate_frame():
    frame = pd.DataFrame({c: [0., 1., 2., 3.] for c in EXPANDED_FEATURES},
                         index=['a', 'b', 'c', 'd'])
    frame['income_gt_50k'] = [0, 1, 0, 1]
    return frame


def stub_chain():
    chain = ConditionalDonorChain(['MAR', 'WKHP'])
    first, second = StubTransform(), StubTransform()
    forest = SimpleNamespace(estimators_=[StubTree(), StubTree()])
    donors = [{0: np.array([1., 2.]), 1: np.array([3., 4.])} for _ in range(2)]
    chain.stages = [('MAR', ['AGEP'], first, forest, donors),
                    ('WKHP', ['MAR'], second, forest, donors)]
    chain.checkpoint_signature_ = 'stub-model-signature'
    return chain, first, second


class RuntimeTests(unittest.TestCase):
    def test_preprocessing_reuse_preserves_seeded_draws_and_hidden_values_stay_hidden(self):
        frame = candidate_frame()
        chain, first, second = stub_chain()
        expected = [chain.sample(frame, seed=seed) for seed in [10, 11, 12]]
        first.calls = second.calls = 0
        actual = list(chain.iter_samples(frame, seeds=[10, 11, 12]))
        self.assertEqual(first.calls, 1)
        self.assertEqual(second.calls, 3)
        for left, right in zip(expected, actual):
            pd.testing.assert_frame_equal(left, right)
        changed = frame.copy()
        changed[['MAR', 'WKHP']] = -999
        for left, right in zip(actual, chain.iter_samples(changed, seeds=[10, 11, 12])):
            pd.testing.assert_frame_equal(left, right)

    def test_atomic_cache_rejects_mismatches_and_leaves_no_partial_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'cache.joblib'
            self.assertIsNone(load_verified(path, 's1'))
            atomic_save(path, 's1', [1, 2])
            self.assertEqual(load_verified(path, 's1'), [1, 2])
            with self.assertRaisesRegex(ValueError, 'mismatch'):
                load_verified(path, 's2')
            self.assertEqual(list(Path(directory).glob('*.part')), [])

    def test_frame_signature_tracks_data_and_row_order(self):
        frame = candidate_frame()
        self.assertNotEqual(frame_signature(frame), frame_signature(frame.iloc[::-1]))
        changed = frame.copy()
        changed.iloc[0, 0] += 1
        self.assertNotEqual(frame_signature(frame), frame_signature(changed))

    def test_fitted_checkpoint_is_saved_without_auditing_and_loaded_without_fitting(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(ConditionalDonorChain, 'fit', return_value=SimpleNamespace()) as fit:
                cached_reconstructor(candidate_frame(), ['MAR'], seed=42, directory=directory)
                fit.assert_called_once()
            with patch.object(ConditionalDonorChain, 'fit', side_effect=AssertionError('must not fit')):
                loaded = cached_reconstructor(candidate_frame(), ['MAR'], seed=42, directory=directory)
                self.assertTrue(hasattr(loaded, 'checkpoint_signature_'))
            changed = candidate_frame()
            changed.iloc[0, 0] += 1
            with self.assertRaisesRegex(ValueError, 'mismatch'):
                cached_reconstructor(changed, ['MAR'], seed=42, directory=directory)

    def test_audit_resumes_saved_draw_after_interruption(self):
        chain, _, _ = stub_chain()
        original = chain.iter_samples

        def interrupted(frame, *, seeds):
            yield next(original(frame, seeds=seeds))
            raise RuntimeError('simulated interruption')

        with tempfile.TemporaryDirectory() as directory:
            with patch.object(chain, 'iter_samples', side_effect=interrupted):
                with self.assertRaisesRegex(RuntimeError, 'simulated interruption'):
                    list(cached_audit_draws(chain, candidate_frame(), k=3, seed=10, directory=directory))
            with patch.object(chain, 'iter_samples', wraps=original) as samples:
                resumed = list(cached_audit_draws(chain, candidate_frame(), k=3, seed=10, directory=directory))
                self.assertEqual(samples.call_args.kwargs['seeds'], [11, 12])
            expected = list(original(candidate_frame(), seeds=[10, 11, 12]))
            for left, right in zip(resumed, expected):
                pd.testing.assert_frame_equal(left, right[chain.hidden])

    def test_score_resume_skips_feature_queries_and_rejects_changed_oracle(self):
        chain, _, _ = stub_chain()
        base = StubClassifier()
        calls = []

        def features(query, frame, labels, **kwargs):
            calls.append(1)
            return pd.DataFrame({'signal': frame['MAR'].to_numpy() / 10}), {'record_queries': len(frame)}

        with tempfile.TemporaryDirectory() as directory:
            settings = dict(k=3, seed=10, walk_config={'seed': 42},
                            checkpoint_dir=directory, context_signature='oracle1')
            with patch('src.uncertainty_mf_mia.extract_paper_features', side_effect=features):
                first, audits = completion_scores(None, candidate_frame(), chain, base, **settings)
                self.assertEqual(len(calls), 3)
            with patch('src.uncertainty_mf_mia.extract_paper_features', side_effect=AssertionError('must not query')):
                second, repeated = completion_scores(None, candidate_frame(), chain, base, **settings)
            np.testing.assert_array_equal(first, second)
            self.assertEqual(audits, repeated)
            with self.assertRaisesRegex(ValueError, 'mismatch'):
                completion_scores(None, candidate_frame(), chain, base,
                                  **(settings | {'context_signature': 'oracle2'}))


class StubClassifier:
    feature_names_in_ = ['signal']

    def predict_proba(self, frame):
        values = frame['signal'].to_numpy()
        return np.column_stack([1 - values, values])


if __name__ == '__main__':
    unittest.main()
