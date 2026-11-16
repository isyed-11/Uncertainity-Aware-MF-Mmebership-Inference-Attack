"""Donor-storage regressions using stubs, without training or querying models."""
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from scipy import sparse

from src.expanded_data import EXPANDED_FEATURES
from src.uncertainty_mf_mia import ConditionalDonorChain


class StubTransform:
    def fit_transform(self, frame):
        return self.transform(frame)

    def transform(self, frame):
        return frame.to_numpy(dtype=np.float32)


class StubTree:
    def __init__(self, offset):
        self.offset = offset

    def apply(self, x):
        return (x.toarray()[:, 0].astype(int) + self.offset) % 3


class StubForest:
    def __init__(self):
        self.estimators_ = [StubTree(0), StubTree(1), StubTree(2)]

    def fit(self, x, y):
        # Deliberately no training; only exercise donor-storage construction.
        return self

    def apply(self, x):
        return np.column_stack([tree.apply(x) for tree in self.estimators_])


def frame_fixture(dtype=np.float64):
    frame = pd.DataFrame({c: np.arange(120, dtype=float) for c in EXPANDED_FEATURES})
    # Repeated values must retain their empirical multiplicities.
    frame['MAR'] = np.resize([1., 1., 2., 4., 4., 4.], len(frame))
    frame['WKHP'] = np.resize([10., 20., 20., 40., 50.], len(frame))
    return frame.astype(dtype)


def stub_fitted_chain(hidden, dtype=np.float64, training=None):
    with patch('src.uncertainty_mf_mia.ColumnTransformer', return_value=StubTransform()), \
            patch('src.uncertainty_mf_mia.RandomForestClassifier', side_effect=lambda **kw: StubForest()), \
            patch('src.uncertainty_mf_mia.RandomForestRegressor', side_effect=lambda **kw: StubForest()), \
            patch('builtins.print'):
        return ConditionalDonorChain(hidden, n_estimators=3).fit(
            frame_fixture(dtype) if training is None else training)


def legacy_copy(chain):
    legacy = ConditionalDonorChain(chain.hidden)
    legacy.stages = []
    for field, columns, transform, forest, groups, values in chain.stages:
        old_groups = [{leaf: values[indices] for leaf, indices in tree.items()}
                      for tree in groups]
        legacy.stages.append((field, columns, transform, forest, old_groups))
    return legacy


class ConditionalDonorTests(unittest.TestCase):
    def test_compact_codes_preserve_leaf_order_and_repeated_donor_weights(self):
        for dtype in [np.float32, np.float64]:
            training = frame_fixture(dtype)
            chain = stub_fitted_chain(['MAR', 'WKHP'], dtype)
            for field, columns, transform, forest, donors, codebook in chain.stages:
                np.testing.assert_array_equal(codebook, np.unique(training[field]))
                self.assertEqual(codebook.dtype, np.dtype(dtype))
                leaves = forest.apply(sparse.csr_matrix(transform.transform(training[columns])))
                for tree_number, groups in enumerate(donors):
                    for leaf, codes in groups.items():
                        self.assertEqual(codes.dtype, np.dtype('uint8'))
                        expected = training.loc[leaves[:, tree_number] == leaf, field].to_numpy()
                        np.testing.assert_array_equal(codebook[codes], expected)
                pool_bytes = sum(group.nbytes for tree in donors for group in tree.values())
                legacy_bytes = len(forest.estimators_) * training[field].to_numpy().nbytes
                self.assertEqual(pool_bytes * np.dtype(dtype).itemsize, legacy_bytes)
                self.assertLess(pool_bytes + codebook.nbytes, legacy_bytes)

    def test_code_dtype_widens_without_truncating_values(self):
        training = pd.DataFrame({c: np.arange(300, dtype=np.float32)
                                 for c in EXPANDED_FEATURES})
        chain = stub_fitted_chain(['WKHP'], training=training)
        _, columns, transform, forest, donors, codebook = chain.stages[0]
        self.assertEqual(len(codebook), 300)
        self.assertEqual(next(iter(donors[0].values())).dtype, np.dtype('uint16'))
        legacy = legacy_copy(chain)
        pd.testing.assert_frame_equal(chain.sample(training, seed=11),
                                      legacy.sample(training, seed=11))

    def test_sampling_is_exactly_equal_to_legacy_value_pools(self):
        for dtype in [np.float32, np.float64]:
            candidates = frame_fixture(dtype).iloc[[17, 4, 3, 90, 81, 16, 117, 18, 45, 68]]
            for hidden in [['MAR'], ['MAR', 'WKHP']]:
                chain = stub_fitted_chain(hidden, dtype)
                legacy = legacy_copy(chain)
                for seed in range(30):
                    pd.testing.assert_frame_equal(chain.sample(candidates, seed=seed),
                                                  legacy.sample(candidates, seed=seed))

    def test_iterated_sampling_preserves_draws_and_ignores_true_hidden_values(self):
        chain = stub_fitted_chain(['MAR', 'WKHP'])
        legacy = legacy_copy(chain)
        candidates = frame_fixture().iloc[:24].copy()
        expected = [legacy.sample(candidates, seed=seed) for seed in [8, 13, 21]]
        candidates[chain.hidden] = -999.
        actual = list(chain.iter_samples(candidates, seeds=[8, 13, 21]))
        for left, right in zip(expected, actual):
            pd.testing.assert_frame_equal(left, right)


if __name__ == '__main__':
    unittest.main()
