"""Same-year attack contracts with fixtures/stubs only; no experiments."""
import math
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
from scipy import sparse
from scipy.stats import norm

from src.same_year_mia import (
    disjoint_same_year_pool, output_features_from_logits, score_metrics,
    gaussian_out_score, fit_offline_calibration, exact_outlier_mask,
    subgroup_metrics, fit_outlier_slices, generalization_row, verify_target_recipe,
    run_same_year_study,
    query_output_features,
    fit_rich_attack, RICH_FEATURES,
    split_same_year_shadow, assign_attack_record_roles, balanced_shadow_attack_ids,
)
from src.multiyear_baseline import BASELINE_FEATURES


def fixture(count=100):
    frame = pd.DataFrame({key: np.ones(count, dtype=np.float32) for key in BASELINE_FEATURES},
        index=pd.Index([f'acs_2023_{i:03d}' for i in range(count)], name='record_id'))
    frame['AGEP'] = np.arange(count, dtype=np.float32) + 20
    frame['SEX'] = np.arange(count) % 2 + 1
    frame['RAC1P'] = np.arange(count) % 3 + 1
    frame['income_gt_50k'] = np.arange(count) % 2
    return frame


def calibration_fixture():
    ids = pd.Index(list('abcdef'))
    # The query model has extreme scores. Its own output must not influence
    # per-record OUT centers, even for a known nonmember query.
    margins = np.tile(np.arange(7, dtype=float)[:, None], (1, 6))
    margins[0] = [100., 200., 100., 200., 100., 200.]
    pure_out = np.ones_like(margins, dtype=bool)
    pure_out[0, ::2] = False
    rows = pd.DataFrame(dict(record_id=ids, membership=[1, 0, 1, 0, 1, 0], reference_index=0))
    roles = dict(train={'a', 'b'}, validation={'c', 'd'}, test={'e', 'f'})
    return margins, pure_out, rows, roles, ids


class SameYearMiaTests(unittest.TestCase):
    def test_new_split_is_seeded_stratified_disjoint_and_has_expected_rounding(self):
        pool = fixture(1003)
        parts = split_same_year_shadow(pool, seed=42, early_stopping=True)
        self.assertEqual({key: len(value) for key, value in parts.items()},
            dict(train=751, validation=101, nonmembers=151))
        seen = set()
        for frame in parts.values():
            self.assertFalse(seen & set(frame.index))
            self.assertEqual(set(frame['income_gt_50k'].unique()), {0, 1})
            seen.update(frame.index)
        self.assertEqual(seen, set(pool.index))
        repeated = split_same_year_shadow(pool, seed=42, early_stopping=True)
        changed = split_same_year_shadow(pool, seed=52, early_stopping=True)
        self.assertEqual(list(parts['train'].index), list(repeated['train'].index))
        self.assertNotEqual(set(parts['train'].index), set(changed['train'].index))

    def test_no_stopping_split_has_no_validation_and_keeps_full_test(self):
        parts = split_same_year_shadow(fixture(1000), seed=52, early_stopping=False)
        self.assertEqual({key: len(value) for key, value in parts.items()}, dict(train=800, nonmembers=200))
        self.assertTrue(set(parts['train'].index).isdisjoint(parts['nonmembers'].index))

    def test_attack_sampling_keeps_all_out_and_exact_role_balance_without_validation_rows(self):
        pool = fixture(1000)
        parts = split_same_year_shadow(pool)
        roles = assign_attack_record_roles(pool.index)
        selected, audit = balanced_shadow_attack_ids(parts, roles, seed=42)
        self.assertEqual(set(selected['nonmembers']), set(parts['nonmembers'].index))
        self.assertEqual(len(selected['train']), len(parts['nonmembers']))
        self.assertEqual(len(set(selected['train'])), len(selected['train']))
        self.assertTrue(set(selected['train']) <= set(parts['train'].index))
        self.assertTrue(set(parts['validation'].index).isdisjoint(set(selected['train']) | set(selected['nonmembers'])))
        self.assertTrue(audit.selected_in.eq(audit.selected_out).all())
        for role, ids in roles.items():
            self.assertEqual(len(set(selected['train']) & ids), len(set(selected['nonmembers']) & ids))
        repeated, _ = balanced_shadow_attack_ids(parts, roles, seed=42)
        changed, _ = balanced_shadow_attack_ids(parts, roles, seed=52)
        np.testing.assert_array_equal(selected['train'], repeated['train'])
        self.assertNotEqual(set(selected['train']), set(changed['train']))

    def test_balancing_refuses_to_drop_out_rows_when_role_has_insufficient_in(self):
        parts = split_same_year_shadow(fixture(100))
        roles = dict(train=set(parts['nonmembers'].index), validation=set(parts['train'].index),
            test=set(parts['validation'].index))
        with self.assertRaisesRegex(ValueError, 'Insufficient shadow IN'):
            balanced_shadow_attack_ids(parts, roles)

    def test_pool_excludes_every_target_role_and_preserves_unused_same_year_rows(self):
        source = fixture()
        partitions = dict(train=source.iloc[:20], validation=source.iloc[20:30],
            test=source.iloc[30:40], mia_members=source.iloc[:10], mia_nonmembers=source.iloc[40:50])
        pool, audit = disjoint_same_year_pool(source, partitions)
        self.assertEqual(list(pool.index), list(source.index[50:]))
        self.assertEqual(audit.overlapping_record_ids.sum(), 0)
        self.assertEqual(len(audit), 5)
        for frame in partitions.values():
            self.assertTrue(set(pool.index).isdisjoint(frame.index))

    def test_pool_rejects_missing_target_roles_and_different_year_ids(self):
        source = fixture()
        with self.assertRaisesRegex(ValueError, 'five target partitions'):
            disjoint_same_year_pool(source, dict(train=source.iloc[:20]))
        parts = {key: source.iloc[:2].copy() for key in
            ['train', 'validation', 'test', 'mia_members', 'mia_nonmembers']}
        parts['train'].index = ['acs_2024_0', 'acs_2024_1']
        with self.assertRaisesRegex(ValueError, 'same-year source'):
            disjoint_same_year_pool(source, parts)

    def test_stable_full_output_features_preserve_extreme_logits_and_shift_invariance(self):
        logits = np.array([[0., 1000.], [1000., 0.], [0., 0.], [-100., -99.]])
        labels = np.array([1, 1, 0, 0])
        outputs = output_features_from_logits(logits, labels, list('abcd'))
        self.assertTrue(np.isfinite(outputs.to_numpy()).all())
        np.testing.assert_allclose(outputs.p_income_le_50k + outputs.p_income_gt_50k, 1.)
        np.testing.assert_allclose(outputs.class_scaled_logit, [1000., -1000., 0., -1.])
        self.assertAlmostEqual(outputs.loc['b', 'cross_entropy'], 1000.)
        self.assertAlmostEqual(outputs.loc['b', 'modified_entropy'], 2000.)
        shifted = output_features_from_logits(logits + 1e4, labels, list('abcd'))
        np.testing.assert_allclose(outputs.to_numpy(), shifted.to_numpy(), atol=1e-12)

    def test_batched_logit_queries_never_fit_preprocessing_or_update_weights(self):
        class Processor:
            def __init__(self):
                self.queried = []

            def transform(self, frame):
                self.queried.append(frame.index.tolist())
                self.columns = list(frame.columns)
                return sparse.csr_matrix(frame[['AGEP', 'WKHP']].to_numpy(dtype=np.float32))

        source, processor = fixture(10), Processor()
        model = torch.nn.Linear(2, 2)
        before = {name: value.detach().clone() for name, value in model.state_dict().items()}
        output = query_output_features(model, processor, source, batch_size=3)
        self.assertEqual(list(output.index), list(source.index))
        self.assertEqual(len(processor.queried), 4)
        self.assertEqual(processor.columns, BASELINE_FEATURES)
        self.assertFalse(model.training)
        for name, value in model.state_dict().items():
            torch.testing.assert_close(value, before[name])

    def test_score_metrics_reports_real_slice_baseline_and_undefined_auc(self):
        metrics = score_metrics([0, 0, 0, 1], [.1, .2, .8, .9], threshold=.5)
        self.assertEqual((metrics['tn'], metrics['fp'], metrics['fn'], metrics['tp']), (2, 1, 0, 1))
        self.assertEqual(metrics['majority_class_accuracy'], .75)
        self.assertAlmostEqual(metrics['balanced_accuracy'], (2/3+1)/2)
        self.assertLess(metrics['accuracy_ci95_low'], metrics['accuracy'])
        self.assertGreater(metrics['accuracy_ci95_high'], metrics['accuracy'])
        one_class = score_metrics([1, 1], [.1, .9], threshold=.5)
        self.assertTrue(math.isnan(one_class['roc_auc']))
        self.assertEqual(one_class['majority_class_accuracy'], 1.)
        self.assertEqual(one_class['metric_status'], 'single_membership_class_auc_undefined')

    def test_rich_rf_keeps_record_groups_and_fits_only_training_output_features(self):
        features = RICH_FEATURES + ['blind_walk']
        rows = pd.DataFrame({column: np.ones(120) for column in features})
        rows['record_id'] = np.repeat([f'aux_{i}' for i in range(60)], 2)
        rows['membership'] = np.tile([0, 1], 60)
        rows['confidence'] = np.where(rows.membership, .8, .2)

        class Classifier:
            def fit(self, x, y):
                self.fit_features = list(x.columns)
                self.fit_index = x.index.copy()
                self.fit_labels = y.copy()

            def predict_proba(self, x):
                return np.column_stack([1-x.confidence, x.confidence])

        classifier = Classifier()
        with patch('src.same_year_mia.RandomForestClassifier', return_value=classifier) as factory:
            result = fit_rich_attack(rows, features, role_ids=assign_attack_record_roles(rows.record_id.unique()))
        self.assertEqual(classifier.fit_features, features)
        self.assertNotIn('membership', classifier.fit_features)
        self.assertEqual(list(classifier.fit_index), list(result['splits']['train'].index))
        role_ids = {role: set(frame.record_id) for role, frame in result['splits'].items()}
        self.assertTrue(role_ids['train'].isdisjoint(role_ids['validation'] | role_ids['test']))
        self.assertTrue(role_ids['validation'].isdisjoint(role_ids['test']))
        for frame in result['splits'].values():
            self.assertEqual(frame.membership.value_counts()[0], frame.membership.value_counts()[1])
        self.assertEqual(result['metrics']['test']['roc_auc'], 1.)
        self.assertEqual(factory.call_args.kwargs['n_estimators'], 300)
        self.assertEqual(factory.call_args.kwargs['max_depth'], 12)

    def test_outlier_selection_is_exact_top_five_percent_with_stable_ties(self):
        mask = exact_outlier_mask(np.ones(100))
        self.assertEqual(np.flatnonzero(mask).tolist(), list(range(5)))
        mask = exact_outlier_mask(np.arange(20))
        self.assertEqual(np.flatnonzero(mask).tolist(), [19])

    def test_outlier_detector_fits_only_disjoint_pool_and_ignores_membership(self):
        source = fixture()
        pool, candidates = source.iloc[:60], source.iloc[60:].copy()
        candidates['membership'] = np.arange(len(candidates)) % 2

        class Processor:
            def fit_transform(self, frame):
                self.fit_ids = frame.index.copy()
                self.columns = list(frame.columns)
                return sparse.csr_matrix(frame[['AGEP']])

            def transform(self, frame):
                self.query_ids = frame.index.copy()
                return sparse.csr_matrix(frame[['AGEP']])

        class Detector:
            def fit(self, values):
                self.fit_values = values.toarray()

            def score_samples(self, values):
                return -np.arange(values.shape[0], dtype=float)

        processor, detector = Processor(), Detector()
        with patch('src.same_year_mia.build_baseline_preprocessor', return_value=processor), \
                patch('src.same_year_mia.IsolationForest', return_value=detector) as constructor:
            scores, mask = fit_outlier_slices(pool, candidates)
        self.assertEqual(list(processor.fit_ids), list(pool.index))
        self.assertEqual(processor.columns, BASELINE_FEATURES)
        self.assertEqual(list(processor.query_ids), list(candidates.index))
        self.assertEqual(constructor.call_args.kwargs['max_samples'], 256)
        self.assertEqual(mask.sum(), 2)
        np.testing.assert_array_equal(scores, np.arange(40))

    def test_offline_score_matches_negative_gaussian_log_likelihood(self):
        values, centers = np.array([-1., 0., 10.]), np.array([0., 1., 2.])
        np.testing.assert_allclose(gaussian_out_score(values, centers, 2.), -norm.logpdf(values, centers, 2.))
        with self.assertRaises(ValueError):
            gaussian_out_score(values, centers, 0)

    def test_offline_centers_exclude_query_model_and_non_out_validation_references(self):
        values, out, rows, roles, ids = calibration_fixture()
        out[6] = False  # Model 6 used these candidates for validation: never OUT.
        result = fit_offline_calibration(values, out, rows, roles, ids)
        scored = pd.concat(result['splits'].values())
        # Only references 1..5 are eligible; median is 3, not the query's 100/200.
        np.testing.assert_allclose(scored.out_center, 3.)
        np.testing.assert_array_equal(scored.out_reference_count, 5)
        self.assertEqual(result['skipped_rows'], 0)
        self.assertEqual(result['out_reference_count_summary'], dict(minimum=5, median=5., maximum=5))
        for role, frame in result['splits'].items():
            self.assertEqual(set(frame.record_id), roles[role])

    def test_offline_variance_uses_only_attack_training_record_ids(self):
        values, out, rows, roles, ids = calibration_fixture()
        before = fit_offline_calibration(values, out, rows, roles, ids)
        changed = values.copy()
        changed[:, 2:] += 500.
        after = fit_offline_calibration(changed, out, rows, roles, ids)
        self.assertEqual(before['scale'], after['scale'])
        # Infinity is a valid frozen "predict all nonmember" cutoff when the
        # predeclared score is uninformative/anti-informative on validation.
        self.assertFalse(np.isnan(before['accuracy_threshold']))

    def test_offline_rejects_overlapping_roles_and_insufficient_out_support(self):
        values, out, rows, roles, ids = calibration_fixture()
        with self.assertRaisesRegex(ValueError, 'disjoint'):
            fit_offline_calibration(values, out, rows, dict(roles, test={'a', 'e', 'f'}), ids)
        with self.assertRaisesRegex(ValueError, 'No usable'):
            fit_offline_calibration(values[:3], out[:3], rows, roles, ids, minimum_out=3)

    def test_subgroups_keep_frozen_cutoffs_and_flag_single_class_slices(self):
        candidates = fixture(20)
        scored = pd.DataFrame(dict(membership=np.arange(20)%2, score=np.linspace(0, 1, 20)), index=candidates.index)
        slices = subgroup_metrics(scored, candidates, exact_outlier_mask(np.arange(20)),
            score_column='score', threshold=.7, low_fpr_threshold=.9)
        self.assertTrue(slices.threshold.eq(.7).all())
        outlier = slices.loc[slices.group.eq('top_5_percent')].iloc[0]
        self.assertEqual(outlier['records'], 1)
        self.assertEqual(outlier['majority_class_accuracy'], 1.)
        self.assertTrue(np.isnan(outlier['roc_auc']))
        with self.assertRaisesRegex(ValueError, 'aligned'):
            subgroup_metrics(scored.iloc[::-1], candidates, np.zeros(20, dtype=bool),
                score_column='score', threshold=.7, low_fpr_threshold=.9)

    def test_gap_sign_and_target_recipe_guard(self):
        row = generalization_row(dict(train=dict(accuracy=.9, log_loss=.2),
            validation=dict(accuracy=.8, log_loss=.4), test=dict(accuracy=.75, log_loss=.5)))
        self.assertAlmostEqual(row['validation_minus_train_log_loss'], .2)
        self.assertAlmostEqual(row['train_minus_holdout_accuracy'], .15)
        target = dict(preprocessor=SimpleNamespace(feature_names_in_=BASELINE_FEATURES),
            training=dict(hidden_dims=[512, 256, 128], epochs_requested=100, early_stopping=True))
        verify_target_recipe(target, dict(hidden_dims=(512, 256, 128), epochs=100, early_stopping=True, seed=999))
        with self.assertRaisesRegex(ValueError, 'Target/config mismatch'):
            verify_target_recipe(target, dict(early_stopping=False))

    def test_study_routes_matched_recipes_and_keeps_targets_out_of_all_reference_fits(self):
        """Exercise orchestration with stub models/classifiers, no fitting."""
        source = fixture()
        partitions = dict(train=source.iloc[:20], validation=source.iloc[20:30],
            test=source.iloc[30:40], mia_members=source.iloc[:10], mia_nonmembers=source.iloc[40:50])
        recipe = dict(hidden_dims=(512, 256, 128), epochs=100, early_stopping=True, seed=42)
        income = {role: dict(accuracy=.8, log_loss=.4) for role in ['train', 'validation', 'test']}
        target = dict(model=SimpleNamespace(tag=42),
            preprocessor=SimpleNamespace(feature_names_in_=BASELINE_FEATURES), partitions=partitions,
            training=dict(hidden_dims=[512, 256, 128], epochs_requested=100, early_stopping=True),
            income_metrics=income,
            attack_features=pd.DataFrame(dict(record_id=list(partitions['mia_members'].index)
                + list(partitions['mia_nonmembers'].index), blind_walk=2.)))

        def train(pool, algorithm, *, seed, parameters, partitions):
            self.assertEqual(algorithm, 'mlp')
            self.assertEqual(parameters, dict(recipe, seed=seed))
            for frame in target['partitions'].values():
                self.assertTrue(set(pool.index).isdisjoint(frame.index))
            parts = partitions
            return dict(model=SimpleNamespace(tag=seed), preprocessor=target['preprocessor'],
                partition_ids={name: part.index for name, part in parts.items()},
                history=[], metadata=dict(parameters),
                metrics={name: dict(accuracy=.8, log_loss=.4) for name in parts})

        def query(model, processor, frame):
            logits = np.column_stack([np.zeros(len(frame)), frame.AGEP.to_numpy()/100+model.tag/1000])
            return output_features_from_logits(logits, frame['income_gt_50k'], frame.index)

        def collect(results, pool, **kwargs):
            name, model = next(iter(results.items()))
            self.assertIsNone(kwargs['per_class'])
            pieces = [pd.DataFrame(dict(record_id=model['partition_ids'][role], membership=membership,
                        blind_walk=3., shadow_model=name)) for role, membership in [('train', 1), ('nonmembers', 0)]]
            return pd.concat(pieces, ignore_index=True), {name: {}}

        class Classifier:
            def predict_proba(self, frame):
                self.features = list(frame.columns)
                return np.tile([.4, .6], (len(frame), 1))

        classifier = Classifier()
        with patch('src.same_year_mia.train_baseline_shadow', side_effect=train) as trainer, \
                patch('src.same_year_mia.query_output_features', side_effect=query) as query_call, \
                patch('src.same_year_mia.collect_shadow_paper_features', side_effect=collect), \
                patch('src.same_year_mia.fit_rich_attack', return_value=dict(classifier=classifier)):
            result = run_same_year_study(source, target, target_year=2023, count=4,
                train_config=recipe, walk_config={}, numeric_protocol={})
        self.assertEqual(trainer.call_count, 4)
        self.assertEqual(query_call.call_count, 9)
        self.assertEqual(result['auxiliary_margins'].shape, (4, 50))
        self.assertEqual(result['target_reference_margins'].shape, (4, 20))
        self.assertIn('blind_walk', classifier.features)
        self.assertNotIn('membership', classifier.features)
        self.assertEqual(result['target_rows'].membership.sum(), 10)
        self.assertTrue(result['target_rows'].blind_walk.eq(2.).all())
        self.assertEqual(result['pure_out_masks'].sum(axis=1).tolist(), [8]*4)
        self.assertTrue(result['attack_balance_audit'].selected_in.eq(result['attack_balance_audit'].selected_out).all())
        self.assertEqual(len(result['shadow_rows']), 4*2*8)
        self.assertEqual(result['shadow_split_policy'], '75/10/15')


if __name__ == '__main__':
    unittest.main()
