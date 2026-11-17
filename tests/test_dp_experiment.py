"""DP study contract tests with fixtures/stubs; no model training or experiments."""
from contextlib import ExitStack
from importlib.util import find_spec
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import torch
from scipy import sparse

from src.dp_experiment import (
    _calibrate_fixed_schedule, _dense_batch, _training_loss, categorical_budget_rows, check_partition_ids,
    prepare_public_preprocessing, train_dp_study_target, plot_budget_accuracy,
)
from src.expanded_data import EXPANDED_FEATURES
from src.training import build_mlp


def frame(prefix, count=8):
    values = pd.DataFrame({column: np.ones(count, dtype=np.float32) for column in EXPANDED_FEATURES},
        index=pd.Index([f'{prefix}_{i}' for i in range(count)], name='record_id'))
    values['AGEP'] = np.arange(count, dtype=np.float32) + 20
    values['WKHP'] = 40.
    values['income_gt_50k'] = np.arange(count) % 2
    return values


def partitions():
    training = frame('train')
    return dict(train=training, validation=frame('validation'), test=frame('test'),
        mia_members=training.iloc[:4].copy(), mia_nonmembers=frame('nonmembers', 4))


class FakeAccountant:
    def __init__(self):
        self.history = []

    def step(self, *, noise_multiplier, sample_rate):
        self.history.append((noise_multiplier, sample_rate, 1))

    def get_epsilon(self, *, delta):
        return 0.1 * sum(item[2] for item in self.history)


class FakeModel:
    def __init__(self):
        self.calls = 0
        self.restored = None
        self.cleaned = False

    def to(self, device):
        return self

    def parameters(self):
        return []

    def train(self):
        return self

    def eval(self):
        return self

    def __call__(self, batch):
        self.calls += 1
        return object()

    def state_dict(self):
        return {'weight_counter': self.calls}

    def load_state_dict(self, values):
        self.restored = values['weight_counter']

    def to_standard_module(self):
        self.cleaned = True
        return self


class FakeLoader:
    sample_rate = 0.5

    def __len__(self):
        return 3

    def __iter__(self):
        for indices, labels in [([0, 1], [0, 1]), ([], []), ([2, 3], [0, 1])]:
            yield torch.tensor(indices, dtype=torch.long), torch.tensor(labels, dtype=torch.long)


class FakeOptimizer:
    noise_multiplier = 1.0
    sample_rate = 0.5

    def __init__(self):
        self.accountant = None
        self.steps = 0

    def zero_grad(self, **kwargs):
        pass

    def step(self):
        self.steps += 1
        if self.accountant:
            self.accountant.step(noise_multiplier=self.noise_multiplier, sample_rate=self.sample_rate)


class FakeEngine:
    last = None

    def __init__(self, **kwargs):
        FakeEngine.last = self
        self.accountant = FakeAccountant()

    def make_private_with_epsilon(self, **kwargs):
        self.settings = kwargs
        kwargs['optimizer'].accountant = self.accountant
        return kwargs['module'], kwargs['optimizer'], FakeLoader()

    def get_epsilon(self, *, delta):
        return self.accountant.get_epsilon(delta=delta)


class FakeMemoryManager:
    calls = []

    def __init__(self, **kwargs):
        self.settings = kwargs
        FakeMemoryManager.calls.append(kwargs)

    def __enter__(self):
        return self.settings['data_loader']

    def __exit__(self, *args):
        pass


class DPExperimentTests(unittest.TestCase):
    def test_preprocessing_never_learns_target_statistics_or_categories(self):
        public, parts = frame('public'), partitions()
        public.loc[public.index[0], 'JWMNP'] = np.nan
        parts['train']['AGEP'] = 80.
        parts['train']['OCCP'] = 9999.
        parts['train']['JWRIP'] = np.nan
        processor, arrays = prepare_public_preprocessing(public, parts)
        numeric = processor.named_transformers_['numeric']
        self.assertAlmostEqual(numeric.named_steps['scale'].mean_[0], public['AGEP'].mean())
        self.assertEqual(numeric.named_steps['impute'].indicator_.features_.tolist(), [2])
        columns = next(columns for name, _, columns in processor.transformers if name == 'categorical')
        encoder = processor.named_transformers_['categorical'].named_steps['encode']
        self.assertNotIn(9999., encoder.categories_[columns.index('OCCP')])
        for name, values in arrays.items():
            self.assertEqual(values['X'].shape[1], arrays['train']['X'].shape[1])
            self.assertTrue(np.isfinite(values['X'].data).all())
            np.testing.assert_array_equal(values['record_ids'], parts[name].index)

    def test_public_boundaries_and_membership_ids_are_enforced(self):
        public, parts = frame('public'), partitions()
        parts['validation'] = parts['train'].copy()
        with self.assertRaisesRegex(ValueError, 'Overlapping'):
            check_partition_ids(parts, public)
        parts = partitions()
        parts['mia_members'] = frame('outsider', 4)
        with self.assertRaisesRegex(ValueError, 'must be target training'):
            check_partition_ids(parts, public)

    def test_sparse_physical_batches_and_empty_poisson_loss(self):
        source = sparse.csr_matrix(np.arange(12).reshape(4, 3))
        values = _dense_batch(source, torch.tensor([3, 1]), torch.device('cpu'))
        np.testing.assert_array_equal(values.numpy(), [[9, 10, 11], [3, 4, 5]])
        empty = _dense_batch(source, torch.empty(0, dtype=torch.long), torch.device('cpu'))
        self.assertEqual(tuple(empty.shape), (0, 3))
        loss = _training_loss(torch.empty((0, 2), requires_grad=True), torch.empty(0, dtype=torch.long))
        self.assertEqual(loss.item(), 0.)
        self.assertTrue(loss.requires_grad)

    def test_fixed_schedule_bound_survives_best_epoch_restoration_and_empty_samples(self):
        model, optimizer, loader = FakeModel(), FakeOptimizer(), FakeLoader()
        losses = [.4, .3, .35, .36]
        public_x = sparse.csr_matrix(np.ones((4, 2), dtype=np.float32))
        labels = np.array([0, 1, 0, 1])
        loss_mock = MagicMock()
        validation = [dict(accuracy=.75, roc_auc=.8, log_loss=value) for value in losses]
        FakeMemoryManager.calls = []
        with ExitStack() as stack:
            stack.enter_context(patch('src.dp_experiment._load_opacus', return_value=(
                FakeEngine, lambda **kwargs: FakeAccountant(), FakeMemoryManager, MagicMock())))
            stack.enter_context(patch('src.dp_experiment.build_mlp', return_value=model))
            stack.enter_context(patch('src.dp_experiment.torch.optim.Adam', return_value=optimizer))
            loader_factory = stack.enter_context(patch('src.dp_experiment.DataLoader', return_value=loader))
            loss_call = stack.enter_context(patch('src.dp_experiment._training_loss', return_value=loss_mock))
            evaluator = stack.enter_context(patch('src.dp_experiment.evaluate_model', side_effect=validation))
            result, history, metadata = train_dp_study_target(public_x, labels, public_x, labels,
                epsilon=5, delta=1e-6, max_epochs=5, patience=2, batch_size=2, max_physical_batch_size=1)
        self.assertIs(result, model)
        self.assertTrue(model.cleaned)
        self.assertEqual(metadata['best_epoch'], 2)
        self.assertEqual(metadata['epochs_run'], 4)
        self.assertEqual(model.restored, 6)
        self.assertEqual(metadata['planned_optimizer_steps'], 15)
        self.assertEqual(metadata['actual_optimizer_steps'], 12)
        self.assertAlmostEqual(metadata['epsilon_fixed_schedule_bound'], 1.5)
        self.assertAlmostEqual(metadata['epsilon_at_stop_diagnostic'], 1.2)
        self.assertEqual(optimizer.steps, 12)  # Includes all four empty samples.
        self.assertEqual(sum(len(call.args[1]) == 0 for call in loss_call.call_args_list), 4)
        self.assertEqual(loss_mock.backward.call_count, 12)
        self.assertTrue(FakeEngine.last.settings['poisson_sampling'])
        self.assertEqual(FakeEngine.last.settings['max_grad_norm'], 1.)
        self.assertEqual(FakeEngine.last.settings['epochs'], 5)
        self.assertIsNot(FakeEngine.last.settings['noise_generator'], loader_factory.call_args.kwargs['generator'])
        self.assertTrue(all(call['max_physical_batch_size'] == 1 for call in FakeMemoryManager.calls))
        self.assertEqual(len(history), 4)
        for call in evaluator.call_args_list:
            self.assertIs(call.args[1], public_x)

    def test_invalid_privacy_parameters_fail_before_model_creation(self):
        values, labels = np.ones((4, 2)), np.array([0, 1, 0, 1])
        for settings in [dict(epsilon=0), dict(epsilon=float('inf')), dict(delta=.25),
                         dict(selection_metric='test_accuracy'), dict(strict_accounting='yes'),
                         dict(early_stopping='no'), dict(restore_best=0)]:
            with self.subTest(settings=settings), patch('src.dp_experiment.build_mlp') as build:
                with self.assertRaises(ValueError):
                    train_dp_study_target(values, labels, values, labels, **settings)
                build.assert_not_called()

    def test_accuracy_selection_ties_and_strict_accounting_use_conservative_rate(self):
        """Stub updates only: verify selection and accountant contracts together."""
        model, optimizer = FakeModel(), FakeOptimizer()
        optimizer.sample_rate = 1 / 3
        values = sparse.csr_matrix(np.ones((4, 2), dtype=np.float32))
        labels = np.array([0, 1, 0, 1])
        validations = [dict(accuracy=accuracy, roc_auc=.9, log_loss=loss)
            for accuracy, loss in [(.8, .3), (.85, .5), (.85, .4), (.84, .2), (.83, .1)]]
        planned_accountants = []

        def create_accountant(**kwargs):
            result = FakeAccountant()
            planned_accountants.append(result)
            return result

        with ExitStack() as stack:
            stack.enter_context(patch('src.dp_experiment._load_opacus', return_value=(
                FakeEngine, create_accountant, FakeMemoryManager, MagicMock())))
            stack.enter_context(patch('src.dp_experiment.build_mlp', return_value=model))
            stack.enter_context(patch('src.dp_experiment.torch.optim.Adam', return_value=optimizer))
            stack.enter_context(patch('src.dp_experiment.DataLoader', return_value=FakeLoader()))
            stack.enter_context(patch.object(FakeLoader, 'sample_rate', .25))
            stack.enter_context(patch('src.dp_experiment._training_loss', return_value=MagicMock()))
            stack.enter_context(patch('src.dp_experiment.evaluate_model', side_effect=validations))
            calibration = stack.enter_context(patch('src.dp_experiment._calibrate_fixed_schedule', return_value=2.))
            _, history, metadata = train_dp_study_target(values, labels, values, labels,
                epsilon=5, max_epochs=6, patience=2, selection_metric='accuracy', strict_accounting=True)
        calibration.assert_called_once_with(5, 1e-6, 1 / 3, 18)
        self.assertEqual(metadata['best_epoch'], 3)
        self.assertEqual(metadata['epochs_run'], 5)
        self.assertEqual(model.restored, 9)
        self.assertEqual(metadata['sampler_sample_rate'], .25)
        self.assertEqual(metadata['accountant_sample_rate'], 1 / 3)
        self.assertEqual(metadata['bound_sample_rate'], 1 / 3)
        self.assertEqual(metadata['noise_multiplier'], 2.)
        self.assertEqual(len(history), 5)
        self.assertTrue(all(item[:2] == (2., 1 / 3) for item in planned_accountants[0].history))
        self.assertTrue(all(item[:2] == (2., 1 / 3) for item in FakeEngine.last.accountant.history))

    def test_fixed_100_epoch_schedule_returns_final_weights_despite_worsening_validation(self):
        """Stub-only control/DP runs: no real optimizer updates or experiment."""
        values = sparse.csr_matrix(np.ones((4, 2), dtype=np.float32))
        labels = np.array([0, 1, 0, 1])
        for epsilon in (None, 50):
            with self.subTest(epsilon=epsilon), ExitStack() as stack:
                model, optimizer = FakeModel(), FakeOptimizer()
                optimizer.sample_rate = 1 / 3
                stack.enter_context(patch('src.dp_experiment._load_opacus', return_value=(
                    FakeEngine, lambda **kwargs: FakeAccountant(), FakeMemoryManager, MagicMock())))
                stack.enter_context(patch('src.dp_experiment.build_mlp', return_value=model))
                stack.enter_context(patch('src.dp_experiment.torch.optim.Adam', return_value=optimizer))
                stack.enter_context(patch('src.dp_experiment.DataLoader', return_value=FakeLoader()))
                stack.enter_context(patch.object(FakeLoader, 'sample_rate', .25))
                stack.enter_context(patch('src.dp_experiment._training_loss', return_value=MagicMock()))
                stack.enter_context(patch('src.dp_experiment._calibrate_fixed_schedule', return_value=2.))
                stack.enter_context(patch('src.dp_experiment.evaluate_model', side_effect=[
                    dict(accuracy=.8, roc_auc=.9, log_loss=.1 + epoch / 100)
                    for epoch in range(100)]))
                result, history, metadata = train_dp_study_target(values, labels, values, labels,
                    epsilon=epsilon, max_epochs=100, patience=1, dropout=0., weight_decay=0.,
                    learning_rate=.001, early_stopping=False, restore_best=False, strict_accounting=True)
            self.assertIs(result, model)
            self.assertEqual(metadata['best_epoch'], 1)  # Diagnostic only.
            self.assertEqual(metadata['returned_epoch'], 100)
            self.assertEqual(metadata['epochs_run'], 100)
            self.assertEqual(len(history), 100)
            self.assertIsNone(model.restored)
            self.assertEqual(model.calls, 300)
            self.assertEqual(metadata['planned_optimizer_steps'], metadata['actual_optimizer_steps'])
            self.assertFalse(metadata['early_stopping'])
            self.assertFalse(metadata['restore_best'])
            if epsilon is not None:
                self.assertEqual(metadata['epsilon_fixed_schedule_bound'], metadata['epsilon_at_stop_diagnostic'])

    def test_budget_categories_preserve_declared_order_and_reject_incomplete_runs(self):
        results = pd.DataFrame(dict(private=[True, False, True, True, True, True],
            epsilon_requested=[50, None, 10, 1, 25, 5],
            target_test_accuracy=[.8] * 6, mia_accuracy=[.51] * 6))
        ordered, labels = categorical_budget_rows(results, (1, 5, 10, 25, 50))
        self.assertEqual(ordered['epsilon_requested'].tolist(), [1, 5, 10, 25, 50])
        self.assertEqual(labels, ['ε = 1', 'ε = 5', 'ε = 10', 'ε = 25', 'ε = 50'])
        with self.assertRaisesRegex(ValueError, 'Exactly one'):
            categorical_budget_rows(results.iloc[:-1], (1, 5, 10, 25, 50))

    @unittest.skipUnless(find_spec('opacus'), 'Optional DP dependency not installed')
    def test_real_opacus_wrap_gradients_memory_manager_and_cleanup_without_training(self):
        """Check actual library compatibility; never call optimizer.step()."""
        from opacus import PrivacyEngine
        from opacus.utils.batch_memory_manager import BatchMemoryManager
        from opacus.validators import ModuleValidator
        from torch.utils.data import DataLoader, TensorDataset

        model = build_mlp(3, hidden_dims=(8, 4), dropout=.1)
        initial = {key: value.clone() for key, value in model.state_dict().items()}
        ModuleValidator.validate(model, strict=True)
        source = sparse.csr_matrix(np.ones((64, 3), dtype=np.float32))
        dataset = TensorDataset(torch.arange(64), torch.arange(64) % 2)
        loader = DataLoader(dataset, batch_size=16, generator=torch.Generator().manual_seed(7))
        engine = PrivacyEngine(accountant='rdp', secure_mode=False)
        private_model, optimizer, private_loader = engine.make_private_with_epsilon(module=model,
            optimizer=torch.optim.Adam(model.parameters()), data_loader=loader,
            target_epsilon=5., target_delta=1e-6, epochs=30,
            max_grad_norm=1., poisson_sampling=True, grad_sample_mode='hooks')
        try:
            with BatchMemoryManager(data_loader=private_loader, max_physical_batch_size=3,
                    optimizer=optimizer) as physical_loader:
                indices, labels = next(iter(physical_loader))
                self.assertLessEqual(len(indices), 3)
            for indices in [torch.tensor([0, 1, 2]), torch.empty(0, dtype=torch.long)]:
                optimizer.zero_grad(set_to_none=True)
                batch = _dense_batch(source, indices, torch.device('cpu'))
                loss = _training_loss(private_model(batch), (indices % 2).long())
                self.assertTrue(torch.isfinite(loss).item())
                loss.backward()
                for parameter in model.parameters():
                    self.assertEqual(len(parameter.grad_sample), len(indices))
            self.assertEqual(engine.accountant.history, [])
        finally:
            restored = private_model.to_standard_module()
        self.assertIs(restored, model)
        for key, value in restored.state_dict().items():
            torch.testing.assert_close(value, initial[key])

    @unittest.skipUnless(find_spec('opacus'), 'Optional DP dependency not installed')
    def test_all_five_real_noise_calibrations_respect_full_schedule_budget(self):
        """Accountant arithmetic only; no model, data loading or training."""
        from opacus.accountants import create_accountant
        from opacus.accountants.utils import get_noise_multiplier

        logical_batches = int(np.ceil(100_000 / 256))
        sample_rate, epochs, delta = 1 / logical_batches, 30, 1e-6
        sigmas = []
        for epsilon in (1, 5, 10, 25, 50):
            with self.subTest(epsilon=epsilon):
                sigma = get_noise_multiplier(target_epsilon=epsilon, target_delta=delta,
                    sample_rate=sample_rate, epochs=epochs, accountant='rdp')
                accountant = create_accountant(mechanism='rdp')
                for _ in range(logical_batches * epochs):
                    accountant.step(noise_multiplier=sigma, sample_rate=sample_rate)
                self.assertLessEqual(accountant.get_epsilon(delta=delta), epsilon + 1e-6)
                self.assertGreater(sigma, 0)
                sigmas.append(sigma)
        self.assertTrue(all(a > b for a, b in zip(sigmas, sigmas[1:])))

    @unittest.skipUnless(find_spec('opacus'), 'Optional DP dependency not installed')
    def test_strict_calibration_covers_exact_private_loader_schedule_at_all_budgets(self):
        """Accountant arithmetic only, covering the 391-to-390 rounding case."""
        from opacus.accountants import create_accountant

        steps, accountant_rate, sampler_rate, delta = 390 * 30, 1 / 390, 1 / 391, 1e-6
        for epsilon in (1, 5, 10, 25, 50):
            with self.subTest(epsilon=epsilon):
                sigma = _calibrate_fixed_schedule(epsilon, delta, accountant_rate, steps)
                bounds = []
                for rate in (sampler_rate, accountant_rate):
                    accountant = create_accountant(mechanism='rdp')
                    for _ in range(steps):
                        accountant.step(noise_multiplier=sigma, sample_rate=rate)
                    bounds.append(accountant.get_epsilon(delta=delta))
                self.assertLessEqual(bounds[0], bounds[1])
                self.assertLessEqual(bounds[1], epsilon + 1e-6)

    @unittest.skipUnless(find_spec('matplotlib'), 'Optional plotting dependency not installed')
    def test_real_graph_exports_have_categorical_ticks_and_both_reference_lines(self):
        results = pd.DataFrame(dict(private=[False] + [True] * 5,
            epsilon_requested=[None, 1, 5, 10, 25, 50], delta=[None] + [1e-6] * 5,
            target_test_accuracy=[.82, .75, .78, .8, .81, .82],
            mia_accuracy=[.57, .5, .51, .53, .54, .56]))
        with TemporaryDirectory() as directory, patch.dict('os.environ',
                {'MPLCONFIGDIR': str(Path(directory) / 'font_cache')}):
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            for metric, stem in [('target_test_accuracy', 'target_accuracy_vs_epsilon'),
                                 ('mia_accuracy', 'mia_accuracy_vs_epsilon')]:
                figure, axes = plot_budget_accuracy(results, (1, 5, 10, 25, 50),
                    metric=metric, output_dir=directory, delta=1e-6)
                try:
                    self.assertEqual([tick.get_text() for tick in axes.get_xticklabels()],
                        ['ε = 1', 'ε = 5', 'ε = 10', 'ε = 25', 'ε = 50'])
                    np.testing.assert_allclose(np.diff(axes.get_xticks()), np.ones(4))
                    self.assertEqual(len(axes.lines), 3 if metric == 'mia_accuracy' else 2)
                    for suffix in ['png', 'svg']:
                        self.assertGreater((Path(directory) / f'{stem}.{suffix}').stat().st_size, 100)
                finally:
                    plt.close(figure)


if __name__ == '__main__':
    unittest.main()
