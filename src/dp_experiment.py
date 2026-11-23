"""Complete-feature DP study helpers; imports never fit or query models.

The privacy claim concerns an individual target model with public preprocessing
and public validation, conditional on the fixed research train/evaluation split.
Private evaluation statistics and membership-labelled research files are not DP
releases. Opacus is optional until private training is explicitly invoked.
"""
import copy
import math
import random
import time

import numpy as np
import pandas as pd
import torch
from scipy import sparse
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .data import TARGET
from .expanded_data import EXPANDED_FEATURES, build_expanded_preprocessor
from .training import build_mlp, evaluate_model


def check_partition_ids(partitions, public_preprocessing):
    """Protect the explicit membership definition and public-data boundaries."""
    required = {'train', 'validation', 'test', 'mia_members', 'mia_nonmembers'}
    if required - set(partitions):
        raise ValueError(f'Missing partitions: {sorted(required - set(partitions))}')
    groups = {}
    for name, frame in {**partitions, 'public_preprocessing': public_preprocessing}.items():
        if frame.empty or not frame.index.is_unique:
            raise ValueError(f'{name} requires nonempty, unique record IDs.')
        if set(frame[TARGET].unique()) != {0, 1}:
            raise ValueError(f'{name} requires both income classes.')
        groups[name] = set(frame.index)
    if not groups['mia_members'] <= groups['train']:
        raise ValueError('MIA members must be target training records.')
    exclusive = ['train', 'validation', 'test', 'mia_nonmembers', 'public_preprocessing']
    for index, name in enumerate(exclusive):
        for other in exclusive[index + 1:]:
            if groups[name] & groups[other]:
                raise ValueError(f'Overlapping record IDs: {name} / {other}')


def prepare_public_preprocessing(public_preprocessing, partitions):
    """Learn medians, scaling, missing indicators and vocabulary on public data.

    All target variants share this exact fitted representation. No target row
    updates its statistics, vocabulary, dimension or missing-indicator selection.
    """
    check_partition_ids(partitions, public_preprocessing)
    preprocessor = build_expanded_preprocessor()
    preprocessor.fit(public_preprocessing[EXPANDED_FEATURES])
    arrays = {}
    for name, frame in partitions.items():
        values = sparse.csr_matrix(
            preprocessor.transform(frame[EXPANDED_FEATURES]), dtype=np.float32)
        if not np.isfinite(values.data).all():
            raise ValueError(f'Non-finite encoded inputs in {name}.')
        arrays[name] = dict(X=values, y=frame[TARGET].to_numpy(dtype=np.int64, copy=True),
            record_ids=frame.index.to_numpy(copy=True))
    return preprocessor, arrays


def _load_opacus():
    try:
        from opacus import PrivacyEngine
        from opacus.accountants import create_accountant
        from opacus.utils.batch_memory_manager import BatchMemoryManager
        from opacus.validators import ModuleValidator
    except ImportError as error:
        raise ImportError('DP training requires: python -m pip install -r requirements.txt') from error
    return PrivacyEngine, create_accountant, BatchMemoryManager, ModuleValidator


def _dense_batch(values, indices, device):
    """Densify only the current physical batch, including a Poisson empty batch."""
    selected = values[indices.cpu().numpy()]
    if sparse.issparse(selected):
        selected = selected.toarray()
    return torch.as_tensor(np.asarray(selected), dtype=torch.float32, device=device)


def _training_loss(logits, labels):
    # Empty Poisson samples must still perform the noise/accountant step.
    # A sum over zero logits gives a finite differentiable zero, unlike mean CE.
    return nn.functional.cross_entropy(logits, labels) if len(labels) else logits.sum()


def _calibrate_fixed_schedule(epsilon, delta, sample_rate, steps):
    """Calibrate against an explicit logical-step count, without training."""
    from opacus.accountants.utils import get_noise_multiplier

    return float(get_noise_multiplier(target_epsilon=float(epsilon), target_delta=delta,
        sample_rate=sample_rate, steps=steps, accountant='rdp'))


def train_dp_study_target(X_train, y_train, X_public_validation, y_public_validation, *,
                          epsilon=None, delta=1e-6, hidden_dims=(512, 256, 128),
                          dropout=0.1, max_epochs=30, patience=5, min_delta=0.0,
                          batch_size=256, max_physical_batch_size=32,
                          learning_rate=1e-3, weight_decay=1e-4,
                          max_grad_norm=1.0, seed=42, device='cpu', progress=None,
                          selection_metric='log_loss', strict_accounting=False,
                          early_stopping=True, restore_best=True,
                          checkpoint_path=None, checkpoint_signature=None):
    """Matched Adam/DP-Adam targets with public-validation early stopping.

    epsilon=None is the fresh non-private control. Private targets use Poisson
    sampling, flat per-example clipping, Gaussian noise and the RDP accountant.
    The announced bound covers the FIXED maximum schedule, including all epochs
    considered during selection. Epsilon at the realised stopping time is only
    a diagnostic, not a tighter guarantee for adaptive early stopping.

    Initialisation uses the experiment seed; private noise and Poisson sampling
    use separate nondeterministically seeded generators whose seeds are not
    exported. secure_mode=False is a research setting, not hardened deployment.
    No train losses, private metrics, test records or MIA inputs select an epoch.
    selection_metric='accuracy' maximizes public accuracy, breaking ties by loss.
    strict_accounting=True aligns calibration and reporting with the conservative
    Opacus 1.6 accountant rate, 1 / len(private_loader), before the first update.
    Legacy defaults preserve the original notebook's configuration.
    early_stopping=False, restore_best=False runs the fixed full schedule and
    returns the final epoch; validation is then monitoring only.
    Optional internal epoch checkpoints preserve optimizer, accountant and RNG
    states. They are not DP releases. No checkpoint I/O occurs by default.
    """
    if epsilon is not None and (not math.isfinite(epsilon) or epsilon <= 0):
        raise ValueError('epsilon must be a finite positive number or None.')
    if selection_metric not in {'log_loss', 'accuracy'}:
        raise ValueError('selection_metric must be log_loss or accuracy.')
    if not isinstance(strict_accounting, bool):
        raise ValueError('strict_accounting must be a boolean.')
    if not isinstance(early_stopping, bool) or not isinstance(restore_best, bool):
        raise ValueError('early_stopping and restore_best must be booleans.')
    for name, value in [('max_epochs', max_epochs), ('patience', patience),
                        ('batch_size', batch_size), ('max_physical_batch_size', max_physical_batch_size)]:
        if not isinstance(value, int) or value < 1:
            raise ValueError(f'{name} must be a positive integer.')
    if not 0 < delta < 1 / max(len(y_train), 1):
        raise ValueError('delta must be positive and smaller than 1 / training size.')
    if not 0 <= dropout < 1 or min_delta < 0 or max_grad_norm <= 0:
        raise ValueError('Invalid dropout, min_delta or clipping norm.')
    if not math.isfinite(learning_rate) or learning_rate <= 0 or not math.isfinite(weight_decay) or weight_decay < 0:
        raise ValueError('Learning rate must be positive and weight decay nonnegative.')
    if X_train.ndim != 2 or X_public_validation.ndim != 2 or X_train.shape[1] != X_public_validation.shape[1]:
        raise ValueError('Training and public-validation dimensions must agree.')
    for values, labels in [(X_train, y_train), (X_public_validation, y_public_validation)]:
        if values.shape[0] != len(labels) or set(np.unique(labels)) != {0, 1}:
            raise ValueError('Each partition needs aligned rows and both binary income classes.')
        finite = values.data if sparse.issparse(values) else values
        if not np.isfinite(finite).all():
            raise ValueError('Inputs must be finite.')

    saved = None
    if checkpoint_path is not None:
        if not checkpoint_signature:
            raise ValueError('Training checkpoints require a data/settings signature.')
        from .dp_mf_checkpoints import capture_rng, restore_rng
        from .uncertainty_checkpoints import atomic_save, load_verified
        import joblib
        # Bind the caller's run signature to this condition and its actual data.
        checkpoint_signature = joblib.hash((checkpoint_signature, epsilon, delta,
            hidden_dims, dropout, max_epochs, patience, min_delta, batch_size,
            max_physical_batch_size, learning_rate, weight_decay, max_grad_norm,
            seed, device, selection_metric, strict_accounting, early_stopping,
            restore_best, X_train, y_train, X_public_validation, y_public_validation))
        saved = load_verified(checkpoint_path, checkpoint_signature)

    private = epsilon is not None
    opacus_parts = _load_opacus() if private else None
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = torch.device(device)
    base_model = build_mlp(X_train.shape[1], hidden_dims, dropout).to(device)
    model = base_model
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    dataset = TensorDataset(torch.arange(len(y_train)), torch.tensor(np.asarray(y_train).copy(), dtype=torch.long))
    sampling_generator = torch.Generator()
    sampling_generator.seed() if private else sampling_generator.manual_seed(seed)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
        generator=sampling_generator, num_workers=0)
    engine, planned_epsilon, noise_multiplier, sample_rate = None, None, None, None
    noise_generator = None
    accountant_sample_rate, bound_sample_rate = None, None
    planned_steps = len(loader) * max_epochs
    if private:
        PrivacyEngine, create_accountant, BatchMemoryManager, ModuleValidator = opacus_parts
        ModuleValidator.validate(model, strict=True)
        noise_generator = torch.Generator(device=device)
        noise_generator.seed()
        engine = PrivacyEngine(accountant='rdp', secure_mode=False)
        model, optimizer, loader = engine.make_private_with_epsilon(
            module=model, optimizer=optimizer, data_loader=loader,
            target_epsilon=float(epsilon), target_delta=delta, epochs=max_epochs,
            max_grad_norm=max_grad_norm, poisson_sampling=True, clipping='flat',
            grad_sample_mode='hooks', noise_generator=noise_generator)
        sample_rate = float(loader.sample_rate)
        accountant_sample_rate = 1 / len(loader)
        planned_steps = len(loader) * max_epochs
        bound_sample_rate = accountant_sample_rate if strict_accounting else sample_rate
        if strict_accounting:
            if sample_rate > accountant_sample_rate + 1e-12:
                raise RuntimeError('Sampler rate exceeds the conservative Opacus accountant rate.')
            # Opacus's hook reads optimizer.noise_multiplier on each step.
            # Change it BEFORE any update; calibrate to the exact same hook rate.
            optimizer.noise_multiplier = _calibrate_fixed_schedule(
                epsilon, delta, accountant_sample_rate, planned_steps)
        noise_multiplier = float(optimizer.noise_multiplier)
        # A fixed-schedule upper bound remains valid for selecting/stopping from
        # the DP trajectory on independent public validation data.
        planned_accountant = create_accountant(mechanism='rdp')
        for _ in range(planned_steps):
            planned_accountant.step(noise_multiplier=noise_multiplier, sample_rate=bound_sample_rate)
        planned_epsilon = float(planned_accountant.get_epsilon(delta=delta))
        if planned_epsilon > epsilon + 1e-6:
            raise RuntimeError(f'Calibrated privacy bound {planned_epsilon} exceeds requested {epsilon}.')

    history, best_state = [], None
    best_loss, best_epoch, stale = float('inf'), None, 0
    best_accuracy = -float('inf')
    elapsed_before, start_epoch = 0.0, 1
    if saved is not None:
        base_model.load_state_dict(saved['model'])
        optimizer.load_state_dict(saved['optimizer'])
        if private:
            engine.accountant.load_state_dict(saved['accountant'])
            if saved['noise_multiplier'] != noise_multiplier:
                raise ValueError('Checkpoint noise calibration does not match this run.')
            if sum(item[2] for item in engine.accountant.history) != len(loader) * saved['epoch']:
                raise ValueError('Checkpoint accountant does not cover all completed epochs.')
        history, best_state = saved['history'], saved['best_state']
        best_loss, best_epoch = saved['best_loss'], saved['best_epoch']
        best_accuracy, stale = saved['best_accuracy'], saved['stale']
        elapsed_before, start_epoch = saved['training_seconds'], saved['epoch'] + 1
        restore_rng(saved['rng'], sampling_generator, noise_generator)
        if progress:
            progress(f'Resumed training after epoch {start_epoch - 1}; privacy accountant restored')
    started = time.monotonic()
    try:
        epoch_range = range(start_epoch, max_epochs + 1)
        if early_stopping and stale >= patience:
            epoch_range = ()
        for epoch in epoch_range:
            model.train()
            if private:
                with BatchMemoryManager(data_loader=loader,
                        max_physical_batch_size=max_physical_batch_size, optimizer=optimizer) as physical_loader:
                    for indices, labels in physical_loader:
                        optimizer.zero_grad(set_to_none=True)
                        batch = _dense_batch(X_train, indices, device)
                        labels = labels.to(device)
                        loss = _training_loss(model(batch), labels)
                        loss.backward()
                        optimizer.step()
            else:
                for indices, labels in loader:
                    optimizer.zero_grad(set_to_none=True)
                    batch = _dense_batch(X_train, indices, device)
                    loss = _training_loss(model(batch), labels.to(device))
                    loss.backward()
                    optimizer.step()
            validation = evaluate_model(model, X_public_validation, y_public_validation, device=device)
            if not all(math.isfinite(value) for value in validation.values()):
                raise RuntimeError('Non-finite public-validation metrics; model selection aborted.')
            history.append(dict(epoch=epoch,
                **{f'validation_{key}': value for key, value in validation.items()}))
            improved = validation['log_loss'] < best_loss - min_delta
            if selection_metric == 'accuracy':
                improved = (validation['accuracy'] > best_accuracy + min_delta or
                    (validation['accuracy'] == best_accuracy and validation['log_loss'] < best_loss))
            if improved:
                best_accuracy = validation['accuracy']
                best_loss, best_epoch, stale = validation['log_loss'], epoch, 0
                if restore_best:
                    best_state = copy.deepcopy(base_model.state_dict())
            else:
                stale += 1
            if progress:
                name = 'Non-private' if not private else f'epsilon={epsilon:g}'
                progress(f'{name}: epoch {epoch}/{max_epochs}; public validation '
                    f'accuracy={validation["accuracy"]:.3%}, loss={validation["log_loss"]:.4f}; best epoch={best_epoch}')
            if checkpoint_path is not None:
                if private and (getattr(optimizer, '_is_last_step_skipped', False)
                                or getattr(optimizer, '_step_skip_queue', [])):
                    raise RuntimeError('Refusing to checkpoint inside an unfinished logical DP batch.')
                atomic_save(checkpoint_path, checkpoint_signature, dict(
                    epoch=epoch, model=base_model.state_dict(), optimizer=optimizer.state_dict(),
                    accountant=engine.accountant.state_dict() if private else None,
                    noise_multiplier=noise_multiplier, history=history, best_state=best_state,
                    best_loss=best_loss, best_epoch=best_epoch, best_accuracy=best_accuracy, stale=stale,
                    rng=capture_rng(sampling_generator, noise_generator),
                    training_seconds=elapsed_before + time.monotonic() - started))
            if early_stopping and stale >= patience:
                break
        spent_epsilon = float(engine.get_epsilon(delta=delta)) if private else None
        logical_steps = (sum(item[2] for item in engine.accountant.history) if private
                         else len(loader) * len(history))
    finally:
        if private:
            model = model.to_standard_module()
    if restore_best:
        if best_state is None:
            raise RuntimeError('No finite validation-selected weights were produced.')
        model.load_state_dict(best_state)
    model.eval()
    metadata = dict(private=private, epsilon_requested=epsilon,
        epsilon_fixed_schedule_bound=planned_epsilon,
        epsilon_at_stop_diagnostic=spent_epsilon, delta=delta if private else None,
        accountant='rdp' if private else None,
        epsilon_reporting='Fixed maximum-schedule bound; stopped-time epsilon is diagnostic only',
        noise_multiplier=noise_multiplier, sample_rate=sample_rate,
        sampler_sample_rate=sample_rate, accountant_sample_rate=accountant_sample_rate,
        bound_sample_rate=bound_sample_rate, strict_accounting=strict_accounting,
        max_grad_norm=max_grad_norm if private else None,
        planned_optimizer_steps=planned_steps, actual_optimizer_steps=logical_steps,
        max_epochs=max_epochs, epochs_run=len(history), best_epoch=best_epoch,
        returned_epoch=best_epoch if restore_best else len(history), patience=patience, min_delta=min_delta,
        selection_metric=selection_metric,
        selection_rule=('Final epoch; validation does not select weights' if not restore_best else
            ('Highest public-validation accuracy; lower log loss breaks exact ties'
             if selection_metric == 'accuracy' else 'Lowest public-validation log loss')),
        input_dim=X_train.shape[1], hidden_dims=list(hidden_dims), dropout=dropout,
        batch_size=batch_size, max_physical_batch_size=max_physical_batch_size,
        learning_rate=learning_rate, weight_decay=weight_decay, optimizer='Adam',
        seed=seed, device=str(device), training_seconds=elapsed_before + time.monotonic() - started,
        secure_mode=False if private else None,
        private_rng='Independent nondeterministic sampling/noise generators; seeds not saved' if private else None,
        early_stopping=early_stopping, restore_best=restore_best,
        validation_scope='Fixed public held-out records outside protected training set')
    if checkpoint_path is not None:
        metadata.update(training_checkpoint_enabled=True, resumed_after_epoch=start_epoch - 1)
        if private:
            metadata['private_rng'] = 'Independent nondeterministic generators; state saved in internal research checkpoint only'
    return model, history, metadata


def categorical_budget_rows(results, epsilon_budgets):
    """Select requested DP rows in predeclared order; never sort by accuracy."""
    epsilons = list(epsilon_budgets)
    if not epsilons or len(set(epsilons)) != len(epsilons):
        raise ValueError('Budget categories must be nonempty and unique.')
    private = results.loc[results['private']].copy()
    if private['epsilon_requested'].duplicated().any() or set(private['epsilon_requested']) != set(epsilons):
        raise ValueError('Exactly one completed result is required per requested epsilon.')
    ordered = private.set_index('epsilon_requested').loc[epsilons].reset_index()
    for column in ['target_test_accuracy', 'mia_accuracy']:
        values = ordered[column].to_numpy(dtype=float)
        if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
            raise ValueError(f'{column} must contain finite probabilities.')
    return ordered, [f'ε = {epsilon:g}' for epsilon in epsilons]


def plot_budget_accuracy(results, epsilon_budgets, *, metric, output_dir, delta):
    """Scientific PNG/SVG figure with equally spaced, categorical epsilon labels.

    Non-private control is a horizontal reference, never a numeric infinity.
    Values are requested maximum-budget categories, not realised stopping epsilons.
    """
    from pathlib import Path
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    choices = {'target_test_accuracy': ('Target income-model test accuracy', 'target_accuracy_vs_epsilon'),
               'mia_accuracy': ('Membership inference accuracy', 'mia_accuracy_vs_epsilon')}
    if metric not in choices:
        raise ValueError(f'metric must be one of {list(choices)}')
    ordered, labels = categorical_budget_rows(results, epsilon_budgets)
    control = results.loc[~results['private']]
    if len(control) != 1 or not math.isfinite(float(control.iloc[0][metric])):
        raise ValueError('Exactly one finite non-private control is required.')
    if ordered['delta'].nunique() != 1 or not np.allclose(ordered['delta'], delta, rtol=0, atol=0):
        raise ValueError('All plotted DP models must use the declared constant delta.')
    title, stem = choices[metric]
    fig, ax = plt.subplots(figsize=(8, 4.8), constrained_layout=True)
    # Strings force category spacing; there is no continuous numeric epsilon axis.
    ax.plot(labels, ordered[metric].to_numpy(), marker='o', linewidth=2, label='DP target')
    ax.axhline(float(control.iloc[0][metric]), color='tab:orange', linestyle='--', label='Non-private control')
    if metric == 'mia_accuracy':
        ax.axhline(0.5, color='0.45', linestyle=':', label='Chance (balanced MIA records)')
    ax.set(title=f'{title} vs privacy budget',
        xlabel='Requested privacy budget ε (equally spaced categories)', ylabel=title, ylim=(0, 1))
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1))
    ax.grid(axis='y', alpha=0.25)
    ax.legend(loc='best')
    ax.text(0.01, 0.02, f'Constant δ = {delta:g}; one training run per condition',
        transform=ax.transAxes, fontsize=9, bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    for suffix in ['png', 'svg']:
        fig.savefig(destination / f'{stem}.{suffix}', dpi=200)
    return fig, ax
