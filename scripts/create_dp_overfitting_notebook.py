"""Prepare the fixed 100-epoch DP study without executing notebook cells."""
import ast
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DESTINATION = ROOT / 'exp_4_differential_privacy_overfitting.ipynb'


def replace_once(source, old, new):
    if source.count(old) != 1:
        raise ValueError(f'Expected one template occurrence: {old!r}')
    return source.replace(old, new)


def build_notebook():
    # Extract literals only: importing the original generator would overwrite
    # the original notebook and its saved outputs.
    tree = ast.parse((ROOT / 'scripts/create_dp_notebook.py').read_text(encoding='utf-8'))
    templates = [tuple(ast.literal_eval(arg) for arg in node.value.args)
        for node in tree.body if isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name)
        and node.value.func.id == 'add']
    if len(templates) != 20:
        raise ValueError('Review changes to the original DP notebook template.')
    cells = []
    for index, (kind, source) in enumerate(templates):
        if index == 0:
            source = '''# Experiment 4: differential privacy under a fixed 100-epoch overfitting configuration
Compare a freshly trained non-private ACS 2023 MLP with seven DP MLPs at
**epsilon = 0.5, 1, 2, 5, 10, 25, 50**, holding **delta = 1e-6** constant.
Use 100,000 target training records, 48 complete inputs and nominal PINCP >$50,000
labels. ACS 2014 supplies public preprocessing data and 16 tree shadow models.
There is no missing-feature training, masking or reconstruction in this study.

All eight targets use hidden layers **512/256/128**, Adam learning rate **0.001**,
batch size **256**, **no dropout**, **no weight decay**, and **exactly 100 epochs**.
Disable early stopping and best-epoch restoration. Return **epoch 100** regardless
of validation performance; validation histories are monitoring only. The same
predeclared settings apply to every privacy budget. DP may prevent overfitting;
the configuration does not guarantee that every resulting model overfits.

The historical Exp 3 configuration recorded **62.625% target MIA accuracy**,
99.849% target training accuracy and 79.52% target test accuracy. These are
historical context, not new observations or promised results. This notebook
retains the DP experiment's public-only preprocessing and consistent blind-walk
perturbations across shadows and targets. The historical run fitted target
preprocessing on target training rows and used different commuting-field walk
rules for shadows and targets, so this is **not an exact reproduction** of that
62.625% run. No old model or score replaces the fresh non-private control.

Use the same frozen four-feature RF membership attack for every target. The
graphs show equally spaced epsilon categories with a non-private horizontal
reference. Record target utility, generalization gap, MIA accuracy, ROC-AUC and
TPR@1%FPR without selecting a budget using test or membership results.

Noise is calibrated for all 100 epochs using the exact logical-step count and
the conservative Opacus accountant rate, 1 / len(private_loader). The actual
Poisson sampler rate is recorded separately. The full-schedule bound must be
within the requested epsilon before training begins. No privacy budget is reset
between epochs. Compared with 30 epochs, this schedule requires different noise
at the same epsilon, so cross-notebook differences are not an epoch-only effect.

Run experiments only after explicit user approval. No cells have been executed
by creating this notebook. No checkpoints, model files, intermediate predictions
or logs are saved. Final metrics, histories, protocol and PNG/SVG graphs are
exported to a separate directory. The known sklearn parallel warning is hidden;
the three known DP research warnings each appear once. Other warnings and errors
remain visible. Eight 100-epoch runs can take substantially longer than the
early-stopped study.
'''
        elif index == 2:
            source = replace_once(source, 'dropout=0.1,', 'dropout=0.0,')
            source = replace_once(source, 'max_epochs=30,', 'max_epochs=100,')
            source = replace_once(source, 'weight_decay=1e-4,', 'weight_decay=0.0,')
            source = replace_once(source, "max_grad_norm=1.0, seed=SEED, device='cpu')",
                "max_grad_norm=1.0, seed=SEED, device='cpu',\n"
                "    selection_metric='log_loss', strict_accounting=True,\n"
                "    early_stopping=False, restore_best=False)")
            source = replace_once(source, 'EPSILON_BUDGETS = (1, 5, 10, 25, 50)',
                'EPSILON_BUDGETS = (0.5, 1, 2, 5, 10, 25, 50)')
            source = replace_once(source, 'exp_4_2023_2014_dp_e1_5_10_25_50_d1e6',
                'exp_4_2023_2014_dp_overfitting100_e0p5_1_2_5_10_25_50_d1e6')
            source = replace_once(source,
                "print('Epoch cap:', TRAIN_CONFIG['max_epochs'], '| Patience:', TRAIN_CONFIG['patience'])",
                "print('Fixed epochs:', TRAIN_CONFIG['max_epochs'], '| Early stopping: disabled; return final epoch')")
        elif index == 3:
            source = replace_once(source,
                'outside the protected target training set; it selects\nepochs only.',
                'outside the protected target training set; it provides\nmonitoring only and never selects an epoch.')
        elif index == 7:
            source = '''## Train and evaluate each target at epoch 100
The following helper defines the work; the run cell performs it. Every model
starts from scratch with the same architecture, initialization seed and
unregularized Adam settings. DP adds per-example clipping and Gaussian noise
with Poisson sampling. Opacus sampling/noise generators are independently,
nondeterministically seeded and their seeds are not saved.

Run all 100 epochs and evaluate the final weights. Validation cannot stop a run
or select an earlier epoch. `best_epoch` is a monitoring diagnostic only;
`returned_epoch` must equal 100. Private train metrics and test/MIA measurements
are computed only after training, with no effect on training or thresholds.

Before the first update, calibrate noise using the full private-loader logical
step count and the same conservative rate used by Opacus's accountant. Record
both sampler and accountant rates. After training, assert that all planned
logical steps were executed and the realised accountant epsilon agrees with
the full-schedule bound. Keep delta at 1e-6.

Query the same balanced MIA records with blind walks and the frozen auxiliary
RF attack for every target. This is a transferred attack from non-private tree
shadows, not an attack matched to each DP budget. Final train/test accuracy and
their gap establish whether the intended overfitting actually occurred.
'''
        elif index == 8:
            source = replace_once(source,
                '    target_progress.update(f\'{condition}: evaluating selected epoch {metadata["best_epoch"]}\')',
                '''    assert metadata['epochs_run'] == TRAIN_CONFIG['max_epochs']
    assert metadata['returned_epoch'] == TRAIN_CONFIG['max_epochs']
    assert metadata['actual_optimizer_steps'] == metadata['planned_optimizer_steps']
    if epsilon is not None:
        assert metadata['epsilon_fixed_schedule_bound'] <= epsilon + 1e-6
        assert math.isclose(metadata['epsilon_at_stop_diagnostic'],
            metadata['epsilon_fixed_schedule_bound'], rel_tol=1e-9, abs_tol=1e-9)
    target_progress.update(f'{condition}: evaluating final epoch {metadata["returned_epoch"]}')''')
            source = replace_once(source, "best_epoch=metadata['best_epoch'], training_seconds=metadata['training_seconds'],",
                "best_epoch=metadata['best_epoch'], returned_epoch=metadata['returned_epoch'],\n"
                "        planned_optimizer_steps=metadata['planned_optimizer_steps'],\n"
                "        actual_optimizer_steps=metadata['actual_optimizer_steps'],\n"
                "        sampler_sample_rate=metadata['sampler_sample_rate'],\n"
                "        accountant_sample_rate=metadata['accountant_sample_rate'],\n"
                "        bound_sample_rate=metadata['bound_sample_rate'],\n"
                "        training_seconds=metadata['training_seconds'],")
        elif index == 9:
            source = '''### Run the eight fixed 100-epoch conditions
This cell trains a fresh non-private control and seven private targets, each
for all 100 epochs. It requires explicit approval to execute. Existing Exp 3
models, predictions and the historical 62.625% score are never substituted.
Report every predeclared budget. Known DP warnings are displayed once per run.
'''
        elif index == 12:
            source = replace_once(source, "'delta', 'epochs_run', 'best_epoch',",
                "'delta', 'epochs_run', 'returned_epoch',")
            source += '''
display(study_results.set_index('condition')[[
    'sampler_sample_rate', 'accountant_sample_rate', 'bound_sample_rate',
    'noise_multiplier', 'planned_optimizer_steps', 'actual_optimizer_steps'
]])
'''
        elif index == 17:
            source = '''## Save results and interpret the comparison
Save final CSV metrics, all 100 monitoring epochs, protocol and both graphs to
the separate overfitting-study directory. No models, row-level predictions,
checkpoints or logs are saved. Private evaluation statistics in these research
outputs are not DP-sanitized releases.

The epsilon/delta claim concerns each individual target model with its fixed
public preprocessor, conditional on the fixed research split. Releasing several
models trained on the same records requires composition; the non-private
control has no DP guarantee. secure_mode=False is a research setting.

This experiment evaluates the final epoch after a fixed schedule. There is no
adaptive stopping or best-epoch selection. The full-schedule bound and final
accountant diagnostic use the same conservative rate and should agree.

The training settings are intended to expose a larger non-private train/test
gap, but observed gaps and leakage must be measured. The historical 62.625%
MIA score came from a different preprocessing and shadow-walk protocol and is
not a guaranteed baseline here. A near-chance tree-shadow transfer attack does
not establish robustness to stronger attacks. There is one run per condition;
the early-stopped study and this study also differ in regularization, learning
rate and DP noise calibration, so comparing them is not an epoch-only ablation.
'''
        elif index == 18:
            source = replace_once(source, "protocol = dict(execution_status='complete',",
                "protocol = dict(study_variant='Fixed 100-epoch unregularized DP study',\n"
                "    historical_reference=dict(target_mia_accuracy=0.62625,\n"
                "        source='artifacts/exp_3_2023_2014_100000_nominal_paper_algorithms/summary.json',\n"
                "        role='Historical context only; not a new result or exact reproduction'),\n"
                "    accounting_calibration='Exact private-loader steps; conservative Opacus accountant rate',\n"
                "    execution_status='complete',")
            source = replace_once(source,
                "privacy_bound='Fixed maximum-schedule epsilon; epsilon at adaptive stop is diagnostic only',",
                "privacy_bound='Fixed full 100-epoch schedule; final accountant epsilon agrees with bound',")
        for old, new in [
            ('all six targets', 'all eight targets'),
            ('All six conditions completed', 'All eight conditions completed'),
            ('training histories, selected', 'training histories, monitored'),
            ('selected training histories', 'full training histories'),
            ('actual accountant values and stopping epochs', 'actual accountant values and final epochs'),
        ]:
            source = source.replace(old, new)
        if kind == 'code':
            ast.parse(source)
        cell = dict(cell_type=kind, id=f'exp4-dp-overfit-{index}', metadata={},
            source=source.splitlines(keepends=True))
        if kind == 'code':
            cell.update(execution_count=None, outputs=[])
        cells.append(cell)
    return dict(nbformat=4, nbformat_minor=5, cells=cells, metadata=dict(
        kernelspec=dict(display_name='Python 3 (ipykernel)', language='python', name='python3'),
        language_info=dict(name='python', version='3.12')))


if __name__ == '__main__':
    notebook = build_notebook()
    DESTINATION.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + '\n', encoding='utf-8')
    print(f'Prepared {DESTINATION.name}: {len(notebook["cells"])} cells; none executed.')
