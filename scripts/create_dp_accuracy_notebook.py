"""Prepare an accuracy-focused DP notebook; never execute cells or the old generator."""
import ast
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DESTINATION = ROOT / 'exp_4_differential_privacy_accuracy.ipynb'


def replace_once(source, old, new):
    if source.count(old) != 1:
        raise ValueError(f'Expected one template occurrence: {old!r}')
    return source.replace(old, new)


def build_notebook():
    # Read literal templates without importing/executing the old generator,
    # which would overwrite the original notebook and its saved outputs.
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
            source = replace_once(source, '# Experiment 4: differential privacy and complete-feature MIA',
                '# Experiment 4: accuracy-focused differential privacy comparison')
            source = replace_once(source,
                'Adam learning rate=1e-3, and a **30-epoch maximum**. Stop after five epochs without\n'
                'public-validation-loss improvement and restore the best validation epoch.',
                'Adam learning rate=3e-4, and a **30-epoch maximum**. Select the epoch with\n'
                'highest public-validation accuracy, breaking exact ties by lower log loss.\n'
                'Stop after five epochs without improvement and restore the selected epoch.')
            source += '''
This separate, exploratory follow-up was prepared after inspecting the original
DP validation histories. Those histories sometimes showed rising accuracy after
the minimum-loss epoch. A lower learning rate and accuracy-based selection may
improve utility; improvement and a small generalization gap are not guaranteed.
The test set and target MIA results never select epochs or settings in this run.
The original notebook and its saved results remain available for comparison.
This version uses 16 mixed shadows: four RF, four gradient boosting, four
XGBoost and four generic MLPs. MLP shadows have hidden layers 128/64, no dropout,
Adam learning rate 0.001 and weight decay 0.0001, with validation-loss early
stopping (patience five, maximum 50 epochs, best weights restored). Their
architecture is predeclared independently of the target. All shadows are
non-private; this remains a frozen transfer attack across DP budgets.

Noise calibration, the full-schedule bound and the realised-stop diagnostic all
use the conservative Opacus accountant rate, **1 / len(private_loader)**. The
actual Poisson sampler rate is recorded separately. This avoids a small rounding
discrepancy between those two rates in Opacus 1.6. The full 30-epoch bound must
remain within each requested epsilon before training can start.
'''
        elif index == 2:
            source = replace_once(source,
                'from src.expanded_data import EXPANDED_FEATURES, load_expanded_year, reference_split_config',
                'import src.expanded_data as expanded_data_module\n'
                'importlib.reload(expanded_data_module)\n'
                'from src.expanded_data import EXPANDED_FEATURES, load_expanded_year, reference_split_config')
            source = replace_once(source,
                'from src.shadow_models import train_shadow_model, summarize_shadow_models, PAPER_SHADOW_ALGORITHMS',
                'import src.shadow_models as shadow_models_module\n'
                'importlib.reload(shadow_models_module)\n'
                'from src.shadow_models import (train_shadow_model, summarize_shadow_models,\n'
                '    PAPER_SHADOW_ALGORITHMS, MLP_SHADOW_CONFIG)')
            source = replace_once(source, 'SHADOW_ALGORITHMS = PAPER_SHADOW_ALGORITHMS',
                "SHADOW_ALGORITHMS = PAPER_SHADOW_ALGORITHMS + ('mlp',)\n"
                'SHADOW_MLP_CONFIG = dict(MLP_SHADOW_CONFIG)\n'
                'assert N_SHADOW_MODELS == 4 * len(SHADOW_ALGORITHMS)')
            source = replace_once(source, 'learning_rate=1e-3', 'learning_rate=3e-4')
            source = replace_once(source, "max_grad_norm=1.0, seed=SEED, device='cpu')",
                "max_grad_norm=1.0, seed=SEED, device='cpu',\n"
                "    selection_metric='accuracy', strict_accounting=True)")
            source = replace_once(source, 'exp_4_2023_2014_dp_e1_5_10_25_50_d1e6',
                'exp_4_2023_2014_dp_accuracy_e0p5_1_2_5_10_25_50_d1e6_mixed_mlp')
            source = replace_once(source, 'EPSILON_BUDGETS = (1, 5, 10, 25, 50)',
                'EPSILON_BUDGETS = (0.5, 1, 2, 5, 10, 25, 50)')
        elif index == 5:
            source = replace_once(source,
                'Train 16 public 2014 shadows (6 RF, 5 gradient boosting, 5 XGBoost), keeping their\nexisting estimator settings and 40/20/40 splits.',
                'Train 16 public 2014 shadows: four each of RF, gradient boosting, XGBoost\n'
                'and generic MLP. Existing tree settings and 40/20/40 splits are retained.\n'
                'MLP configuration is recorded in SHADOW_MLP_CONFIG; only its shadow\n'
                'validation partition selects epochs, never nonmembers or target records.')
        elif index == 6:
            source = replace_once(source,
                "result = train_shadow_model(auxiliary, algorithm, seed=seed, schema='expanded_acs')",
                "result = train_shadow_model(auxiliary, algorithm, seed=seed, schema='expanded_acs',\n"
                "            parameters=SHADOW_MLP_CONFIG if algorithm == 'mlp' else None)")
        elif index == 7:
            source = replace_once(source,
                'and noise generators; their seeds are not saved. Public-validation loss alone\nselects the returned epoch.',
                'and noise generators; their seeds are not saved. Highest public-validation\n'
                'accuracy selects the returned epoch; lower log loss breaks exact ties.')
            source += '''
Before the first update, recalibrate noise using the private loader's exact
logical step count and the same conservative rate used by the Opacus accountant.
Record both the actual sampler rate and the accountant/bound rate. Public
preprocessing, splits, architecture, regularization, auxiliary attack and blind
walks follow the original DP experiment. Learning rate, epoch selection,
accounting calibration, expanded budget grid and mixed shadow composition differ
in this follow-up. Target training settings are unaffected by adding MLP shadows.
'''
        elif index == 8:
            source = replace_once(source,
                "noise_multiplier=metadata['noise_multiplier'], epochs_run=metadata['epochs_run'],",
                "noise_multiplier=metadata['noise_multiplier'],\n"
                "        sampler_sample_rate=metadata['sampler_sample_rate'],\n"
                "        accountant_sample_rate=metadata['accountant_sample_rate'],\n"
                "        bound_sample_rate=metadata['bound_sample_rate'],\n"
                "        selection_metric=metadata['selection_metric'], epochs_run=metadata['epochs_run'],")
        elif index == 12:
            source += '''
display(study_results.loc[study_results['private'], [
    'condition', 'sampler_sample_rate', 'accountant_sample_rate', 'bound_sample_rate',
    'noise_multiplier', 'epsilon_fixed_schedule_bound'
]].set_index('condition'))
'''
        elif index == 14:
            source = replace_once(source, 'display(figure)',
                'axes.set_ylim(0.70, 0.90)\n'
                "for suffix in ('png', 'svg'):\n"
                "    figure.savefig(RUN_DIR / f'target_accuracy_vs_epsilon.{suffix}', dpi=200)\n"
                'display(figure)')
        elif index == 16:
            source = replace_once(source, 'display(figure)',
                'axes.set_ylim(0.40, 0.60)\n'
                "for suffix in ('png', 'svg'):\n"
                "    figure.savefig(RUN_DIR / f'mia_accuracy_vs_epsilon.{suffix}', dpi=200)\n"
                'display(figure)')
        elif index == 18:
            source = replace_once(source, "protocol = dict(execution_status='complete',",
                "protocol = dict(study_variant='accuracy-focused exploratory follow-up',\n"
                "    accounting_calibration='Exact private-loader steps; conservative Opacus accountant rate',\n"
                "    execution_status='complete',")
            source = replace_once(source,
                'n_shadow_models=N_SHADOW_MODELS, shadow_algorithms=SHADOW_ALGORITHMS, shadow_seeds=shadow_seeds,',
                'n_shadow_models=N_SHADOW_MODELS, shadow_algorithms=SHADOW_ALGORITHMS, shadow_seeds=shadow_seeds,\n'
                '    shadow_models_per_algorithm=4, shadow_mlp_config=SHADOW_MLP_CONFIG,')
            source = replace_once(source,
                'Frozen complete-feature RF transfer attack; non-DP tree shadows',
                'Frozen complete-feature RF transfer attack; non-DP tree and generic MLP shadows')
        # Keep narrative counts and run-status messages aligned with seven DP
        # budgets plus the single non-private control.
        for old, new in [
            ('epsilon = 1, 5, 10, 25, 50', 'epsilon = 0.5, 1, 2, 5, 10, 25, 50'),
            ('six matched conditions', 'eight matched conditions'),
            ('five private targets', 'seven private targets'),
            ('All five', 'All seven'), ('all five', 'all seven'),
            ('All six', 'All eight'), ('all six', 'all eight'),
            ('the six\nconditions', 'the eight\nconditions'),
            ('transferred tree-shadow attacks', 'transferred mixed tree/MLP-shadow attacks'),
        ]:
            source = source.replace(old, new)
        if kind == 'code':
            ast.parse(source)
        cell = dict(cell_type=kind, id=f'exp4-dp-accuracy-{index}', metadata={},
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
