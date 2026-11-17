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

Noise calibration, the full-schedule bound and the realised-stop diagnostic all
use the conservative Opacus accountant rate, **1 / len(private_loader)**. The
actual Poisson sampler rate is recorded separately. This avoids a small rounding
discrepancy between those two rates in Opacus 1.6. The full 30-epoch bound must
remain within each requested epsilon before training can start.
'''
        elif index == 2:
            source = replace_once(source, 'learning_rate=1e-3', 'learning_rate=3e-4')
            source = replace_once(source, "max_grad_norm=1.0, seed=SEED, device='cpu')",
                "max_grad_norm=1.0, seed=SEED, device='cpu',\n"
                "    selection_metric='accuracy', strict_accounting=True)")
            source = replace_once(source, 'exp_4_2023_2014_dp_e1_5_10_25_50_d1e6',
                'exp_4_2023_2014_dp_accuracy_e0p5_1_2_5_10_25_50_d1e6')
            source = replace_once(source, 'EPSILON_BUDGETS = (1, 5, 10, 25, 50)',
                'EPSILON_BUDGETS = (0.5, 1, 2, 5, 10, 25, 50)')
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
accounting calibration and the expanded budget grid differ in this follow-up.
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
        elif index == 18:
            source = replace_once(source, "protocol = dict(execution_status='complete',",
                "protocol = dict(study_variant='accuracy-focused exploratory follow-up',\n"
                "    accounting_calibration='Exact private-loader steps; conservative Opacus accountant rate',\n"
                "    execution_status='complete',")
        # Keep narrative counts and run-status messages aligned with seven DP
        # budgets plus the single non-private control.
        for old, new in [
            ('epsilon = 1, 5, 10, 25, 50', 'epsilon = 0.5, 1, 2, 5, 10, 25, 50'),
            ('six matched conditions', 'eight matched conditions'),
            ('five private targets', 'seven private targets'),
            ('All five', 'All seven'), ('all five', 'all seven'),
            ('All six', 'All eight'), ('all six', 'all eight'),
            ('the six\nconditions', 'the eight\nconditions'),
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
