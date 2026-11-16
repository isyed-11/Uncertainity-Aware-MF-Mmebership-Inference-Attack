"""Prepare the ACS-only PFV notebook without executing notebook cells."""
import ast
import copy
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
source = json.loads((ROOT / 'exp_3_expanded_missing_feature_baseline.ipynb').read_text(encoding='utf-8'))
notebook = copy.deepcopy(source)
# Preserve the existing ACS-only pipeline, stopping before Adult comparisons.
notebook['cells'] = notebook['cells'][:18]
for cell in notebook['cells']:
    if cell['cell_type'] == 'code':
        cell['execution_count'] = None
        cell['outputs'] = []
        text = ''.join(cell['source'])
        text = text.replace("ADULT_RUN_DIR = RUN_DIR / 'adult_auxiliary'\n", '')
        text = text.replace("execution_status='all_cells_completed'", "execution_status='acs_baseline_completed'")
        cell['source'] = text.splitlines(keepends=True)
notebook['cells'][0]['source'] = ['# Expanded ACS missing-feature baseline: ACS 2014 auxiliary only\n',
    '\nComplete-feature ACS baseline, PFV reconstruction and uncertainty-aware MF-MIA.\n',
    'Target: ACS 2023; 16 ACS 2014 shadows. No Adult data are loaded.\n',
    'Existing target, splits, training, attack settings and asymmetric blind-walk protocol are retained.\n',
    'PFV samples one nonmissing value from the empirical 2014 marginal distribution.\n',
    'Seven preselected masks cover one to four missing fields; related predictors remain visible.\n',
    'The complete-feature attack classifier and auxiliary-selected thresholds remain frozen.\n',
    'Separate cells add conditional single/average/mean-only/uncertainty-aware comparisons.\n',
    'Run only after explicit user approval. No model checkpoints or logs are loaded or saved. The historical baseline configuration recorded 62.625% accuracy; a rerun is not guaranteed to reproduce it.\n']


def add(kind, text):
    cell = dict(cell_type=kind, metadata={}, id=f'missing-pfv-{len(notebook["cells"])}',
                source=text.splitlines(keepends=True))
    if kind == 'code':
        cell.update(execution_count=None, outputs=[])
    notebook['cells'].append(cell)


add('markdown', '''## Missing-feature PFV baseline
Use seven preselected masks: two singles, two pairs, two triples and one quadruple.
These retain the reduced K=6 study; the complete-feature baseline uses the historical settings.
All 48 target inputs remain required; the target is not retrained after masking.
Each hidden field is replaced before querying, including before blind walks.
The reconstruction pool contains ACS 2014 predictors only, with no income-label
conditioning. Naturally missing auxiliary values are excluded from the sampling pool.
Use stable-code fields only; do not extend this to occupation/industry without
an explicit cross-year category mapping. Related marriage and employment fields
remain observed. Income labels remain known for the original attack's loss features.

The same complete-feature shadow attack is used in every condition, isolating the
effect of target completion. There is no mask-specific attack retraining. Auxiliary
attack train/validation/test grouping and thresholds therefore remain unchanged.
A single seed is a baseline realization, not a measure of reconstruction variability.
''')
add('code', '''from itertools import combinations
from src.missing_feature_pfv import complete_pfv

MASK_FIELDS = ['MAR', 'WKHP', 'COW', 'SEX']
MISSING_MASKS = {
    'MAR': ['MAR'], 'WKHP': ['WKHP'],
    'MAR+COW': ['MAR', 'COW'], 'COW+SEX': ['COW', 'SEX'],
    'MAR+WKHP+SEX': ['MAR', 'WKHP', 'SEX'],
    'WKHP+COW+SEX': ['WKHP', 'COW', 'SEX'],
    'MAR+WKHP+COW+SEX': ['MAR', 'WKHP', 'COW', 'SEX'],
}
MISSING_DIR = RUN_DIR / 'missing_feature_pfv'
MISSING_DIR.mkdir(parents=True, exist_ok=True)
missing_rows = [{'condition': 'complete', 'n_missing': 0, **target_mia_metrics}]
missing_audits = {}
for mask_name, hidden in MISSING_MASKS.items():
    completed_partitions = {}
    # One shared completion pass: membership never selects a seed or distribution.
    candidates = pd.concat([expanded_partitions[name] for name in
        ['mia_members', 'mia_nonmembers']]).sort_index()
    completed_all = complete_pfv(candidates[EXPANDED_FEATURES],
        acs_auxiliary[EXPANDED_FEATURES], hidden, seed=SEED)
    completed_all['income_gt_50k'] = candidates['income_gt_50k']
    for name in ['mia_members', 'mia_nonmembers']:
        original = expanded_partitions[name]
        completed_partitions[name] = completed_all.loc[original.index].copy()
    features, audit = collect_target_paper_features(
        expanded_model, expanded_preprocessor, completed_partitions,
        config=WALK_CONFIG, seed=SEED, numeric_protocol=EXPANDED_NUMERIC_WALK)
    metrics, scores = evaluate_attack(attack_result['classifier'], features,
        threshold=attack_result['accuracy_threshold'],
        low_fpr_threshold=attack_result['low_fpr_threshold'])
    metrics['tpr_at_accuracy_threshold'] = metrics['tp'] / metrics['members']
    metrics['fpr_at_accuracy_threshold'] = metrics['fp'] / (metrics['records'] - metrics['members'])
    missing_rows.append({'condition': mask_name, 'n_missing': len(hidden), **metrics})
    missing_audits[mask_name] = audit
    features.assign(membership_score=scores).to_csv(MISSING_DIR / f'{mask_name}_scores.csv', index=False)
missing_comparison = pd.DataFrame(missing_rows).set_index('condition')
missing_comparison['auc_change_from_complete'] = missing_comparison['roc_auc'] - target_mia_metrics['roc_auc']
display(missing_comparison)
missing_comparison.to_csv(MISSING_DIR / 'comparison.csv')
(MISSING_DIR / 'protocol.json').write_text(json.dumps(dict(
    method='PFV_single_draw', auxiliary_year=2014, target_year=2023,
    masks=MISSING_MASKS, seed=SEED,
    naturally_missing_auxiliary_values='excluded',
    classifier='frozen complete-feature ACS 2014 attack',
    thresholds='frozen auxiliary validation', diagnostics=missing_audits,
    baseline_summary='../summary.json'), indent=2), encoding='utf-8')
''')
from uncertainty_notebook_cells import append_uncertainty_cells
append_uncertainty_cells(add)

def wrap_verbose_calls(text):
    lines = text.splitlines(keepends=True)
    names = {'train_target_model', 'train_shadow_model',
        'collect_shadow_paper_features', 'collect_target_paper_features'}
    tree = ast.parse(text)
    parents = {child: parent for parent in ast.walk(tree) for child in ast.iter_child_nodes(parent)}
    def already_quiet(node):
        while node in parents:
            node = parents[node]
            if isinstance(node, ast.With) and any(isinstance(item.context_expr, ast.Call)
                    and isinstance(item.context_expr.func, ast.Name)
                    and item.context_expr.func.id == 'quiet_output' for item in node.items):
                return True
        return False
    nodes = [node for node in ast.walk(tree)
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Name) and node.value.func.id in names
        and not already_quiet(node)]
    for node in sorted(nodes, key=lambda item: item.lineno, reverse=True):
        indent = ' ' * node.col_offset
        block = lines[node.lineno - 1:node.end_lineno]
        lines[node.lineno - 1:node.end_lineno] = [indent + 'with quiet_output():\n',
            *['    ' + line for line in block]]
    return ''.join(lines)


for cell in notebook['cells']:
    text = ''.join(cell['source'])
    if cell['cell_type'] == 'code':
        text = text.replace('from src.exp3_checkpoints import cached_target_train, cached_expanded_shadows\n', '')
        text = text.replace('load_adult_data, train_shadow_models,', 'load_adult_data, train_shadow_model, train_shadow_models,')
        text = text.replace("RUN_DIR = Path('artifacts/exp_3_2023_2014_100000_nominal_paper_algorithms')",
            "EXPERIMENT_DIR = Path('artifacts/exp_3_2023_2014_100000_nominal_paper_algorithms')\nRUN_DIR = EXPERIMENT_DIR / 'reduced_k6_s16_m7'")
        text = text.replace("expanded_model, expanded_history, expanded_metadata = cached_target_train(\n    expanded_arrays, TRAIN_CONFIG, RUN_DIR)",
            "expanded_model, expanded_history, expanded_metadata = train_target_model(\n    expanded_arrays['train']['X'], expanded_arrays['train']['y'],\n    expanded_arrays['validation']['X'], expanded_arrays['validation']['y'], **TRAIN_CONFIG)")
        text = text.replace("shadow_results = cached_expanded_shadows(acs_auxiliary, seed=SEED, n_models=N_SHADOW_MODELS,\n    run_dir=RUN_DIR, algorithms=SHADOW_ALGORITHMS)",
            "shadow_results = {}\nshadow_progress = CellProgress()\nfor index in range(N_SHADOW_MODELS):\n    algorithm = SHADOW_ALGORITHMS[index % len(SHADOW_ALGORITHMS)]\n    name = f'{algorithm}_{index + 1:02d}'\n    shadow_progress.update(f'Training shadow {index + 1}/{N_SHADOW_MODELS}: {name}')\n    result = train_shadow_model(acs_auxiliary, algorithm, seed=SEED + 10 * index, schema='expanded_acs')\n    result['attack_data']['shadow_model'] = name\n    shadow_results[name] = result\nshadow_progress.update(f'Completed {len(shadow_results)} shadows')")
        text = text.replace('compact_output', 'quiet_output')
        import re
        text = re.sub(r'with quiet_output\([^\n]+\):', 'with quiet_output():', text)
        text = '\n'.join(line for line in text.split('\n')
            if not line.startswith('UNCERTAINTY_LOG_DIR =')
            and not line.startswith("print('Detailed progress and warnings:'")
            and line != 'import joblib')
        text = wrap_verbose_calls(text)
    else:
        text = text.replace('Completed target and\nshadow checkpoints are saved during training to support interruption/restart.',
            'Models stay in memory; no training checkpoints or log files are written.')
        text = text.replace('Only final metrics, the protocol and compact progress logs are written to disk.',
            'Only final metrics, row assignments and the protocol are written to disk.')
    cell['source'] = text.splitlines(keepends=True)

config = ''.join(notebook['cells'][1]['source'])
config = config.split('\nimport importlib\nimport src.notebook_output as notebook_output_module')[0]
config += "\nimport importlib\nimport src.notebook_output as notebook_output_module\nimportlib.reload(notebook_output_module)\nnotebook_output_module.suppress_sklearn_parallel_warning()\nfrom src.notebook_output import CellProgress, quiet_output\nassert len(EXPANDED_FEATURES) == 48 and 'VPS' in EXPANDED_FEATURES\n"
notebook['cells'][1]['source'] = config.splitlines(keepends=True)
notebook['cells'][16]['source'] = [
    '## Small reproducibility artifacts\n',
    'Save summaries, the feature inventory and row assignments. Models remain in memory;\n',
    'no checkpoints or log files are loaded or saved.\n']
notebook.get('metadata', {}).pop('widgets', None)

for cell in notebook['cells']:
    if cell['cell_type'] == 'code':
        ast.parse(''.join(cell['source']))
destination = ROOT / 'exp_3_expanded_missing_feature_baseline.ipynb'
destination.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + '\n', encoding='utf-8')
print(f'Prepared {destination.name}: {len(notebook["cells"])} cells; no cells executed.')
