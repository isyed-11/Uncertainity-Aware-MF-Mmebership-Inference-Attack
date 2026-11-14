"""Prepare the ACS-only PFV notebook without executing notebook cells."""
import ast
import copy
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
source = json.loads((ROOT / 'exp_3_expanded dataset.ipynb').read_text(encoding='utf-8'))
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
    'Masks cover all 15 nonempty combinations of MAR, WKHP, COW and SEX; related predictors remain visible.\n',
    'The complete-feature attack classifier and auxiliary-selected thresholds remain frozen.\n',
    'Separate cells add conditional single/average/mean-only/uncertainty-aware comparisons.\n',
    'Run only after explicit user approval. Existing training caches may be reused by the original pipeline.\n']


def add(kind, text):
    cell = dict(cell_type=kind, metadata={}, id=f'missing-pfv-{len(notebook["cells"])}',
                source=text.splitlines(keepends=True))
    if kind == 'code':
        cell.update(execution_count=None, outputs=[])
    notebook['cells'].append(cell)


add('markdown', '''## Missing-feature PFV baseline
Mask all combinations of one, two, three and four selected original fields on the
same target member/nonmember records (4 + 6 + 4 + 1 = 15 scenarios).
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
MISSING_MASKS = {'+'.join(fields): list(fields)
    for count in range(1, len(MASK_FIELDS) + 1)
    for fields in combinations(MASK_FIELDS, count)}
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

for cell in notebook['cells']:
    if cell['cell_type'] == 'code':
        ast.parse(''.join(cell['source']))
destination = ROOT / 'exp_3_expanded_missing_feature_baseline.ipynb'
destination.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + '\n', encoding='utf-8')
print(f'Prepared {destination.name}: {len(notebook["cells"])} cells; no cells executed.')
