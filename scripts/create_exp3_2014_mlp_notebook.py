"""Build Exp 3 with the original 16 trees plus four target-recipe MLPs.

Notebook generation only; no cells or experiments are executed.
"""
import argparse
import ast
import copy
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'exp_3_expanded_missing_feature_baseline.ipynb'
DESTINATION = ROOT / 'exp_3_expanded_missing_feature_baseline_mlp.ipynb'


def create_notebook(*, overwrite=False):
    if DESTINATION.exists() and not overwrite:
        raise FileExistsError(f'Refusing to overwrite {DESTINATION.name}')
    original_bytes = SOURCE.read_bytes()
    original_hash = hashlib.sha256(original_bytes).hexdigest()
    notebook = copy.deepcopy(json.loads(original_bytes.decode('utf-8')))

    def read(index):
        return ''.join(notebook['cells'][index]['source'])

    def write(index, source):
        notebook['cells'][index]['source'] = source.splitlines(keepends=True)

    def replace(index, old, new):
        source = read(index)
        if source.count(old) != 1:
            raise ValueError(f'Expected one occurrence in cell {index}: {old!r}')
        write(index, source.replace(old, new))

    for index, cell in enumerate(notebook['cells']):
        cell['id'] = f'exp3-2014-tree16-mlp4-{index:02d}'
        for key in ('execution', 'ExecuteTime', 'collapsed', 'scrolled'):
            cell['metadata'].pop(key, None)
        if cell['cell_type'] == 'code':
            cell.update(execution_count=None, outputs=[])

    replace(0, '# Expanded ACS missing-feature baseline: ACS 2014 auxiliary only',
            '# Expanded ACS missing-feature baseline: 16 tree shadows plus four target-recipe MLPs')
    replace(0, 'Target: ACS 2023; 16 ACS 2014 shadows. No Adult data are loaded.',
            'Target: ACS 2023; 20 ACS 2014 shadows: six RF, five gradient boosting, five XGBoost and four MLPs.\n'
            'No Adult data are loaded. The original 16-tree notebook and saved observations remain separate.')
    replace(0, 'Existing target, splits, training, attack settings and asymmetric blind-walk protocol are retained.',
            'Target, data splits, target training, attack settings and asymmetric blind-walk protocol are retained.\n'
            'Retain all original 16 tree shadows and append four MLPs matching the target training recipe.\n'
            'This variant assumes knowledge of that recipe, while queries remain black box.\n'
            'Refit the base attack and recompute grouped\n'
            'auxiliary/meta/calibration roles using this new pool. Target evaluation records are unchanged.')
    replace(1, 'import time\n', '''import time
import importlib
import src.expanded_data as expanded_data_module
import src.shadow_models as shadow_models_module
importlib.reload(expanded_data_module)
importlib.reload(shadow_models_module)
''')
    replace(1, 'artifacts/exp_3_2023_2014_100000_nominal_paper_algorithms',
            'artifacts/exp_3_2023_2014_100000_nominal_tree16_mlp4_target_recipe')
    replace(1, 'reduced_k6_s16_m7', 'reduced_k6_s20_m7')
    replace(1, 'N_SHADOW_MODELS = 16\n', 'N_SHADOW_MODELS = 20\n')
    replace(1, 'SHADOW_ALGORITHMS = PAPER_SHADOW_ALGORITHMS\n',
            "SHADOW_ALGORITHMS = PAPER_SHADOW_ALGORITHMS + ('mlp',)\n"
            '# Copy the target recipe, but let each shadow use its own seed.\n'
            "SHADOW_MLP_CONFIG = {key: value for key, value in TRAIN_CONFIG.items() if key != 'seed'}\n"
            'N_TREE_SHADOWS = 16\n'
            'SHADOW_PLAN = tuple(PAPER_SHADOW_ALGORITHMS[index % len(PAPER_SHADOW_ALGORITHMS)]\n'
            "    for index in range(N_TREE_SHADOWS)) + ('mlp',) * 4\n"
            'assert N_SHADOW_MODELS == len(SHADOW_PLAN) == 20\n'
            "assert {algorithm: SHADOW_PLAN.count(algorithm) for algorithm in SHADOW_ALGORITHMS} == {\n"
            "    'random_forest': 6, 'gradient_boosting': 5, 'xgboost': 5, 'mlp': 4}\n")

    write(10, '''## 20 ACS 2014 shadows: original 16 trees plus four target-recipe MLPs
Keep the original six RF, five gradient boosting and five XGBoost shadows in slots
1-16 with the same algorithms, names, settings and seeds. Append MLPs in slots
17-20. Every model keeps the original 40/20/40 train/validation/OUT split
procedure with seed = 42 + 10 * index. Each preprocessor fits its own train split.

MLPs copy TRAIN_CONFIG except for their distinct seeds: hidden layers (512, 256,
128), ReLU, dropout 0, Adam learning rate 0.001, weight decay 0, batch size 256,
CPU and exactly 100 epochs. Early stopping and best-weight restoration are off;
return the final epoch, as for the target. Validation is monitored but never
selects weights. Shadow OUT records are used only for evaluation/attack examples.
Matching the target recipe does not require equal observed generalization gaps.

The same 48-field preprocessing procedure is used, fitted separately on each
model's training records. Encoded input widths can differ because observed
categories differ. Each shadow trains on about 73,576 auxiliary rows under the
retained split, while the target trains on 100,000 rows; training sizes are not
matched. The MLP seeds are 202, 212, 222 and 232.

This is a stronger-attacker variant that assumes the target architecture and
training recipe are known, without access to target weights or training records.
It changes both pool size and composition, so it does not isolate shadow count.
Four 100-epoch MLPs add training cost; improved attack accuracy is not guaranteed.

All models use the full eligible ACS 2014 pool. Records can be reused across
shadows, while grouped attack splits keep repeated record IDs together.
The attack classifier, thresholds and uncertainty roles are rebuilt for this pool.
''')
    replace(11, '''for index in range(N_SHADOW_MODELS):
    algorithm = SHADOW_ALGORITHMS[index % len(SHADOW_ALGORITHMS)]''',
            'for index, algorithm in enumerate(SHADOW_PLAN):')
    replace(11, "seed=SEED + 10 * index, schema='expanded_acs')",
            "seed=SEED + 10 * index, schema='expanded_acs',\n"
            "            parameters=SHADOW_MLP_CONFIG if algorithm == 'mlp' else None)")
    replace(11, "shadow_progress.update(f'Completed {len(shadow_results)} shadows')",
            "shadow_progress.update(f'Completed {len(shadow_results)} shadows')\n"
            'assert len(shadow_results) == N_SHADOW_MODELS == 20\n'
            "assert sum(result['algorithm'] == 'mlp' for result in shadow_results.values()) == 4\n"
            'for name, result in shadow_results.items():\n'
            "    if result['algorithm'] == 'mlp':\n"
            "        assert result['metadata']['epochs_run'] == TRAIN_CONFIG['epochs']\n"
            "        assert result['metadata']['returned_epoch'] == TRAIN_CONFIG['epochs']\n"
            "        assert result['metadata']['hidden_dims'] == list(TRAIN_CONFIG['hidden_dims'])\n"
            "        assert result['metadata']['seed'] == result['seed']")
    replace(11, 'Probability oracle consistency verified for all 16 shadows.',
            'Probability oracle consistency verified for all 20 shadows.')
    replace(17, 'shadow_algorithms=list(SHADOW_ALGORITHMS),',
            'shadow_algorithms=list(SHADOW_ALGORITHMS),\n'
            '    shadow_algorithm_plan=list(SHADOW_PLAN),\n'
            '    shadow_counts={algorithm: SHADOW_PLAN.count(algorithm) for algorithm in SHADOW_ALGORITHMS},\n'
            '    original_tree_shadow_count=N_TREE_SHADOWS, shadow_mlp_config=SHADOW_MLP_CONFIG,\n'
            "    shadow_architecture_assumption='Target architecture and training recipe known; weights and training records unavailable',\n"
            "    shadow_mlp_training={name: result['metadata'] for name, result in shadow_results.items()\n"
            "        if result['algorithm'] == 'mlp'},")
    replace(20, 'Use the seven preselected masks, target, 16 ACS 2014 shadows, four base',
            'Use the seven preselected masks, target, 20 ACS 2014 shadows (16 trees plus four target-recipe MLPs), four base')
    replace(20, 'using all 16 shadows,', 'using all 20 shadows,')
    replace(20, 'Results are saved under reduced_k6_s16_m7, separate from the eight-shadow run and full experiment.',
            'Results use the separate exp_3_2023_2014_100000_nominal_tree16_mlp4_target_recipe artifact root\n'
            'and reduced_k6_s20_m7 subdirectory. Previous tree-only and 16-shadow mixed results are not overwritten.')
    replace(21, "assert N_SHADOW_MODELS == len(shadow_results) == 16, 'Load the 16-shadow pool first.'",
            "assert N_SHADOW_MODELS == len(shadow_results) == 20, 'Load the 20-shadow pool first.'")
    replace(21, 'Rebuild base attack data using all 16 shadows.',
            'Rebuild base attack data using all 20 shadows.')
    replace(21, 'Refit the base attack on the 16-shadow pool.',
            'Refit the base attack on the 20-shadow pool.')
    replace(28, 'n_shadow_models=N_SHADOW_MODELS, shadow_names=list(shadow_results),',
            'n_shadow_models=N_SHADOW_MODELS, shadow_names=list(shadow_results),\n'
            '    shadow_algorithm_plan=list(SHADOW_PLAN), shadow_mlp_config=SHADOW_MLP_CONFIG,')

    for index, cell in enumerate(notebook['cells']):
        if cell['cell_type'] == 'code':
            ast.parse(read(index), filename=f'{DESTINATION.name}:cell{index}')
    import nbformat
    nbformat.validate(nbformat.from_dict(notebook))
    assert hashlib.sha256(SOURCE.read_bytes()).hexdigest() == original_hash
    DESTINATION.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + '\n', encoding='utf-8')
    return DESTINATION


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--overwrite', action='store_true',
                        help='Replace the existing MLP notebook and clear its outputs.')
    args = parser.parse_args()
    print(create_notebook(overwrite=args.overwrite).name)
