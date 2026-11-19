"""Build Exp 3 with 2014 auxiliary data, 16 balanced shadows and early stopping.

Keep the existing target and generic MLP training recipes; never execute cells.
"""
import argparse
import ast
import copy
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'exp_3_expanded_missing_feature_2023_auxiliary_mlp.ipynb'
DESTINATION = ROOT / 'exp_3_expanded_missing_feature_2023_early_stopping.ipynb'


def create_notebook(*, overwrite=False):
    if DESTINATION.exists() and not overwrite:
        raise FileExistsError(f'Refusing to overwrite {DESTINATION.name}')
    source_bytes = SOURCE.read_bytes()
    source_hash = hashlib.sha256(source_bytes).hexdigest()
    notebook = copy.deepcopy(json.loads(source_bytes.decode('utf-8')))

    def read(index):
        return ''.join(notebook['cells'][index]['source'])

    def write(index, source):
        notebook['cells'][index]['source'] = source.splitlines(keepends=True)

    def replace(index, old, new):
        source = read(index)
        if old not in source:
            raise ValueError(f'Expected source missing in cell {index}: {old!r}')
        write(index, source.replace(old, new))

    for index, cell in enumerate(notebook['cells']):
        cell['id'] = f'exp3-2014-early-stop-s16-{index:02d}'
        for key in ('execution', 'ExecuteTime', 'collapsed', 'scrolled'):
            cell['metadata'].pop(key, None)
        if cell['cell_type'] == 'code':
            cell.update(execution_count=None, outputs=[])

    replace(0, '# Expanded ACS missing-feature study: disjoint 2023 auxiliary data + MLP shadows',
            '# Expanded ACS missing-feature study: early-stopped target and MLP shadows')
    replace(0, '100-epoch target configuration, seven masks, K=6 and original blind walks.',
            'seven masks, K=6 and original blind walks. Target training now uses a\n'
            '100-epoch maximum with validation-loss early stopping (patience 5) and\n'
            'restores the best validation weights. The four MLP shadows retain their\n'
            '50-epoch maximum, patience 5 and best-weight restoration. Tree shadows\n'
            'retain fixed settings; early stopping does not apply to them in this study.')
    replace(0, 'this and the new algorithm mix mean results need not match the historical 62.625%.',
            'this, the algorithm mix and early stopping mean results need not match\n'
            'the historical 62.625%. Lower membership accuracy can reflect reduced\n'
            'leakage; better target generalization does not guarantee a stronger attack.')
    replace(1, 'artifacts/exp_3_2023_2023_100000_nominal_mixed_mlp',
            'artifacts/exp_3_2023_2023_100000_nominal_mixed_mlp_early_stopping')
    replace(1, "device='cpu', early_stopping=False, restore_best=False, monitor_train=True)",
            "device='cpu', patience=5, early_stopping=True, restore_best=True, monitor_train=True)")
    replace(1, 'SHADOW_MLP_CONFIG = dict(MLP_SHADOW_CONFIG)',
            'SHADOW_MLP_CONFIG = dict(MLP_SHADOW_CONFIG, patience=5,\n'
            '    early_stopping=True, restore_best=True)')
    replace(1, 'assert N_SHADOW_MODELS == 4 * len(SHADOW_ALGORITHMS)',
            'assert N_SHADOW_MODELS == 4 * len(SHADOW_ALGORITHMS)\n'
            "assert TRAIN_CONFIG['early_stopping'] and TRAIN_CONFIG['restore_best']\n"
            "assert SHADOW_MLP_CONFIG['early_stopping'] and SHADOW_MLP_CONFIG['restore_best']")
    write(6, '''## Train the target with validation-loss early stopping
Keep the same architecture, optimizer, training records and preprocessing.
Stop after five consecutive epochs without a strictly lower validation loss,
with 100 epochs as the maximum. Restore the lowest-validation-loss weights
from memory. Target test and MIA records never select the returned epoch.
No model checkpoint is written to disk.
''')
    replace(7, 'display(pd.DataFrame(expanded_history).iloc[::10])',
            "print('Epochs run:', expanded_metadata['epochs_run'],\n"
            "      '| Best/returned epoch:', expanded_metadata['returned_epoch'])\n"
            '# Include the last epoch even when stopping before the next tenth epoch.\n'
            'history_table = pd.DataFrame(expanded_history)\n'
            'history_indices = sorted(set(range(0, len(history_table), 10)) | {len(history_table) - 1})\n'
            'display(history_table.iloc[history_indices])')
    replace(10, 'Keep existing tree settings.',
            'Keep existing fixed tree settings; only the target and MLP shadows\n'
            'use validation-loss early stopping.')
    replace(11, 'display(summarize_shadow_models(shadow_results))',
            '''display(summarize_shadow_models(shadow_results))
shadow_training_summary = pd.DataFrame([
    dict(shadow_model=name, algorithm=result['algorithm'],
         early_stopping=result['metadata'].get('early_stopping', False),
         epochs_run=result['metadata'].get('epochs_run'),
         returned_epoch=result['metadata'].get('returned_epoch'))
    for name, result in shadow_results.items()
]).set_index('shadow_model')
display(shadow_training_summary.loc[shadow_training_summary['algorithm'] == 'mlp'])''')
    replace(17, "summarize_shadow_models(shadow_results).to_csv(result_dir / 'shadow_metrics.csv')",
            "summarize_shadow_models(shadow_results).to_csv(result_dir / 'shadow_metrics.csv')\n"
            "shadow_training_summary.to_csv(result_dir / 'shadow_training_summary.csv')")
    replace(17, 'shadow_mlp_config=SHADOW_MLP_CONFIG,',
            "shadow_mlp_config=SHADOW_MLP_CONFIG,\n"
            "    target_training_config=TRAIN_CONFIG,\n"
            "    early_stopping_protocol=dict(monitor='validation log_loss', patience=5,\n"
            "        improvement='strictly lower', restore_best=True,\n"
            "        scope='target and four MLP shadows only; tree settings fixed'),\n"
            "    shadow_training_metadata={name: result['metadata'] for name, result\n"
            "        in shadow_results.items() if result['algorithm'] == 'mlp'},")
    replace(18, 'the complete-feature baseline uses the historical settings.',
            'the complete-feature baseline uses the early-stopped target and MLP shadows.')
    replace(20, 'separate 2023-auxiliary mixed-MLP artifact root',
            'separate 2023-auxiliary mixed-MLP early-stopping artifact root')
    replace(28, 'n_shadow_models=N_SHADOW_MODELS, shadow_names=list(shadow_results),',
            'n_shadow_models=N_SHADOW_MODELS, shadow_names=list(shadow_results),\n'
            '    target_training=expanded_metadata, shadow_mlp_config=SHADOW_MLP_CONFIG,')

    # Requested pool: four shadows per algorithm, keeping the early-stopping recipes.
    write(0, '''# Expanded ACS missing-feature study: 2014 auxiliary data, 16 shadows and early stopping

The target uses ACS 2023; shadows and reconstruction use the full eligible ACS
2014 pool. The notebook filename retains 2023 to identify the target year.
Check auxiliary record IDs against all target partitions before training.

Retain 48 features, nominal PINCP > 50,000 labels, the 100,000-row target training
set, seven masks, K=6 and the original blind walks. Target training uses at most
100 epochs, validation-loss early stopping with patience 5, and best weights.

Use four RF, four gradient boosting, four XGBoost and four MLP shadows.
Cycle these four algorithms with a unique seed per model. Each shadow has
its own seeded 40/20/40 split and fits
preprocessing on its training split only. MLPs keep the existing generic recipe:
hidden layers (128, 64), up to 50 epochs, patience 5 and best-weight restoration.
They do not copy the target architecture. Tree settings remain fixed.

Includes complete-feature MIA, marginal PFV and conditional single/average/
mean-only/uncertainty-aware comparisons. Cross-year distribution shift and the
historical shadow/target commuting-walk asymmetry limit interpretation. Results
need not match the historical 62.625%; early stopping can reduce membership
signals without guaranteeing that the attack is strong.

No experiments are run when generating this notebook. No checkpoints or logs
are used. Results, row IDs and protocols use a separate 2014/s16 artifact folder.
''')
    replace(1, 'from src.same_year_mia import disjoint_same_year_pool\n', '')
    replace(1, 'artifacts/exp_3_2023_2023_100000_nominal_mixed_mlp_early_stopping',
            'artifacts/exp_3_2023_2014_100000_nominal_mixed_mlp_early_stopping')
    replace(1, 'assert N_SHADOW_MODELS == 4 * len(SHADOW_ALGORITHMS)',
            'SHADOW_PLAN = tuple(SHADOW_ALGORITHMS[index % len(SHADOW_ALGORITHMS)]\n'
            '    for index in range(N_SHADOW_MODELS))\n'
            'assert N_SHADOW_MODELS == len(SHADOW_PLAN) == 16\n'
            'assert {algorithm: SHADOW_PLAN.count(algorithm) for algorithm in SHADOW_ALGORITHMS} == {\n'
            "    'random_forest': 4, 'gradient_boosting': 4, 'xgboost': 4, 'mlp': 4}")
    write(10, '''## 16 shadows on ACS 2014 records: four per algorithm
Use the full eligible 2014 pool, disjoint from the 2023 target record IDs.
Cycle RF, gradient boosting, XGBoost and MLP four times, with
seed = 42 + 10 * index. The MLP seeds are 72, 112, 152 and 192.
Each model keeps its own 40/20/40 train/validation/OUT
split. Fit imputation, scaling and encoding on that shadow's train records only.

| Shadow algorithm | Count | Main settings |
| --- | ---: | --- |
| Random forest | 4 | 200 trees, minimum leaf size 2 |
| Gradient boosting | 4 | 300 trees, depth 3, learning rate 0.05 |
| XGBoost | 4 | 300 trees, depth 5, learning rate 0.05 |
| MLP | 4 | Hidden layers (128, 64), maximum 50 epochs, early stopping with patience 5 |

Keep fixed tree settings and the existing generic MLP recipe: hidden layers
(128, 64), dropout 0, learning rate 0.001, weight decay 0.0001, batch size 256,
at most 50 epochs, validation-loss early stopping with patience 5 and best weights.
The target retains its separate (512, 256, 128) architecture and 100-epoch cap.
Only the target and MLPs use early stopping; OUT records never select weights.

Models may reuse auxiliary records across shadows. The attack samples 2,000 IN
and 2,000 OUT records per model (64,000 attack examples before grouped splitting).
Repeated record IDs stay in the same attack partition. The base attack and its
grouped auxiliary splits are rebuilt for this 16-model pool.
''')
    replace(11, '''acs_auxiliary, auxiliary_disjointness_audit = disjoint_same_year_pool(
    expanded_source, expanded_partitions)''',
            '''acs_auxiliary = load_expanded_year(2014, income_base_year=INCOME_BASE_YEAR)
assert acs_auxiliary.index.is_unique
assert set(acs_auxiliary.index).isdisjoint(expanded_source.index)
auxiliary_disjointness_audit = pd.DataFrame([
    dict(target_partition=name, target_records=len(part),
         shadow_pool_records=len(acs_auxiliary),
         overlapping_record_ids=len(set(acs_auxiliary.index) & set(part.index)))
    for name, part in expanded_partitions.items()
])''')
    replace(11, 'Disjoint ACS 2023 auxiliary pool:', 'Disjoint ACS 2014 auxiliary pool:')
    replace(11, '''for index in range(N_SHADOW_MODELS):
    algorithm = SHADOW_ALGORITHMS[index % len(SHADOW_ALGORITHMS)]''',
            'for index, algorithm in enumerate(SHADOW_PLAN):')
    replace(11, "shadow_progress.update(f'Completed {len(shadow_results)} shadows')",
            "shadow_progress.update(f'Completed {len(shadow_results)} shadows')\n"
            'assert len(shadow_results) == N_SHADOW_MODELS == 16\n'
            "assert sum(result['algorithm'] == 'mlp' for result in shadow_results.values()) == 4")
    replace(12, 'repeated 2023 records', 'repeated 2014 records')
    replace(15, '2023 auxiliary attack test', '2014 auxiliary attack test')
    replace(17, 'auxiliary_year=2023', 'auxiliary_year=2014')
    replace(17, 'shadow_algorithms=list(SHADOW_ALGORITHMS),',
            'shadow_algorithms=list(SHADOW_ALGORITHMS),\n'
            '    shadow_algorithm_plan=list(SHADOW_PLAN),\n'
            '    shadow_counts={algorithm: SHADOW_PLAN.count(algorithm) for algorithm in SHADOW_ALGORITHMS},')
    replace(17, "auxiliary_pool_source='unused ACS 2023 records after excluding all target partitions'",
            "auxiliary_pool_source='full eligible ACS 2014 pool; record IDs disjoint from ACS 2023 target'")
    replace(18, 'ACS 2023 predictors only', 'ACS 2014 predictors only')
    replace(18, "Both sources use the same year's feature codes. The original masks are retained;\nrelated marriage and employment fields remain observed.",
            'Use the original stable-code masks across 2014 and 2023; related marriage\n'
            'and employment fields remain observed. Extending to occupation/industry\n'
            'would require an explicit cross-year category mapping.')
    replace(19, 'auxiliary_year=2023', 'auxiliary_year=2014')
    replace(19, 'frozen complete-feature ACS 2023 attack', 'frozen complete-feature ACS 2014 auxiliary attack')
    replace(20, 'target, 16 ACS 2023 shadows', 'target, 16 ACS 2014 shadows')
    replace(20, 'separate 2023-auxiliary mixed-MLP early-stopping artifact root and reduced_k6_s16_m7 subdirectory.',
            'separate 2014-auxiliary mixed-MLP early-stopping artifact root and reduced_k6_s16_m7 subdirectory.')
    replace(21, 'unique_2023_records', 'unique_2014_records')
    replace(22, '2023', '2014')
    replace(26, '2023_auxiliary_test', '2014_auxiliary_test')
    replace(28, 'auxiliary_year=2023', 'auxiliary_year=2014')
    replace(28, 'target_training=expanded_metadata, shadow_mlp_config=SHADOW_MLP_CONFIG,',
            'target_training=expanded_metadata, shadow_mlp_config=SHADOW_MLP_CONFIG,\n'
            '    shadow_algorithm_plan=list(SHADOW_PLAN),')
    replace(28, 'smaller disjoint same-year auxiliary pool; shadows train on fewer records than target',
            'cross-year distribution shift; shadows train on fewer records than target')

    for index, cell in enumerate(notebook['cells']):
        if cell['cell_type'] == 'code':
            ast.parse(read(index), filename=f'{DESTINATION.name}:cell{index}')
    import nbformat
    nbformat.validate(nbformat.from_dict(notebook))
    assert hashlib.sha256(SOURCE.read_bytes()).hexdigest() == source_hash
    DESTINATION.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + '\n', encoding='utf-8')
    return DESTINATION


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--overwrite', action='store_true',
                        help='Replace the existing early-stopping notebook and clear outputs.')
    args = parser.parse_args()
    print(create_notebook(overwrite=args.overwrite).name)
