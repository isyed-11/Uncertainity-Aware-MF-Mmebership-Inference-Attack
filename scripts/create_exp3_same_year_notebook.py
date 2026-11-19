"""Create a separate same-year Exp 3 notebook without running notebook cells."""
import ast
import copy
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'exp_3_expanded_missing_feature_baseline.ipynb'
DESTINATION = ROOT / 'exp_3_expanded_missing_feature_2023_auxiliary_mlp.ipynb'


def create_notebook():
    if DESTINATION.exists():
        raise FileExistsError(f'Refusing to overwrite {DESTINATION.name}')
    notebook = copy.deepcopy(json.loads(SOURCE.read_text(encoding='utf-8')))

    def read(index):
        return ''.join(notebook['cells'][index]['source'])

    def write(index, source):
        notebook['cells'][index]['source'] = source.splitlines(keepends=True)

    def replace(index, old, new):
        source = read(index)
        if old not in source:
            raise ValueError(f'Expected source missing in cell {index}: {old!r}')
        write(index, source.replace(old, new))

    # All copied observations belong to the old experiment, so clear them.
    for index, cell in enumerate(notebook['cells']):
        cell['id'] = f'exp3-2023-aux-mlp-{index:02d}'
        for key in ('execution', 'ExecuteTime', 'collapsed', 'scrolled'):
            cell['metadata'].pop(key, None)
        if cell['cell_type'] == 'code':
            cell.update(execution_count=None, outputs=[])
        write(index, read(index).replace('2014', '2023'))

    write(0, '''# Expanded ACS missing-feature study: disjoint 2023 auxiliary data + MLP shadows

ACS 2023 is used for both the target and auxiliary source. The auxiliary pool
contains only unused records: all target train, validation, test, MIA member and
MIA nonmember IDs are excluded. Records may be reused across shadow models.

Retain Exp 3's 48 features, nominal PINCP > 50,000 label, target training size,
100-epoch target configuration, seven masks, K=6 and original blind walks.
The 16 shadows comprise four random forests, four gradient-boosting models,
four XGBoost models and four MLPs. Each has a separately seeded 40/20/40 split;
preprocessing fits its training records only. MLPs use the existing generic
shadow settings, not the target architecture or target training schedule.

Includes the complete-feature baseline, marginal PFV, and conditional
single/average/mean-only/uncertainty-aware comparisons. The historical shadow
versus target commuting-walk asymmetry is preserved and limits interpretation.
Same-year auxiliary data remove the year mismatch but provide a smaller pool;
this and the new algorithm mix mean results need not match the historical 62.625%.

No experiments have been run in this notebook. No checkpoints or logs are used.
Only final results, row IDs and protocols are saved in a separate artifact directory.
''')

    replace(1, 'import time\n', '''import time
import importlib
import src.expanded_data as expanded_data_module
import src.shadow_models as shadow_models_module
importlib.reload(expanded_data_module)
importlib.reload(shadow_models_module)
''')
    replace(1, 'summarize_shadow_models, PAPER_SHADOW_ALGORITHMS)',
            'summarize_shadow_models, PAPER_SHADOW_ALGORITHMS, MLP_SHADOW_CONFIG)\n'
            'from src.same_year_mia import disjoint_same_year_pool')
    replace(1, 'artifacts/exp_3_2023_2023_100000_nominal_paper_algorithms',
            'artifacts/exp_3_2023_2023_100000_nominal_mixed_mlp')
    replace(1, 'SHADOW_ALGORITHMS = PAPER_SHADOW_ALGORITHMS\n',
            "SHADOW_ALGORITHMS = PAPER_SHADOW_ALGORITHMS + ('mlp',)\n"
            'SHADOW_MLP_CONFIG = dict(MLP_SHADOW_CONFIG)\n'
            'assert N_SHADOW_MODELS == 4 * len(SHADOW_ALGORITHMS)\n')

    write(10, '''## 16 shadows on unused ACS 2023 records
Exclude every target training and evaluation record before creating the pool.
Train four models per algorithm: random forest, gradient boosting, XGBoost and MLP.
Each shadow uses its own seed and randomized 40/20/40 train/validation/nonmember
split. Fit scaling, encoding and imputation strictly on that shadow's train split.
Models can reuse auxiliary records across shadows; grouped attack splits keep
repeated record IDs together. The complete-feature attack samples 2,000 IN and
2,000 OUT records per shadow, as in Exp 3.

Keep existing tree settings. MLP shadows use hidden layers (128, 64), dropout 0,
learning rate 0.001, weight decay 0.0001, batch size 256 and at most 50 epochs,
with validation-loss early stopping (patience 5) and best-weight restoration.
They share the target's preprocessing implementation and 48 input fields, but
are not an architecture-matched attack. The smaller disjoint auxiliary pool
also gives each shadow fewer training records than the target.
''')
    replace(11, 'acs_auxiliary = load_expanded_year(2023, income_base_year=INCOME_BASE_YEAR)',
            '''acs_auxiliary, auxiliary_disjointness_audit = disjoint_same_year_pool(
    expanded_source, expanded_partitions)
assert list(acs_auxiliary.columns) == list(expanded_source.columns)
assert set(EXPANDED_FEATURES) <= set(acs_auxiliary.columns)
print(f'Disjoint ACS 2023 auxiliary pool: {len(acs_auxiliary):,} records')
display(auxiliary_disjointness_audit)''')
    replace(11, 'assert set(acs_auxiliary.index).isdisjoint(expanded_source.index)',
            "assert auxiliary_disjointness_audit['overlapping_record_ids'].eq(0).all()")
    replace(11, "seed=SEED + 10 * index, schema='expanded_acs')",
            "seed=SEED + 10 * index, schema='expanded_acs',\n"
            "            parameters=SHADOW_MLP_CONFIG if algorithm == 'mlp' else None)")
    replace(11, "    result['attack_data']['shadow_model'] = name", """    # Verify every actual shadow partition belongs to the audited pool.
    for partition_ids in result['partition_ids'].values():
        assert set(partition_ids).issubset(acs_auxiliary.index)
    result['attack_data']['shadow_model'] = name""")

    replace(17, "feature_inventory.to_csv(result_dir / 'feature_inventory.csv')", """feature_inventory.to_csv(result_dir / 'feature_inventory.csv')
auxiliary_disjointness_audit.to_csv(result_dir / 'auxiliary_disjointness_audit.csv', index=False)
pd.Series(acs_auxiliary.index, name='record_id').to_csv(
    result_dir / 'auxiliary_pool_ids.csv', index=False)
for shadow_name, result in shadow_results.items():
    for partition_name, record_ids in result['partition_ids'].items():
        pd.Series(record_ids, name='record_id').to_csv(
            result_dir / f'{shadow_name}_{partition_name}_ids.csv', index=False)""")
    replace(17, "'correction': 'Prior assistant statements that the shadow commuting perturbation protocol changed were incorrect. Original and current perturbation call sources are identical.'",
            "'interpretation': 'Historical shadow/target commuting-field walk asymmetry is retained.'")
    replace(17, 'shadow_algorithms=list(SHADOW_ALGORITHMS),', '''shadow_algorithms=list(SHADOW_ALGORITHMS),
    shadow_mlp_config=SHADOW_MLP_CONFIG,
    shadow_split_fractions=dict(train=0.4, validation=0.2, nonmembers=0.4),
    shadow_records_per_membership_class=SHADOW_RECORDS_PER_CLASS,
    auxiliary_pool_records=len(acs_auxiliary),
    auxiliary_pool_source='unused ACS 2023 records after excluding all target partitions',
    auxiliary_disjointness_audit=auxiliary_disjointness_audit.to_dict('records'),''')
    replace(18, '''Use stable-code fields only; do not extend this to occupation/industry without
an explicit cross-year category mapping. Related marriage and employment fields
remain observed.''', '''Both sources use the same year's feature codes. The original masks are retained;
related marriage and employment fields remain observed.''')
    replace(20, 'Results are saved under reduced_k6_s16_m7, separate from the eight-shadow run and full experiment.',
            'Results use a separate 2023-auxiliary mixed-MLP artifact root and reduced_k6_s16_m7 subdirectory.')
    replace(28, "'cross-year distribution shift'", "'smaller disjoint same-year auxiliary pool; shadows train on fewer records than target'")

    # Syntax checks only: never execute copied cells while generating the file.
    for index, cell in enumerate(notebook['cells']):
        if cell['cell_type'] == 'code':
            ast.parse(read(index), filename=f'{DESTINATION.name}:cell{index}')
    import nbformat
    nbformat.validate(nbformat.from_dict(notebook))
    DESTINATION.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + '\n', encoding='utf-8')
    return DESTINATION


if __name__ == '__main__':
    print(create_notebook().name)
