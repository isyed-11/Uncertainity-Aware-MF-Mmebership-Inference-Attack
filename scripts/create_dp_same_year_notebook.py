"""Prepare a same-year DP notebook; never run notebook cells or experiments."""
import argparse
import ast
import copy
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'exp_4_differential_privacy_accuracy.ipynb'
DESTINATION = ROOT / 'exp_4_differential_privacy_same_year_2023.ipynb'


def build_notebook(public_preprocessing_year):
    if public_preprocessing_year not in (2014, 2023):
        raise ValueError('Choose public preprocessing from 2014 or 2023 explicitly.')
    notebook = copy.deepcopy(json.loads(SOURCE.read_text(encoding='utf-8')))

    def read(index):
        return ''.join(notebook['cells'][index]['source'])

    def write(index, source):
        notebook['cells'][index]['source'] = source.splitlines(keepends=True)

    def replace(index, old, new):
        source = read(index)
        if source.count(old) != 1:
            raise ValueError(f'Expected exactly one occurrence in cell {index}: {old!r}')
        write(index, source.replace(old, new))

    for index, cell in enumerate(notebook['cells']):
        cell['id'] = f'exp4-dp-same-year-{index:02d}'
        for key in ('execution', 'ExecuteTime', 'collapsed', 'scrolled'):
            cell['metadata'].pop(key, None)
        if cell['cell_type'] == 'code':
            cell.update(execution_count=None, outputs=[])

    write(0, '''# Experiment 4: DP utility and membership leakage with disjoint 2023 shadows

Compare a fresh non-private ACS 2023 target (epsilon = infinity) and five private
targets at epsilon = 0.5, 1, 2, 4, 8, with delta = 1e-6 fixed. Use 100,000
target training records, all 48 expanded inputs and nominal PINCP > 50,000 labels.
There is no missing-feature reconstruction, PFV or uncertainty aggregation.

Reserve every target train, validation, test, MIA member and MIA nonmember ID
before constructing the unused ACS 2023 auxiliary pool (approximately 73,000
eligible records; report the actual count). Each of 16 non-private shadows gets
its own stratified 80/20 train/test split. Fit each shadow's imputation, scaling
and encoding strictly on its own training rows. Retain four RF, four gradient
boosting, four XGBoost and four generic MLP shadows. MLP shadows train a fixed
50 epochs without validation selection; the OUT test split is never used to
choose epochs. Generic shadow architectures and settings are not target-matched.

Retain the accuracy-focused Exp 4 target recipe: hidden layers 512/256/128,
dropout 0.1, Adam learning rate 0.0003, weight decay 0.0001, maximum 30 epochs,
public-validation accuracy selection (loss breaks ties), patience 5 and best
weight restoration. Logical batches remain 256, physical batches at most 32.
Private targets use Opacus clipping, Gaussian noise, Poisson sampling and the
strict RDP accounting calibration of the original accuracy-focused study.

Train the complete-feature four-signal RF membership classifier once on balanced
2,000 IN and 2,000 OUT examples per shadow, with grouped auxiliary record-ID
train/validation/test roles. Freeze that classifier and its auxiliary-selected
thresholds across all six targets. Blind walks and commuting-field perturbations
are identical for shadows and targets. All shadows are non-private: this remains
a transfer attack, rather than an attack adapted separately to each DP budget.

Plot target test accuracy, MIA accuracy and low-FPR membership leakage against
equally spaced epsilon categories, with a non-private horizontal reference.
Report both evaluation ROC TPR at FPR <= 1% and TPR/actual FPR at the frozen
auxiliary threshold. The evaluation ROC summary uses target membership labels
only to summarize results; it does not select a deployed attack threshold.

Same-year shadows control the source-year mismatch. They do not prove the attack
is optimal, guarantee monotonic curves, or make near-chance MIA a privacy proof.
One run per condition has no repeated-run error bars. Each target's DP guarantee
uses its fixed maximum-schedule bound; stopped-time epsilon is diagnostic only.

No notebook cells have been run. No models, checkpoints, logs or row-level target
scores are saved. Final metrics, public auxiliary row assignments, validation
histories, protocol and PNG/SVG figures are research outputs, not DP releases.
''')

    replace(2, 'AUXILIARY_YEAR = 2014',
            f'AUXILIARY_YEAR = 2023\nPUBLIC_PREPROCESSING_YEAR = {public_preprocessing_year}')
    replace(2, 'EPSILON_BUDGETS = (0.5, 1, 2, 5, 10, 25, 50)',
            'EPSILON_BUDGETS = (0.5, 1, 2, 4, 8)')
    replace(2, 'SHADOW_MLP_CONFIG = dict(MLP_SHADOW_CONFIG)',
            'SHADOW_MLP_CONFIG = dict(MLP_SHADOW_CONFIG,\n'
            '    early_stopping=False, restore_best=False)')
    replace(2, 'from src.membership_attack import train_membership_classifier, evaluate_attack',
            'from src.same_year_mia import disjoint_same_year_pool, split_same_year_shadow\n'
            'from src.dp_same_year import plot_low_fpr_budget\n'
            'from src.membership_attack import train_membership_classifier, evaluate_attack')
    replace(2, 'exp_4_2023_2014_dp_accuracy_e0p5_1_2_5_10_25_50_d1e6_mixed_mlp',
            f'exp_4_2023_2023_dp_e0p5_1_2_4_8_d1e6_s16_train80_test20_public{public_preprocessing_year}')
    replace(2, "for year, filename in [(TARGET_YEAR, 'psam_p06.csv'), (AUXILIARY_YEAR, 'ss14pca.csv')]:",
            "for year in sorted({TARGET_YEAR, AUXILIARY_YEAR, PUBLIC_PREPROCESSING_YEAR}):\n"
            "    filename = 'ss14pca.csv' if year == 2014 else 'psam_p06.csv'")
    replace(2, "print('Epoch cap:', TRAIN_CONFIG['max_epochs'], '| Patience:', TRAIN_CONFIG['patience'])",
            "print('Epoch cap:', TRAIN_CONFIG['max_epochs'], '| Patience:', TRAIN_CONFIG['patience'])\n"
            "print('Shadow split: 80/20; fixed-schedule MLP shadows, no shadow validation.')\n"
            "print('Target public preprocessing year:', PUBLIC_PREPROCESSING_YEAR)")

    common_preprocessing = '''## Fixed public preprocessing and disjoint target/shadow data
Use the original stratified target partitions and 100,000-record training subset.
The fixed target validation set is public and selects epochs only. Test and MIA
records never select model settings or thresholds. Exclude the union of all five
target partitions before constructing the same-year auxiliary pool. Report and
save zero-overlap audits, exact public pool IDs and the actual pool size.

'''
    if public_preprocessing_year == 2023:
        detail = '''Fit the target preprocessor on a seeded 10,000-row public subset of the
unused 2023 auxiliary pool, then freeze it for all six targets. These public
preprocessing rows remain eligible for shadow train/test splits so all roughly
73,000 unused rows remain in the shadow pool. Overlap with shadow records is
intentional; they are disjoint from every target partition. Each shadow still
fits its own independent preprocessor on its own training split only; target
preprocessing is never reused to fit a shadow. Shadow OUT rows never fit their
own shadow's representation or model.
'''
        preprocessing_code = '''_, public_preprocessing = train_test_split(shadow_auxiliary,
    test_size=PUBLIC_PREPROCESSING_SIZE, stratify=shadow_auxiliary[TARGET], random_state=SEED)
public_preprocessing = public_preprocessing.sort_index().copy()
assert set(public_preprocessing.index).issubset(shadow_auxiliary.index)
'''
        scope = 'Seeded public 2023 subset of unused target-source records; may overlap shadow splits; shared by all targets'
    else:
        detail = '''Retain the original seeded 10,000-row public 2014 preprocessing subset,
independent of every 2023 target/shadow record. Freeze this target representation
across all six targets. Shadow preprocessors fit only their own 2023 training
split. The public 2014 vocabulary can miss 2023 categories: the shared fixed
representation controls this across budgets but this limitation remains.
'''
        preprocessing_code = '''public_source = load_expanded_year(PUBLIC_PREPROCESSING_YEAR, income_base_year=None)
_, public_preprocessing = train_test_split(public_source,
    test_size=PUBLIC_PREPROCESSING_SIZE, stratify=public_source[TARGET], random_state=SEED)
public_preprocessing = public_preprocessing.sort_index().copy()
assert set(public_preprocessing.index).isdisjoint(shadow_auxiliary.index)
del public_source
'''
        scope = 'Original seeded public 2014 subset, disjoint from 2023 target/shadow records; shared by all targets'
    write(3, common_preprocessing + detail + '''
The record-level DP statement concerns each target's protected training records,
conditional on the fixed research partition and independently available public
data. Public preprocessing and public validation do not consume that target
training budget. Public shadow models themselves have no DP guarantee.
''')
    write(4, '''target_source = load_expanded_year(TARGET_YEAR, income_base_year=None)
reference_partitions = split_acs_data(target_source, **SPLIT_CONFIG)
target_partitions = subset_target_training(reference_partitions, train_size=TRAIN_SIZE, seed=SEED)
for name in ['validation', 'test', 'mia_members', 'mia_nonmembers']:
    np.testing.assert_array_equal(target_partitions[name].index, reference_partitions[name].index)
assert len(target_partitions['train']) == TRAIN_SIZE
assert len(target_partitions['mia_members']) == len(target_partitions['mia_nonmembers'])
shadow_auxiliary, auxiliary_disjointness_audit = disjoint_same_year_pool(
    target_source, target_partitions)
assert list(shadow_auxiliary.columns) == list(target_source.columns)
assert auxiliary_disjointness_audit['overlapping_record_ids'].eq(0).all()
''' + preprocessing_code + '''
target_preprocessor, target_arrays = prepare_public_preprocessing(public_preprocessing, target_partitions)
partition_counts = {name: len(frame) for name, frame in target_partitions.items()}
public_preprocessing_shadow_overlap = len(set(public_preprocessing.index) & set(shadow_auxiliary.index))
display(auxiliary_disjointness_audit)
display(pd.Series({**partition_counts, 'public_preprocessing': len(public_preprocessing),
    'disjoint_2023_shadow_pool': len(shadow_auxiliary)}, name='records'))
print('Public preprocessing / shadow pool overlap:', public_preprocessing_shadow_overlap)
print('Encoded input dimensions shared by all targets:', target_arrays['train']['X'].shape[1])
del target_source, reference_partitions
gc.collect()
''')
    write(5, '''## Fit and freeze the same-year shadow attack
Train 16 non-private ACS 2023 shadows: four each RF, gradient boosting, XGBoost
and generic MLP. Each shadow independently splits the entire unused pool 80/20
with seed 42 + 10 * shadow_index. There is no shadow validation partition. MLPs
use the existing 128/64 architecture and optimizer, a fixed 50 epochs, and final
weights. Tree settings are unchanged. Each preprocessor fits training rows only.

Sample 2,000 IN training rows and 2,000 OUT test rows per shadow for scoring,
keeping attack examples balanced without querying every shadow training row.
Group repeated record IDs into attack train/validation/test roles; those meta
roles are distinct from a shadow's 80/20 model-data split. The RF attack and
accuracy/low-FPR thresholds are selected using auxiliary data only and frozen
before target evaluation. Release each shadow after extracting its features.

This controls source year but remains a non-private, mixed-model transfer attack.
It does not establish the strongest possible DP-aware attack. Do not interpret
weak auxiliary held-out performance as evidence of target privacy.
''')
    replace(6, '    with quiet_output():\n        result = train_shadow_model',
            '    partitions = split_same_year_shadow(auxiliary, seed=seed, early_stopping=False)\n'
            '    assert set(partitions) == {\'train\', \'nonmembers\'}\n'
            '    with quiet_output():\n        result = train_shadow_model')
    replace(6, "parameters=SHADOW_MLP_CONFIG if algorithm == 'mlp' else None)",
            "parameters=SHADOW_MLP_CONFIG if algorithm == 'mlp' else None, partitions=partitions)")
    replace(6, '    return features, audits[name], summary', '''    counts = features['membership'].value_counts()
    assert counts[0] == counts[1] == SHADOW_RECORDS_PER_CLASS
    audit = dict(audits[name], seed=seed,
        train_records=len(partitions['train']), out_test_records=len(partitions['nonmembers']),
        validation_records=0, sampled_in=int(counts[1]), sampled_out=int(counts[0]),
        epochs_run=result['metadata'].get('epochs_run'),
        early_stopping=result['metadata'].get('early_stopping', False))
    return features, audit, summary''')
    # Matched target helper/training and accounting remain exactly as before.
    replace(7, 'Public\npreprocessing, splits, architecture, regularization, auxiliary attack and blind\nwalks follow the original DP experiment. Learning rate, epoch selection,\naccounting calibration, expanded budget grid and mixed shadow composition differ\nin this follow-up. Target training settings are unaffected by adding MLP shadows.',
            'Target architecture, regularization, learning rate, epoch selection,\n'
            'splits, physical/logical batch sizes, DP accounting and blind walks\n'
            'follow accuracy-focused Exp 4. This variant changes auxiliary source\n'
            'year, shadow splits/schedule and the declared budget grid. The public\n'
            'preprocessing source is recorded explicitly in this notebook.')
    write(9, '''### Run six matched conditions
This cell trains a fresh non-private control (epsilon = infinity) and five
private targets at epsilon = 0.5, 1, 2, 4, 8. It is an experiment and requires
explicit approval to execute. No historical targets or cached scores are used.
Report every budget without selecting one using target test or MIA outcomes.
The three known DP research warnings are shown once; errors remain visible.
''')
    replace(10, 'All eight conditions completed', 'All six conditions completed')
    replace(12, "'generalization_gap', 'mia_accuracy', 'mia_roc_auc', 'mia_tpr_at_1pct_fpr_roc'",
            "'generalization_gap', 'mia_accuracy', 'mia_roc_auc', 'mia_tpr_at_1pct_fpr_roc',\n"
            "    'mia_tpr_at_shadow_threshold', 'mia_fpr_at_shadow_threshold'")
    replace(17, 'both graphs', 'all three figures')
    replace(17, 'native year-specific Census\ncodes, an auxiliary-derived vocabulary,',
            'same-year Census shadow data,\na fixed public vocabulary,')
    replace(17, 'tables, fitted models, checkpoints or logs',
            'tables, fitted models, checkpoints or logs')
    replace(18, "study_variant='accuracy-focused exploratory follow-up'",
            "study_variant='same-year 2023 shadows, 80/20, accuracy-focused DP targets'")
    replace(18, "preprocessing_scope='Fixed public 2014 subset, disjoint from shadow pool; shared by all targets',",
            f'preprocessing_scope={scope!r},\n'
            '    public_preprocessing_year=PUBLIC_PREPROCESSING_YEAR,\n'
            '    public_preprocessing_shadow_overlap=public_preprocessing_shadow_overlap,\n'
            '    auxiliary_pool_records=len(shadow_auxiliary),\n'
            "    auxiliary_pool_source='Unused ACS 2023 source records excluding every target partition',\n"
            "    auxiliary_disjointness_audit=auxiliary_disjointness_audit.to_dict('records'),\n"
            '    shadow_split_fractions=dict(train=0.8, nonmembers=0.2),\n'
            "    shadow_mlp_selection='Fixed 50 epochs; final weights; no validation/test selection',")
    replace(18, "shadow_income_metrics.to_csv(RUN_DIR / 'shadow_income_metrics.csv')",
            "shadow_income_metrics.to_csv(RUN_DIR / 'shadow_income_metrics.csv')\n"
            "auxiliary_disjointness_audit.to_csv(RUN_DIR / 'auxiliary_disjointness_audit.csv', index=False)\n"
            "pd.Series(shadow_auxiliary.index, name='record_id').to_csv(RUN_DIR / 'auxiliary_pool_ids.csv', index=False)\n"
            "pd.Series(public_preprocessing.index, name='record_id').to_csv(RUN_DIR / 'public_preprocessing_ids.csv', index=False)")
    replace(18, 'target_accuracy_vs_epsilon and mia_accuracy_vs_epsilon (PNG and SVG)',
            'target_accuracy_vs_epsilon, mia_accuracy_vs_epsilon, mia_low_fpr_vs_epsilon (PNG and SVG)')

    graph_cells = [
        dict(cell_type='markdown', metadata={}, id='exp4-dp-same-year-low-fpr-notes', source='''## Graph 3: low-FPR membership leakage versus privacy budget
Left: evaluation ROC TPR at empirical FPR <= 1%, summarized using target labels.
Right: TPR and actual target FPR at the single frozen auxiliary-validation cutoff.
The fixed cutoff is not recalibrated on target records and need not achieve 1%
target FPR. Report the actual FPR even when the TPR appears favorable. Epsilon
uses categorical spacing; the non-private control is a horizontal reference.
These single-run measurements do not force or establish monotonic privacy gains.
'''.splitlines(keepends=True)),
        dict(cell_type='code', metadata={}, id='exp4-dp-same-year-low-fpr-code', execution_count=None,
             outputs=[], source='''figure, axes = plot_low_fpr_budget(study_results, EPSILON_BUDGETS,
    output_dir=RUN_DIR, delta=DELTA)
display(figure)
plt.close(figure)
'''.splitlines(keepends=True)),
    ]
    notebook['cells'][17:17] = graph_cells
    for index, cell in enumerate(notebook['cells']):
        if cell['cell_type'] == 'code':
            ast.parse(''.join(cell['source']), filename=f'{DESTINATION.name}:cell{index}')
    import nbformat
    nbformat.validate(nbformat.from_dict(notebook))
    return notebook


def create_notebook(public_preprocessing_year):
    if DESTINATION.exists():
        raise FileExistsError(f'Refusing to overwrite {DESTINATION.name}')
    notebook = build_notebook(public_preprocessing_year)
    DESTINATION.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + '\n', encoding='utf-8')
    return DESTINATION


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--public-preprocessing-year', type=int, choices=(2014, 2023), required=True)
    args = parser.parse_args()
    print(create_notebook(args.public_preprocessing_year).name)
