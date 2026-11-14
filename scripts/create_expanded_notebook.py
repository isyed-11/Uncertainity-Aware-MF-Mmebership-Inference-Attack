"""Create the expanded experiment notebook; does not execute it."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
cells = []


def markdown(text):
    cells.append({'cell_type': 'markdown', 'metadata': {}, 'id': f'exp3-{len(cells)}',
                  'source': text.splitlines(keepends=True)})


def code(text):
    cells.append({'cell_type': 'code', 'metadata': {}, 'id': f'exp3-{len(cells)}',
                  'execution_count': None, 'outputs': [], 'source': text.splitlines(keepends=True)})


markdown("""# Experiment 3: ACS 2023 target and ACS 2014 auxiliary shadows
Train the target on 100,000 California 2023 records with expanded predictors.
Compare 16 California 2014 shadows with 16 UCI Adult shadows in separate cells.
Both pools cycle random forest (6), gradient boosting (5), and XGBoost (5).
Each model fits its own preprocessing on its training partition only.
The label is nominal personal income (PINCP) >$50,000 in both years, without
ADJINC or CPI inflation adjustment. Original eligibility filters are retained. Occupation and other category
codes remain native to each year; no cross-year code equivalence is assumed.
Income-derived predictors, identifiers, survey weights and allocation flags are
excluded. MIA uses logit, blind walk, entropy and cross-entropy.
Perturbation calls retain the original protocol: categorical commuting changes
for shadows and numeric commuting changes for the target. No protocol correction
was applied in the 100,000-record run.
Prepared for review; execution requires user approval.
""")
code('''from pathlib import Path
import json
import time
import numpy as np
import pandas as pd
import torch
from IPython.display import display
from src.data import load_acs_income, split_acs_data, subset_target_training
from src.expanded_data import (
    load_expanded_acs, preprocess_expanded_partitions, expanded_feature_inventory,
    reference_split_config, EXPANDED_FEATURES, EXCLUDED, load_expanded_year,
)
from src.training import train_target_model, evaluate_model
from src.exp3_checkpoints import cached_target_train, cached_expanded_shadows
from src.shadow_models import (load_adult_data, train_shadow_models,
                               summarize_shadow_models, PAPER_SHADOW_ALGORITHMS)
from src.membership_attack import train_membership_classifier, evaluate_attack
from src.paper_attack_features import (collect_shadow_paper_features, collect_target_paper_features,
                                       make_probability_oracle)

torch.set_num_threads(4)
SEED = 42
TRAIN_SIZE = 100_000
INCOME_BASE_YEAR = None
NOMINAL_INCOME_METADATA = dict(enabled=False, label='PINCP > 50000', adjustment='None')
RUN_DIR = Path('artifacts/exp_3_2023_2014_100000_nominal_paper_algorithms')
ADULT_RUN_DIR = RUN_DIR / 'adult_auxiliary'
RUN_DIR.mkdir(parents=True, exist_ok=True)
SPLIT_CONFIG = reference_split_config()
assert SPLIT_CONFIG['seed'] == SEED, 'Use the reference seed consistently.'
TRAIN_CONFIG = dict(hidden_dims=(512, 256, 128), dropout=0.0, epochs=100,
    batch_size=256, learning_rate=1e-3, weight_decay=0.0, seed=SEED,
    device='cpu', early_stopping=False, restore_best=False, monitor_train=True)
WALK_CONFIG = dict(max_steps=20, repeats=2, categorical_change_probability=0.05,
                   batch_size=512, seed=SEED)
SHADOW_RECORDS_PER_CLASS = 2000
N_SHADOW_MODELS = 16
SHADOW_ALGORITHMS = PAPER_SHADOW_ALGORITHMS
# Native commuting units; all other added codes are categorical in the walk.
EXPANDED_NUMERIC_WALK = {'JWMNP': (2.0, 0, 200), 'JWRIP': (1.0, 1, 10)}
print('Training size:', TRAIN_SIZE, '| Input fields:', len(EXPANDED_FEATURES))
print('Reference split configuration:', SPLIT_CONFIG)
''')
markdown('''## Load 2023 target records and make stratified partitions
Split sizes and seed follow Experiment 2, applied to the new 2023 records.
''')
code('''expanded_source = load_expanded_year(2023, income_base_year=INCOME_BASE_YEAR)
reference_partitions = split_acs_data(expanded_source, **SPLIT_CONFIG)
expanded_base_partitions = reference_partitions
expanded_partitions = subset_target_training(
    expanded_base_partitions, train_size=TRAIN_SIZE, seed=SEED)
for name in ['validation', 'test', 'mia_members', 'mia_nonmembers']:
    np.testing.assert_array_equal(expanded_partitions[name].index, reference_partitions[name].index)
assert set(expanded_partitions['mia_members'].index).issubset(expanded_partitions['train'].index)
for name in ['validation', 'test', 'mia_nonmembers']:
    assert set(expanded_partitions[name].index).isdisjoint(expanded_partitions['train'].index)
print('Target income adjustment:', expanded_source.attrs.get('income_adjustment', NOMINAL_INCOME_METADATA))
print('Eligible records:', len(expanded_source))
print('Unused records:', len(expanded_source) - TRAIN_SIZE
      - sum(len(expanded_partitions[name]) for name in ['validation', 'test', 'mia_nonmembers']))
display(pd.DataFrame({name: {'records': len(part), 'positive_rate': part['income_gt_50k'].mean()}
    for name, part in expanded_partitions.items()}).T)
feature_inventory = expanded_feature_inventory(expanded_partitions['train'])
display(feature_inventory)
''')
markdown('''## Preprocess using training records only
The output is sparse; the MLP densifies only the current batch. Cardinality
counts above refer to the original fields, not one-hot dimensions.
''')
code('''expanded_preprocessor, expanded_arrays = preprocess_expanded_partitions(expanded_partitions)
print('Encoded input dimensions:', expanded_arrays['train']['X'].shape[1])
print('Training CSR storage (MiB):', round(sum(
    value.nbytes for value in [expanded_arrays['train']['X'].data,
    expanded_arrays['train']['X'].indices, expanded_arrays['train']['X'].indptr]) / 2**20, 2))
pd.DataFrame({name: {'rows': values['X'].shape[0], 'encoded_features': values['X'].shape[1]}
    for name, values in expanded_arrays.items()}).T
''')
markdown('''## Train the target
Fixed 100 epochs, final weights retained. Validation is monitored but does not
select an epoch in this deliberately overfit experimental condition.
''')
code('''target_started = time.monotonic()
expanded_model, expanded_history, expanded_metadata = cached_target_train(
    expanded_arrays, TRAIN_CONFIG, RUN_DIR)
target_training_seconds = time.monotonic() - target_started
print('Target training seconds:', round(target_training_seconds, 1))
display(pd.DataFrame(expanded_history).iloc[::10])
''')
markdown('''## Income-prediction evaluation
These are target income metrics, not membership metrics. The test set was not
used for model selection. Do not choose target settings using these results.
''')
code('''expanded_metrics = {name: evaluate_model(expanded_model, expanded_arrays[name]['X'],
    expanded_arrays[name]['y']) for name in ['train', 'validation', 'test']}
display(pd.DataFrame(expanded_metrics).T)
print('Train-test accuracy gap (percentage points):',
      100 * (expanded_metrics['train']['accuracy'] - expanded_metrics['test']['accuracy']))
''')
markdown('''## 16 expanded ACS 2014 shadows using the paper's three named algorithms
Keep existing RF/XGBoost settings and 40/20/40 splits.
Gradient boosting uses 300 trees, learning rate 0.05, maximum depth 3, seed per shadow.
These are project settings; the paper does not provide its exact configurations.
Each preprocessor fits its own training records.
These shadows do not assume knowledge of the target architecture.
Cycle the three algorithms using unique seeds and splits. Models may reuse auxiliary
records across shadows; grouped attack splits keep repeated records together.
''')
code('''acs_auxiliary = load_expanded_year(2014, income_base_year=INCOME_BASE_YEAR)
print('Auxiliary income adjustment:', acs_auxiliary.attrs.get('income_adjustment', NOMINAL_INCOME_METADATA))
assert set(acs_auxiliary.index).isdisjoint(expanded_source.index)
shadow_results = cached_expanded_shadows(acs_auxiliary, seed=SEED, n_models=N_SHADOW_MODELS,
    run_dir=RUN_DIR, algorithms=SHADOW_ALGORITHMS)
# Verify native-input queries reproduce probabilities computed during training.
for name, result in shadow_results.items():
    audit_rows = result['attack_data'].iloc[:32]
    native = acs_auxiliary.loc[audit_rows['record_id'], EXPANDED_FEATURES]
    query = make_probability_oracle(result['model'], result['preprocessor'], schema='acs')
    np.testing.assert_allclose(query(native),
        audit_rows[['p_income_le_50k', 'p_income_gt_50k']].to_numpy(),
        rtol=1e-5, atol=1e-6, err_msg=f'Query mismatch for {name}')
print('Probability oracle consistency verified for all 16 shadows.')
display(summarize_shadow_models(shadow_results))
''')
markdown('''## Extract shadow features and fit the random forest attack
Features: class-scaled logit, blind walk, entropy and cross-entropy. Use 2,000
members and 2,000 non-members per shadow. Attack record-ID grouping prevents
repeated 2014 records from crossing attack partitions. Select both thresholds
on auxiliary validation only; no target membership labels train this classifier.
''')
code('''shadow_attack_data, shadow_query_diagnostics = collect_shadow_paper_features(
    shadow_results, acs_auxiliary, per_class=SHADOW_RECORDS_PER_CLASS,
    config=WALK_CONFIG, seed=SEED)
display(pd.DataFrame(shadow_query_diagnostics).T)
attack_result = train_membership_classifier(
    shadow_attack_data, seed=SEED, algorithm='random_forest', feature_set='paper')
display(pd.DataFrame(attack_result['metrics']).T)
''')
markdown('''## Final MIA evaluation on the expanded target
Query the expanded native input schema, preserving categorical codes and
handling numeric commute fields in native units. The walk still generates
field-valid but potentially implausible combinations, and its dimensionality
matches the expanded auxiliary schema. Thresholds are frozen from auxiliary validation.
''')
code('''target_attack_data, target_query_diagnostics = collect_target_paper_features(
    expanded_model, expanded_preprocessor, expanded_partitions,
    config=WALK_CONFIG, seed=SEED, numeric_protocol=EXPANDED_NUMERIC_WALK)
display(pd.Series(target_query_diagnostics, name='expanded target query diagnostics'))
target_mia_metrics, target_membership_scores = evaluate_attack(
    attack_result['classifier'], target_attack_data,
    threshold=attack_result['accuracy_threshold'],
    low_fpr_threshold=attack_result['low_fpr_threshold'])
target_mia_metrics['tpr_at_accuracy_threshold'] = target_mia_metrics['tp'] / target_mia_metrics['members']
target_mia_metrics['fpr_at_accuracy_threshold'] = target_mia_metrics['fp'] / (
    target_mia_metrics['records'] - target_mia_metrics['members'])
display(pd.DataFrame({'2014 auxiliary attack test': attack_result['metrics']['test'],
                      'Expanded ACS target MIA': target_mia_metrics}).T)
''')
markdown('''## Small reproducibility artifacts
Save summaries, feature inventory and row assignments. Completed target and
shadow checkpoints are saved during training to support recovery after interruption.
Source CSVs and the other experiment notebooks remain unchanged.
''')
code('''result_dir = RUN_DIR
result_dir.mkdir(parents=True, exist_ok=True)
feature_inventory.to_csv(result_dir / 'feature_inventory.csv')
for name, part in expanded_partitions.items():
    pd.Series(part.index, name='record_id').to_csv(result_dir / f'{name}_ids.csv', index=False)
summarize_shadow_models(shadow_results).to_csv(result_dir / 'shadow_metrics.csv')
summary = dict(income_base_year=INCOME_BASE_YEAR,
    target_income_adjustment=expanded_source.attrs.get('income_adjustment', NOMINAL_INCOME_METADATA),
    auxiliary_income_adjustment=acs_auxiliary.attrs.get('income_adjustment', NOMINAL_INCOME_METADATA),
    perturbation_protocol_audit={'shadow_numeric_fields': {'AGEP': [1.0, 17, 99], 'WKHP': [2.0, 1, 99]}, 'target_extra_numeric_fields': {'JWMNP': [2.0, 0, 200], 'JWRIP': [1.0, 1, 10]}, 'shadow_commuting_fields': 'Categorical replacements, unchanged from original run', 'target_commuting_fields': 'Numeric perturbations, unchanged from original run', 'correction': 'Prior assistant statements that the shadow commuting perturbation protocol changed were incorrect. Original and current perturbation call sources are identical.'},
    walk_numeric_protocol_scope='Extra target numeric fields only',
    walk_numeric_protocol=EXPANDED_NUMERIC_WALK, execution_status='all_cells_completed',
    attack_metrics_status='oracle_consistency_verified', target_year=2023, auxiliary_year=2014, n_shadow_models=N_SHADOW_MODELS,
    shadow_algorithms=list(SHADOW_ALGORITHMS),
    shadow_seeds={name: result['seed'] for name, result in shadow_results.items()},
    original_feature_count=len(EXPANDED_FEATURES),
    encoded_feature_count=expanded_arrays['train']['X'].shape[1],
    train_size=TRAIN_SIZE, split_config=SPLIT_CONFIG, training=expanded_metadata,
    target_training_seconds=target_training_seconds, target_metrics=expanded_metrics,
    shadow_attack_metrics=attack_result['metrics'], target_mia_metrics=target_mia_metrics,
    target_query_diagnostics=target_query_diagnostics)
(result_dir / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
print('Summary and exact row IDs written to', result_dir)
''')

markdown('''## UCI Adult auxiliary comparison: load every record and train 16 shadows
Pool the original Adult training and test files as auxiliary information only.
Use all 48,842 rows: per shadow, 40% training members, 20% validation and 40%
held-out non-members. Partitions are disjoint within each shadow. Across shadows,
records can be reused; grouped attack splits prevent record leakage.
Adult uses its native 12 predictors and nominal income label, not a fabricated
mapping to the 48 ACS predictors. The same four output/query features transfer
to the same ACS target. Differences in input geometry and population limit comparison.
No Adult feature-extraction subsampling is used. This increases query cost.
''')
code('''adult_auxiliary = load_adult_data()
assert len(adult_auxiliary) == 48_842, 'Expected complete UCI Adult dataset.'
assert adult_auxiliary.index.is_unique
assert set(adult_auxiliary.index).isdisjoint(expanded_source.index)
print('Adult auxiliary records:', len(adult_auxiliary))
adult_shadow_results = cached_expanded_shadows(
    adult_auxiliary, seed=SEED, n_models=N_SHADOW_MODELS,
    run_dir=ADULT_RUN_DIR, algorithms=SHADOW_ALGORITHMS, schema='adult')
for name, result in adult_shadow_results.items():
    assigned = [set(ids) for ids in result['partition_ids'].values()]
    assert sum(map(len, assigned)) == len(adult_auxiliary)
    assert set.union(*assigned) == set(adult_auxiliary.index)
    assert all(a.isdisjoint(b) for i, a in enumerate(assigned) for b in assigned[i+1:])
    audit_rows = result['attack_data'].iloc[:32]
    query = make_probability_oracle(result['model'], result['preprocessor'], schema='adult')
    np.testing.assert_allclose(query(adult_auxiliary.loc[audit_rows['record_id']]),
        audit_rows[['p_income_le_50k', 'p_income_gt_50k']].to_numpy(), rtol=1e-5, atol=1e-6)
display(summarize_shadow_models(adult_shadow_results))
''')
markdown('''## Adult attack features: all available members and non-members
Validation rows retain their existing model-validation role. Every training member
and held-out non-member contributes attack features (per_class=None).
Use the existing native Adult perturbation protocol and unchanged RF attack.
Thresholds are selected on auxiliary attack validation only.
''')
code('''adult_attack_data, adult_query_diagnostics = collect_shadow_paper_features(
    adult_shadow_results, adult_auxiliary, per_class=None, config=WALK_CONFIG, seed=SEED)
assert len(adult_attack_data) == sum(
    len(result['attack_data']) for result in adult_shadow_results.values())
display(adult_attack_data.groupby(['shadow_model', 'membership']).size().unstack())
adult_attack_result = train_membership_classifier(
    adult_attack_data, seed=SEED, algorithm='random_forest', feature_set='paper')
display(pd.DataFrame(adult_attack_result['metrics']).T)
''')
markdown('''## Compare both auxiliary sources against the same saved target queries
Reuse target_attack_data from above: no retraining or additional target queries.
Both attacks use thresholds frozen on their own auxiliary validation data.
''')
code('''adult_target_mia_metrics, adult_target_scores = evaluate_attack(
    adult_attack_result['classifier'], target_attack_data,
    threshold=adult_attack_result['accuracy_threshold'],
    low_fpr_threshold=adult_attack_result['low_fpr_threshold'])
comparison = pd.DataFrame({
    'ACS 2014 auxiliary': target_mia_metrics,
    'UCI Adult auxiliary (all records)': adult_target_mia_metrics}).T
display(comparison)
ADULT_RUN_DIR.mkdir(parents=True, exist_ok=True)
comparison.to_csv(RUN_DIR / 'auxiliary_comparison.csv')
summarize_shadow_models(adult_shadow_results).to_csv(ADULT_RUN_DIR / 'shadow_metrics.csv')
adult_summary = dict(execution_status='all_adult_cells_completed', auxiliary='UCI Adult',
    auxiliary_records=len(adult_auxiliary), shadow_schema='adult',
    n_shadow_models=N_SHADOW_MODELS, algorithms=list(SHADOW_ALGORITHMS),
    split_fractions=dict(train=0.4, validation=0.2, nonmembers=0.4),
    feature_sampling_per_membership_class=None,
    shadow_seeds={name: result['seed'] for name, result in adult_shadow_results.items()},
    shadow_attack_metrics=adult_attack_result['metrics'],
    target_mia_metrics=adult_target_mia_metrics, query_diagnostics=adult_query_diagnostics,
    target_summary='../summary.json', target_queries_reused=True)
(ADULT_RUN_DIR / 'summary.json').write_text(json.dumps(adult_summary, indent=2), encoding='utf-8')
print('Adult comparison saved to', ADULT_RUN_DIR)
''')

notebook = {'cells': cells, 'metadata': {
    'kernelspec': {'display_name': 'Python 3 (ipykernel)', 'language': 'python', 'name': 'python3'},
    'language_info': {'name': 'python', 'version': '3.12'}}, 'nbformat': 4, 'nbformat_minor': 5}
path = ROOT / 'exp_3_expanded dataset.ipynb'
path.write_text(json.dumps(notebook, indent=1) + '\n', encoding='utf-8')
print(path)
