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
Train the target on 50,000 California 2023 records with expanded predictors.
Train 16 diverse shadows exclusively on California 2014 auxiliary records.
Each model fits its own preprocessing on its training partition only.
The label is nominal personal income >$50,000 in each year; inflation and survey
year shifts are part of this transfer experiment. Occupation and other category
codes remain native to each year; no cross-year code equivalence is assumed.
Income-derived predictors, identifiers, survey weights and allocation flags are
excluded. MIA uses logit, blind walk, entropy and cross-entropy.
Execution was authorized for this run. Ask before future cell/experiment runs.
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
from src.shadow_models import load_adult_data, train_shadow_models, summarize_shadow_models
from src.membership_attack import train_membership_classifier, evaluate_attack
from src.paper_attack_features import (collect_shadow_paper_features, collect_target_paper_features,
                                       make_probability_oracle)

torch.set_num_threads(4)
SEED = 42
TRAIN_SIZE = 50_000
SPLIT_CONFIG = reference_split_config()
assert SPLIT_CONFIG['seed'] == SEED, 'Use the reference seed consistently.'
TRAIN_CONFIG = dict(hidden_dims=(512, 256, 128), dropout=0.0, epochs=100,
    batch_size=256, learning_rate=1e-3, weight_decay=0.0, seed=SEED,
    device='cpu', early_stopping=False, restore_best=False, monitor_train=True)
WALK_CONFIG = dict(max_steps=20, repeats=2, categorical_change_probability=0.05,
                   batch_size=512, seed=SEED)
SHADOW_RECORDS_PER_CLASS = 2000
N_SHADOW_MODELS = 16
# Native commuting units; all other added codes are categorical in the walk.
EXPANDED_NUMERIC_WALK = {'JWMNP': (2.0, 0, 200), 'JWRIP': (1.0, 1, 10)}
print('Training size:', TRAIN_SIZE, '| Input fields:', len(EXPANDED_FEATURES))
print('Reference split configuration:', SPLIT_CONFIG)
''')
markdown('''## Load 2023 target records and make stratified partitions
Split sizes and seed follow Experiment 2, applied to the new 2023 records.
''')
code('''expanded_source = load_expanded_year(2023)
reference_partitions = split_acs_data(expanded_source, **SPLIT_CONFIG)
expanded_base_partitions = reference_partitions
expanded_partitions = subset_target_training(
    expanded_base_partitions, train_size=TRAIN_SIZE, seed=SEED)
for name in ['validation', 'test', 'mia_members', 'mia_nonmembers']:
    np.testing.assert_array_equal(expanded_partitions[name].index, reference_partitions[name].index)
assert set(expanded_partitions['mia_members'].index).issubset(expanded_partitions['train'].index)
for name in ['validation', 'test', 'mia_nonmembers']:
    assert set(expanded_partitions[name].index).isdisjoint(expanded_partitions['train'].index)
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
expanded_model, expanded_history, expanded_metadata = train_target_model(
    expanded_arrays['train']['X'], expanded_arrays['train']['y'],
    expanded_arrays['validation']['X'], expanded_arrays['validation']['y'], **TRAIN_CONFIG)
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
markdown('''## 16 diverse expanded ACS 2014 shadow models
Diverse algorithms/settings and 40/20/40 splits follow the
restored Experiment 2 attack. Each preprocessor fits its own training records.
These shadows do not assume knowledge of the target architecture.
Cycle the five algorithms using unique seeds and splits. Models may reuse Adult
records across shadows; grouped attack splits keep repeated records together.
''')
code('''acs_auxiliary = load_expanded_year(2014)
assert set(acs_auxiliary.index).isdisjoint(expanded_source.index)
shadow_results = train_shadow_models(acs_auxiliary, seed=SEED, n_models=N_SHADOW_MODELS, schema='expanded_acs')
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
Save only summaries, feature inventory and row assignments (no model checkpoint).
Source CSVs and the other experiment notebooks remain unchanged.
''')
code('''result_dir = Path('artifacts/exp_3_2023_2014')
result_dir.mkdir(parents=True, exist_ok=True)
feature_inventory.to_csv(result_dir / 'feature_inventory.csv')
for name, part in expanded_partitions.items():
    pd.Series(part.index, name='record_id').to_csv(result_dir / f'{name}_ids.csv', index=False)
summarize_shadow_models(shadow_results).to_csv(result_dir / 'shadow_metrics.csv')
summary = dict(execution_status='all_cells_completed',
    attack_metrics_status='oracle_consistency_verified', target_year=2023, auxiliary_year=2014, n_shadow_models=N_SHADOW_MODELS,
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

notebook = {'cells': cells, 'metadata': {
    'kernelspec': {'display_name': 'Python 3 (ipykernel)', 'language': 'python', 'name': 'python3'},
    'language_info': {'name': 'python', 'version': '3.12'}}, 'nbformat': 4, 'nbformat_minor': 5}
path = ROOT / 'exp_3_expanded dataset.ipynb'
path.write_text(json.dumps(notebook, indent=1) + '\n', encoding='utf-8')
print(path)
