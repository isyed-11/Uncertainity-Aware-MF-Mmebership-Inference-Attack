"""Create a separate auxiliary-attack comparison notebook; never execute it."""
import ast
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DESTINATION = ROOT / 'exp_4_auxiliary_attack_baseline.ipynb'
notebook = dict(nbformat=4, nbformat_minor=5, cells=[], metadata=dict(
    kernelspec=dict(display_name='Python 3 (ipykernel)', language='python', name='python3'),
    language_info=dict(name='python', version='3.12')))


def add(kind, source):
    cell = dict(cell_type=kind, id=f'auxiliary-baseline-{len(notebook["cells"])}',
        metadata={}, source=source.splitlines(keepends=True))
    if kind == 'code':
        ast.parse(source)
        cell.update(execution_count=None, outputs=[])
    notebook['cells'].append(cell)


add('markdown', '''# Experiment 4 follow-up: auxiliary attack selection and baseline MIA
Improve the auxiliary attack comparison while retaining the generalization-focused
non-private ACS 2023 target from Experiment 4. Use ACS 2014 auxiliary data, 48
expanded inputs, nominal PINCP >$50,000 labels, 100,000 target training records,
16 shadows and the same fixed target/MIA splits. Target training remains an MLP
with hidden layers (512,256,128), dropout=0.1, weight decay=1e-4, Adam lr=1e-3,
30 maximum epochs, patience=5 and best-public-validation-loss weights. No extra
overfitting is introduced to obtain a desired attack result.

Predeclare **84 auxiliary attack candidates**:

| Comparison | Settings |
|---|---|
| Shadow pools | All 16; RF-only (6); gradient-boosting-only (5); XGBoost-only (5) |
| Fixed loss score | One loss-threshold baseline per pool (4) |
| Learned attack features | Four paper-inspired features; the same features without blind walk |
| Logistic attack | StandardScaler + logistic regression, C=1 (8 candidates) |
| RF attack | 300 trees; depth 4/8/12; minimum leaf 20/50/100 (72 candidates) |

Fit on auxiliary attack training records only; calibrate thresholds and select
using auxiliary validation only. Preserve one master record-ID role assignment
across pools and feature sets. Rank by validation ROC-AUC, then balanced accuracy,
then predeclared candidate order. Keep the original pooled four-feature RF
(depth=12, leaf=20) as a reference. Only after freezing selection, evaluate that
reference and the selected attack on their auxiliary test pools and the same
target MIA records. Comparisons use different pool sizes, not matched-size pools.

Extract four features once per shadow and once for the target; the no-walk
ablation drops the already extracted walk column and needs no new queries.
The loss score is exp(-cross_entropy), equivalent to thresholding loss, and is
not a calibrated membership probability. Income labels are known to the attacker.
MLP shadows are an optional later architecture-informed comparison and are not
included here. This notebook ends at **complete-feature baseline MIA**: no DP
training, masking, PFV, reconstruction or uncertainty aggregation.

Existing Exp 4 auxiliary test and target results have already been inspected,
so this is an exploratory follow-up, not fresh confirmatory evidence. No method
is chosen using target membership labels. No checkpoints, models, logs or
intermediate feature files are loaded or saved. Final summaries, record roles
and protocol are exported. Run cells only after explicit user approval.
''')
add('markdown', '''## Configuration and optional reuse of an existing Exp 4 kernel
The default is a standalone run. If the original DP notebook's kernel still
contains its data, shadow feature table, frozen attack and completed protocol,
connect this notebook to **that same kernel** and set `REUSE_EXP4_CONTEXT=True`.
Separate kernels do not share variables. Reuse skips shadow fitting/querying
and preserves the original attack's record-group assignment. It does not load
files or reuse DP target models. The non-private target is trained fresh with
the same settings. If context is missing or incompatible, the reuse check fails
before training instead of silently combining different experiments.
''')
add('code', '''from pathlib import Path
from importlib.util import find_spec
import csv
import gc
import importlib.metadata
import json
import math
import numpy as np
import pandas as pd
import torch
from IPython.display import display
from sklearn.model_selection import train_test_split
from src.data import split_acs_data, subset_target_training, TARGET
from src.expanded_data import EXPANDED_FEATURES, load_expanded_year, reference_split_config
from src.dp_experiment import prepare_public_preprocessing
from src.training import train_target_model, evaluate_model
from src.shadow_models import train_shadow_model, summarize_shadow_models, PAPER_SHADOW_ALGORITHMS
from src.paper_attack_features import collect_shadow_paper_features, collect_target_paper_features
from src.auxiliary_attack_selection import (
    POOLS, candidate_specs, master_record_roles, fixed_pool_splits, pool_frame,
    select_auxiliary_attack, evaluate_frozen_attack, validation_feature_diagnostics,
)
from src.notebook_output import CellProgress, quiet_output, suppress_sklearn_parallel_warning

suppress_sklearn_parallel_warning()
torch.set_num_threads(4)
SEED = 42
TARGET_YEAR = 2023
AUXILIARY_YEAR = 2014
TRAIN_SIZE = 100_000
PUBLIC_PREPROCESSING_SIZE = 10_000
N_SHADOW_MODELS = 16
SHADOW_RECORDS_PER_CLASS = 2000
SHADOW_ALGORITHMS = PAPER_SHADOW_ALGORITHMS
REUSE_EXP4_CONTEXT = False
SPLIT_CONFIG = reference_split_config()
assert SPLIT_CONFIG['seed'] == SEED
assert len(EXPANDED_FEATURES) == 48
TRAIN_CONFIG = dict(hidden_dims=(512, 256, 128), dropout=0.1, epochs=30,
    patience=5, batch_size=256, learning_rate=1e-3, weight_decay=1e-4,
    seed=SEED, device='cpu', early_stopping=True, restore_best=True, monitor_train=False)
WALK_CONFIG = dict(max_steps=20, repeats=2, categorical_change_probability=0.05,
    batch_size=512, seed=SEED)
EXPANDED_NUMERIC_WALK = {'JWMNP': (2.0, 0, 200), 'JWRIP': (1.0, 1, 10)}
RUN_DIR = Path('artifacts/exp_4_2023_2014_auxiliary_attack_baseline')
if find_spec('xgboost') is None:
    raise ImportError('Install in this kernel environment: python -m pip install -r requirements.txt')
versions = {name: importlib.metadata.version(name) for name in
    ['torch', 'numpy', 'pandas', 'scikit-learn', 'xgboost']}
for year, filename in [(TARGET_YEAR, 'psam_p06.csv'), (AUXILIARY_YEAR, 'ss14pca.csv')]:
    path = Path('data/raw/folktables') / str(year) / '1-Year' / filename
    with path.open(newline='', encoding='utf-8-sig') as handle:
        header = set(next(csv.reader(handle)))
    assert set(EXPANDED_FEATURES + ['PINCP', 'PWGTP']) <= header, f'Missing source columns: {path}'
print('Candidates:', len(candidate_specs()), '| Reuse existing Exp 4 kernel:', REUSE_EXP4_CONTEXT)
print('Target epoch cap:', TRAIN_CONFIG['epochs'], '| No private target training in this notebook.')
''')
add('markdown', '''## Prepare fixed data and public preprocessing
Use the same 10,000-row public 2014 preprocessing subset as Exp 4, disjoint
from the 2014 shadow pool. Freeze imputation, scaling, missing-indicator choices
and category vocabulary before transforming target records. The fixed 2023
validation partition is public and selects the target epoch only. Test and MIA
nonmember records do not select target settings. Native category codes remain
year-specific; this experiment does not introduce a cross-year code mapping.
''')
add('code', '''if REUSE_EXP4_CONTEXT:
    required = ['target_partitions', 'target_preprocessor', 'target_arrays',
        'public_preprocessing', 'shadow_auxiliary', 'shadow_attack_data',
        'shadow_income_metrics', 'shadow_query_audits', 'attack_result', 'protocol']
    missing = [name for name in required if name not in globals()]
    if missing:
        raise ValueError('Connect to the completed Exp 4 kernel or disable reuse. Missing: ' + ', '.join(missing))
    assert protocol.get('target_year') == TARGET_YEAR and protocol.get('auxiliary_year') == AUXILIARY_YEAR
    assert protocol.get('feature_count') == 48 and protocol.get('train_size') == TRAIN_SIZE
    assert protocol.get('n_shadow_models') == N_SHADOW_MODELS
    assert protocol.get('public_preprocessing_records') == PUBLIC_PREPROCESSING_SIZE
    assert protocol.get('walk_config') == WALK_CONFIG
    assert protocol.get('include_blind_walk') is True
    assert protocol.get('records_per_class_per_shadow') == SHADOW_RECORDS_PER_CLASS
    assert protocol.get('extra_numeric_protocol') == EXPANDED_NUMERIC_WALK
    assert protocol.get('split_config') == SPLIT_CONFIG
    prior_training = protocol['training_config']
    for name in ['hidden_dims', 'dropout', 'batch_size', 'learning_rate', 'weight_decay', 'seed', 'device', 'patience']:
        assert prior_training[name] == TRAIN_CONFIG[name], f'Incompatible target setting: {name}'
    assert prior_training['max_epochs'] == TRAIN_CONFIG['epochs'] and prior_training['min_delta'] == 0.
    master_roles = {name: set(part['record_id']) for name, part in attack_result['splits'].items()}
    print('Reusing in-memory Exp 4 data/features and the existing master record roles.')
else:
    source = load_expanded_year(TARGET_YEAR, income_base_year=None)
    reference_partitions = split_acs_data(source, **SPLIT_CONFIG)
    target_partitions = subset_target_training(reference_partitions, train_size=TRAIN_SIZE, seed=SEED)
    for name in ['validation', 'test', 'mia_members', 'mia_nonmembers']:
        np.testing.assert_array_equal(target_partitions[name].index, reference_partitions[name].index)
    auxiliary = load_expanded_year(AUXILIARY_YEAR, income_base_year=None)
    shadow_auxiliary, public_preprocessing = train_test_split(auxiliary,
        test_size=PUBLIC_PREPROCESSING_SIZE, stratify=auxiliary[TARGET], random_state=SEED)
    shadow_auxiliary = shadow_auxiliary.sort_index().copy()
    public_preprocessing = public_preprocessing.sort_index().copy()
    target_preprocessor, target_arrays = prepare_public_preprocessing(public_preprocessing, target_partitions)
    del source, auxiliary, reference_partitions

assert set(shadow_auxiliary.index).isdisjoint(public_preprocessing.index)
assert len(target_partitions['train']) == TRAIN_SIZE
assert len(target_partitions['mia_members']) == len(target_partitions['mia_nonmembers'])
assert list(target_preprocessor.feature_names_in_) == EXPANDED_FEATURES
for name, values in target_arrays.items():
    np.testing.assert_array_equal(values['record_ids'], target_partitions[name].index)
    assert values['X'].shape[1] == target_arrays['train']['X'].shape[1]
    assert np.isfinite(values['X'].data).all()
display(pd.Series({name: len(frame) for name, frame in target_partitions.items()}, name='target_records'))
gc.collect()
''')
add('markdown', '''## Extract shadow features once, then freeze the master roles
Retain the existing 16 shadows, seeds 42+10*i, tree-estimator settings and
40/20/40 partitions. Sample 2,000 members and 2,000 nonmembers per shadow.
Train/query one shadow at a time and release it after extracting four features.
When reuse is enabled this expensive stage is skipped. Roles are assigned once
on the pooled feature table (or inherited from Exp 4), never after filtering a
family. A record retains its role even when observed under multiple shadows.
''')
add('code', '''def fit_and_query_baseline_shadow(auxiliary, name, algorithm, seed):
    with quiet_output():
        shadow = train_shadow_model(auxiliary, algorithm, seed=seed, schema='expanded_acs')
        shadow['attack_data']['shadow_model'] = name
        features, audits = collect_shadow_paper_features({name: shadow}, auxiliary,
            per_class=SHADOW_RECORDS_PER_CLASS, config=WALK_CONFIG, seed=SEED,
            include_blind_walk=True, numeric_protocol=EXPANDED_NUMERIC_WALK)
    return features, audits[name], summarize_shadow_models({name: shadow})

if not REUSE_EXP4_CONTEXT:
    progress = CellProgress()
    feature_parts, summary_parts, shadow_query_audits = [], [], {}
    for index in range(N_SHADOW_MODELS):
        algorithm = SHADOW_ALGORITHMS[index % len(SHADOW_ALGORITHMS)]
        name = f'{algorithm}_{index + 1:02d}'
        progress.update(f'Shadow {index + 1}/{N_SHADOW_MODELS}: {name}')
        features, audit, summary = fit_and_query_baseline_shadow(shadow_auxiliary,
            name, algorithm, SEED + 10 * index)
        feature_parts.append(features)
        summary_parts.append(summary)
        shadow_query_audits[name] = audit
        del features, audit, summary
        gc.collect()
    shadow_attack_data = pd.concat(feature_parts, ignore_index=True)
    shadow_income_metrics = pd.concat(summary_parts)
    master_roles = master_record_roles(shadow_attack_data, seed=SEED)
    del feature_parts, summary_parts
    progress.update('Shadow features extracted; master record roles frozen')

expected_names = {f'{SHADOW_ALGORITHMS[i % len(SHADOW_ALGORITHMS)]}_{i + 1:02d}' for i in range(N_SHADOW_MODELS)}
assert set(shadow_attack_data['shadow_model']) == expected_names
assert len(shadow_attack_data) == N_SHADOW_MODELS * 2 * SHADOW_RECORDS_PER_CLASS
for pool in POOLS:
    parts = fixed_pool_splits(pool_frame(shadow_attack_data, pool), master_roles)
    assert all(set(frame['record_id']) <= master_roles[name] for name, frame in parts.items())
print('Feature rows:', len(shadow_attack_data), '| Master record groups:', {k: len(v) for k, v in master_roles.items()})
''')
add('markdown', '''## Diagnose membership signals before tuning
Report shadow income generalization gaps and auxiliary-validation-only attack
feature distributions. Univariate score directions are predeclared: larger
logit/walk, lower entropy/loss are treated as member-like, without flipping a
direction using test or target labels. Full shadow walk censoring is a descriptive
query audit; it does not choose a candidate. A gap alone does not guarantee MIA.
''')
add('code', '''income_wide = shadow_income_metrics.reset_index().pivot(index='shadow_model', columns='partition', values='accuracy')
income_wide['family'] = income_wide.index.to_series().str.rsplit('_', n=1).str[0]
income_wide['gap'] = income_wide['train'] - income_wide['nonmembers']
display(income_wide.groupby('family')[['train', 'nonmembers', 'gap']].mean().round(4))
feature_distributions, univariate_validation = validation_feature_diagnostics(shadow_attack_data, master_roles)
display(univariate_validation.pivot(index='pool', columns='feature', values='validation_roc_auc').round(4))
walk_audit = pd.DataFrame(shadow_query_audits).T
display(walk_audit[['record_queries', 'walk_censored_fraction']])
''')
add('markdown', '''## Select the attack using auxiliary validation only
Fit 84 candidates sequentially. This stage performs no additional oracle queries
and does not evaluate auxiliary test or target data. Scaling for logistic
regression fits auxiliary training records only. All thresholds are selected
on each candidate's auxiliary validation pool, using the existing balanced-accuracy
cutoff and 1%-FPR calibration procedure. RF estimators retain 300 trees,
max_features='sqrt', n_jobs=2 and seed=42. Only depth and minimum leaf size vary.
No target settings are tuned here. Keep just the overall winner and original
pooled RF reference in memory; do not retain 72 fitted forests.
''')
add('code', '''selection_progress = CellProgress()
with quiet_output():
    attack_comparison = select_auxiliary_attack(shadow_attack_data, master_roles,
        seed=SEED, progress=selection_progress.update)
validation_comparison = attack_comparison['validation_table']
assert len(validation_comparison) == 84
selected_attack = attack_comparison['selected']
reference_attack = attack_comparison['reference']
selected_candidate = selected_attack['spec']['candidate']
assert selected_candidate == validation_comparison.iloc[0]['candidate']
selection_progress.update('Selection frozen: ' + selected_candidate)
columns = ['candidate', 'train_roc_auc', 'validation_roc_auc', 'validation_accuracy', 'validation_balanced_accuracy']
display(validation_comparison[columns].head(10).round(4))
best_by_pool_features = validation_comparison.drop_duplicates(['pool', 'feature_set'])
display(best_by_pool_features[columns].round(4))
print('Original reference:', reference_attack['spec']['candidate'])
''')
add('markdown', '''## Auxiliary test evaluation after selection is frozen
Evaluate only the selected candidate and original RF reference on their own
pool's test rows. Pool sizes and shadow families differ, so these test samples
are not identical across family choices. They share the same master record-ID
assignment. Selection is already complete; these numbers cannot choose another
candidate. The test data are reused research data, not a new confirmation set.
''')
add('code', '''auxiliary_test_rows = []
for label, result in [('selected', selected_attack), ('original_reference', reference_attack)]:
    pool = result['spec']['pool']
    test = fixed_pool_splits(pool_frame(shadow_attack_data, pool), master_roles)['test']
    metrics, _ = evaluate_frozen_attack(result, test)
    auxiliary_test_rows.append(dict(method=label, candidate=result['spec']['candidate'], pool=pool, **metrics))
auxiliary_test_metrics = pd.DataFrame(auxiliary_test_rows)
display(auxiliary_test_metrics[['method', 'candidate', 'records', 'accuracy', 'balanced_accuracy', 'roc_auc',
    'tpr_at_1pct_fpr_roc']].round(4))
''')
add('markdown', '''## Train the unchanged non-private target and measure utility
Train once from scratch, with the same generalization-focused settings as the
non-private Exp 4 control. Public validation selects the returned epoch; test
and membership records do not. This is not the historical 100-epoch target.
Attack selection above is frozen before any target scores are inspected.
''')
add('code', '''target_progress = CellProgress()
target_progress.update('Training one non-private target; at most 30 epochs with early stopping')
with quiet_output():
    baseline_target_model, target_history, target_training = train_target_model(
        target_arrays['train']['X'], target_arrays['train']['y'],
        target_arrays['validation']['X'], target_arrays['validation']['y'], **TRAIN_CONFIG)
target_income_metrics = {name: evaluate_model(baseline_target_model,
    target_arrays[name]['X'], target_arrays[name]['y']) for name in ['train', 'validation', 'test']}
target_progress.update(f'Target ready; returned validation-selected epoch {target_training["returned_epoch"]}')
display(pd.DataFrame(target_income_metrics).T.round(4))
print('Train-minus-test accuracy gap:', round(target_income_metrics['train']['accuracy'] - target_income_metrics['test']['accuracy'], 4))
''')
add('markdown', '''## Complete-feature target baseline MIA
Extract the four original features **once** on the fixed balanced target
member/nonmember sample. Both frozen attacks consume this same query table,
selecting their own one/three/four columns internally. Use each attack's frozen
auxiliary-validation cutoffs. Target membership labels score results only.
There is no target-based threshold adjustment or candidate reselection.
''')
add('code', '''target_progress.update('Extracting complete-feature target attack inputs once')
with quiet_output():
    target_attack_features, target_query_audit = collect_target_paper_features(
        baseline_target_model, target_preprocessor, target_partitions,
        config=WALK_CONFIG, seed=SEED, include_blind_walk=True,
        numeric_protocol=EXPANDED_NUMERIC_WALK)
target_mia_rows = []
for label, result in [('selected', selected_attack), ('original_reference', reference_attack)]:
    metrics, _ = evaluate_frozen_attack(result, target_attack_features)
    assert metrics['records'] == 2 * metrics['members']
    target_mia_rows.append(dict(method=label, candidate=result['spec']['candidate'], **metrics))
baseline_mia_metrics = pd.DataFrame(target_mia_rows)
target_progress.update('Baseline MIA completed; no DP or missing-feature sections follow')
display(baseline_mia_metrics[['method', 'candidate', 'accuracy', 'balanced_accuracy', 'roc_auc',
    'tpr_at_1pct_fpr_roc', 'tpr_at_shadow_threshold', 'fpr_at_shadow_threshold']].round(4))
''')
add('markdown', '''## Save final summaries and protocol
Export the complete validation grid, two frozen auxiliary-test and target
comparisons, feature diagnostics, shadow/target income metrics, target history,
master auxiliary role IDs and protocol. Fitted models and intermediate feature
tables stay in memory. No log files or checkpoints are written. A new kernel
requires recomputation unless an existing Exp 4 context is explicitly reused.
An increase is not guaranteed; near-chance results remain valid exploratory findings.
''')
add('code', '''def json_value(value):
    if isinstance(value, dict):
        return {str(key): json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(item) for item in value]
    if isinstance(value, np.generic):
        return json_value(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    return value

RUN_DIR.mkdir(parents=True, exist_ok=True)
for filename, frame in {
    'validation_candidate_grid.csv': validation_comparison,
    'auxiliary_test_metrics.csv': auxiliary_test_metrics,
    'baseline_mia_metrics.csv': baseline_mia_metrics,
    'validation_feature_distributions.csv': feature_distributions,
    'validation_univariate_auc.csv': univariate_validation,
}.items():
    frame.to_csv(RUN_DIR / filename, index=False)
shadow_income_metrics.to_csv(RUN_DIR / 'shadow_income_metrics.csv')
pd.DataFrame(target_income_metrics).T.to_csv(RUN_DIR / 'target_income_metrics.csv')
pd.DataFrame(target_history).to_csv(RUN_DIR / 'target_training_history.csv', index=False)
role_rows = [dict(record_id=record_id, role=role) for role, ids in master_roles.items() for record_id in sorted(ids)]
pd.DataFrame(role_rows).to_csv(RUN_DIR / 'master_auxiliary_roles.csv', index=False)
attack_protocol = dict(execution_status='baseline_mia_completed', exploratory=True,
    target_year=TARGET_YEAR, auxiliary_year=AUXILIARY_YEAR, train_size=TRAIN_SIZE,
    features=EXPANDED_FEATURES, label='nominal PINCP > 50000', split_config=SPLIT_CONFIG,
    public_preprocessing_records=PUBLIC_PREPROCESSING_SIZE,
    preprocessing='Frozen public 2014 preprocessing subset; same as Exp 4 non-private control',
    target_config=TRAIN_CONFIG, target_training=target_training, target_income_metrics=target_income_metrics,
    reused_exp4_context=REUSE_EXP4_CONTEXT, shadow_count=N_SHADOW_MODELS,
    shadow_algorithms=SHADOW_ALGORITHMS, shadow_seeds={name: SEED + 10*i for i, name in enumerate(
        [f'{SHADOW_ALGORITHMS[j % len(SHADOW_ALGORITHMS)]}_{j + 1:02d}' for j in range(N_SHADOW_MODELS)])},
    candidate_count=len(candidate_specs()), candidates=candidate_specs(),
    selection='Auxiliary validation ROC-AUC, balanced accuracy, then predeclared candidate order',
    role_assignment='One master grouped record-ID assignment reused across pools and feature subsets',
    selected=dict(selected_attack['spec'], accuracy_threshold=selected_attack['accuracy_threshold'],
        low_fpr_threshold=selected_attack['low_fpr_threshold']),
    reference=dict(reference_attack['spec'], accuracy_threshold=reference_attack['accuracy_threshold'],
        low_fpr_threshold=reference_attack['low_fpr_threshold']),
    walk_config=WALK_CONFIG, extra_numeric_protocol=EXPANDED_NUMERIC_WALK,
    walk_scope='Consistent shadows and target; one four-feature extraction, no new queries for column ablation',
    shadow_query_audits=shadow_query_audits, target_query_audit=target_query_audit,
    versions=versions, checkpoints_enabled=False, logging_enabled=False,
    dp_training=False, missing_feature_experiments=False, mlp_shadows=False,
    limitations=['Existing auxiliary test and target results already inspected',
        'Unequal pool sizes and shadow families', 'One target/attack seed',
        'Native year-specific codes and auxiliary vocabulary', 'No guarantee of improved MIA'])
_ = (RUN_DIR / 'protocol.json').write_text(json.dumps(json_value(attack_protocol), indent=2, allow_nan=False), encoding='utf-8')
print('Baseline MIA only. Final summaries:', RUN_DIR)
''')

DESTINATION.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + '\n', encoding='utf-8')
print(f'Prepared {DESTINATION.name}: {len(notebook["cells"])} cells; none executed.')
