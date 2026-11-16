"""Write Experiment 4 without executing notebook cells or experiments."""
import ast
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DESTINATION = ROOT / 'exp_4_differential_privacy.ipynb'
notebook = dict(nbformat=4, nbformat_minor=5, cells=[], metadata=dict(
    kernelspec=dict(display_name='Python 3 (ipykernel)', language='python', name='python3'),
    language_info=dict(name='python', version='3.12')))


def add(kind, source):
    cell = dict(cell_type=kind, id=f'exp4-dp-{len(notebook["cells"])}', metadata={},
        source=source.splitlines(keepends=True))
    if kind == 'code':
        ast.parse(source)
        cell.update(execution_count=None, outputs=[])
    notebook['cells'].append(cell)


add('markdown', '''# Experiment 4: differential privacy and complete-feature MIA
Compare one fresh non-private ACS 2023 MLP with private targets at requested
privacy budgets **epsilon = 1, 5, 10, 25, 50**, holding **delta = 1e-6** constant.
Use 100,000 target training records, the original 48 expanded inputs and nominal
PINCP >$50,000 labels. ACS 2014 supplies public preprocessing data and 16
separate shadow models. No features are masked and there is no PFV, reconstruction,
K-completion scoring or uncertainty aggregation in this experiment.

The matched targets use the same architecture, dropout=0.1, weight decay=1e-4,
Adam learning rate=1e-3, and a **30-epoch maximum**. Stop after five epochs without
public-validation-loss improvement and restore the best validation epoch.
Actual stopping epochs may differ. This is a generalization-focused study,
separate from the historical 100-epoch overfitting configuration and its 62.625%
MIA accuracy. Low generalization gap and high accuracy are measured, not promised.

Both requested graphs use **equally spaced epsilon categories**, not a continuous
numeric axis. A horizontal non-private reference avoids treating infinity as a
numeric budget. CSV results retain both requested budgets and accountant results.

Run cells in order only after explicit approval. Creating or validating this
notebook does not authorize or start training. No checkpoints, model files,
intermediate predictions or log files are loaded or saved. Final metrics,
training histories, protocol and PNG/SVG graphs are exported. Only the known
sklearn parallel warning is suppressed. The three known DP research warnings
(non-secure RNG, RDP alpha boundary and backward hooks) each appear once per run;
other warnings and errors remain visible. Warning display does not change DP settings.
''')
add('markdown', '''## Dependencies and fixed study configuration
Use the project virtual environment. If the dependency check below fails, install
`python -m pip install -r requirements.txt` in that environment before running.
Installation is separate from executing this notebook.

Private training uses Opacus per-example gradient clipping and Gaussian noise
with Adam (DP-Adam), Poisson sampling and an RDP accountant. Physical batches
are capped at 32 to limit per-sample-gradient memory; logical batches remain 256.
The existing four paper-inspired attack features, including blind walks, are
retained with consistent numeric perturbations for shadows and targets.
''')
add('code', '''from pathlib import Path
from importlib.util import find_spec
import csv
import gc
import importlib.metadata
import json
import math

missing_dependencies = [name for name in ['opacus', 'matplotlib', 'xgboost'] if find_spec(name) is None]
if missing_dependencies:
    raise ImportError('Missing ' + ', '.join(missing_dependencies)
        + '. Install in this notebook kernel: python -m pip install -r requirements.txt')

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from IPython.display import display
from sklearn.model_selection import train_test_split
from src.data import split_acs_data, subset_target_training, TARGET
from src.expanded_data import EXPANDED_FEATURES, load_expanded_year, reference_split_config
from src.dp_experiment import (
    prepare_public_preprocessing, train_dp_study_target, plot_budget_accuracy,
)
from src.training import evaluate_model
from src.shadow_models import train_shadow_model, summarize_shadow_models, PAPER_SHADOW_ALGORITHMS
from src.membership_attack import train_membership_classifier, evaluate_attack
from src.paper_attack_features import collect_shadow_paper_features, collect_target_paper_features
from src.notebook_output import CellProgress, quiet_output, suppress_sklearn_parallel_warning

suppress_sklearn_parallel_warning()
torch.set_num_threads(4)
SEED = 42
TARGET_YEAR = 2023
AUXILIARY_YEAR = 2014
TRAIN_SIZE = 100_000
PUBLIC_PREPROCESSING_SIZE = 10_000
EPSILON_BUDGETS = (1, 5, 10, 25, 50)
DELTA = 1e-6
assert 0 < DELTA < 1 / TRAIN_SIZE
assert len(EXPANDED_FEATURES) == 48
N_SHADOW_MODELS = 16
SHADOW_RECORDS_PER_CLASS = 2000
SHADOW_ALGORITHMS = PAPER_SHADOW_ALGORITHMS
SPLIT_CONFIG = reference_split_config()
assert SPLIT_CONFIG['seed'] == SEED
TRAIN_CONFIG = dict(hidden_dims=(512, 256, 128), dropout=0.1,
    max_epochs=30, patience=5, min_delta=0.0, batch_size=256,
    max_physical_batch_size=32, learning_rate=1e-3, weight_decay=1e-4,
    max_grad_norm=1.0, seed=SEED, device='cpu')
WALK_CONFIG = dict(max_steps=20, repeats=2, categorical_change_probability=0.05,
    batch_size=512, seed=SEED)
EXPANDED_NUMERIC_WALK = {'JWMNP': (2.0, 0, 200), 'JWRIP': (1.0, 1, 10)}
RUN_DIR = Path('artifacts/exp_4_2023_2014_dp_e1_5_10_25_50_d1e6')

for year, filename in [(TARGET_YEAR, 'psam_p06.csv'), (AUXILIARY_YEAR, 'ss14pca.csv')]:
    raw_path = Path('data/raw/folktables') / str(year) / '1-Year' / filename
    with raw_path.open(newline='', encoding='utf-8-sig') as handle:
        columns = set(next(csv.reader(handle)))
    missing_columns = set(EXPANDED_FEATURES + ['PINCP', 'PWGTP']) - columns
    if missing_columns:
        raise ValueError(f'{raw_path}: missing columns {sorted(missing_columns)}')

versions = {name: importlib.metadata.version(name) for name in
    ['torch', 'opacus', 'numpy', 'pandas', 'scikit-learn', 'xgboost', 'matplotlib']}
print('Budgets:', EPSILON_BUDGETS, '| Constant delta:', DELTA)
print('Epoch cap:', TRAIN_CONFIG['max_epochs'], '| Patience:', TRAIN_CONFIG['patience'])
print('Complete-feature MIA; four features including blind walks; no reconstruction.')
display(pd.Series(versions, name='package_version'))
''')
add('markdown', '''## Public preprocessing and fixed target partitions
Make the same stratified target splits and 100,000-record training subset as
Exp 3. For this research experiment, the fixed 2023 validation partition is
explicitly **public**, outside the protected target training set; it selects
epochs only. Test and MIA nonmember records never select models or thresholds.
The privacy statement is conditional on this fixed partition and concerns
record-level target-training membership, not person-level or survey-selection privacy.

Reserve 10,000 public 2014 rows for preprocessing, disjoint from shadow records.
Learn all imputation/scaling statistics, missing-indicator choices and category
vocabularies on those public rows only, then freeze them for all six targets.
There is no target-fitted preprocessing. The auxiliary data and validation data
are assumed available independently of the protected training records. Native
category codes remain year-specific; public 2014 vocabulary may not cover all
2023 categories. This limitation applies equally to every target here.
''')
add('code', '''target_source = load_expanded_year(TARGET_YEAR, income_base_year=None)
reference_partitions = split_acs_data(target_source, **SPLIT_CONFIG)
target_partitions = subset_target_training(reference_partitions, train_size=TRAIN_SIZE, seed=SEED)
for name in ['validation', 'test', 'mia_members', 'mia_nonmembers']:
    np.testing.assert_array_equal(target_partitions[name].index, reference_partitions[name].index)
assert len(target_partitions['train']) == TRAIN_SIZE
assert len(target_partitions['mia_members']) == len(target_partitions['mia_nonmembers'])

auxiliary_source = load_expanded_year(AUXILIARY_YEAR, income_base_year=None)
shadow_auxiliary, public_preprocessing = train_test_split(auxiliary_source,
    test_size=PUBLIC_PREPROCESSING_SIZE, stratify=auxiliary_source[TARGET], random_state=SEED)
shadow_auxiliary = shadow_auxiliary.sort_index().copy()
public_preprocessing = public_preprocessing.sort_index().copy()
assert set(shadow_auxiliary.index).isdisjoint(public_preprocessing.index)
assert set(auxiliary_source.index).isdisjoint(target_source.index)
target_preprocessor, target_arrays = prepare_public_preprocessing(public_preprocessing, target_partitions)
partition_counts = {name: len(frame) for name, frame in target_partitions.items()}
display(pd.Series({**partition_counts, 'public_2014_preprocessing': len(public_preprocessing),
    'public_2014_shadow_pool': len(shadow_auxiliary)}, name='records'))
print('Encoded input dimensions shared by all targets:', target_arrays['train']['X'].shape[1])
del target_source, reference_partitions, auxiliary_source
gc.collect()
''')
add('markdown', '''## Fit the complete-feature auxiliary attack once
Train 16 public 2014 shadows (6 RF, 5 gradient boosting, 5 XGBoost), keeping their
existing estimator settings and 40/20/40 splits. Each shadow fits its own
training-only preprocessor. Query 2,000 members and 2,000 nonmembers per shadow.
Keep repeated auxiliary IDs in one grouped attack train/validation/test role.
Train one RF membership classifier and freeze its auxiliary-selected thresholds
before evaluating any target. Shadow models are released after feature extraction.

This is a **frozen transfer-attack evaluation** across privacy budgets. The
shadows are not DP MLPs matched to each budget. A weak transferred attack alone
does not establish resistance to an attacker adapted to DP training.
''')
add('code', '''def fit_and_query_shadow(auxiliary, name, algorithm, seed):
    with quiet_output():
        result = train_shadow_model(auxiliary, algorithm, seed=seed, schema='expanded_acs')
        result['attack_data']['shadow_model'] = name
        features, audits = collect_shadow_paper_features({name: result}, auxiliary,
            per_class=SHADOW_RECORDS_PER_CLASS, config=WALK_CONFIG, seed=SEED,
            include_blind_walk=True, numeric_protocol=EXPANDED_NUMERIC_WALK)
    summary = summarize_shadow_models({name: result})
    return features, audits[name], summary

shadow_progress = CellProgress()
shadow_feature_parts, shadow_summary_parts, shadow_query_audits, shadow_seeds = [], [], {}, {}
for index in range(N_SHADOW_MODELS):
    algorithm = SHADOW_ALGORITHMS[index % len(SHADOW_ALGORITHMS)]
    name = f'{algorithm}_{index + 1:02d}'
    shadow_seeds[name] = SEED + 10 * index
    shadow_progress.update(f'Shadow {index + 1}/{N_SHADOW_MODELS}: training/querying {name}')
    features, audit, summary = fit_and_query_shadow(shadow_auxiliary, name, algorithm, shadow_seeds[name])
    shadow_feature_parts.append(features)
    shadow_summary_parts.append(summary)
    shadow_query_audits[name] = audit
    del features, audit, summary
    gc.collect()

shadow_attack_data = pd.concat(shadow_feature_parts, ignore_index=True)
shadow_income_metrics = pd.concat(shadow_summary_parts)
assert len(shadow_attack_data) == N_SHADOW_MODELS * 2 * SHADOW_RECORDS_PER_CLASS
assert shadow_attack_data['shadow_model'].nunique() == N_SHADOW_MODELS
with quiet_output():
    attack_result = train_membership_classifier(shadow_attack_data, seed=SEED,
        algorithm='random_forest', feature_set='paper')
split_ids = {name: set(frame['record_id']) for name, frame in attack_result['splits'].items()}
assert split_ids['train'].isdisjoint(split_ids['validation'] | split_ids['test'])
assert split_ids['validation'].isdisjoint(split_ids['test'])
auxiliary_attack_metrics = pd.DataFrame(attack_result['metrics']).T
shadow_progress.update('All shadows scored; auxiliary attack and thresholds frozen')
display(auxiliary_attack_metrics[['accuracy', 'balanced_accuracy', 'roc_auc', 'tpr_at_1pct_fpr_roc']].round(4))
del shadow_feature_parts, shadow_summary_parts
gc.collect()
''')
add('markdown', '''## Train and evaluate one target at a time
The helper below defines the work; the following run cell performs it. Each
model starts from scratch with the same architecture/initialization seed and
regularization. Opacus uses independent, nondeterministically seeded sampling
and noise generators; their seeds are not saved. Public-validation loss alone
selects the returned epoch. The test set, private training metrics and MIA
labels do not influence training or model selection.

Each private run calibrates noise for the **fixed maximum 30-epoch schedule**.
The fixed-schedule epsilon bound covers the early-stopped, validation-selected
model. Epsilon at the actual stopping time is recorded separately as a diagnostic;
it is not advertised as a tighter adaptive-stopping guarantee. Restoring an
earlier epoch does not reset the privacy expenditure. Delta remains 1e-6.

After selection, measure train/test utility and the signed train-minus-test
accuracy gap. Query the same balanced target MIA records using the frozen
four-feature attack. Only one target model and its query table are held at a time.
''')
add('code', '''target_progress = CellProgress()

def run_target_condition(epsilon):
    condition = 'non_private' if epsilon is None else f'epsilon_{epsilon:g}'
    target_progress.update(f'{condition}: training from scratch')
    with quiet_output():
        model, history, metadata = train_dp_study_target(
            target_arrays['train']['X'], target_arrays['train']['y'],
            target_arrays['validation']['X'], target_arrays['validation']['y'],
            epsilon=epsilon, delta=DELTA, progress=target_progress.update, **TRAIN_CONFIG)
    target_progress.update(f'{condition}: evaluating selected epoch {metadata["best_epoch"]}')
    income_metrics = {name: evaluate_model(model, target_arrays[name]['X'], target_arrays[name]['y'])
        for name in ['train', 'validation', 'test']}
    target_progress.update(f'{condition}: querying complete-feature membership attack')
    with quiet_output():
        attack_features, query_audit = collect_target_paper_features(model, target_preprocessor,
            target_partitions, config=WALK_CONFIG, seed=SEED,
            include_blind_walk=True, numeric_protocol=EXPANDED_NUMERIC_WALK)
        mia_metrics, _ = evaluate_attack(attack_result['classifier'], attack_features,
            threshold=attack_result['accuracy_threshold'],
            low_fpr_threshold=attack_result['low_fpr_threshold'])
    assert mia_metrics['records'] == 2 * mia_metrics['members']
    row = dict(condition=condition, private=epsilon is not None,
        epsilon_requested=epsilon, delta=DELTA if epsilon is not None else None,
        epsilon_fixed_schedule_bound=metadata['epsilon_fixed_schedule_bound'],
        epsilon_at_stop_diagnostic=metadata['epsilon_at_stop_diagnostic'],
        noise_multiplier=metadata['noise_multiplier'], epochs_run=metadata['epochs_run'],
        best_epoch=metadata['best_epoch'], training_seconds=metadata['training_seconds'],
        target_train_accuracy=income_metrics['train']['accuracy'],
        target_validation_accuracy=income_metrics['validation']['accuracy'],
        target_test_accuracy=income_metrics['test']['accuracy'],
        target_test_roc_auc=income_metrics['test']['roc_auc'],
        target_test_log_loss=income_metrics['test']['log_loss'],
        generalization_gap=income_metrics['train']['accuracy'] - income_metrics['test']['accuracy'],
        mia_accuracy=mia_metrics['accuracy'], mia_balanced_accuracy=mia_metrics['balanced_accuracy'],
        mia_roc_auc=mia_metrics['roc_auc'], mia_tpr_at_1pct_fpr_roc=mia_metrics['tpr_at_1pct_fpr_roc'],
        mia_tpr_at_shadow_threshold=mia_metrics['tpr_at_shadow_threshold'],
        mia_fpr_at_shadow_threshold=mia_metrics['fpr_at_shadow_threshold'])
    metadata.update(income_metrics=income_metrics, membership_metrics=mia_metrics, query_audit=query_audit)
    # Return metrics only: no fitted target, private row scores or model files.
    return row, history, metadata
''')
add('markdown', '''### Run the six matched conditions
This cell trains the fresh non-private control and five private targets. It is
an experiment and requires explicit user approval to execute. No existing Exp 3
models, scores, checkpoints or historical metrics are substituted. All five
budgets are reported without selecting one based on test or MIA performance.
Each of the three known DP research warnings is shown once across the six
conditions. The first warning remains visible; duplicate display is omitted.
''')
add('code', '''# Reload presentation support only; keep existing data, shadows and attack.
import importlib
import src.notebook_output as notebook_output_module
importlib.reload(notebook_output_module)
from src.notebook_output import compact_dp_warnings

study_rows, training_histories, target_metadata = [], {}, {}
with compact_dp_warnings():
    for epsilon in (None,) + EPSILON_BUDGETS:
        row, history, metadata = run_target_condition(epsilon)
        study_rows.append(row)
        training_histories[row['condition']] = history
        target_metadata[row['condition']] = metadata
        gc.collect()
study_results = pd.DataFrame(study_rows)
assert len(study_results) == 1 + len(EPSILON_BUDGETS)
assert study_results.loc[study_results['private'], 'delta'].eq(DELTA).all()
target_progress.update('All six conditions completed; final metrics ready')
''')
add('markdown', '''## Compare utility, generalization and membership leakage
Accuracies below are fractions; graphs show percentages. The generalization gap
is training accuracy minus test accuracy. A small gap with poor test accuracy can
indicate underfitting, so both are reported. Ordinary MIA accuracy uses one
auxiliary-validation-selected threshold, frozen across targets. ROC-AUC and
TPR@1%FPR supplement accuracy so threshold transfer is visible.
''')
add('code', '''display(study_results.set_index('condition')[[
    'epsilon_requested', 'epsilon_fixed_schedule_bound', 'epsilon_at_stop_diagnostic',
    'delta', 'epochs_run', 'best_epoch', 'target_train_accuracy', 'target_test_accuracy',
    'generalization_gap', 'mia_accuracy', 'mia_roc_auc', 'mia_tpr_at_1pct_fpr_roc'
]].round(6))
''')
add('markdown', '''## Graph 1: target model accuracy versus privacy budget
Use held-out **income-prediction test accuracy**. Epsilon labels are categories
with equal spacing, not numeric coordinates. The dashed line is the matched
non-private control. Budget labels describe requested maximum-schedule budgets;
actual accountant values and stopping epochs are available in the result table.
''')
add('code', '''figure, axes = plot_budget_accuracy(study_results, EPSILON_BUDGETS,
    metric='target_test_accuracy', output_dir=RUN_DIR, delta=DELTA)
display(figure)
plt.close(figure)
''')
add('markdown', '''## Graph 2: membership inference accuracy versus privacy budget
Use complete-feature **target MIA accuracy**, with the same categorical epsilon
axis. The non-private control and 50% chance level are horizontal references.
Chance is 50% because members and nonmembers are equally represented. Neither
curve is forced to be monotonic. One run per condition does not support error
bars or a claim that a near-chance attack proves privacy.
''')
add('code', '''figure, axes = plot_budget_accuracy(study_results, EPSILON_BUDGETS,
    metric='mia_accuracy', output_dir=RUN_DIR, delta=DELTA)
display(figure)
plt.close(figure)
''')
add('markdown', '''## Save final research results and interpretation limits
Save final CSV metrics, selected training histories, protocol and both graphs.
These research outputs include private-data evaluation statistics and are **not
DP-sanitized releases**. No private training rows, membership-labelled prediction
tables, fitted models, checkpoints or logs are written by this notebook.

The stated epsilon/delta bound concerns each individual target inference model
with its fixed public preprocessor. Releasing several models trained on the same
records requires privacy composition; the non-private control has no DP
guarantee. `secure_mode=False` is explicitly a research setting. Early stopping
uses public validation only. The recorded realised-stop accountant value is a
diagnostic; the fixed maximum-schedule bound is the claimed bound.

This study has one initialization seed per condition and independently sampled
DP noise. It uses transferred tree-shadow attacks, native year-specific Census
codes, an auxiliary-derived vocabulary, record-level protection, and a fixed
research partition. Repeated runs and attacks matched to DP targets are follow-up
checks. Lower measured MIA does not by itself demonstrate stronger protection.
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
study_results.to_csv(RUN_DIR / 'dp_target_and_mia_metrics.csv', index=False)
auxiliary_attack_metrics.to_csv(RUN_DIR / 'auxiliary_attack_metrics.csv')
shadow_income_metrics.to_csv(RUN_DIR / 'shadow_income_metrics.csv')
for condition, history in training_histories.items():
    pd.DataFrame(history).to_csv(RUN_DIR / f'{condition}_public_validation_history.csv', index=False)
protocol = dict(execution_status='complete', target_year=TARGET_YEAR, auxiliary_year=AUXILIARY_YEAR,
    epsilon_budgets=EPSILON_BUDGETS, delta=DELTA, train_size=TRAIN_SIZE,
    feature_count=len(EXPANDED_FEATURES), features=EXPANDED_FEATURES,
    label='nominal PINCP > 50000', split_config=SPLIT_CONFIG, target_partition_sizes=partition_counts,
    public_preprocessing_records=PUBLIC_PREPROCESSING_SIZE,
    preprocessing_scope='Fixed public 2014 subset, disjoint from shadow pool; shared by all targets',
    public_validation='Fixed 2023 validation split outside protected training records',
    training_config=TRAIN_CONFIG, target_results=target_metadata,
    privacy_scope='Individual target model; record-level DP conditional on fixed partition/public data',
    privacy_bound='Fixed maximum-schedule epsilon; epsilon at adaptive stop is diagnostic only',
    secure_mode=False, research_outputs_are_dp_releases=False,
    multiple_models='Per-model bounds; joint release requires composition; control is non-private',
    n_shadow_models=N_SHADOW_MODELS, shadow_algorithms=SHADOW_ALGORITHMS, shadow_seeds=shadow_seeds,
    records_per_class_per_shadow=SHADOW_RECORDS_PER_CLASS,
    attack='Frozen complete-feature RF transfer attack; non-DP tree shadows',
    attack_features=list(attack_result['classifier'].feature_names_in_), include_blind_walk=True,
    walk_config=WALK_CONFIG, extra_numeric_protocol=EXPANDED_NUMERIC_WALK,
    walk_scope='Consistent extra numeric protocol for shadows and targets',
    shadow_query_audits=shadow_query_audits, auxiliary_attack_metrics=attack_result['metrics'],
    accuracy_threshold=attack_result['accuracy_threshold'], low_fpr_threshold=attack_result['low_fpr_threshold'],
    plots='Equally spaced categorical requested epsilon labels; non-private horizontal reference',
    versions=versions, checkpoints_enabled=False, logging_enabled=False,
    missing_feature_experiments=False, repeated_seed_confirmation=False)
_ = (RUN_DIR / 'protocol.json').write_text(json.dumps(json_value(protocol), indent=2, allow_nan=False), encoding='utf-8')
print('Final metrics:', RUN_DIR / 'dp_target_and_mia_metrics.csv')
print('Graphs: target_accuracy_vs_epsilon and mia_accuracy_vs_epsilon (PNG and SVG)')
''')
add('markdown', '''Implementation references:
[Opacus PrivacyEngine](https://opacus.ai/api/privacy_engine.html),
[BatchMemoryManager](https://opacus.ai/api/batch_memory_manager.html),
[DPDataLoader](https://opacus.ai/api/data_loader.html),
[Opacus FAQ](https://opacus.ai/docs/faq).
''')

DESTINATION.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + '\n', encoding='utf-8')
print(f'Prepared {DESTINATION.name}: {len(notebook["cells"])} cells; none executed.')
