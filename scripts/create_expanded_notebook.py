# Generate the four-year, complete-feature Exp 3 baseline without running cells.
import ast
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DESTINATION = ROOT / 'exp_3_expanded dataset.ipynb'
metadata = json.loads(DESTINATION.read_text(encoding='utf-8')).get('metadata', {})
metadata.pop('widgets', None)
notebook = dict(nbformat=4, nbformat_minor=5, metadata=metadata, cells=[])


def add(kind, source):
    cell = dict(cell_type=kind, metadata={}, id=f'multiyear-baseline-{len(notebook["cells"])}',
        source=source.splitlines(keepends=True))
    if kind == 'code':
        ast.parse(source)
        cell.update(execution_count=None, outputs=[])
    notebook['cells'].append(cell)


add('markdown', '''# Exp 3 baseline: two targets and two auxiliary years
Train separate ACS 2023 and ACS 2024 income MLPs. Train a separate membership
attack from each auxiliary year, ACS 2014 and ACS 2018, using 16 shadows per year
(6 random forests, 5 gradient boosting, 5 XGBoost). Report all four combinations:
2023/2014, 2023/2018, 2024/2014 and 2024/2018 (target/auxiliary).

Use the explicitly approved common 47-feature schema: VPS is absent from the
2024 file and is excluded from all four years. Other feature definitions,
nominal income >$50,000 labels, eligibility, training settings and split sizes
follow Exp 3. The historical 48-input metrics are a separate configuration.
Every target and shadow fits its own preprocessor on training records only.

Blind walks are enabled, with the same numeric protocol for shadows and targets.
All 16 shadows for one auxiliary year share that year's grouped attack split;
auxiliary validation alone selects thresholds. Target membership labels only
score the frozen attacks. Native category codes are retained within each year.

Extract target attack features once per target and reuse them for both auxiliary
attacks. Train and query one shadow at a time to keep memory use bounded. No
model checkpoints or log files are loaded or saved. Compact progress, metrics,
row assignments and the study protocol are exported. The known sklearn/joblib
warning is suppressed; other warnings and exceptions remain visible.

This notebook stops after complete-feature baseline MIA. PFV, missing-feature
reconstruction and uncertainty-aware methods are not run here. Notebook creation
and validation execute no experiments; ask before running the cells.
''')
add('code', '''from pathlib import Path
import csv
import gc
import importlib
import json
import time
import numpy as np
import pandas as pd
import torch
from IPython.display import display
from src.data import split_acs_data, subset_target_training
from src.expanded_data import reference_split_config
from src.multiyear_baseline import (
    BASELINE_FEATURES, baseline_raw_path, load_baseline_year,
    preprocess_baseline_partitions, baseline_feature_inventory, train_baseline_shadow,
)
from src.training import train_target_model, evaluate_model
from src.shadow_models import PAPER_SHADOW_ALGORITHMS, summarize_shadow_models
from src.membership_attack import train_membership_classifier, evaluate_attack
from src.paper_attack_features import (
    collect_shadow_paper_features, collect_target_paper_features, make_probability_oracle,
)
import src.notebook_output as notebook_output_module
importlib.reload(notebook_output_module)
notebook_output_module.suppress_sklearn_parallel_warning()
from src.notebook_output import CellProgress, quiet_output

torch.set_num_threads(4)
SEED = 42
TARGET_YEARS = (2023, 2024)
AUXILIARY_YEARS = (2014, 2018)
TRAIN_SIZE = 100_000
N_SHADOW_MODELS = 16
SHADOW_RECORDS_PER_CLASS = 2000
SHADOW_ALGORITHMS = PAPER_SHADOW_ALGORITHMS
FEATURE_COLUMNS = list(BASELINE_FEATURES)
assert len(FEATURE_COLUMNS) == 47 and 'VPS' not in FEATURE_COLUMNS
DATA_ROOT = Path('data/raw/folktables')
SPLIT_CONFIG = reference_split_config()
assert SPLIT_CONFIG['seed'] == SEED
TRAIN_CONFIG = dict(hidden_dims=(512, 256, 128), dropout=0.0, epochs=100,
    batch_size=256, learning_rate=1e-3, weight_decay=0.0, seed=SEED,
    device='cpu', early_stopping=False, restore_best=False, monitor_train=True)
WALK_CONFIG = dict(max_steps=20, repeats=2, categorical_change_probability=0.05,
    batch_size=512, seed=SEED)
EXPANDED_NUMERIC_WALK = {'JWMNP': (2.0, 0, 200), 'JWRIP': (1.0, 1, 10)}
INCLUDE_BLIND_WALK = True
ATTACK_FEATURE_SET = 'paper' if INCLUDE_BLIND_WALK else 'paper_no_walk'
ATTACK_FEATURES = (['class_scaled_logit', 'blind_walk', 'entropy', 'cross_entropy']
    if INCLUDE_BLIND_WALK else ['class_scaled_logit', 'entropy', 'cross_entropy'])
ATTACK_MODE = 'with_walk_consistent_v1' if INCLUDE_BLIND_WALK else 'probability_only_v1'
RUN_DIR = Path('artifacts/exp_3_targets2023_2024_aux2014_2018_s16_common47') / ATTACK_MODE
RUN_DIR.mkdir(parents=True, exist_ok=True)
print('Targets:', TARGET_YEARS, '| Auxiliary pools:', AUXILIARY_YEARS)
print('Inputs:', len(FEATURE_COLUMNS), '| Shadows per auxiliary year:', N_SHADOW_MODELS)
print('Attack features:', ATTACK_FEATURES)
print('Results:', RUN_DIR)
''')
add('markdown', '''## Check all source files before training
Check headers for all four years first. Income and survey weights are used only
for eligibility/labels. They are excluded from the 47 predictors. Missing source
columns cause an early error rather than silent imputation of unavailable inputs.
''')
add('code', '''source_checks = []
for year in TARGET_YEARS + AUXILIARY_YEARS:
    path = baseline_raw_path(year, DATA_ROOT)
    if not path.is_file():
        raise FileNotFoundError(f'ACS {year} source CSV is required: {path}')
    with path.open(encoding='utf-8-sig', newline='') as handle:
        columns = set(next(csv.reader(handle, skipinitialspace=True)))
    missing = set(FEATURE_COLUMNS + ['PINCP', 'PWGTP']) - columns
    if missing:
        raise ValueError(f'ACS {year} missing required columns: {sorted(missing)}')
    source_checks.append(dict(year=year, role='target' if year in TARGET_YEARS else 'auxiliary',
        file=str(path), predictors=len(FEATURE_COLUMNS)))
display(pd.DataFrame(source_checks))
''')
add('markdown', '''## Train and evaluate two independent target MLPs
Each target uses 100,000 training records and its own stratified validation,
income-test and MIA partitions. MIA members belong to training; MIA nonmembers
are held out. Each year has 10,000 MIA members and 10,000 MIA nonmembers.
Targets train for 100 epochs with final weights retained. The same fixed split
configuration is applied independently within each year.
''')
add('code', '''def prepare_target(target_year, progress):
    started = time.monotonic()
    progress.update(f'Target {target_year}: loading and splitting')
    source = load_baseline_year(target_year, data_root=DATA_ROOT)
    reference = split_acs_data(source, **SPLIT_CONFIG)
    partitions = subset_target_training(reference, train_size=TRAIN_SIZE, seed=SEED)
    for name in ['validation', 'test', 'mia_members', 'mia_nonmembers']:
        np.testing.assert_array_equal(partitions[name].index, reference[name].index)
    train_ids = set(partitions['train'].index)
    assert set(partitions['mia_members'].index).issubset(train_ids)
    for name in ['validation', 'test', 'mia_nonmembers']:
        assert train_ids.isdisjoint(partitions[name].index)
    progress.update(f'Target {target_year}: preprocessing and training {TRAIN_CONFIG["epochs"]} epochs')
    with quiet_output():
        preprocessor, arrays = preprocess_baseline_partitions(partitions)
        model, history, training = train_target_model(
            arrays['train']['X'], arrays['train']['y'],
            arrays['validation']['X'], arrays['validation']['y'], **TRAIN_CONFIG)
        income_metrics = {name: evaluate_model(model, arrays[name]['X'], arrays[name]['y'])
            for name in ['train', 'validation', 'test']}
    assert list(preprocessor.feature_names_in_) == FEATURE_COLUMNS
    progress.update(f'Target {target_year}: extracting baseline attack features once')
    with quiet_output():
        attack_features, query_audit = collect_target_paper_features(
            model, preprocessor, partitions, config=WALK_CONFIG, seed=SEED,
            numeric_protocol=EXPANDED_NUMERIC_WALK, include_blind_walk=INCLUDE_BLIND_WALK)
    assert list(attack_features[ATTACK_FEATURES].columns) == ATTACK_FEATURES
    directory = RUN_DIR / f'target_{target_year}'
    directory.mkdir(parents=True, exist_ok=True)
    for name, frame in partitions.items():
        pd.Series(frame.index, name='record_id').to_csv(directory / f'{name}_ids.csv', index=False)
    baseline_feature_inventory(partitions['train']).to_csv(directory / 'feature_inventory.csv')
    pd.DataFrame(income_metrics).T.to_csv(directory / 'income_metrics.csv')
    progress.update(f'Target {target_year}: complete ({time.monotonic() - started:.1f}s)')
    return dict(model=model, preprocessor=preprocessor, partitions=partitions,
        attack_features=attack_features, income_metrics=income_metrics, history=history,
        training=training, query_audit=query_audit, eligible_records=len(source),
        income_adjustment=source.attrs['income_adjustment'])
''')
add('code', '''target_results = {}
target_progress = CellProgress()
for target_year in TARGET_YEARS:
    target_results[target_year] = prepare_target(target_year, target_progress)
    gc.collect()
target_income_rows = [dict(target_year=year, partition=name, **metrics)
    for year, result in target_results.items()
    for name, metrics in result['income_metrics'].items()]
target_income_metrics = pd.DataFrame(target_income_rows)
display(target_income_metrics.set_index(['target_year', 'partition']).round(4))
print('Two target models are available in target_results[2023] and target_results[2024].')
''')
add('markdown', '''## Fit separate baseline attacks from ACS 2014 and ACS 2018
For each auxiliary year, train 16 shadows independently: cycle RF, gradient
boosting and XGBoost with seeds 42, 52, ..., 192. Preserve the original 40/20/40
shadow train/validation/nonmember splits and estimator settings. Each shadow
contributes 2,000 sampled members and 2,000 sampled nonmembers to the attack.

Query one shadow at a time and discard its fitted model after feature extraction.
The resulting features match collecting the same ordered models as one pool.
Record-ID-grouped 60/20/20 attack splits keep repeated records in one role. Fit
one RF attack per auxiliary year and select its thresholds on that year's
validation records. The two auxiliary pools are not combined or selected using
target scores. Both targets use each auxiliary attack's frozen thresholds.
''')
add('code', '''def fit_query_one_shadow(auxiliary, name, algorithm, seed):
    with quiet_output():
        shadow = train_baseline_shadow(auxiliary, algorithm, seed=seed)
        shadow['attack_data']['shadow_model'] = name
        audit_rows = shadow['attack_data'].iloc[:32]
        oracle = make_probability_oracle(shadow['model'], shadow['preprocessor'], schema='acs')
        native = auxiliary.loc[audit_rows['record_id'], FEATURE_COLUMNS]
        np.testing.assert_allclose(oracle(native),
            audit_rows[['p_income_le_50k', 'p_income_gt_50k']].to_numpy(),
            rtol=1e-5, atol=1e-6, err_msg=f'Query mismatch: {name}')
        features, audits = collect_shadow_paper_features({name: shadow}, auxiliary,
            per_class=SHADOW_RECORDS_PER_CLASS, config=WALK_CONFIG, seed=SEED,
            numeric_protocol=EXPANDED_NUMERIC_WALK, include_blind_walk=INCLUDE_BLIND_WALK)
        metrics = summarize_shadow_models({name: shadow})
    return features, audits[name], metrics


def prepare_auxiliary_attack(auxiliary_year, progress):
    started = time.monotonic()
    auxiliary = load_baseline_year(auxiliary_year, data_root=DATA_ROOT)
    for target_year, target in target_results.items():
        for frame in target['partitions'].values():
            assert set(auxiliary.index).isdisjoint(frame.index)
    feature_tables, shadow_metric_tables, query_audits, shadow_seeds = [], [], {}, {}
    for index in range(N_SHADOW_MODELS):
        algorithm = SHADOW_ALGORITHMS[index % len(SHADOW_ALGORITHMS)]
        name = f'{algorithm}_{index + 1:02d}'
        seed = SEED + 10 * index
        progress.update(f'Auxiliary {auxiliary_year}: shadow [{index + 1}/{N_SHADOW_MODELS}] {name}')
        features, audit, metrics = fit_query_one_shadow(auxiliary, name, algorithm, seed)
        feature_tables.append(features)
        shadow_metric_tables.append(metrics)
        query_audits[name] = audit
        shadow_seeds[name] = seed
        gc.collect()
    attack_data = pd.concat(feature_tables, ignore_index=True)
    assert len(attack_data) == N_SHADOW_MODELS * 2 * SHADOW_RECORDS_PER_CLASS
    progress.update(f'Auxiliary {auxiliary_year}: fitting attack and calibrating thresholds')
    with quiet_output():
        attack = train_membership_classifier(attack_data, seed=SEED,
            algorithm='random_forest', feature_set=ATTACK_FEATURE_SET)
    assert list(attack['classifier'].feature_names_in_) == ATTACK_FEATURES
    role_ids = {name: set(frame['record_id']) for name, frame in attack['splits'].items()}
    assert role_ids['train'].isdisjoint(role_ids['validation'])
    assert role_ids['train'].isdisjoint(role_ids['test'])
    assert role_ids['validation'].isdisjoint(role_ids['test'])
    directory = RUN_DIR / f'auxiliary_{auxiliary_year}'
    directory.mkdir(parents=True, exist_ok=True)
    pd.concat(shadow_metric_tables).to_csv(directory / 'shadow_income_metrics.csv')
    pd.DataFrame(attack['metrics']).T.to_csv(directory / 'attack_metrics.csv')
    for name, frame in attack['splits'].items():
        frame[['record_id', 'shadow_model', 'membership']].to_csv(
            directory / f'attack_{name}_rows.csv', index=False)
    progress.update(f'Auxiliary {auxiliary_year}: complete ({time.monotonic() - started:.1f}s)')
    return dict(attack_result=attack, query_audits=query_audits, shadow_seeds=shadow_seeds,
        eligible_records=len(auxiliary), income_adjustment=auxiliary.attrs['income_adjustment'])
''')
add('code', '''auxiliary_results = {}
auxiliary_progress = CellProgress()
for auxiliary_year in AUXILIARY_YEARS:
    auxiliary_results[auxiliary_year] = prepare_auxiliary_attack(auxiliary_year, auxiliary_progress)
    gc.collect()
auxiliary_attack_rows = [dict(auxiliary_year=year, partition=name, **metrics)
    for year, result in auxiliary_results.items()
    for name, metrics in result['attack_result']['metrics'].items()]
auxiliary_attack_metrics = pd.DataFrame(auxiliary_attack_rows)
display(auxiliary_attack_metrics.set_index(['auxiliary_year', 'partition'])[
    ['accuracy', 'balanced_accuracy', 'roc_auc', 'tpr_at_1pct_fpr_roc']].round(4))
''')
add('markdown', '''## Baseline MIA: evaluate all four target/auxiliary pairs
Target features were extracted once per target, independently of auxiliary
membership labels. Reuse them here; changing auxiliary year adds no target-oracle
queries. Every target uses identical MIA records for its two auxiliary comparisons.

Accuracy and balanced accuracy use the auxiliary-validation-selected cutoff.
ROC-derived TPR@1% FPR is an evaluation summary. Separate TPR/FPR columns report
the actual operating point of the frozen auxiliary low-FPR cutoff; its target FPR
need not equal 1%. No target labels choose cutoffs or the auxiliary year.
''')
add('code', '''baseline_rows = []
for target_year in TARGET_YEARS:
    features = target_results[target_year]['attack_features']
    for auxiliary_year in AUXILIARY_YEARS:
        attack = auxiliary_results[auxiliary_year]['attack_result']
        with quiet_output():
            metrics, scores = evaluate_attack(attack['classifier'], features,
                threshold=attack['accuracy_threshold'], low_fpr_threshold=attack['low_fpr_threshold'])
        metrics['tpr_at_accuracy_threshold'] = metrics['tp'] / metrics['members']
        metrics['fpr_at_accuracy_threshold'] = metrics['fp'] / (metrics['records'] - metrics['members'])
        baseline_rows.append(dict(target_year=target_year, auxiliary_year=auxiliary_year,
            n_shadow_models=N_SHADOW_MODELS, **metrics))
        features.assign(membership_score=scores).to_csv(
            RUN_DIR / f'target_{target_year}_auxiliary_{auxiliary_year}_scores.csv', index=False)
baseline_mia_metrics = pd.DataFrame(baseline_rows).set_index(['target_year', 'auxiliary_year'])
assert len(baseline_mia_metrics) == len(TARGET_YEARS) * len(AUXILIARY_YEARS)
display(baseline_mia_metrics[['accuracy', 'balanced_accuracy', 'roc_auc',
    'tpr_at_1pct_fpr_roc', 'tpr_at_shadow_threshold', 'fpr_at_shadow_threshold']].round(4))
''')
add('markdown', '''## Save final metrics and protocol
CSV/JSON outputs contain metrics, predictions and row assignments, not fitted
models or resumable checkpoints. Each year's attack result remains in memory for
later inspection; there is no missing-feature or uncertainty computation here.
''')
add('code', '''baseline_mia_metrics.to_csv(RUN_DIR / 'baseline_mia_metrics.csv')
target_income_metrics.to_csv(RUN_DIR / 'target_income_metrics.csv', index=False)
auxiliary_attack_metrics.to_csv(RUN_DIR / 'auxiliary_attack_metrics.csv', index=False)
protocol = dict(execution_status='baseline_mia_completed',
    target_years=TARGET_YEARS, auxiliary_years=AUXILIARY_YEARS,
    pairs=[dict(target_year=t, auxiliary_year=a) for t in TARGET_YEARS for a in AUXILIARY_YEARS],
    feature_count=len(FEATURE_COLUMNS), features=FEATURE_COLUMNS,
    schema_change='VPS excluded from every year with user approval; absent from 2024 file',
    label='nominal PINCP > 50000', eligibility='AGEP > 16; PINCP > 100; WKHP > 0; PWGTP >= 1',
    train_size=TRAIN_SIZE, split_config=SPLIT_CONFIG, target_training_config=TRAIN_CONFIG,
    shadow_models_per_auxiliary=N_SHADOW_MODELS, total_shadow_models=N_SHADOW_MODELS * len(AUXILIARY_YEARS),
    shadow_algorithms=list(SHADOW_ALGORITHMS), records_per_class_per_shadow=SHADOW_RECORDS_PER_CLASS,
    attack_feature_set=ATTACK_FEATURE_SET, attack_features=ATTACK_FEATURES,
    include_blind_walk=INCLUDE_BLIND_WALK, walk_config=WALK_CONFIG,
    extra_numeric_protocol=EXPANDED_NUMERIC_WALK,
    walk_protocol_scope='Same numeric fields and perturbation settings for all target and shadow years',
    threshold_selection='Each auxiliary validation only; frozen for both targets',
    target_features='Queried once per target and reused for both auxiliary attacks',
    checkpoints_enabled=False, logging_enabled=False,
    targets={str(year): dict(training=result['training'], eligible_records=result['eligible_records'],
        income_metrics=result['income_metrics'], query_audit=result['query_audit'])
        for year, result in target_results.items()},
    auxiliary={str(year): dict(eligible_records=result['eligible_records'],
        shadow_seeds=result['shadow_seeds'], attack_metrics=result['attack_result']['metrics'],
        query_audits=result['query_audits']) for year, result in auxiliary_results.items()},
    limitations=['native category codes differ across years', 'one training/attack seed configuration',
        '47-feature schema differs from historical 48-feature baseline'])
_ = (RUN_DIR / 'protocol.json').write_text(json.dumps(protocol, indent=2), encoding='utf-8')
print('Complete-feature baseline MIA finished. Results:', RUN_DIR / 'baseline_mia_metrics.csv')
''')
add('markdown', '''The notebook ends at baseline MIA. No PFV or uncertainty section is appended.
''')

DESTINATION.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + '\n', encoding='utf-8')
print(f'Prepared {DESTINATION.name}: {len(notebook["cells"])} cells; none executed.')
