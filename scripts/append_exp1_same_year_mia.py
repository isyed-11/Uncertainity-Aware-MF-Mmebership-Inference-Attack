"""Append the authorized same-year diagnostics; never run any notebook cells."""
import ast
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / 'exp_1_expanded_dataset_generalization.ipynb'


def update_notebook():
    notebook = json.loads(PATH.read_text(encoding='utf-8'))
    if any(cell.get('id') == 'exp1-same-year-intro' for cell in notebook['cells']):
        raise ValueError('Same-year section already exists; edit it without appending duplicates.')
    notebook['cells'][0]['source'] += '''
Additional end sections test matched MLP-only shadows from unused ACS 2023 and
2024 records against the respective same-year target only. They add rich-output
RF attacks, subgroup and top-5% outlier diagnostics, generalization-loss gaps,
and offline LiRA. Existing 2014/2018 mixed-shadow comparisons remain unchanged.
'''.splitlines(keepends=True)
    notebook['cells'][14]['source'] = ['The cross-year baseline ends here. The separate same-year sections below add\n',
        'matched MLP shadow attacks and diagnostics; no PFV or reconstruction is added.\n']
    if notebook['cells'][-1]['cell_type'] == 'code' and not ''.join(notebook['cells'][-1]['source']).strip():
        notebook['cells'].pop()

    def add(kind, identity, source):
        cell = dict(cell_type=kind, id=identity, metadata={}, source=source.splitlines(keepends=True))
        if kind == 'code':
            ast.parse(source)
            cell.update(execution_count=None, outputs=[])
        notebook['cells'].append(cell)

    add('markdown', 'exp1-same-year-intro', '''## Same-year matched MLP shadows: 2023→2023 and 2024→2024
Use **16 additional MLP shadows per year**. Match each target's MLP hidden layers,
optimizer, learning rate, dropout, weight decay, batch size, maximum epochs and
validation-loss stopping/restoration rule. Reference split/initialization seeds
remain independent (42, 52, ..., 192). Fit the same 47-feature preprocessing
recipe independently on each shadow's training rows. Fitted statistics,
vocabularies and hence encoded input dimensions need not equal the target's;
no target-fitted preprocessing or weights are given to shadows.

The source for each pool is that year's eligible rows **unused by ALL target
partitions**, including validation, income test and MIA nonmembers. Target
training remains 100,000 records with all its current IDs unchanged. The new
75/10/15 train/validation/test split is used when early stopping is enabled.
Without early stopping, use 80/20 train/test and disable best-weight restoration;
no validation or test data enters the training helper in that fixed-schedule mode.
Each model independently randomizes its stratified split with its own seed.
Approximately 73,000 unused records remain per year, giving about 55,000 training
rows per shadow with 75/10/15, so **training sample size is still not matched**.
Counts and zero-overlap assertions are reported; records are row
IDs, not deduplicated people. References can overlap each other, as before.

The new RF attack sees both output probabilities, true-class probability,
confidence, entropy, stable cross entropy, correctness, true-class logit margin,
modified entropy and the existing blind-walk feature. This is a predeclared
richer-feature comparison, not an isolated test of auxiliary year alone.
Assign record-group attack roles before sampling predictions. Retain every
shadow OUT-test prediction; independently subsample shadow IN-training rows to
match that exact OUT count within each attack role. The standard meta-classifier's
training, validation and test datasets are therefore each exactly 50/50 IN/OUT.
Shadow validation rows are never attack IN or OUT. Using all OUT rows increases
blind-walk query work compared with the earlier 2,000-per-class cap.
Repeated IDs keep one grouped attack train/validation/test role. No target MIA
labels choose settings, score direction, thresholds, groups or the anomaly model.

Creating these cells does not authorize execution. Running this section trains
32 additional non-private MLPs (in addition to the 32 existing cross-year
shadows). No DP notebook is modified and no target is retrained here.
''')
    add('code', 'exp1-same-year-setup', '''import src.training as same_year_training_module
importlib.reload(same_year_training_module)
import src.multiyear_baseline as same_year_baseline_module
importlib.reload(same_year_baseline_module)
import src.same_year_mia as same_year_mia_module
importlib.reload(same_year_mia_module)
from src.same_year_mia import (
    disjoint_same_year_pool, verify_target_recipe, run_same_year_study,
    fit_outlier_slices, score_metrics, subgroup_metrics,
    fit_offline_calibration, gaussian_out_score,
)

SAME_YEAR_REFERENCE_COUNT = N_SHADOW_MODELS  # 16, matching the existing pool count
SAME_YEAR_MINIMUM_OUT = 3
SAME_YEAR_TARGETS = (2023, 2024)
SAME_YEAR_SPLIT_POLICY = ('train75_val10_test15' if TRAIN_CONFIG['early_stopping'] else 'train80_test20')
SAME_YEAR_DIR = RUN_DIR / f'same_year_matched_mlp_{SAME_YEAR_SPLIT_POLICY}'
assert SAME_YEAR_REFERENCE_COUNT == 16
assert set(SAME_YEAR_TARGETS) <= set(target_results)
same_year_sources, same_year_audit_parts = {}, []
for year in SAME_YEAR_TARGETS:
    verify_target_recipe(target_results[year], TRAIN_CONFIG)
    same_year_sources[year] = load_baseline_year(year, data_root=DATA_ROOT)
    unused, audit = disjoint_same_year_pool(same_year_sources[year], target_results[year]['partitions'])
    same_year_audit_parts.append(audit.assign(target_year=year))
    del unused
same_year_disjointness = pd.concat(same_year_audit_parts, ignore_index=True)
display(same_year_disjointness.set_index(['target_year', 'target_partition']))
print('Additional matched MLP shadows:', SAME_YEAR_REFERENCE_COUNT * len(SAME_YEAR_TARGETS))
''')
    add('markdown', 'exp1-same-year-run-note', '''### Run the two same-year shadow pools and rich-output RF attacks
This is an experiment: execute only after explicit approval. Each pool is used
only against its respective target. The targets and their original MIA record
sets are reused. Keep blind-walk settings unchanged; query each target's existing
walk feature from memory rather than running its walks again. The references
also query all shadow-pool and target-candidate unaugmented logits once, so the
offline LiRA section reuses scores instead of fitting a second reference ensemble.
Training models and reference scores stay in memory; no checkpoints are used.
''')
    add('code', 'exp1-same-year-run', '''same_year_results = {}
same_year_progress = CellProgress()
for year in SAME_YEAR_TARGETS:
    same_year_results[year] = run_same_year_study(
        same_year_sources[year], target_results[year], target_year=year,
        count=SAME_YEAR_REFERENCE_COUNT, train_config=TRAIN_CONFIG,
        walk_config=WALK_CONFIG, numeric_protocol=EXPANDED_NUMERIC_WALK,
        include_blind_walk=INCLUDE_BLIND_WALK,
        seed=SEED, progress=same_year_progress.update)
    gc.collect()
same_year_progress.update('Both same-year matched shadow pools and rich RF attacks completed')
''')
    add('markdown', 'exp1-same-year-slices-note', '''## Same-year RF metrics, subgroup/outlier breakdown and loss gaps
Fit a fixed Isolation Forest (100 trees, max_samples=256, seed 42) using only the
disjoint same-year shadow pool and its own training-fitted preprocessing. Rank
target candidates without membership labels and select exactly the top 5%
(1,000 of 20,000) by anomaly score, with stable tie handling. This is an input
anomaly diagnostic, not proof of statistical leverage or rare-group coverage.

Report whole-set, top-5%, remaining-95%, income class, SEX and RAC1P slices for
both attacks. Use the same global shadow-validation-selected cutoffs in every
slice. Report membership counts/proportions, majority-class baseline, balanced
accuracy, AUC, TPR@1%FPR and actual transferred-cutoff FPR/TPR alongside accuracy.
Slices need not have 50% members: raw accuracy alone can be misleading. AUC and
balanced accuracy are undefined for a single-membership-class slice; keep NaN
with an explicit status. Small slices have coarse low-FPR resolution.

Report target and shadow train-versus-validation loss/accuracy gaps. The signed
loss gap is **validation loss minus training loss** (positive means validation
loss is worse), not training minus validation. Accuracy confidence intervals are
descriptive Wilson intervals; they do not prove equivalence to chance or account
for all training randomness and multiple subgroup comparisons.
In 80/20 mode there is no shadow validation split, so its validation-gap columns
are undefined; final shadow test gaps are still reported after training.
''')
    add('code', 'exp1-same-year-rf-metrics', '''same_year_metric_rows, same_year_aux_rows, same_year_slice_parts = [], [], []
same_year_gap_rows, same_year_shadow_gap_parts = [], []
for year, result in same_year_results.items():
    same_year_progress.update(f'{year}: fitting input-only outlier diagnostic and scoring RF slices')
    anomaly_scores, outlier_mask = fit_outlier_slices(result['pool'], result['candidates'], seed=SEED)
    result['target_rows']['anomaly_score'] = anomaly_scores
    result['target_rows']['outlier_top5'] = outlier_mask
    attack = result['rich_attack']
    row = score_metrics(result['target_rows'].membership, result['target_rows'].rich_rf_score,
        threshold=attack['accuracy_threshold'], low_fpr_threshold=attack['low_fpr_threshold'])
    same_year_metric_rows.append(dict(target_year=year, auxiliary_year=year, method='matched_rich_rf', **row))
    for role, metrics in attack['metrics'].items():
        same_year_aux_rows.append(dict(auxiliary_year=year, method='matched_rich_rf', partition=role, **metrics))
    same_year_slice_parts.append(subgroup_metrics(result['target_rows'], result['candidates'], outlier_mask,
        score_column='rich_rf_score', threshold=attack['accuracy_threshold'],
        low_fpr_threshold=attack['low_fpr_threshold']).assign(target_year=year, method='matched_rich_rf'))
    same_year_gap_rows.append(dict(target_year=year, **result['target_generalization']))
    same_year_shadow_gap_parts.append(result['shadow_generalization'].assign(auxiliary_year=year))

same_year_target_generalization = pd.DataFrame(same_year_gap_rows)
same_year_shadow_generalization = pd.concat(same_year_shadow_gap_parts, ignore_index=True)
display(pd.DataFrame(same_year_metric_rows).set_index(['target_year', 'method'])[
    ['accuracy', 'balanced_accuracy', 'roc_auc', 'tpr_at_1pct_fpr_roc']].round(4))
display(same_year_target_generalization.set_index('target_year').round(4))
''')
    add('markdown', 'exp1-offline-lira-note', '''## Per-sample calibration: offline LiRA on the same-year reference pools
Use the already-queried raw binary true-class logit margin, avoiding softmax
rounding/saturation. Each target candidate is absent from every reference's
training **and validation** data. Estimate its OUT location as the median margin
across the 16 reference models. Use a pooled OUT standard deviation fitted only
on attack-training record IDs (floor 1e-6). The membership score is the negative
Gaussian OUT log likelihood, matching the fixed-variance offline LiRA form.
This is a likelihood-based score, not a calibrated probability of membership.

Calibrate cutoffs using shadows as pseudo-targets. For every pseudo-target query,
exclude its own model from the OUT distribution. Also exclude references where
that row was training or validation data: validation can affect early stopping.
Require at least three remaining pure-OUT references; report skipped rows and
the remaining support. Reuse the RF attack's fixed record-group roles. Fit the
accuracy/low-FPR cutoffs on shadow validation only, then freeze them for the
target and every subgroup. No target membership labels fit any distribution,
threshold or variance. Test data never chooses score direction or a method.

**This is offline LiRA, not online LiRA.** Online LiRA would train candidate-IN
and candidate-OUT references and conflict with the requested disjoint data.
The smaller shadow training size, limited reference count, learned-vocabulary
differences and finite calibration support remain limitations. Near-chance
results strengthen evidence about these tested attacks; they do not prove
privacy or establish that no vulnerable subgroup exists.
The 15% OUT-test fraction gives fewer pure-OUT calibration references per
pseudo-target than the former 40% fraction. The minimum remains three; skipped
rows and actual reference support are reported rather than silently relaxed.

References: [LiRA paper](https://arxiv.org/abs/2112.03570),
[reference offline implementation](https://github.com/tensorflow/privacy/blob/master/research/mi_lira_2021/plot.py).
''')
    add('code', 'exp1-offline-lira-scores', '''for year, result in same_year_results.items():
    same_year_progress.update(f'{year}: offline LiRA calibration using existing reference queries')
    role_ids = {role: set(frame.record_id) for role, frame in result['rich_attack']['splits'].items()}
    offline = fit_offline_calibration(result['auxiliary_margins'], result['pure_out_masks'],
        result['shadow_rows'], role_ids, result['pool'].index, minimum_out=SAME_YEAR_MINIMUM_OUT)
    rows = result['target_rows']
    assert result['target_reference_margins'].shape == (SAME_YEAR_REFERENCE_COUNT, len(rows))
    rows['offline_out_center'] = np.median(result['target_reference_margins'], axis=0)
    rows['offline_lira_score'] = gaussian_out_score(rows.class_scaled_logit.to_numpy(),
        rows.offline_out_center.to_numpy(), offline['scale'])
    result['offline_lira'] = offline
    metrics = score_metrics(rows.membership, rows.offline_lira_score,
        threshold=offline['accuracy_threshold'], low_fpr_threshold=offline['low_fpr_threshold'])
    same_year_metric_rows.append(dict(target_year=year, auxiliary_year=year, method='offline_lira', **metrics))
    for role, metrics in offline['metrics'].items():
        same_year_aux_rows.append(dict(auxiliary_year=year, method='offline_lira', partition=role, **metrics))
    same_year_slice_parts.append(subgroup_metrics(rows, result['candidates'], rows.outlier_top5,
        score_column='offline_lira_score', threshold=offline['accuracy_threshold'],
        low_fpr_threshold=offline['low_fpr_threshold']).assign(target_year=year, method='offline_lira'))

same_year_mia_metrics = pd.DataFrame(same_year_metric_rows)
same_year_auxiliary_metrics = pd.DataFrame(same_year_aux_rows)
same_year_subgroups = pd.concat(same_year_slice_parts, ignore_index=True)
same_year_progress.update('Same-year RF, offline LiRA and subgroup diagnostics completed')
''')
    add('code', 'exp1-same-year-comparison', '''display(same_year_mia_metrics.set_index(['target_year', 'method'])[
    ['accuracy', 'balanced_accuracy', 'roc_auc', 'tpr_at_1pct_fpr_roc',
     'tpr_at_shadow_threshold', 'fpr_at_shadow_threshold']].round(4))
display(same_year_auxiliary_metrics.set_index(['auxiliary_year', 'method', 'partition'])[
    ['records', 'accuracy', 'balanced_accuracy', 'roc_auc']].round(4))
display(pd.concat([result['attack_balance_audit'].assign(auxiliary_year=year)
    for year, result in same_year_results.items()], ignore_index=True).groupby(
    ['auxiliary_year', 'attack_role'])[['selected_in', 'selected_out']].sum())
display(same_year_subgroups.loc[same_year_subgroups.group_type.eq('outlier')].set_index(
    ['target_year', 'method', 'group'])[['records', 'members', 'nonmembers',
    'majority_class_accuracy', 'accuracy', 'balanced_accuracy', 'roc_auc']].round(4))
display(same_year_subgroups.loc[same_year_subgroups.group_type.isin(['income_gt_50k', 'SEX', 'RAC1P'])].set_index(
    ['target_year', 'method', 'group_type', 'group'])[['records', 'members', 'nonmembers',
    'accuracy', 'balanced_accuracy', 'roc_auc', 'metric_status']].round(4))
display(pd.DataFrame([dict(auxiliary_year=year, reference_count=SAME_YEAR_REFERENCE_COUNT,
    pool_records=len(result['pool']), shadow_train_records=result['model_protocols']['same_year_mlp_01']['partition_sizes']['train'],
    target_train_records=len(target_results[year]['partitions']['train']),
    offline_out_std=result['offline_lira']['scale'], skipped_calibration_rows=result['offline_lira']['skipped_rows'],
    calibration_min_out=result['offline_lira']['out_reference_count_summary']['minimum'],
    calibration_median_out=result['offline_lira']['out_reference_count_summary']['median'],
    calibration_max_out=result['offline_lira']['out_reference_count_summary']['maximum'])
    for year, result in same_year_results.items()]).set_index('auxiliary_year'))
''')
    add('markdown', 'exp1-same-year-save-note', '''## Export the additional same-year results
Save these results under the split-specific `same_year_matched_mlp_train75_val10_test15`
(or `same_year_matched_mlp_train80_test20`) directory, retaining the
existing cross-year baseline files. Save metrics, subgroup counts, disjointness
audits, reference recipes/epoch diagnostics, histories and final target scores.
No reference models, fitted anomaly models, checkpoints or log files are saved.
The reference score matrices remain in memory for this run only.

For the report, describe this as an architecture-informed, same-year attack
comparison with matched MLP hyperparameters and preprocessing recipe but smaller
reference training sets. Report adverse subgroup results as well as near-chance
results; do not select or suppress groups after inspecting scores. Whole-set
chance is 50% because target MIA candidates are balanced. Slice baselines depend
on their actual membership proportions. One target seed per year and these
attacks cannot certify privacy or rule out untested attacks.
''')
    add('code', 'exp1-same-year-save', '''def same_year_json_value(value):
    if isinstance(value, dict):
        return {str(key): same_year_json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [same_year_json_value(item) for item in value]
    if isinstance(value, np.generic):
        return same_year_json_value(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None if np.isnan(value) else str(value)
    return value

SAME_YEAR_DIR.mkdir(parents=True, exist_ok=True)
same_year_mia_metrics.to_csv(SAME_YEAR_DIR / 'same_year_mia_metrics.csv', index=False)
same_year_auxiliary_metrics.to_csv(SAME_YEAR_DIR / 'same_year_auxiliary_attack_metrics.csv', index=False)
same_year_subgroups.to_csv(SAME_YEAR_DIR / 'same_year_subgroup_metrics.csv', index=False)
same_year_target_generalization.to_csv(SAME_YEAR_DIR / 'target_generalization.csv', index=False)
same_year_shadow_generalization.to_csv(SAME_YEAR_DIR / 'shadow_generalization.csv', index=False)
same_year_disjointness.to_csv(SAME_YEAR_DIR / 'disjointness_audit.csv', index=False)
for year, result in same_year_results.items():
    directory = SAME_YEAR_DIR / str(year)
    directory.mkdir(parents=True, exist_ok=True)
    pd.Series(result['pool'].index, name='record_id').to_csv(directory / 'shadow_pool_ids.csv', index=False)
    result['attack_balance_audit'].to_csv(directory / 'attack_balance_audit.csv', index=False)
    pd.DataFrame([dict(record_id=record_id, attack_role=role)
        for role, ids in result['attack_role_ids'].items() for record_id in sorted(ids)]).to_csv(
            directory / 'attack_record_roles.csv', index=False)
    result['target_rows'].to_csv(directory / 'target_mia_scores.csv')
    for name, history in result['histories'].items():
        pd.DataFrame(history).to_csv(directory / f'{name}_validation_history.csv', index=False)
same_year_protocol = dict(execution_status='same_year_attacks_completed',
    pairs=[dict(target_year=year, auxiliary_year=year) for year in SAME_YEAR_TARGETS],
    references_per_year=SAME_YEAR_REFERENCE_COUNT, shadow_algorithm='MLP only',
    matched_target_training_recipe=TRAIN_CONFIG, independent_reference_seeds=True,
    preprocessing='Same common47 recipe, fitted on each model training partition only; vocabulary may differ',
    shadow_data='Unused same-year rows excluding every target training/validation/test/MIA ID',
    shadow_split=('75% train, 10% validation, 15% test' if TRAIN_CONFIG['early_stopping']
        else '80% train, 20% test; no validation or best-epoch restoration'),
    shadow_split_rounding='Ceil test and validation counts; remainder to training; independent seeds',
    standard_attack_sampling='All OUT-test rows; exact matching IN-training subsample within each fixed record role',
    attack_roles_before_subsampling=True, standard_attack_exact_balance_per_role=True,
    training_size_matched=False, target_retrained=False,
    attack_features=next(iter(same_year_results.values()))['features'],
    rf_attack='300 trees, max_depth12, min_samples_leaf20, sqrt features; grouped 60/20/20 roles',
    thresholds='Shadow validation only; frozen for target and subgroup evaluation',
    blind_walk=INCLUDE_BLIND_WALK, walk_config=WALK_CONFIG, numeric_protocol=EXPANDED_NUMERIC_WALK,
    offline_lira=dict(method='Negative OUT Gaussian log likelihood, per-record median, fixed pooled OUT std',
        variance_fit='Attack-training IDs only; no target scores', minimum_out=SAME_YEAR_MINIMUM_OUT,
        variance_floor=1e-6, reference_query_exclusion=True, exclude_shadow_validation_from_out=True,
        online_in_models=False, score_direction='Higher negative OUT log likelihood means member',
        implementation_reference='https://github.com/tensorflow/privacy/blob/master/research/mi_lira_2021/plot.py'),
    outliers=dict(method='IsolationForest', fit_data='Disjoint same-year shadow pool only',
        n_estimators=100, max_samples=256, seed=SEED, top_fraction=.05,
        ranking='Target inputs only, stable tie handling; membership labels not used'),
    subgroup_fields=['income_gt_50k', 'SEX', 'RAC1P'], loss_gap='validation log loss minus training log loss',
    years={str(year): dict(pool_records=len(result['pool']), model_protocols=result['model_protocols'],
        offline_calibration_skipped_rows=result['offline_lira']['skipped_rows'],
        offline_calibration_kept_rows=result['offline_lira']['kept_rows'],
        offline_out_reference_count_summary=result['offline_lira']['out_reference_count_summary'],
        offline_scale=result['offline_lira']['scale'], target_generalization=result['target_generalization'],
        rich_rf_threshold=result['rich_attack']['accuracy_threshold'],
        rich_rf_low_fpr_threshold=result['rich_attack']['low_fpr_threshold'],
        offline_threshold=result['offline_lira']['accuracy_threshold'],
        offline_low_fpr_threshold=result['offline_lira']['low_fpr_threshold'])
        for year, result in same_year_results.items()},
    checkpoints_enabled=False, logging_enabled=False,
    limitations=['Smaller reference training sets than target', 'Finite OUT reference support',
        'Learned encoder vocabularies/input dimensions may differ', 'Offline rather than online LiRA',
        'One target seed per year', 'Near-chance scores do not prove privacy'])
_ = (SAME_YEAR_DIR / 'protocol.json').write_text(
    json.dumps(same_year_json_value(same_year_protocol), indent=2, allow_nan=False), encoding='utf-8')
print('Additional same-year results:', SAME_YEAR_DIR)
''')
    PATH.write_text(json.dumps(notebook, indent=1, ensure_ascii=False)+'\n', encoding='utf-8')
    print(f'Prepared same-year sections in {PATH.name}; {len(notebook["cells"])} total cells; none executed.')


if __name__ == '__main__':
    update_notebook()
