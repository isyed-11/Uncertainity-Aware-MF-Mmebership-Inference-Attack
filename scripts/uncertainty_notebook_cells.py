"""Append reviewable uncertainty-study cells; contains no execution runner."""


def append_uncertainty_cells(add):
    add('markdown', '''## Uncertainty-aware MF-MIA: study protocol
Research question: does reconstruction-induced score standard deviation improve
membership inference beyond the mean? The following cells are additional experiments;
they are not executed when this notebook is generated.

Use the seven preselected masks, target, 16 ACS 2014 shadows, four base
attack features and existing blind-walk protocol. The base attack is frozen.
Compare conditional single imputation, average score, learned mean-only RF and
matched mean+standard-deviation RF on identical completion draws. PFV above is a
separate reconstruction baseline, not the single-imputation control for this section.

Sequential forest-leaf donor sampling conditions each hidden field on visible
fields and earlier sampled hidden fields (MAR, WKHP, COW, SEX order). It preserves
observed auxiliary values but is an approximate predictive distribution, not a
calibrated posterior. Reconstruction training excludes all base-attack, meta-fit,
calibration and auxiliary-test candidate IDs. No income label conditions imputation.
Auxiliary validation IDs are divided into meta-fit and calibration groups. Auxiliary
test IDs remain untouched until evaluation. The base attack's earlier validation
thresholds are not reused for this comparison. No target labels select methods,
masks, K or thresholds. Income class labels remain known to the base loss feature.

K=6 and one reconstruction seed are predeclared exploratory settings, not a
confirmatory multi-seed study. All seven masks are reported without choosing a winner
on target results. Raw draws remain in memory during this run.
The original commuting walk asymmetry is retained and limits interpretation.
Use two single-field, two double-field, two triple-field and one four-field mask;
each of MAR, WKHP, COW and SEX is hidden in four masks. Six draws give noisier
variability estimates than the original K=20 setting. The base attack is refitted
using all 16 shadows, with grouped auxiliary splits recomputed from
that pool. Reconstruction/meta/calibration roles follow those new splits.
Results are saved under reduced_k6_s16_m7, separate from the eight-shadow run and full experiment.
Common walk seeds reduce additional Monte Carlo noise; score spread can still
include completion-dependent effects of the finite blind-walk procedure.

This section can be expensive: each candidate is evaluated K times per mask,
with up to 41 record queries per completion under the original walk settings.
DP training/evaluation and repeated-seed confirmation are not implemented here.

Execution uses memory only for reconstruction models, audit draws, completion
scores and aggregators. No checkpoints are loaded or saved. Run these cells in
order in one kernel session; restarting requires computing them again.
Only final metrics, the protocol and compact progress logs are written to disk.
''')
    add('code', '''# 1. Fixed configuration and disjoint auxiliary roles
import joblib
import importlib
from sklearn.model_selection import train_test_split
from src.notebook_output import CellProgress, compact_output
import src.uncertainty_mf_mia as uncertainty_module
# Pick up runtime-support edits without retraining/reloading the target or shadows.
importlib.reload(uncertainty_module)
from src.uncertainty_mf_mia import (
    ConditionalDonorChain, auxiliary_roles, completion_scores, summarize_scores,
    fit_aggregation, evaluate_aggregation,
)

UNCERTAINTY_K = 6
assert N_SHADOW_MODELS == len(shadow_results) == 16, 'Load the 16-shadow pool first.'
assert set(shadow_attack_data['shadow_model']) == set(shadow_results), 'Rebuild base attack data using all 16 shadows.'
assert set().union(*(set(part['shadow_model']) for part in attack_result['splits'].values())) == set(shadow_results), 'Refit the base attack on the 16-shadow pool.'
assert len(MISSING_MASKS) == 7, 'Use the seven preselected masks for the reduced study.'
RECONSTRUCTION_SEED = SEED + 1000
UNCERTAINTY_DIR = RUN_DIR / 'uncertainty_mf_mia'
UNCERTAINTY_DIR.mkdir(parents=True, exist_ok=True)
UNCERTAINTY_LOG_DIR = UNCERTAINTY_DIR / 'logs'
uncertainty_roles = auxiliary_roles(acs_auxiliary, attack_result, seed=SEED)
recon_fit_ids, recon_check_ids = train_test_split(
    sorted(uncertainty_roles['reconstruction']), test_size=0.2, random_state=SEED)
recon_training = acs_auxiliary.loc[recon_fit_ids, EXPANDED_FEATURES]
recon_check = acs_auxiliary.loc[recon_check_ids, EXPANDED_FEATURES]
display(pd.Series({name: len(ids) for name, ids in uncertainty_roles.items()}, name='unique_2014_records'))
uncertainty_reconstructors = {}
''')
    add('markdown', '''### Fit and audit conditional reconstruction using 2014 only
Held-out reconstruction checks use only reserved 2014 records, not attack candidates.
Numeric coverage is descriptive empirical coverage of the sampled 90% interval.
Categorical sample-match fraction measures draw accuracy; it is not calibrated
probability accuracy. Report these checks even when attack performance is poor.
Related fields remain visible, so reconstruction may be easy through proxies.
''')
    add('code', '''# 2. Fit one conditional reconstruction chain per mask; no target queries
from src.notebook_output import CellProgress, compact_output
UNCERTAINTY_LOG_DIR = UNCERTAINTY_DIR / 'logs'
reconstruction_audit_rows = []
reconstruction_progress = CellProgress()
for mask_index, (mask_name, hidden) in enumerate(MISSING_MASKS.items(), start=1):
    mask_started = time.monotonic()
    reconstruction_progress.update(f'[{mask_index}/{len(MISSING_MASKS)}] {mask_name}: fitting reconstruction')
    with compact_output(UNCERTAINTY_LOG_DIR / 'reconstruction.log'):
        reconstructor = ConditionalDonorChain(hidden, seed=RECONSTRUCTION_SEED).fit(recon_training)
    uncertainty_reconstructors[mask_name] = reconstructor
    reconstruction_progress.update(f'[{mask_index}/{len(MISSING_MASKS)}] {mask_name}: auditing {UNCERTAINTY_K} completions')
    with compact_output(UNCERTAINTY_LOG_DIR / 'reconstruction.log'):
        samples = list(reconstructor.iter_samples(recon_check,
            seeds=[RECONSTRUCTION_SEED + draw for draw in range(UNCERTAINTY_K)]))
    for field in hidden:
        observed = recon_check[field].notna().to_numpy()
        if not observed.any():
            continue
        draws = np.column_stack([sample[field].to_numpy()[observed] for sample in samples])
        truth = recon_check[field].to_numpy()[observed]
        row = dict(mask=mask_name, field=field, records=int(observed.sum()))
        if field == 'WKHP':
            lower, upper = np.quantile(draws, [0.05, 0.95], axis=1)
            row.update(mae_of_sample_mean=float(np.abs(draws.mean(axis=1) - truth).mean()),
                sampled_90pct_coverage=float(((truth >= lower) & (truth <= upper)).mean()),
                mean_interval_width=float((upper - lower).mean()))
        else:
            row['sample_match_fraction'] = float((draws == truth[:, None]).mean())
        reconstruction_audit_rows.append(row)
    reconstruction_progress.update(f'[{mask_index}/{len(MISSING_MASKS)}] {mask_name}: complete ({time.monotonic() - mask_started:.1f}s)')
reconstruction_audit = pd.DataFrame(reconstruction_audit_rows)
display(reconstruction_audit.round(4))
reconstruction_audit.to_csv(UNCERTAINTY_DIR / 'reconstruction_audit.csv', index=False)
print('Detailed progress and warnings:', UNCERTAINTY_LOG_DIR / 'reconstruction.log')
''')
    add('markdown', '''### Generate held-out auxiliary completion scores
The original 2,000 member/nonmember samples per shadow are reused. Only meta-fit,
calibration and auxiliary-test records are queried here; base-attack training
records are excluded. Repeated IDs retain the same role across all shadows.
Completion never receives membership labels as a conditioning variable.
''')
    add('code', '''# 3. K scores per held-out shadow observation, separately for every mask
from src.notebook_output import CellProgress, compact_output
UNCERTAINTY_LOG_DIR = UNCERTAINTY_DIR / 'logs'
uncertainty_auxiliary_tables = {}
uncertainty_query_audits = {}
auxiliary_progress = CellProgress()
heldout_ids = set().union(*(uncertainty_roles[role] for role in ['meta', 'calibration', 'test']))
for mask_name, reconstructor in uncertainty_reconstructors.items():
    mask_tables = []
    for shadow_name, shadow in shadow_results.items():
        selected = shadow_attack_data.loc[
            (shadow_attack_data['shadow_model'] == shadow_name)
            & shadow_attack_data['record_id'].isin(heldout_ids),
            ['record_id', 'membership']].sort_values('record_id').reset_index(drop=True)
        if selected.empty:
            continue
        assert selected['record_id'].is_unique
        candidates = acs_auxiliary.loc[selected['record_id']]
        oracle = make_probability_oracle(shadow['model'], shadow['preprocessor'], schema='acs')
        auxiliary_progress.update(f'Auxiliary scoring: {mask_name} / {shadow_name}, {len(selected):,} records, K={UNCERTAINTY_K}')
        with compact_output(UNCERTAINTY_LOG_DIR / 'auxiliary_scoring.log'):
            draw_scores, audit = completion_scores(oracle, candidates, reconstructor,
                attack_result['classifier'], k=UNCERTAINTY_K, seed=RECONSTRUCTION_SEED,
                walk_config=WALK_CONFIG)  # preserve existing shadow walk protocol
        table = summarize_scores(draw_scores)
        table['record_id'] = selected['record_id'].to_numpy()
        table['membership'] = selected['membership'].to_numpy()
        table['shadow_model'] = shadow_name
        table['role'] = [next(role for role in ['meta', 'calibration', 'test']
            if record_id in uncertainty_roles[role]) for record_id in table['record_id']]
        mask_tables.append(table)
        uncertainty_query_audits[f'{mask_name}/{shadow_name}'] = audit
    uncertainty_auxiliary_tables[mask_name] = pd.concat(mask_tables, ignore_index=True)
auxiliary_progress.update(f'Auxiliary scoring complete: {len(uncertainty_auxiliary_tables)} masks')
print('Detailed progress and warnings:', UNCERTAINTY_LOG_DIR / 'auxiliary_scoring.log')
''')
    add('code', '''# 4. Fit aggregators, freeze thresholds on calibration, evaluate auxiliary test
from src.notebook_output import compact_output
UNCERTAINTY_LOG_DIR = UNCERTAINTY_DIR / 'logs'
uncertainty_aggregators = {}
uncertainty_auxiliary_result_rows = []
for mask_name, table in uncertainty_auxiliary_tables.items():
    meta = table.loc[table['role'] == 'meta']
    calibration = table.loc[table['role'] == 'calibration']
    auxiliary_test = table.loc[table['role'] == 'test']
    with compact_output(UNCERTAINTY_LOG_DIR / 'aggregation.log'):
        learned, thresholds = fit_aggregation(meta, calibration, seed=SEED)
    uncertainty_aggregators[mask_name] = (learned, thresholds)
    with compact_output(UNCERTAINTY_LOG_DIR / 'aggregation.log'):
        metrics, predictions = evaluate_aggregation(auxiliary_test, learned, thresholds)
    uncertainty_auxiliary_result_rows.extend(metrics.assign(mask=mask_name,
        n_missing=len(MISSING_MASKS[mask_name]), dataset='2014_auxiliary_test').to_dict('records'))
print('Auxiliary test: macro averages across masks of the same size')
display(pd.DataFrame(uncertainty_auxiliary_result_rows).groupby(['n_missing', 'method'])[
    ['roc_auc', 'balanced_accuracy', 'tpr_at_1pct_fpr_roc']].mean().round(4))
''')
    add('markdown', '''### Final target evaluation: frozen methods and thresholds
Use the same target member/nonmember IDs as the complete-feature baseline.
Membership labels are attached only after scores are generated. The same K draws
feed single, averaging and both learned aggregators. Do not tune on these results.
The complete-feature and PFV results above are context; the controlled test for
the standard-deviation signal compares uncertainty against learned_mean here.
''')
    add('code', '''# 5. Query the existing ACS 2023 target; no target retraining
from src.notebook_output import CellProgress, compact_output
UNCERTAINTY_LOG_DIR = UNCERTAINTY_DIR / 'logs'
uncertainty_result_rows = list(uncertainty_auxiliary_result_rows)
target_candidates = pd.concat([expanded_partitions[name]
    for name in ['mia_members', 'mia_nonmembers']]).sort_index()
target_member_ids = set(expanded_partitions['mia_members'].index)
target_oracle = make_probability_oracle(expanded_model, expanded_preprocessor, schema='acs')
target_progress = CellProgress()
for mask_name, reconstructor in uncertainty_reconstructors.items():
    target_progress.update(f'Target scoring: {mask_name}, K={UNCERTAINTY_K}')
    with compact_output(UNCERTAINTY_LOG_DIR / 'target_scoring.log'):
        draw_scores, audit = completion_scores(target_oracle, target_candidates,
            reconstructor, attack_result['classifier'], k=UNCERTAINTY_K,
            seed=RECONSTRUCTION_SEED, walk_config=WALK_CONFIG,
            numeric_protocol=EXPANDED_NUMERIC_WALK)
    table = summarize_scores(draw_scores)
    table['record_id'] = target_candidates.index.to_numpy()
    table['membership'] = table['record_id'].isin(target_member_ids).astype(int)
    learned, thresholds = uncertainty_aggregators[mask_name]
    with compact_output(UNCERTAINTY_LOG_DIR / 'target_scoring.log'):
        metrics, predictions = evaluate_aggregation(table, learned, thresholds)
    uncertainty_result_rows.extend(metrics.assign(mask=mask_name,
        n_missing=len(MISSING_MASKS[mask_name]), dataset='2023_target').to_dict('records'))
    uncertainty_query_audits[f'{mask_name}/target'] = audit
uncertainty_results = pd.DataFrame(uncertainty_result_rows)
uncertainty_results.to_csv(UNCERTAINTY_DIR / 'comparison.csv', index=False)
target_progress.update(f'Target scoring complete: {len(uncertainty_reconstructors)} masks')
target_summary = uncertainty_results.loc[uncertainty_results['dataset'] == '2023_target']
print('Target: macro averages across masks of the same size')
display(target_summary.groupby(['n_missing', 'method'])[
    ['roc_auc', 'balanced_accuracy', 'tpr_at_1pct_fpr_roc']].mean().round(4))
print('Target ROC-AUC by mask; complete metrics are saved in comparison.csv')
display(target_summary.pivot(index='mask', columns='method', values='roc_auc').round(4))
_ = (UNCERTAINTY_DIR / 'protocol.json').write_text(json.dumps(dict(
    k=UNCERTAINTY_K, reconstruction_seed=RECONSTRUCTION_SEED, masks=MISSING_MASKS,
    n_shadow_models=N_SHADOW_MODELS, shadow_names=list(shadow_results),
    mask_selection='preselected coverage: 2 singles, 2 pairs, 2 triples, 1 quadruple; each field hidden four times',
    reconstruction='sequential random-forest leaf donor sampling; approximate predictive distribution',
    reconstruction_chain_order=MASK_FIELDS, score_std_ddof=0,
    auxiliary_year=2014, target_year=2023, target_labels_for_tuning=False,
    meta_source='half of base-attack validation record groups',
    calibration_source='other half of base-attack validation record groups',
    methods=['single', 'average', 'learned_mean', 'uncertainty'],
    queries=uncertainty_query_audits,
    limitations=['one reconstruction seed', 'no DP evaluation yet',
        'cross-year distribution shift', 'original blind-walk asymmetry retained',
        'no confirmatory significance claim; repeated shadow rows require grouped inference']),
    indent=2), encoding='utf-8')
''')
