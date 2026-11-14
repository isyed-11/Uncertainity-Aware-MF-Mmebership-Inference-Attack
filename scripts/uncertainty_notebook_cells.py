"""Append reviewable uncertainty-study cells; contains no execution runner."""


def append_uncertainty_cells(add):
    add('markdown', '''## Uncertainty-aware MF-MIA: study protocol
Research question: does reconstruction-induced score standard deviation improve
membership inference beyond the mean? The following cells are additional experiments;
they are not executed when this notebook is generated.

Use the same 15 masks, target, 16 ACS 2014 shadows, record assignments, four base
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

K=20 and one reconstruction seed are predeclared exploratory settings, not a
confirmatory multi-seed study. All 15 masks are reported without choosing a winner
on target results. Raw draws are saved for later paired/record-grouped analysis.
The original commuting walk asymmetry is retained and limits interpretation.
Common walk seeds reduce additional Monte Carlo noise; score spread can still
include completion-dependent effects of the finite blind-walk procedure.

This section can be expensive: each candidate is evaluated K times per mask,
with up to 41 record queries per completion under the original walk settings.
DP training/evaluation and repeated-seed confirmation are not implemented here.
''')
    add('code', '''# 1. Fixed configuration and disjoint auxiliary roles
import joblib
from sklearn.model_selection import train_test_split
from src.uncertainty_mf_mia import (
    ConditionalDonorChain, auxiliary_roles, completion_scores, summarize_scores,
    fit_aggregation, evaluate_aggregation,
)

UNCERTAINTY_K = 20
RECONSTRUCTION_SEED = SEED + 1000
UNCERTAINTY_DIR = RUN_DIR / 'uncertainty_mf_mia'
UNCERTAINTY_DIR.mkdir(parents=True, exist_ok=True)
uncertainty_roles = auxiliary_roles(acs_auxiliary, attack_result, seed=SEED)
recon_fit_ids, recon_check_ids = train_test_split(
    sorted(uncertainty_roles['reconstruction']), test_size=0.2, random_state=SEED)
recon_training = acs_auxiliary.loc[recon_fit_ids, EXPANDED_FEATURES]
recon_check = acs_auxiliary.loc[recon_check_ids, EXPANDED_FEATURES]
for role, ids in uncertainty_roles.items():
    pd.Series(sorted(ids), name='record_id').to_csv(
        UNCERTAINTY_DIR / f'{role}_ids.csv', index=False)
for role, ids in [('reconstruction_fit', recon_fit_ids), ('reconstruction_check', recon_check_ids)]:
    pd.Series(ids, name='record_id').to_csv(UNCERTAINTY_DIR / f'{role}_ids.csv', index=False)
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
reconstruction_audit_rows = []
for mask_index, (mask_name, hidden) in enumerate(MISSING_MASKS.items(), start=1):
    mask_started = time.monotonic()
    print(f'[{mask_index}/{len(MISSING_MASKS)}] Reconstruction mask: {mask_name}', flush=True)
    reconstructor = ConditionalDonorChain(hidden, seed=RECONSTRUCTION_SEED)
    reconstructor.fit(recon_training)
    uncertainty_reconstructors[mask_name] = reconstructor
    samples = []
    for draw in range(UNCERTAINTY_K):
        print(f'  {mask_name}: reconstruction audit draw {draw + 1}/{UNCERTAINTY_K}', flush=True)
        samples.append(reconstructor.sample(recon_check, seed=RECONSTRUCTION_SEED + draw)[hidden])
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
    joblib.dump(reconstructor, UNCERTAINTY_DIR / f'{mask_name}_reconstructor.joblib')
    print(f'[{mask_index}/{len(MISSING_MASKS)}] {mask_name}: saved; elapsed {time.monotonic() - mask_started:.1f}s', flush=True)
reconstruction_audit = pd.DataFrame(reconstruction_audit_rows)
display(reconstruction_audit)
reconstruction_audit.to_csv(UNCERTAINTY_DIR / 'reconstruction_audit.csv', index=False)
''')
    add('markdown', '''### Generate held-out auxiliary completion scores
The original 2,000 member/nonmember samples per shadow are reused. Only meta-fit,
calibration and auxiliary-test records are queried here; base-attack training
records are excluded. Repeated IDs retain the same role across all shadows.
Completion never receives membership labels as a conditioning variable.
''')
    add('code', '''# 3. K scores per held-out shadow observation, separately for every mask
uncertainty_auxiliary_tables = {}
uncertainty_query_audits = {}
heldout_ids = set().union(*(uncertainty_roles[role] for role in ['meta', 'calibration', 'test']))
for mask_name, reconstructor in uncertainty_reconstructors.items():
    mask_tables = []
    mask_dir = UNCERTAINTY_DIR / mask_name
    mask_dir.mkdir(parents=True, exist_ok=True)
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
        draw_scores, audit = completion_scores(oracle, candidates, reconstructor,
            attack_result['classifier'], k=UNCERTAINTY_K, seed=RECONSTRUCTION_SEED,
            walk_config=WALK_CONFIG)  # preserve existing shadow walk protocol
        table = summarize_scores(draw_scores)
        table['record_id'] = selected['record_id'].to_numpy()
        table['membership'] = selected['membership'].to_numpy()
        table['shadow_model'] = shadow_name
        table['role'] = [next(role for role in ['meta', 'calibration', 'test']
            if record_id in uncertainty_roles[role]) for record_id in table['record_id']]
        np.savez_compressed(mask_dir / f'{shadow_name}_draws.npz', scores=draw_scores)
        table.to_csv(mask_dir / f'{shadow_name}_rows.csv', index=False)
        mask_tables.append(table)
        uncertainty_query_audits[f'{mask_name}/{shadow_name}'] = audit
    uncertainty_auxiliary_tables[mask_name] = pd.concat(mask_tables, ignore_index=True)
''')
    add('code', '''# 4. Fit aggregators, freeze thresholds on calibration, evaluate auxiliary test
uncertainty_aggregators = {}
uncertainty_auxiliary_result_rows = []
for mask_name, table in uncertainty_auxiliary_tables.items():
    meta = table.loc[table['role'] == 'meta']
    calibration = table.loc[table['role'] == 'calibration']
    auxiliary_test = table.loc[table['role'] == 'test']
    learned, thresholds = fit_aggregation(meta, calibration, seed=SEED)
    uncertainty_aggregators[mask_name] = (learned, thresholds)
    metrics, predictions = evaluate_aggregation(auxiliary_test, learned, thresholds)
    uncertainty_auxiliary_result_rows.extend(metrics.assign(mask=mask_name,
        n_missing=len(MISSING_MASKS[mask_name]), dataset='2014_auxiliary_test').to_dict('records'))
    auxiliary_test.assign(**{f'score_{name}': values for name, values in predictions.items()}).to_csv(
        UNCERTAINTY_DIR / mask_name / 'auxiliary_test_predictions.csv', index=False)
    joblib.dump(dict(models=learned, thresholds=thresholds),
        UNCERTAINTY_DIR / mask_name / 'aggregation.joblib')
display(pd.DataFrame(uncertainty_auxiliary_result_rows))
''')
    add('markdown', '''### Final target evaluation: frozen methods and thresholds
Use the same target member/nonmember IDs as the complete-feature baseline.
Membership labels are attached only after scores are generated. The same K draws
feed single, averaging and both learned aggregators. Do not tune on these results.
The complete-feature and PFV results above are context; the controlled test for
the standard-deviation signal compares uncertainty against learned_mean here.
''')
    add('code', '''# 5. Query the existing ACS 2023 target; no target retraining
uncertainty_result_rows = list(uncertainty_auxiliary_result_rows)
target_candidates = pd.concat([expanded_partitions[name]
    for name in ['mia_members', 'mia_nonmembers']]).sort_index()
target_member_ids = set(expanded_partitions['mia_members'].index)
target_oracle = make_probability_oracle(expanded_model, expanded_preprocessor, schema='acs')
for mask_name, reconstructor in uncertainty_reconstructors.items():
    draw_scores, audit = completion_scores(target_oracle, target_candidates,
        reconstructor, attack_result['classifier'], k=UNCERTAINTY_K,
        seed=RECONSTRUCTION_SEED, walk_config=WALK_CONFIG,
        numeric_protocol=EXPANDED_NUMERIC_WALK)
    table = summarize_scores(draw_scores)
    table['record_id'] = target_candidates.index.to_numpy()
    table['membership'] = table['record_id'].isin(target_member_ids).astype(int)
    learned, thresholds = uncertainty_aggregators[mask_name]
    metrics, predictions = evaluate_aggregation(table, learned, thresholds)
    uncertainty_result_rows.extend(metrics.assign(mask=mask_name,
        n_missing=len(MISSING_MASKS[mask_name]), dataset='2023_target').to_dict('records'))
    np.savez_compressed(UNCERTAINTY_DIR / mask_name / 'target_draws.npz', scores=draw_scores)
    table.assign(**{f'score_{name}': values for name, values in predictions.items()}).to_csv(
        UNCERTAINTY_DIR / mask_name / 'target_predictions.csv', index=False)
    uncertainty_query_audits[f'{mask_name}/target'] = audit
uncertainty_results = pd.DataFrame(uncertainty_result_rows)
uncertainty_results.to_csv(UNCERTAINTY_DIR / 'comparison.csv', index=False)
display(uncertainty_results)
display(uncertainty_results.pivot(index=['dataset', 'n_missing', 'mask'],
    columns='method', values='roc_auc'))
(UNCERTAINTY_DIR / 'protocol.json').write_text(json.dumps(dict(
    k=UNCERTAINTY_K, reconstruction_seed=RECONSTRUCTION_SEED, masks=MISSING_MASKS,
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
