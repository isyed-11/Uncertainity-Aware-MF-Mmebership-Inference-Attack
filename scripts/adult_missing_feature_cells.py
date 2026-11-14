"""Append reviewable Adult baseline/completion cells without executing them."""
import json
from pathlib import Path


def append_adult_cells(cells):
    if any(c.get('id','').startswith('adult-mfmia-') for c in cells):
        raise ValueError('Adult comparison already appended.')
    def md(text):
        cells.append(dict(cell_type='markdown',metadata={},source=text.splitlines(keepends=True),id=f'adult-mfmia-{len(cells)}'))
    def code(text):
        cells.append(dict(cell_type='code',metadata={},source=text.splitlines(keepends=True),
            execution_count=None,outputs=[],id=f'adult-mfmia-{len(cells)}'))
    md('''## UCI Adult auxiliary comparison — prepared, execution pending approval
Use all 48,842 Adult records for a separate auxiliary branch. Keep the same saved
10-input ACS2023 target and membership candidates. Train 16 native Adult shadows
(12 predictors) with the same RF/gradient-boosting/XGBoost cycle and 40/20/40 splits.
Extract attack features from every member/non-member (no 2,000-row cap).
This branch therefore changes auxiliary population, input schema and attack sample count;
it is not an isolated auxiliary-year comparison.

The Adult completion AE uses seven common predictors only: age, workclass, education,
marital status, weekly hours, sex, and race. See data/adult_missing_feature_mapping.md.
Observed target OCCP, POBP and HISP are preserved in target queries but never passed
to the Adult completion model. Only COW, WKHP, MAR and SCHL may be masked.
Adult cannot recover finer distinctions such as nonprofit/private or GED/diploma;
declared representative ACS codes are used. These are approximate mappings.
No target records or ACS2014 records train the Adult AE, classifier or thresholds.
The Adult attack is trained on complete Adult shadow records, then frozen for all 15 masks.
Imputation MSE uses the seven-field common encoding and is not directly comparable
to the ten-field ACS AE's encoded MSE. Raw-field accuracy/MAE are saved per mask.
''')
    code('''from src.shadow_models import load_adult_data, PAPER_SHADOW_ALGORITHMS
from src.exp3_checkpoints import cached_expanded_shadows
from src.adult_missing_features import load_aligned_adult, AdultCompletion, SHARED
ADULT_RUN_DIR = RUN_DIR / 'adult_auxiliary'
ADULT_RUN_DIR.mkdir(parents=True,exist_ok=True)
adult_native = load_adult_data()
adult_aligned = load_aligned_adult()
assert len(adult_native)==len(adult_aligned)==48842
assert adult_native.index.equals(adult_aligned.index)
np.testing.assert_array_equal(adult_native[LABEL],adult_aligned[LABEL])
assert set(adult_native.index).isdisjoint(source.index)
assert set(adult_native.index).isdisjoint(auxiliary.index)
print('Adult rows:',len(adult_native),'| Completion fields:',SHARED)
adult_shadows = cached_expanded_shadows(adult_native,seed=SEED,n_models=16,
    run_dir=ADULT_RUN_DIR,algorithms=PAPER_SHADOW_ALGORITHMS,schema='adult')
for name,result in adult_shadows.items():
    groups=[set(ids) for ids in result['partition_ids'].values()]
    assert set.union(*groups)==set(adult_native.index)
    assert sum(map(len,groups))==len(adult_native)
    audit=result['attack_data'].iloc[:32]
    oracle=make_probability_oracle(result['model'],result['preprocessor'],schema='adult')
    np.testing.assert_allclose(oracle(adult_native.loc[audit.record_id]),
        audit[['p_income_le_50k','p_income_gt_50k']].to_numpy(),rtol=1e-5,atol=1e-6)
summarize_shadow_models(adult_shadows).to_csv(ADULT_RUN_DIR/'shadow_metrics.csv')
display(summarize_shadow_models(adult_shadows))
''')
    md('## Fit the Adult membership attack and evaluate its full-feature baseline')
    code('''adult_shadow_features,adult_shadow_diagnostics=collect_shadow_paper_features(
    adult_shadows,adult_native,per_class=None,config=WALK_CONFIG,seed=SEED)
assert len(adult_shadow_features)==sum(len(r['attack_data']) for r in adult_shadows.values())
adult_attack=train_membership_classifier(adult_shadow_features,seed=SEED,
    algorithm='random_forest',feature_set='paper')
save(ADULT_RUN_DIR/'attack.joblib',adult_attack)
# Reuse the same unmasked target features: no additional baseline target queries.
adult_baseline,adult_baseline_scores=evaluate_attack(adult_attack['classifier'],target_features,
    threshold=adult_attack['accuracy_threshold'],low_fpr_threshold=adult_attack['low_fpr_threshold'])
save(ADULT_RUN_DIR/'baseline.joblib',dict(features=target_features,
    scores=adult_baseline_scores,metrics=adult_baseline))
display(pd.DataFrame({'ACS2014 auxiliary':baseline,'Adult auxiliary':adult_baseline}).T)
''')
    md('## Train the 50-epoch completion AE using Adult shared fields only')
    code('''adult_completer=AdultCompletion().fit(adult_aligned,seed=SEED,epochs=50)
assert set(adult_completer.train_ids).isdisjoint(adult_completer.validation_ids)
assert set(adult_completer.train_ids).isdisjoint(source.index)
save(ADULT_RUN_DIR/'masked_autoencoder.joblib',adult_completer)
pd.DataFrame(adult_completer.history).to_csv(ADULT_RUN_DIR/'autoencoder_history.csv',index=False)
display(pd.DataFrame(adult_completer.history).iloc[::10])
''')
    md('## Evaluate the same 15 masks using Adult-only completion and attack')
    code('''adult_results=run_mask_evaluations(adult_completer,target,pp,candidates,adult_attack,
    adult_baseline,ADULT_RUN_DIR,WALK_CONFIG)
adult_averages=adult_results.groupby('missing_count')[columns].mean()
adult_averages.to_csv(ADULT_RUN_DIR/'missing_count_averages.csv')
display(adult_averages)
comparison=pd.concat({'ACS2014':averages,'UCI Adult':adult_averages},names=['auxiliary_source'])
comparison.to_csv(RUN_DIR/'auxiliary_missing_feature_comparison.csv')
adult_summary=dict(execution_status='all_adult_cells_completed',auxiliary='UCI Adult',
    auxiliary_records=len(adult_native),n_shadow_models=16,native_shadow_predictors=12,
    shared_completion_predictors=SHARED,target_predictors=FEATURES,target_records=TRAIN_SIZE,
    target_year=2023,target_summary='../summary.json',full_target_features_reused=True,
    attack_feature_sampling=None,mapping_document='data/adult_missing_feature_mapping.md',
    mapping_limitations=['private/nonprofit merged','GED/diploma merged','coarse education and race',
        'OCCP,POBP,HISP excluded from completion conditions but observed values preserved'],
    ae_training_records=len(adult_completer.train_ids),ae_validation_records=len(adult_completer.validation_ids),
    ae_epochs=50,classifier_and_thresholds_frozen=True,baseline_mia=adult_baseline,
    mask_results=adult_results.to_dict(orient='records'),
    averages_by_missing_count=adult_averages.reset_index().to_dict(orient='records'))
(ADULT_RUN_DIR/'summary.json').write_text(json.dumps(adult_summary,indent=2),encoding='utf-8')
display(comparison[['accuracy','roc_auc','tpr_at_1pct_fpr_roc']])
print('Adult comparison saved to',ADULT_RUN_DIR)
''')


if __name__=='__main__':
    path=Path(__file__).resolve().parents[1]/'exp_3_missing_features_10inputs.ipynb'
    notebook=json.loads(path.read_text(encoding='utf-8'))
    append_adult_cells(notebook['cells'])
    path.write_text(json.dumps(notebook,indent=1)+'\n',encoding='utf-8')
    print('Appended Adult cells; preserved existing ACS outputs. No cells executed.')
