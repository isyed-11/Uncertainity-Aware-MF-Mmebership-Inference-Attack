"""Prepare Experiment 2 for ACS2023 plus Adult and ACS2014 baseline attacks.

Edits notebook source and clears obsolete outputs; does not execute cells.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
path = ROOT/'experiments_2_overfit.ipynb'
original = ROOT/'artifacts/exp2_attack_repair/before_repair.ipynb'
notebook = json.loads(original.read_text(encoding='utf-8'))
archive = ROOT/'artifacts/exp2_2023_adult_2014_baseline/restored_2018_before_update.ipynb'
archive.parent.mkdir(parents=True,exist_ok=True)
if not archive.exists():
    archive.write_bytes(original.read_bytes())

cells = []
for cell in notebook['cells']:
    text = ''.join(cell['source'])
    if not text.strip():
        continue
    if cell['cell_type']=='code':
        cell['execution_count']=None
        cell['outputs']=[]
    if cell.get('id')=='overfit-notes':
        text='''# Experiment 2: ACS2023 income target with Adult and ACS2014 baseline MIA
Retain the restored Experiment 2 target: 145,000 training records, MLP hidden
layers 512â†’256â†’128, 100 epochs, batch size 256, learning rate .001, no dropout
or weight decay, seed 42, final weights retained. Target data changes to California ACS2023.
Each of validation, test, MIA members and MIA nonmembers has 10,000 records.
Keep the original ten predictor concepts. ACS2023 uses native RELSHIPP relationship
categories where ACS2018/2014 used RELP. Each model fits its own native encoding;
no equivalence between old and new relationship category codes is assumed.
The other nine fields remain AGEP, COW, SCHL, MAR, OCCP, POBP, WKHP, SEX and RAC1P.
Income label is nominal PINCP >$50,000; retain original ACSIncome eligibility.

Compare two separate auxiliary attacks: UCI Adult and California ACS2014.
Each uses eight shadows cycling the original five algorithms: XGBoost, random forest, MLP, decision tree,
logistic regression, retaining their original settings, seeds and 40/20/40 splits.
Extract 2,000 known members and 2,000 nonmembers per shadow. Group attack splits
by auxiliary record ID and choose thresholds on auxiliary validation only.
Both attacks use logit, blind walk, entropy and cross-entropy with the same RF.
The ACS2014 attack reuses the same 20,000 target feature rows as the Adult attack.
This notebook is prepared for review. Ask the user before executing cells.
'''
    elif cell.get('id')=='overfit-imports':
        text=text.replace('from src.data import load_acs_income, split_acs_data, subset_target_training, preprocess_partitions',
            'from src.data import split_acs_data, subset_target_training\n'
            'from src.exp2_year_baseline import (load_baseline_year, features_for_year,\n'
            '    preprocess_partitions, train_acs_shadows)')
        text='from pathlib import Path\nimport json\nimport joblib\nimport numpy as np\nimport torch\ntorch.set_num_threads(4)\nfrom IPython.display import display\n'+text
        text+='''
N_SHADOW_MODELS = 8
RUN_DIR = Path('artifacts/exp2_2023_adult_2014_baseline_8shadows')
RUN_DIR.mkdir(parents=True,exist_ok=True)
TARGET_FEATURES = features_for_year(2023)
assert len(TARGET_FEATURES)==10 and 'RELSHIPP' in TARGET_FEATURES and 'RELP' not in TARGET_FEATURES
print('Target predictors:',TARGET_FEATURES)
'''
    elif cell.get('id')=='overfit-partitions':
        text=text.replace("load_acs_income('data/processed/acs_income_ca_2018.csv')",'load_baseline_year(2023)')
        text+='''
assert len(overfit_partitions['train'])==OVERFIT_TRAIN_SIZE
assert set(overfit_partitions['mia_members'].index)<=set(overfit_partitions['train'].index)
for name in ['validation','test','mia_nonmembers']:
    assert set(overfit_partitions[name].index).isdisjoint(overfit_partitions['train'].index)
'''
    elif cell.get('id')=='overfit-preprocess':
        text=text.replace('preprocess_partitions(overfit_partitions)',
            'preprocess_partitions(overfit_partitions,TARGET_FEATURES)')
    elif cell.get('id')=='overfit-train':
        text+='''
from src.training import save_target_model
save_target_model(overfit_model,overfit_metadata,RUN_DIR/'target_model.pt')
joblib.dump(overfit_preprocessor,RUN_DIR/'target_preprocessor.joblib')
'''
    elif cell.get('id')=='overfit-restored-membership-shadow-prerequisite':
        text=text.replace('the five Adult shadows','the eight Adult shadows')
        text=text.replace('train_shadow_models(adult_auxiliary, seed=42)',
            'train_shadow_models(adult_auxiliary, seed=42, n_models=N_SHADOW_MODELS)')
        text+="\nassert len(shadow_results)==N_SHADOW_MODELS\n"
    else:
        text=text.replace('2018 target','2023 target').replace('ACS 2018 target','ACS 2023 target')
        if cell.get('id')=='overfit-restored-membership-attack-notes':
            text=text.replace('All experiment cells were executed with user authorization for this run.',
                              'The user authorized this execution; ask before future runs.')
            text=text.replace('five diverse Adult shadows','eight diverse Adult shadows')
            text=text.replace('This notebook trains its own attack and reports only its own target result; no cross-notebook results are reused or compared.',
                'The Adult and ACS2014 auxiliary classifiers each evaluate the same target candidates in this notebook.')
    cell['source']=text.splitlines(keepends=True)
    cells.append(cell)

def md(identifier,text):
    cells.append(dict(cell_type='markdown',metadata={},id=identifier,source=text.splitlines(keepends=True)))
def code(identifier,text):
    cells.append(dict(cell_type='code',metadata={},id=identifier,source=text.splitlines(keepends=True),
                      execution_count=None,outputs=[]))

md('exp2-2014-baseline-notes','''## Separate ACS2014 auxiliary baseline
Use California ACS2014 records only for this auxiliary attack. Preserve the same
five algorithm families and eight shadows, using the same settings and seed sequence as the Adult baseline. Native RELP
is retained for these shadows; the target uses native RELSHIPP. Each fitted encoder
is local to its own training records. Category codes do not pass between models.
The common representation transferred to the attack is the four query/output features.
''')
code('exp2-2014-shadows','''acs_2014_auxiliary=load_baseline_year(2014)
ACS_2014_FEATURES=features_for_year(2014)
assert len(ACS_2014_FEATURES)==10
assert set(acs_2014_auxiliary.index).isdisjoint(overfit_source.index)
acs_2014_shadows=train_acs_shadows(acs_2014_auxiliary,ACS_2014_FEATURES,seed=OVERFIT_SEED,n_models=N_SHADOW_MODELS)
assert len(acs_2014_shadows)==N_SHADOW_MODELS
from src.paper_attack_features import make_probability_oracle
for name,result in acs_2014_shadows.items():
    rows=result['attack_data'].iloc[:32]
    query=make_probability_oracle(result['model'],result['preprocessor'],schema='acs')
    np.testing.assert_allclose(query(acs_2014_auxiliary.loc[rows.record_id,ACS_2014_FEATURES]),
        rows[['p_income_le_50k','p_income_gt_50k']].to_numpy(),rtol=1e-5,atol=1e-6)
display(summarize_shadow_models(acs_2014_shadows))
''')
md('exp2-2014-attack-notes','''## Fit the ACS2014 attack and compare against the same target queries
Both attacks select thresholds using their own auxiliary validation records only.
Reuse target_attack_data from the Adult baseline to keep target candidates and
queries identical. Target membership labels do not fit either classifier or select settings.
''')
code('exp2-2014-attack','''acs_2014_attack_data,acs_2014_query_diagnostics=collect_shadow_paper_features(
    acs_2014_shadows,acs_2014_auxiliary,per_class=SHADOW_RECORDS_PER_CLASS,
    config=WALK_CONFIG,seed=OVERFIT_SEED)
acs_2014_attack=train_membership_classifier(acs_2014_attack_data,seed=OVERFIT_SEED,
    algorithm='random_forest',feature_set='paper')
acs_2014_target_metrics,acs_2014_target_scores=evaluate_attack(
    acs_2014_attack['classifier'],target_attack_data,
    threshold=acs_2014_attack['accuracy_threshold'],
    low_fpr_threshold=acs_2014_attack['low_fpr_threshold'])
baseline_comparison=pd.DataFrame({'UCI Adult auxiliary':target_attack_metrics,
    'ACS2014 auxiliary':acs_2014_target_metrics}).T
display(baseline_comparison[['accuracy','balanced_accuracy','roc_auc','tpr_at_1pct_fpr_roc',
    'tpr_at_shadow_threshold','fpr_at_shadow_threshold']])
''')
md('exp2-save-baselines-notes','## Save both baselines and exact target record assignments')
code('exp2-save-baselines','''baseline_comparison.to_csv(RUN_DIR/'baseline_comparison.csv')
summarize_shadow_models(shadow_results).to_csv(RUN_DIR/'adult_shadow_metrics.csv')
summarize_shadow_models(acs_2014_shadows).to_csv(RUN_DIR/'acs2014_shadow_metrics.csv')
for name,part in overfit_partitions.items():
    pd.Series(part.index,name='record_id').to_csv(RUN_DIR/(name+'_ids.csv'),index=False)
joblib.dump(dict(adult=attack_result,acs2014=acs_2014_attack),RUN_DIR/'attack_classifiers.joblib')
joblib.dump(dict(features=target_attack_data,adult_scores=target_membership_scores,
    acs2014_scores=acs_2014_target_scores),RUN_DIR/'target_mia_outputs.joblib')
summary=dict(execution_status='all_cells_completed',target_year=2023,target_train_size=OVERFIT_TRAIN_SIZE,
    target_features=TARGET_FEATURES,acs2014_features=ACS_2014_FEATURES,
    relationship_schema='Native RELSHIPP for 2023; native RELP for 2014; separate encodings',
    target_label='nominal PINCP > 50000',target_training=overfit_metadata,target_metrics=overfit_metrics,
    n_shadows_per_auxiliary=N_SHADOW_MODELS,shadow_algorithms=['xgboost','random_forest','mlp','decision_tree','logistic_regression'],
    shadow_records_per_membership=SHADOW_RECORDS_PER_CLASS,walk_config=WALK_CONFIG,
    split_config=SPLIT_CONFIG,adult_target_mia=target_attack_metrics,acs2014_target_mia=acs_2014_target_metrics,
    adult_auxiliary_attack_metrics=attack_result['metrics'],acs2014_auxiliary_attack_metrics=acs_2014_attack['metrics'])
(RUN_DIR/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
print('Both baselines saved to',RUN_DIR)
''')
notebook['cells']=cells
path.write_text(json.dumps(notebook,indent=1)+'\n',encoding='utf-8')
print('Prepared',path)
print('Original 2018 notebook preserved at',archive)
