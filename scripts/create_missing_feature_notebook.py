"""Build a separate ten-input MF-MIA notebook. Does not run experiments."""
import json
from pathlib import Path

cells = []
def md(text):
    cells.append(dict(cell_type='markdown', metadata={}, source=text.splitlines(keepends=True), id=f'mfmia-{len(cells)}'))
def code(text):
    cells.append(dict(cell_type='code',metadata={},source=text.splitlines(keepends=True),
                      execution_count=None,outputs=[],id=f'mfmia-{len(cells)}'))

md('''# Experiment 3 adaptation: 10-feature income baseline and Missing-Feature MIA
Target: California ACS 2023, nominal personal income >$50,000. Auxiliary: California ACS 2014 only.
Keep ACSIncome eligibility filters. Inputs: AGEP, COW, SCHL, MAR, OCCP, POBP, WKHP, SEX, RAC1P, HISP.
The former RELP field is unavailable in 2023; HISP supplies the tenth common field.
100,000 target training records; validation/test/MIA-member/MIA-nonmember sets each 10,000.
16 shadows cycle RF (6), gradient boosting (5), XGBoost (5), retaining Experiment 3 settings.
Attack: the same four paper-inspired features and RF, grouped auxiliary record splits.
This user-authorized run tests the full-feature baseline and all 15 subsets of 1–4 withheld
fields from COW, WKHP, MAR and SCHL. No income label is hidden or used as an input.

## Explicit adaptation choices
Masked AE trained on an 80% auxiliary split only, with the other 20% held out.
Architecture: input→128→64→64→128→output; ReLU hidden layers, MSE on all encoded coordinates,
50 epochs, Adam lr=.001, batch size 256. Numerics MinMax-scaled, categories one-hot encoded.
Uniformly sample 1–4 whole original fields per row from all ten predictors each training iteration;
replace all coordinates for those fields with -1. This sampling rule and decoding are project choices.
Decode categorical blocks by argmax; clip numeric reconstructions to auxiliary train ranges and round.
Only fill withheld fields; preserve observed inputs. Candidate labels and membership never enter the AE.
Train the attack on complete auxiliary records and freeze its classifier and auxiliary-selected thresholds
for every missing-feature scenario (paper Sections 4.2–4.3). Do not tune using target membership results.
The blind walk starts at each completed record and uses the existing native-field perturbation protocol.
Report per-mask results and macro averages by missing-field count, plus imputation error.
This is an income/ACS adaptation; the paper used gender/GSS with a different feature schema.
''')
code('''from pathlib import Path
import itertools, json, time, joblib
import numpy as np
import pandas as pd
import torch
from IPython.display import display
from src.missing_feature_mia import (FEATURES, WITHHELD, LABEL, load_year, prepare,
    train_shadows, MaskedAutoencoder, run_mask_evaluations)
from src.data import split_acs_data, subset_target_training
from src.exp3_checkpoints import cached_target_train, save
from src.training import evaluate_model
from src.shadow_models import summarize_shadow_models
from src.paper_attack_features import (collect_shadow_paper_features,
    collect_target_paper_features, make_probability_oracle)
from src.membership_attack import train_membership_classifier, evaluate_attack
torch.set_num_threads(4)
SEED = 42
TRAIN_SIZE = 100_000
RUN_DIR = Path('artifacts/exp_3_missing_features_10inputs_2023_2014')
RUN_DIR.mkdir(parents=True, exist_ok=True)
TRAIN_CONFIG = dict(hidden_dims=(512,256,128),dropout=0.0,epochs=100,batch_size=256,
    learning_rate=1e-3,weight_decay=0.0,seed=SEED,device='cpu',early_stopping=False,
    restore_best=False,monitor_train=True)
WALK_CONFIG = dict(max_steps=20,repeats=2,categorical_change_probability=.05,batch_size=512,seed=SEED)
SPLIT_CONFIG = dict(validation_size=10000,test_size=10000,mia_nonmember_size=10000,
    mia_member_size=10000,seed=SEED)
print('Target predictors:', FEATURES)
assert len(FEATURES)==10 and LABEL not in FEATURES
''')
md('## Load native-year data and create disjoint target partitions')
code('''source = load_year(2023)
auxiliary = load_year(2014)
assert set(source.index).isdisjoint(auxiliary.index)
partitions = subset_target_training(split_acs_data(source,**SPLIT_CONFIG),train_size=TRAIN_SIZE,seed=SEED)
assert set(partitions['mia_members'].index) <= set(partitions['train'].index)
for name in ['validation','test','mia_nonmembers']:
    assert set(partitions[name].index).isdisjoint(partitions['train'].index)
pp, arrays = prepare(partitions)
save(RUN_DIR/'partitions_preprocessor.joblib',dict(preprocessor=pp,
    partitions={name:frame.index.to_numpy() for name,frame in partitions.items()}))
print('Eligible 2023:',len(source),'| Eligible 2014:',len(auxiliary))
print('Encoded target inputs:',arrays['train']['X'].shape[1])
display(pd.DataFrame({name:dict(records=len(frame),positive_rate=frame[LABEL].mean())
    for name,frame in partitions.items()}).T)
''')
md('## Train the same target architecture for 100 epochs; retain final weights')
code('''target, target_history, target_metadata = cached_target_train(arrays,TRAIN_CONFIG,RUN_DIR)
target_metrics = {name:evaluate_model(target,arrays[name]['X'],arrays[name]['y'])
    for name in ['train','validation','test']}
display(pd.DataFrame(target_metrics).T)
''')
md('## Train 16 auxiliary shadows and verify the native-input query interface')
code('''shadows = train_shadows(auxiliary,RUN_DIR,count=16,seed=SEED)
for name,result in shadows.items():
    audit = result['attack_data'].iloc[:32]
    oracle = make_probability_oracle(result['model'],result['preprocessor'],schema='acs')
    np.testing.assert_allclose(oracle(auxiliary.loc[audit.record_id,FEATURES]),
        audit[['p_income_le_50k','p_income_gt_50k']].to_numpy(),rtol=1e-5,atol=1e-6)
summarize_shadow_models(shadows).to_csv(RUN_DIR/'shadow_metrics.csv')
display(summarize_shadow_models(shadows))
''')
md('## Full-feature membership attack: freeze classifier and auxiliary thresholds')
code('''shadow_features,shadow_diagnostics = collect_shadow_paper_features(
    shadows,auxiliary,per_class=2000,config=WALK_CONFIG,seed=SEED)
attack = train_membership_classifier(shadow_features,seed=SEED,algorithm='random_forest',feature_set='paper')
save(RUN_DIR/'attack.joblib',attack)
target_features,target_diagnostics = collect_target_paper_features(target,pp,partitions,config=WALK_CONFIG,seed=SEED)
baseline,baseline_scores = evaluate_attack(attack['classifier'],target_features,
    threshold=attack['accuracy_threshold'],low_fpr_threshold=attack['low_fpr_threshold'])
save(RUN_DIR/'baseline.joblib',dict(features=target_features,scores=baseline_scores,metrics=baseline))
display(pd.DataFrame({'full_feature_target_mia':baseline}).T)
''')
md('## Train the auxiliary-only masked autoencoder')
code('''completer = MaskedAutoencoder().fit(auxiliary,seed=SEED,epochs=50)
assert set(completer.train_ids).isdisjoint(completer.validation_ids)
assert set(completer.train_ids).isdisjoint(source.index)
save(RUN_DIR/'masked_autoencoder.joblib',completer)
pd.DataFrame(completer.history).to_csv(RUN_DIR/'autoencoder_history.csv',index=False)
display(pd.DataFrame(completer.history).iloc[::10])
''')
md('''## Evaluate all 15 missing-feature combinations on the same target candidates
Ground-truth hidden values are removed before completion. They are used afterwards only for
reporting imputation quality, with no tuning. Known income labels supply the attack's
true-class features, as in the existing baseline. Each mask's scores and diagnostics are saved.
''')
code('''candidates = pd.concat([partitions['mia_members'].assign(membership=1),
    partitions['mia_nonmembers'].assign(membership=0)]).sample(frac=1,random_state=SEED)
assert candidates.membership.value_counts().to_dict()=={0:10000,1:10000}
assert candidates.index.tolist()==target_features.record_id.tolist()
results = run_mask_evaluations(completer,target,pp,candidates,attack,baseline,RUN_DIR,WALK_CONFIG)
display(results[['missing_count','missing_fields','accuracy','roc_auc','tpr_at_1pct_fpr_roc',
    'tpr_at_shadow_threshold','fpr_at_shadow_threshold']])
''')
md('## Save macro averages, configuration, and execution status')
code('''columns = ['accuracy','balanced_accuracy','roc_auc','tpr_at_1pct_fpr_roc',
    'tpr_at_shadow_threshold','fpr_at_shadow_threshold','imputation_encoded_mse']
averages = results.groupby('missing_count')[columns].mean()
averages.to_csv(RUN_DIR/'missing_count_averages.csv')
display(averages)
summary = dict(execution_status='all_cells_completed',target_year=2023,auxiliary_year=2014,
    target_variable='nominal PINCP > 50000',features=FEATURES,withheld_fields=WITHHELD,
    train_size=TRAIN_SIZE,n_shadow_models=16,shadow_algorithms=['random_forest','gradient_boosting','xgboost'],
    training=target_metadata,target_metrics=target_metrics,baseline_mia=baseline,
    split_config=SPLIT_CONFIG,walk_config=WALK_CONFIG,shadow_feature_records_per_membership=2000,
    ae=dict(epochs=50,learning_rate=.001,batch_size=256,encoder=[128,64],decoder=[64,128],
        loss='MSE on all encoded fields',auxiliary_training_records=len(completer.train_ids),
        auxiliary_validation_records=len(completer.validation_ids),
        masks='uniform 1..4 distinct original fields per row from all ten fields',
        decode='categorical argmax; numeric range clipping and integer rounding'),
    classifier_and_thresholds_frozen=True,mask_results=results.to_dict(orient='records'),
    averages_by_missing_count=averages.reset_index().to_dict(orient='records'))
(RUN_DIR/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
print('Completed experiment saved to',RUN_DIR)
''')
from adult_missing_feature_cells import append_adult_cells
append_adult_cells(cells)
notebook=dict(cells=cells,metadata=dict(kernelspec=dict(display_name='Python 3 (ipykernel)',
    language='python',name='python3'),language_info=dict(name='python',version='3.12')),
    nbformat=4,nbformat_minor=5)
path=Path(__file__).resolve().parents[1]/'exp_3_missing_features_10inputs.ipynb'
path.write_text(json.dumps(notebook,indent=1)+'\n',encoding='utf-8')
print(path)
