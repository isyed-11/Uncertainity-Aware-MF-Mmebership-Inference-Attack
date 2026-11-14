"""Class/difficulty analysis of saved Experiment 2 outputs; no model queries or fitting."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'artifacts/exp2_attack_repair'
REF=RUN/'references'
SCORES=RUN/'threshold_diagnostics'
OUT=RUN/'class_difficulty_analysis'
OUT.mkdir(parents=True,exist_ok=True)
results=json.loads((REF/'reference_results.json').read_text())
source=pd.read_csv(ROOT/'data/processed/acs_income_ca_2018.csv',usecols=['income_gt_50k'])
assignments=np.load(REF/'reference_assignments.npz',allow_pickle=True)
ids=assignments['record_ids'].astype(int)
masks=assignments['in_masks']
margins=assignments['auxiliary_margins']
assert masks.shape==margins.shape
out_count=(~masks).sum(axis=0)
assert (out_count==8).all()
auxiliary_difficulty=pd.Series(np.where(~masks,margins,0).sum(axis=0)/out_count,index=ids)
training_ids=pd.read_csv(REF/'auxiliary_attack_train.csv',usecols=['record_id']).record_id.unique()
# Quartile cuts use one OUT-reference mean per AUXILIARY TRAIN record, never target labels/scores.
cuts=np.quantile(auxiliary_difficulty.loc[training_ids],[.25,.5,.75])
assert np.all(np.diff(cuts)>0)
labels=['hardest','moderate','easy','easiest']
bins=np.r_[-np.inf,cuts,np.inf]
frames={}
for dataset,filename in [('auxiliary_validation','auxiliary_attack_validation.csv'),
                         ('auxiliary_test','auxiliary_attack_test.csv'),
                         ('target','target_calibrated_features.csv')]:
    frame=pd.read_csv(REF/filename)
    frame['income_class']=source.loc[frame.record_id.astype(int),'income_gt_50k'].to_numpy()
    frame['income_label']=np.where(frame.income_class.eq(1),'> $50k','<= $50k')
    if dataset=='target':
        frame['reference_mean_margin']=frame.margin-frame.reference_margin_delta
        np.testing.assert_allclose(frame.reference_mean_margin,assignments['target_margins'].mean(axis=0),rtol=1e-7,atol=1e-7)
    else:
        frame['reference_mean_margin']=frame.record_id.map(auxiliary_difficulty)
    assert frame.reference_mean_margin.notna().all()
    frame['difficulty']=pd.cut(frame.reference_mean_margin,bins=bins,labels=labels,include_lowest=True)
    assert frame.difficulty.notna().all()
    for method in ['stable_margin_rf','calibrated_rf']:
        saved=pd.read_csv(SCORES/f'{method}_{dataset}_scores.csv')
        assert np.array_equal(saved.record_id,frame.record_id)
        assert np.array_equal(saved.membership,frame.membership)
        frame[f'{method}_score']=saved.score.to_numpy()
    frames[dataset]=frame


def summarize(frame,method):
    y=frame.membership.to_numpy(dtype=int)
    scores=frame[f'{method}_score'].to_numpy()
    threshold=results['methods'][method]['target']['threshold']
    low=results['methods'][method]['target']['low_fpr_threshold']
    pred=scores>=threshold
    members=y==1;nonmembers=y==0
    row=dict(records=len(frame),unique_records=int(frame.record_id.nunique()),
        members=int(members.sum()),nonmembers=int(nonmembers.sum()),
        membership_fraction=float(y.mean()),accuracy=float((pred==y).mean()),
        reference_mean_margin_median=float(frame.reference_mean_margin.median()))
    if members.any() and nonmembers.any():
        fpr,tpr,_=roc_curve(y,scores,drop_intermediate=False)
        row.update(balanced_accuracy=float(.5*(pred[members].mean()+(~pred[nonmembers]).mean())),
            roc_auc=float(roc_auc_score(y,scores)),tpr_at_1pct_fpr_roc=float(tpr[fpr<=.01].max()),
            tpr_at_fixed_threshold=float(pred[members].mean()),fpr_at_fixed_threshold=float(pred[nonmembers].mean()),
            tpr_at_fixed_low_threshold=float((scores[members]>=low).mean()),
            fpr_at_fixed_low_threshold=float((scores[nonmembers]>=low).mean()),
            member_score_median=float(np.median(scores[members])),nonmember_score_median=float(np.median(scores[nonmembers])))
    return row


rows=[]
for dataset,frame in frames.items():
    groups=[('all','all',frame)]
    groups += [('income_class',label,part) for label,part in frame.groupby('income_label')]
    groups += [('difficulty',str(label),part) for label,part in frame.groupby('difficulty',observed=True)]
    groups += [('class_x_difficulty',f'{income} | {difficulty}',part)
               for (income,difficulty),part in frame.groupby(['income_label','difficulty'],observed=True)]
    for method in ['stable_margin_rf','calibrated_rf']:
        for group_type,group,part in groups:
            rows.append(dict(method=method,dataset=dataset,group_type=group_type,group=group,**summarize(part,method)))
report=pd.DataFrame(rows)
report.to_csv(OUT/'subgroup_metrics.csv',index=False)
# Target-only uncertainty is conditional on fixed models; no resampling of trained models.
rng=np.random.default_rng(42)
intervals=[]
for group_type in ['income_class','difficulty','class_x_difficulty']:
    if group_type=='income_class': grouped=frames['target'].groupby('income_label')
    elif group_type=='difficulty':grouped=frames['target'].groupby('difficulty',observed=True)
    else:grouped=frames['target'].groupby(['income_label','difficulty'],observed=True)
    for group,part in grouped:
        group=' | '.join(map(str,group)) if isinstance(group,tuple) else str(group)
        positive=part.loc[part.membership.eq(1),'calibrated_rf_score'].to_numpy()
        negative=part.loc[part.membership.eq(0),'calibrated_rf_score'].to_numpy()
        threshold=results['methods']['calibrated_rf']['target']['threshold']
        aucs=[];balanced=[]
        for _ in range(200):
            p=rng.choice(positive,len(positive),replace=True)
            q=rng.choice(negative,len(negative),replace=True)
            aucs.append(roc_auc_score(np.r_[np.ones(len(p)),np.zeros(len(q))],np.r_[p,q]))
            balanced.append(.5*((p>=threshold).mean()+(q<threshold).mean()))
        intervals.append(dict(group_type=group_type,group=group,
            auc_ci_low=float(np.quantile(aucs,.025)),auc_ci_high=float(np.quantile(aucs,.975)),
            balanced_accuracy_ci_low=float(np.quantile(balanced,.025)),balanced_accuracy_ci_high=float(np.quantile(balanced,.975))))
intervals=pd.DataFrame(intervals)
intervals.to_csv(OUT/'target_bootstrap_intervals.csv',index=False)
frames['target'][['record_id','membership','income_label','reference_mean_margin','difficulty',
    'stable_margin_rf_score','calibrated_rf_score']].to_csv(OUT/'target_groups.csv',index=False)
summary=dict(difficulty_definition='Mean true-class margin on OUT reference models; higher means easier. Four quartiles fixed using unique auxiliary TRAIN records only.',
    auxiliary_training_unique_records=len(training_ids),difficulty_cut_points=cuts.tolist(),
    thresholds='Existing global auxiliary-selected thresholds unchanged in every subgroup.',
    access='Saved CSV and NPZ outputs only; no model queries, training, or notebook execution.',
    caveats=['Exploratory subgroup analysis, no subgroup attack chosen or fitted.',
             'Auxiliary rows repeat each record across reference models; use unique-record counts when interpreting support.',
             '200 stratified record-bootstrap intervals condition on fixed models and do not include model-training or household dependence uncertainty.',
             'Low-FPR subgroup summaries can be unstable for small nonmember groups.'],
    target=report.loc[(report.dataset=='target') & (report.method=='calibrated_rf')].to_dict(orient='records'),
    target_intervals=intervals.to_dict(orient='records'))
(OUT/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
for group_type in ['all','income_class','difficulty','class_x_difficulty']:
    print('\nCALIBRATED RF:',group_type)
    print(report.loc[(report.method=='calibrated_rf') & report.group_type.eq(group_type),
        ['dataset','group','records','members','nonmembers','accuracy','balanced_accuracy','roc_auc',
         'tpr_at_1pct_fpr_roc','tpr_at_fixed_low_threshold','fpr_at_fixed_low_threshold']].to_string(index=False))
print('\nDifficulty cuts:',cuts)
print('\nTarget intervals:')
print(intervals.to_string(index=False))
print('Saved:',OUT)
