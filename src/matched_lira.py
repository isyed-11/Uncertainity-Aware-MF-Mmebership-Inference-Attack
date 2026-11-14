"""Matched-size ACS2014 offline LiRA pilot; no import-time experiments.

Gaussian OUT log-likelihood with median locations and pooled fixed variance follows
https://github.com/tensorflow/privacy/blob/master/research/mi_lira_2021/plot.py
(generate_ours_offline, fix_variance=True). This is offline LiRA on a different
year and tabular schema, with one unaugmented query; not an online IN/OUT attack.
"""
import json
import hashlib
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import torch
from scipy.stats import norm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,balanced_accuracy_score,confusion_matrix

from .recent_data import load_recent_acs
from .data import FEATURES,TARGET,build_preprocessor,load_acs_income
from .occupation_mapping import load_occupation_crosswalk
from .training import train_target_model,save_target_model,build_mlp,evaluate_model
from .calibrated_attack import margin_features,balanced_group_splits,select_thresholds,score_metrics


def offline_gaussian_score(margin,location,scale):
    """Higher membership score means lower likelihood under the OUT Gaussian."""
    if not np.isfinite(scale) or scale<=0:
        raise ValueError('OUT standard deviation must be positive and finite.')
    return -norm.logpdf(margin,loc=location,scale=scale)


def evaluate_conditioned(frame,cutoffs,low_cutoffs,group_column):
    threshold=frame[group_column].map(cutoffs).to_numpy()
    low=frame[group_column].map(low_cutoffs).to_numpy()
    if np.isnan(threshold).any() or np.isnan(low).any():
        raise ValueError('Uncovered conditional threshold group.')
    y=frame.membership.to_numpy();scores=frame.lira_score.to_numpy()
    predictions=scores>=threshold;lp=scores>=low
    result=score_metrics(y,scores,(0.,0.))
    tn,fp,fn,tp=confusion_matrix(y,predictions,labels=[0,1]).ravel()
    result.update(accuracy=float(accuracy_score(y,predictions)),
        balanced_accuracy=float(balanced_accuracy_score(y,predictions)),
        tn=int(tn),fp=int(fp),fn=int(fn),tp=int(tp),
        tpr_at_accuracy_threshold=float(predictions[y==1].mean()),
        fpr_at_accuracy_threshold=float(predictions[y==0].mean()),
        tpr_at_shadow_threshold=float(lp[y==1].mean()),fpr_at_shadow_threshold=float(lp[y==0].mean()),
        threshold='group-specific auxiliary validation cutoffs',low_fpr_threshold='group-specific auxiliary validation cutoffs')
    return result


def evaluate_lira_bundle(margins,masks,target_margins,target,auxiliary,run_dir):
    """Fit OUT distributions; choose all cutoffs using grouped auxiliary validation."""
    pooled=margins[~masks]
    scale=max(float(np.std(pooled)),1e-6)
    rows=[]
    for k in range(len(margins)):
        out=~masks.copy();out[k]=False
        usable=out.sum(axis=0)>0
        location=np.nanmedian(np.where(out[:,usable],margins[:,usable],np.nan),axis=0)
        row=pd.DataFrame(dict(record_id=auxiliary.index.to_numpy()[usable],
            membership=masks[k,usable].astype(int),reference=k,
            income_class=auxiliary[TARGET].to_numpy()[usable],
            out_location=location,margin=margins[k,usable]))
        row['lira_score']=offline_gaussian_score(row.margin,location,scale)
        rows.append(row)
    rows=pd.concat(rows,ignore_index=True)
    splits=balanced_group_splits(rows,42)
    # The target's candidates are OUT of every reference model.
    target=target.copy()
    target['out_location']=np.median(target_margins,axis=0)
    target['lira_score']=offline_gaussian_score(target.margin,target.out_location,scale)
    # One difficulty value per unique auxiliary TRAIN record; no target labels used.
    per_record=splits['train'].groupby('record_id').out_location.mean()
    cuts=np.quantile(per_record,[.25,.5,.75])
    if not (np.diff(cuts)>0).all():
        raise ValueError('Nonunique difficulty cuts.')
    for frame in [*splits.values(),target]:
        frame['global_group']='all'
        frame['income_group']=frame.income_class.astype(str)
        frame['difficulty_group']=np.digitize(frame.out_location,cuts).astype(str)
        frame['class_difficulty_group']=frame.income_group+'_'+frame.difficulty_group
    methods={};thresholds={}
    global_thresholds=select_thresholds(splits['validation'].membership,splits['validation'].lira_score)
    for name,column in [('offline_lira','global_group'),('class_conditioned_lira','income_group'),
                        ('difficulty_conditioned_lira','difficulty_group'),
                        ('class_difficulty_conditioned_lira','class_difficulty_group')]:
        cutoffs={};low_cutoffs={};counts={}
        groups=set().union(*(set(f[column]) for f in [*splits.values(),target]))
        for group in sorted(groups):
            part=splits['validation'].loc[splits['validation'][column].eq(group)]
            if part.membership.nunique()==2 and part.membership.value_counts().min()>=50:
                pair=select_thresholds(part.membership,part.lira_score)
            else:pair=global_thresholds
            cutoffs[group],low_cutoffs[group]=pair
            counts[group]=part.membership.value_counts().to_dict()
        methods[name]=dict(auxiliary={part:evaluate_conditioned(frame,cutoffs,low_cutoffs,column)
                                      for part,frame in splits.items()},
                           target=evaluate_conditioned(target,cutoffs,low_cutoffs,column))
        thresholds[name]=dict(accuracy=cutoffs,low_fpr=low_cutoffs,validation_counts=counts)
    subgroups=[]
    for column in ['income_group','difficulty_group','class_difficulty_group']:
        for group,part in target.groupby(column):
            if part.membership.nunique()==2:
                subgroups.append(dict(group_type=column,group=group,
                    **score_metrics(part.membership,part.lira_score,global_thresholds)))
    for name,frame in splits.items():frame.to_csv(run_dir/f'auxiliary_{name}_scores.csv',index=False)
    target.to_csv(run_dir/'target_lira_scores.csv',index=False)
    pd.DataFrame(subgroups).to_csv(run_dir/'target_subgroup_metrics.csv',index=False)
    result=dict(methods=methods,thresholds=thresholds,pooled_out_std=scale,
        difficulty_cuts=cuts.tolist(),reference_count=len(margins),
        auxiliary_attack_rows=len(rows),auxiliary_attack_unique_records=int(rows.record_id.nunique()),
        partition_balance={name:frame.membership.value_counts().to_dict() for name,frame in splits.items()},
        out_count_per_auxiliary_record=pd.Series((~masks).sum(axis=0)).value_counts().to_dict(),
        note='Fixed-variance Gaussian OUT likelihood, median OUT locations, one query. Pseudo-target calibration excludes the query model and requires at least one other OUT model. Four-model calibration has limited per-record OUT support.',
        source='https://github.com/tensorflow/privacy/blob/master/research/mi_lira_2021/plot.py')
    (run_dir/'lira_results.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    return result


def run_matched_2014_pilot(root='.',count=4):
    root=Path(root)
    prior=root/'artifacts/exp2_attack_repair'
    run_dir=prior/'matched_2014_lira'
    run_dir.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4)
    previous=json.loads((prior/'summary.json').read_text())
    target=pd.read_csv(prior/'references/target_calibrated_features.csv')
    target_source=load_acs_income(root/'data/processed/acs_income_ca_2018.csv')
    candidates=target_source.loc[target.record_id].copy()
    target['income_class']=candidates[TARGET].to_numpy()
    auxiliary=load_recent_acs(2014)
    assert set(auxiliary.index).isdisjoint(set(target.record_id))
    crosswalk=load_occupation_crosswalk(root/'data/reference')
    audit=[];replacements={}
    for code,n in auxiliary.OCCP.astype(int).value_counts().sort_index().items():
        entry=crosswalk.get(code,dict(destination=code,status='unchanged_not_listed'))
        replacements[code]=entry['destination']
        audit.append(dict(old_code=code,new_code=entry['destination'],records=int(n),status=entry['status']))
    auxiliary['OCCP']=auxiliary.OCCP.astype(int).map(replacements).astype('float32')
    pd.DataFrame(audit).to_csv(run_dir/'occupation_mapping_audit.csv',index=False)
    config=dict(hidden_dims=(512,256,128),dropout=0.,epochs=100,batch_size=256,
        learning_rate=1e-3,weight_decay=0.,device='cpu',early_stopping=False,
        restore_best=False,monitor_train=False)
    target_size=previous['target_train_size']
    manifest=dict(auxiliary_year=2014,target_year=2018,reference_count=count,
        reference_train_size=target_size,training=config,
        target_checkpoint_sha256=hashlib.sha256((prior/'target_model.pt').read_bytes()).hexdigest(),
        auxiliary_archive_sha256=hashlib.sha256((root/'data/raw/folktables/2014/1-Year/csv_pca.zip').read_bytes()).hexdigest())
    path=run_dir/'manifest.json'
    if path.exists() and json.loads(path.read_text())!=manifest:
        raise ValueError('Existing pilot artifacts have a different manifest; use a new run directory.')
    path.write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    masks=[];aux_margins=[];target_margins=[];diagnostics=[]
    for k in range(count):
        seed=1042+k
        train,out=train_test_split(auxiliary,train_size=target_size,random_state=seed,stratify=auxiliary[TARGET])
        mask=auxiliary.index.isin(train.index)
        masks.append(mask)
        model_path=run_dir/f'reference_{k:02d}.pt'
        preprocessor_path=run_dir/f'reference_{k:02d}_preprocessor.joblib'
        if model_path.exists() and preprocessor_path.exists():
            print(f'Loading completed matched-size reference {k+1}/{count}',flush=True)
            checkpoint=torch.load(model_path,map_location='cpu',weights_only=True)
            m=checkpoint['metadata'];model=build_mlp(m['input_dim'],m['hidden_dims'],m['dropout'],m['num_classes'])
            model.load_state_dict(checkpoint['state_dict']);model.eval()
            preprocessor=joblib.load(preprocessor_path)
        else:
            print(f'Training matched-size reference {k+1}/{count}: {len(train):,} ACS2014 records, 100 epochs',flush=True)
            preprocessor=build_preprocessor()
            X=preprocessor.fit_transform(train[FEATURES]).astype(np.float32)
            validation=out.sample(n=1000,random_state=seed)
            Xv=preprocessor.transform(validation[FEATURES]).astype(np.float32)
            model,history,metadata=train_target_model(X,train[TARGET].to_numpy(),Xv,validation[TARGET].to_numpy(),seed=seed,**config)
            save_target_model(model,metadata,model_path);joblib.dump(preprocessor,preprocessor_path)
            pd.DataFrame(history).to_csv(run_dir/f'reference_{k:02d}_history.csv',index=False)
            del X,Xv
        training_metrics=evaluate_model(model,preprocessor.transform(train[FEATURES]).astype(np.float32),train[TARGET].to_numpy())
        out_metrics=evaluate_model(model,preprocessor.transform(out[FEATURES]).astype(np.float32),out[TARGET].to_numpy())
        diagnostics.append(dict(reference=k,train_records=len(train),out_records=len(out),
            train_accuracy=training_metrics['accuracy'],out_accuracy=out_metrics['accuracy'],
            accuracy_gap_pp=100*(training_metrics['accuracy']-out_metrics['accuracy'])))
        aux_margins.append(margin_features(model,preprocessor,auxiliary).margin.to_numpy())
        target_margins.append(margin_features(model,preprocessor,candidates).margin.to_numpy())
        pd.DataFrame(diagnostics).to_csv(run_dir/'reference_diagnostics.csv',index=False)
        print(f'Reference {k+1} gap: {diagnostics[-1]["accuracy_gap_pp"]:.2f} percentage points',flush=True)
    masks=np.asarray(masks);aux_margins=np.asarray(aux_margins);target_margins=np.asarray(target_margins)
    np.savez_compressed(run_dir/'reference_assignments.npz',in_masks=masks,
        auxiliary_margins=aux_margins,target_margins=target_margins,record_ids=auxiliary.index.to_numpy())
    result=evaluate_lira_bundle(aux_margins,masks,target_margins,target,auxiliary,run_dir)
    # Isolate the Gaussian scoring change using the existing small same-year references.
    small=prior/'small_2018_gaussian_lira';small.mkdir(exist_ok=True)
    saved=np.load(prior/'references/reference_assignments.npz',allow_pickle=True)
    small_aux=target_source.loc[saved['record_ids'].astype(int)]
    small_result=evaluate_lira_bundle(saved['auxiliary_margins'],saved['in_masks'],
        saved['target_margins'],target,small_aux,small)
    comparison=[dict(method='previous_calibrated_rf',**previous['reference_results']['methods']['calibrated_rf']['target'])]
    comparison += [dict(method='2014_matched_'+name,**v['target']) for name,v in result['methods'].items()]
    comparison += [dict(method='2018_small_'+name,**v['target']) for name,v in small_result['methods'].items()]
    comparison=pd.DataFrame(comparison).set_index('method')
    comparison.to_csv(run_dir/'attack_comparison.csv')
    summary=dict(manifest=manifest,matched_2014=result,small_2018=small_result,
        reference_diagnostics=diagnostics,target_metrics=previous['target_metrics'],
        caveats=['Pilot has four matched-size references versus 16 smaller references, so count, year, and training size are confounded.',
                 'Occupation conversion uses Census population-level modal mappings, not exact individual harmonization.',
                 'Income label remains nominal >$50,000 in each year; no inflation adjustment.',
                 'Reference models use logits; global/class/difficulty thresholds selected on auxiliary validation only.',
                 'Fixed target model and target MIA sample are reused; no target-label hyperparameter tuning.'])
    (run_dir/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    return summary,comparison
