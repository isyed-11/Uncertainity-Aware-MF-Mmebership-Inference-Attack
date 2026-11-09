"""Reference-calibrated binary MIA experiments; no import-time execution.

Stable margins require logits/log-probability access. The original probability-only
four-feature attack remains the separate baseline. Calibration below is an offline
LiRA-inspired OUT-reference z-score, not online LiRA or an exact paper reproduction.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.special import expit
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, roc_auc_score, roc_curve
from sklearn.model_selection import GroupShuffleSplit

from .data import FEATURES, TARGET, build_preprocessor
from .training import train_target_model, save_target_model, evaluate_model


def margin_features(model, preprocessor, frame):
    """Compute binary stable logit margins and losses in float64 before softmax."""
    X = np.asarray(preprocessor.transform(frame[FEATURES]), dtype=np.float32)
    labels = frame[TARGET].to_numpy(dtype=int)
    batches = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(frame), 1024):
            logits = model(torch.from_numpy(X[start:start+1024])).double()
            logp = torch.log_softmax(logits, dim=1).numpy()
            y = labels[start:start+len(logp)]
            lp = logp[np.arange(len(y)), y]
            other = logp[np.arange(len(y)), 1-y]
            margin = lp-other
            loss = np.logaddexp(0.0, -margin)
            true_p, other_p = expit(margin), expit(-margin)
            entropy = true_p*loss + other_p*np.logaddexp(0.0, margin)
            # logaddexp retains tiny positive losses after softmax/log_softmax round to 0.
            batches.append(pd.DataFrame(dict(margin=margin, loss=loss,
                entropy=entropy, modified_entropy=2*other_p*loss,
                true_probability=true_p)))
    result = pd.concat(batches, ignore_index=True)
    if not np.isfinite(result.to_numpy()).all():
        raise ValueError('Nonfinite stable attack feature.')
    return result


def balanced_group_splits(frame, seed=42):
    """Keep record IDs disjoint, then downsample each partition to exactly 50/50."""
    splitter = GroupShuffleSplit(n_splits=1, test_size=.4, random_state=seed)
    train, rest = next(splitter.split(frame, groups=frame.record_id))
    remaining = frame.iloc[rest]
    splitter = GroupShuffleSplit(n_splits=1, test_size=.5, random_state=seed+1)
    validation, test = next(splitter.split(remaining, groups=remaining.record_id))
    raw = dict(train=frame.iloc[train], validation=remaining.iloc[validation], test=remaining.iloc[test])
    result = {}
    seen = set()
    for offset, (name, part) in enumerate(raw.items()):
        count = int(part.membership.value_counts().min())
        result[name] = pd.concat([part.loc[part.membership.eq(label)].sample(n=count, random_state=seed+offset)
                                 for label in (0,1)]).sample(frac=1,random_state=seed)
        ids = set(result[name].record_id)
        assert not seen & ids
        seen.update(ids)
        assert result[name].membership.value_counts().nunique() == 1
    return result


def select_thresholds(labels, scores):
    fpr, tpr, cutoffs = roc_curve(labels, scores, drop_intermediate=False)
    cutoff = float(cutoffs[np.argmax(tpr-fpr)])
    eligible = np.flatnonzero(fpr <= .01)
    low = float(cutoffs[eligible[np.argmax(tpr[eligible])]])
    return cutoff, low


def score_metrics(labels, scores, thresholds):
    cutoff, low = thresholds
    labels, scores = np.asarray(labels), np.asarray(scores)
    predictions = scores >= cutoff
    tn, fp, fn, tp = confusion_matrix(labels,predictions,labels=[0,1]).ravel()
    fpr, tpr, _ = roc_curve(labels,scores,drop_intermediate=False)
    return dict(records=len(labels), members=int(labels.sum()),
        accuracy=float(accuracy_score(labels,predictions)),
        balanced_accuracy=float(balanced_accuracy_score(labels,predictions)),
        roc_auc=float(roc_auc_score(labels,scores)),
        tpr_at_1pct_fpr_roc=float(tpr[fpr<=.01].max()),
        threshold=cutoff, low_fpr_threshold=low,
        tpr_at_accuracy_threshold=float(predictions[labels==1].mean()),
        fpr_at_accuracy_threshold=float(predictions[labels==0].mean()),
        tpr_at_shadow_threshold=float((scores[labels==1]>=low).mean()),
        fpr_at_shadow_threshold=float((scores[labels==0]>=low).mean()),
        tn=int(tn),fp=int(fp),fn=int(fn),tp=int(tp))


def run_reference_experiment(auxiliary, candidates, target_model, target_preprocessor,
                             config, run_dir, *, count=16, seed=42):
    """Use only unused ACS2018 records for references; never train on target candidates.

    Every auxiliary record is IN eight and OUT eight reference models. Each
    pseudo-target query is normalized against OTHER models for which that record
    is OUT. Target records use all references, since they are OUT of every one.
    Sample size differs from target and is reported rather than hidden.
    """
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    if count < 8 or count % 2:
        raise ValueError('Use an even reference count of at least eight.')
    if set(auxiliary.index) & set(candidates.index):
        raise ValueError('Auxiliary reference pool overlaps target candidates.')
    rng = np.random.default_rng(seed)
    masks = np.zeros((count,len(auxiliary)),dtype=bool)
    for column in range(len(auxiliary)):
        masks[rng.choice(count,count//2,replace=False),column] = True
    all_features, aux_margins, target_margins, diagnostics = [], [], [], []
    for k in range(count):
        print(f'Reference {k+1}/{count}: training {masks[k].sum():,} unused ACS records',flush=True)
        training = auxiliary.iloc[np.flatnonzero(masks[k])]
        out = auxiliary.iloc[np.flatnonzero(~masks[k])]
        preprocessor = build_preprocessor()
        X = preprocessor.fit_transform(training[FEATURES]).astype(np.float32)
        validation = out.sample(n=min(1000,len(out)),random_state=seed+k)
        Xv = preprocessor.transform(validation[FEATURES]).astype(np.float32)
        settings = dict(config, seed=seed+100+k, monitor_train=False)
        model, history, metadata = train_target_model(X,training[TARGET].to_numpy(),
            Xv,validation[TARGET].to_numpy(),**settings)
        save_target_model(model,metadata,run_dir/f'reference_{k:02d}.pt')
        import joblib
        joblib.dump(preprocessor,run_dir/f'reference_{k:02d}_preprocessor.joblib')
        metrics_in = evaluate_model(model,X,training[TARGET].to_numpy())
        metrics_out = evaluate_model(model,preprocessor.transform(out[FEATURES]).astype(np.float32),out[TARGET].to_numpy())
        diagnostics.append(dict(reference=k,train_records=len(training),out_records=len(out),
            train_accuracy=metrics_in['accuracy'],out_accuracy=metrics_out['accuracy'],
            accuracy_gap_pp=100*(metrics_in['accuracy']-metrics_out['accuracy'])))
        features = margin_features(model,preprocessor,auxiliary)
        aux_margins.append(features.margin.to_numpy())
        target_margins.append(margin_features(model,preprocessor,candidates).margin.to_numpy())
        features['record_id'] = auxiliary.index.to_numpy()
        features['membership'] = masks[k].astype(int)
        features['reference'] = k
        all_features.append(features)
        pd.DataFrame(diagnostics).to_csv(run_dir/'reference_diagnostics.csv',index=False)
    aux_margins, target_margins = np.asarray(aux_margins), np.asarray(target_margins)
    # Use a global OUT scale floor, avoiding spurious certainty with 7-8 models.
    pooled_out = aux_margins[~masks]
    scale_floor = max(float(np.std(pooled_out))*.1, 1e-3)
    for k, features in enumerate(all_features):
        out = (~masks).copy()
        out[k] = False
        n = out.sum(axis=0)
        assert (n >= count//2-1).all()
        mean = np.where(out,aux_margins,0).sum(axis=0)/n
        variance = np.where(out,(aux_margins-mean)**2,0).sum(axis=0)/n
        features['reference_z'] = (aux_margins[k]-mean)/np.maximum(np.sqrt(variance),scale_floor)
        features['reference_margin_delta'] = aux_margins[k]-mean
    rows = pd.concat(all_features,ignore_index=True)
    splits = balanced_group_splits(rows,seed)
    target = margin_features(target_model,target_preprocessor,candidates)
    mean = target_margins.mean(axis=0)
    std = np.maximum(target_margins.std(axis=0),scale_floor)
    target['reference_z'] = (target.margin-mean)/std
    target['reference_margin_delta'] = target.margin-mean
    target['membership'] = candidates.membership.to_numpy()
    target['record_id'] = candidates.index.to_numpy()
    methods, classifiers = fit_reference_methods(splits, target, seed)
    import joblib
    joblib.dump(classifiers,run_dir/'attack_classifiers.joblib')
    target.to_csv(run_dir/'target_calibrated_features.csv',index=False)
    for name,frame in splits.items():
        frame.to_csv(run_dir/f'auxiliary_attack_{name}.csv',index=False)
    np.savez_compressed(run_dir/'reference_assignments.npz',in_masks=masks,
        record_ids=auxiliary.index.to_numpy(),auxiliary_margins=aux_margins,target_margins=target_margins)
    result = dict(methods=methods,reference_count=count,auxiliary_records=len(auxiliary),
        target_saturation_audit={str(label): dict(
            float32_true_probability_one_fraction=float((target.loc[target.membership.eq(label),'true_probability'].astype(np.float32)==1).mean()),
            min_margin=float(target.loc[target.membership.eq(label),'margin'].min()),
            max_margin=float(target.loc[target.membership.eq(label),'margin'].max())) for label in (0,1)},
        reference_scale_floor=scale_floor,reference_diagnostics=diagnostics,
        attack_partition_balance={name:frame.membership.value_counts().to_dict() for name,frame in splits.items()},
        access='stable pre-softmax logits (additional access beyond original probability-only baseline)',
        calibration='offline OUT-reference z-score, LiRA-inspired, not exact online LiRA')
    (run_dir/'reference_results.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    return result


def fit_reference_methods(splits, target, seed=42):
    methods, classifiers = {}, {}
    feature_sets = dict(stable_margin_rf=['margin','loss','entropy','modified_entropy','true_probability'],
        calibrated_rf=['margin','loss','entropy','modified_entropy','reference_z','reference_margin_delta'])
    for name, columns in feature_sets.items():
        classifier = RandomForestClassifier(n_estimators=300,max_depth=12,min_samples_leaf=20,n_jobs=2,random_state=seed)
        classifier.fit(splits['train'][columns],splits['train'].membership)
        thresholds = select_thresholds(splits['validation'].membership,classifier.predict_proba(splits['validation'][columns])[:,1])
        methods[name] = dict(auxiliary={part:score_metrics(frame.membership,classifier.predict_proba(frame[columns])[:,1],thresholds)
                                       for part,frame in splits.items()},
                             target=score_metrics(target.membership,classifier.predict_proba(target[columns])[:,1],thresholds))
        classifiers[name] = classifier
    for name, column, direction in [('stable_margin_threshold','margin',1),('loss_threshold','loss',-1),
                                    ('offline_reference_z','reference_z',1)]:
        thresholds = select_thresholds(splits['validation'].membership,direction*splits['validation'][column])
        methods[name] = dict(auxiliary={part:score_metrics(frame.membership,direction*frame[column],thresholds)
                                       for part,frame in splits.items()},
                             target=score_metrics(target.membership,direction*target[column],thresholds))
    return methods, classifiers


def refit_saved_reference_attacks(run_dir, *, seed=42):
    """Numerical correction using saved margins; no target/reference training."""
    import joblib
    run_dir = Path(run_dir)
    result = json.loads((run_dir/'reference_results.json').read_text())
    splits = {name:pd.read_csv(run_dir/f'auxiliary_attack_{name}.csv')
              for name in ('train','validation','test')}
    target = pd.read_csv(run_dir/'target_calibrated_features.csv')
    for frame in [*splits.values(),target]:
        margin = frame.margin.to_numpy()
        frame['loss'] = np.logaddexp(0.0,-margin)
        frame['true_probability'] = expit(margin)
        frame['modified_entropy'] = 2*expit(-margin)*frame['loss']
        frame['entropy'] = expit(margin)*frame['loss'] + expit(-margin)*np.logaddexp(0.0,margin)
        assert np.isfinite(frame[['margin','loss','entropy','modified_entropy']].to_numpy()).all()
    methods,classifiers = fit_reference_methods(splits,target,seed)
    result['methods'] = methods
    result['loss_calculation'] = 'float64 logaddexp(0, -margin), avoiding log_softmax cancellation'
    joblib.dump(classifiers,run_dir/'attack_classifiers.joblib')
    for name,frame in splits.items():
        frame.to_csv(run_dir/f'auxiliary_attack_{name}.csv',index=False)
    target.to_csv(run_dir/'target_calibrated_features.csv',index=False)
    (run_dir/'reference_results.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    return result
