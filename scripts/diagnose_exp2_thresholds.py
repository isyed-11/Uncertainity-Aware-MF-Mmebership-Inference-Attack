"""Inspect frozen Experiment 2 scores; no training, refitting, or notebook execution."""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, roc_auc_score

ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'artifacts/exp2_attack_repair'
REF=RUN/'references'
OUT=RUN/'threshold_diagnostics'
OUT.mkdir(parents=True,exist_ok=True)
results=json.loads((REF/'reference_results.json').read_text())
models=joblib.load(REF/'attack_classifiers.joblib')
frames={name:pd.read_csv(REF/filename) for name,filename in {
    'auxiliary_validation':'auxiliary_attack_validation.csv',
    'auxiliary_test':'auxiliary_attack_test.csv',
    'target':'target_calibrated_features.csv'}.items()}
rows=[]
score_frames={}
for name,model in models.items():
    stored=results['methods'][name]
    threshold=stored['target']['threshold']
    low=stored['target']['low_fpr_threshold']
    for dataset,frame in frames.items():
        scores=model.predict_proba(frame[list(model.feature_names_in_)])[:,1]
        labels=frame.membership.to_numpy()
        fpr,tpr,cutoffs=roc_curve(labels,scores,drop_intermediate=False)
        best=int(np.argmax(tpr-fpr))
        predictions=scores>=threshold
        low_predictions=scores>=low
        member=scores[labels==1];nonmember=scores[labels==0]
        # 100 fixed bins: descriptive overlap, not an information-theoretic bound.
        bins=np.linspace(0,1,101)
        h0,_=np.histogram(nonmember,bins);h1,_=np.histogram(member,bins)
        overlap=float(np.minimum(h0/len(nonmember),h1/len(member)).sum())
        rows.append(dict(method=name,dataset=dataset,records=len(frame),
            accuracy_at_auxiliary_threshold=float((predictions==labels).mean()),
            auxiliary_threshold=threshold,tpr=float(predictions[labels==1].mean()),
            fpr=float(predictions[labels==0].mean()),roc_auc=float(roc_auc_score(labels,scores)),
            diagnostic_best_balanced_accuracy=float(.5+.5*(tpr[best]-fpr[best])),
            diagnostic_best_threshold=float(cutoffs[best]),
            low_fpr_threshold=low,tpr_at_fixed_low_threshold=float(low_predictions[labels==1].mean()),
            fpr_at_fixed_low_threshold=float(low_predictions[labels==0].mean()),
            histogram_overlap_100_bins=overlap,
            member_p25=float(np.quantile(member,.25)),member_median=float(np.median(member)),
            member_p75=float(np.quantile(member,.75)),
            nonmember_p25=float(np.quantile(nonmember,.25)),nonmember_median=float(np.median(nonmember)),
            nonmember_p75=float(np.quantile(nonmember,.75))))
        if dataset=='target':
            np.testing.assert_allclose(rows[-1]['accuracy_at_auxiliary_threshold'],stored['target']['accuracy'])
        score_frames[(name,dataset)]=pd.DataFrame(dict(record_id=frame.record_id,
            membership=labels,score=scores))
report=pd.DataFrame(rows)
report.to_csv(OUT/'threshold_transfer.csv',index=False)
(OUT/'diagnostics.json').write_text(json.dumps(dict(
    note='All models and thresholds frozen. Best target threshold uses target membership labels and the same evaluation records: optimistic diagnostic ONLY, not a valid attack result or a bound on other classifiers.',
    results=rows),indent=2),encoding='utf-8')
for (name,dataset),frame in score_frames.items():
    frame.to_csv(OUT/f'{name}_{dataset}_scores.csv',index=False)
print(report.to_string(index=False))
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(11,4),sharex=True,sharey=True)
    for ax,dataset,title in zip(axes,['auxiliary_validation','target'],['Auxiliary validation','Target evaluation']):
        frame=score_frames[('calibrated_rf',dataset)]
        for membership,color,label in [(0,'#c65b4b','Nonmembers'),(1,'#3575b8','Members')]:
            ax.hist(frame.loc[frame.membership.eq(membership),'score'],bins=np.linspace(0,1,51),
                density=True,histtype='step',linewidth=1.8,color=color,label=label)
        ax.axvline(results['methods']['calibrated_rf']['target']['threshold'],color='black',linestyle='--',label='Auxiliary-selected threshold')
        if dataset=='target':
            row=report.loc[(report.method=='calibrated_rf') & (report.dataset=='target')].iloc[0]
            ax.axvline(row.diagnostic_best_threshold,color='#7d4b9e',linestyle=':',label='Target-label diagnostic threshold')
        ax.set_title(title);ax.set_xlabel('Membership score');ax.grid(alpha=.15)
    axes[0].set_ylabel('Density');axes[1].legend(fontsize=8)
    fig.suptitle('Calibrated RF: substantial overlap and threshold transfer shift')
    fig.tight_layout();fig.savefig(OUT/'calibrated_score_distributions.png',dpi=180)
    plt.close(fig)
    print('Plot saved:',OUT/'calibrated_score_distributions.png')
except ImportError:
    print('Matplotlib unavailable; tabulated diagnostics and scores saved.')
print('Diagnostics saved:',OUT)
