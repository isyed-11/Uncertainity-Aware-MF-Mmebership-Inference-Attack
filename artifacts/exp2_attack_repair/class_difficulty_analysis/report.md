# Experiment 2: class and difficulty diagnostic

Analysis uses only saved outputs. No model queries, model fitting or notebook execution occurred.
Difficulty quartiles use unique auxiliary-training records and their mean OUT-reference true-class margins. Target membership labels do not define groups. Existing global auxiliary thresholds remain fixed.

## Target calibrated RF

| Group | Records | Balanced accuracy | ROC-AUC |
|---|---:|---:|---:|
| <= $50k | 11,788 | 60.13% | 0.6565 |
| > $50k | 8,212 | 64.81% | 0.7165 |
| hardest | 4,870 | 76.87% | 0.8397 |
| moderate | 5,107 | 62.67% | 0.6608 |
| easy | 5,037 | 57.27% | 0.5928 |
| easiest | 4,986 | 52.30% | 0.5193 |

The strongest signal is concentrated in reference-hard records. The income-class gap is partly consistent with differing difficulty composition: the easiest group has 4,514 low-income records but only 472 high-income records. This is descriptive, not a causal attribution.

Transfer remains imperfect even in the hardest group: auxiliary-test balanced accuracy is 89.52%, versus 76.87% on the target. A limited experiment with better-matched reference training sizes is reasonable; these results do not guarantee an aggregate accuracy increase from simply adding references.

All subgroup results are exploratory. Global low-FPR thresholds do not control subgroup FPR: the hardest target group has 11.41% FPR at the global low-FPR cutoff. Subgroup ROC TPR@1%FPR is a separate evaluation summary.

Bootstrap intervals condition on fixed models, use 200 stratified record resamples, and omit model-training and household-dependence uncertainty. Auxiliary rows repeat reference observations; refer to unique-record counts.
