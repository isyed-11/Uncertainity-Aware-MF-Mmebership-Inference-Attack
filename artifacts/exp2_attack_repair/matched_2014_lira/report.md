# Matched-size ACS2014 offline LiRA pilot

Outcome: no improvement over the previous calibrated RF (62.05% accuracy, 0.6830 ROC-AUC).

Four references trained on 145,000 ACS2014 records each for 100 epochs, hidden dimensions (512,256,128). The original ACS2018 target was loaded from saved artifacts and not retrained.
Mean reference accuracy gap: 19.40 percentage points; target gap: 19.75 points.

## Target evaluation

| Method | Accuracy | ROC-AUC | ROC TPR at 1% FPR |
|---|---:|---:|---:|
| previous_calibrated_rf | 62.05% | 0.6830 | 6.01% |
| 2014_matched_offline_lira | 51.68% | 0.5282 | 1.24% |
| 2014_matched_class_conditioned_lira | 51.42% | 0.5282 | 1.24% |
| 2014_matched_difficulty_conditioned_lira | 53.33% | 0.5282 | 1.24% |
| 2014_matched_class_difficulty_conditioned_lira | 53.24% | 0.5282 | 1.24% |
| 2018_small_offline_lira | 52.65% | 0.5356 | 1.22% |
| 2018_small_class_conditioned_lira | 52.75% | 0.5356 | 1.22% |
| 2018_small_difficulty_conditioned_lira | 53.08% | 0.5356 | 1.22% |
| 2018_small_class_difficulty_conditioned_lira | 53.52% | 0.5356 | 1.22% |

Conditioning changes decision thresholds, not the raw LiRA score ranking, so ROC-AUC is identical across conditional variants within a reference setup.

## Method and limitations

Offline LiRA follows the authors Gaussian OUT-likelihood implementation with median OUT locations and pooled fixed variance: https://github.com/tensorflow/privacy/blob/master/research/mi_lira_2021/plot.py . Scores use a single stable logit-margin query, without image-style augmentation.

All accuracy and low-FPR decision cutoffs come from grouped, balanced auxiliary validation. Target membership labels are used only for final evaluation. Class and difficulty bins are derived from auxiliary data.

This is a four-model pilot, not a full reference-count sweep. References train on about 79% of the 2014 pool, leaving sparse OUT observations for auxiliary pseudo-target calibration; records with no other OUT reference are excluded. All 2018 target candidates are OUT of every 2014 reference. This limitation should not be confused with perfect per-record calibration.

Year, reference count and training size differ from the previous setup, so the experiment does not identify one causal reason for the regression. The unchanged small 2018 references also perform weakly with this Gaussian scoring rule.

Occupation mappings are population-level modal approximations from Census tables; income thresholds remain nominal without inflation adjustment. Logit access is required as in the previous stable-feature attacks.

Verification passed: four 145,000-record reference assignments, unchanged target checkpoint hash, disjoint target/auxiliary records, disjoint attack groups, exact 50/50 attack partitions, finite saved scores and successful notebook cell execution.
