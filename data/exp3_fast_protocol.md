# Exp 3: blind walks enabled and optional probability-only mode

Use `exp_3_expanded_missing_feature_16shadows_k10_4masks.ipynb`.
The pilot retains 16 ACS 2014 shadows, an ACS 2023 target and K=10, with four
nested masks: MAR; MAR+WKHP; MAR+WKHP+COW; MAR+WKHP+COW+SEX.
There is one mask per missing-feature count. This requires 4/15 (26.7%) of the
full study's completion-scoring work; model training and setup costs remain.
The masks follow fixed feature order and coverage, rather than target accuracy.
Results describe these four scenarios and do not average over all 15 subsets.
Pilot results use `pilot_k10_s16_m4`, separate from the previous full study.

## Attack features

`INCLUDE_BLIND_WALK = True` is the default. The base attack uses the class-scaled
logit margin, blind walk, prediction entropy and cross-entropy with the known income label.
The same feature set is used for complete records, PFV, conditional single
completion, averaging, learned mean-only and learned mean-plus-standard-deviation.
Blind walks add perturbation queries after each completion.

Set `INCLUDE_BLIND_WALK = False` in the configuration cell for the faster three-feature
comparison without blind walks. Restart the kernel and run from the first cell so the attack and
aggregators are fitted with that feature set. Shadow and target walks both receive
the same extra numeric protocol for JWMNP and JWRIP. These results use a different
directory from probability-only results and from the historical eight-shadow run.

The supplied MF-MIA paper (Oren et al., Section 5.3) includes blind walk as one of
its four attack features. PFV itself (Section 5.2) samples each missing feature
from its auxiliary empirical frequency distribution. The completed record then
passes through the attack. Removing blind walk is an explicitly labelled adaptation
of that attack; it does not change the PFV completion rule.

## Runtime and interpretation

The current walk settings permit up to 41 predictions per completion: one original
prediction plus two walks of at most 20 steps. The probability-only path uses one.
This reduces the query budget by up to 41 times, not necessarily elapsed time:
reconstruction fitting, preprocessing, sampling and attack fitting still take time.

Reconstruction and uncertainty scoring run one mask at a time. Each mask's
aggregators and thresholds are fitted on the reserved ACS 2014 records before
evaluating its target scores. Target results do not select later masks, settings
or thresholds. Large reconstruction models and temporary arrays are released
before the next mask. Audit sampling retains only reconstructed hidden columns.
Leaf donor pools store compact value codes with a shared codebook. For these
low-cardinality float32 fields, one-byte codes use 75% less raw pool storage;
donor ordering, repeated-value frequencies and seeded sampling are preserved.

Target and shadow models are trained in the session without loading or saving
checkpoints. Reconstruction models, draws and aggregation models also stay in
memory. Final metrics, row assignments and the protocol are exported; no log
files are written. The repeated sklearn/joblib parallel warning is hidden, verbose
batch stdout is discarded, and concise progress stays in the notebook. Other
warnings, exceptions and stderr messages remain visible. A kernel restart requires
retraining and recomputation.

Grouped record roles, seeds, reconstruction forest settings, sample counts and
membership metrics are retained. Better runtime does not guarantee better MIA.
The central comparison remains learned mean-plus-standard-deviation against the
matched learned mean-only classifier on identical completion draws.

## Logistic decision controls

The K=10 notebook also reports `logistic_mean` and `logistic_uncertainty`. They
receive respectively the mean alone and the mean plus standard deviation, using
the exact same completion scores as the RF methods. Both use StandardScaler
followed by L2-regularized logistic regression with fixed C=1, solver lbfgs and
at most 1,000 iterations. Scaler statistics and coefficients are fitted only on
the 2014 meta-training records. Each method's decision thresholds are selected
separately on the existing calibration records. No test or target labels select
coefficients, scaling, regularization or thresholds. There is no additional
imputation or oracle-query pass for these controls.

Compare `logistic_uncertainty` with `logistic_mean` to measure the contribution
of standard deviation within this simpler classifier family. The RF pair remains
unchanged; the K=6 baseline does not enable logistic controls.
