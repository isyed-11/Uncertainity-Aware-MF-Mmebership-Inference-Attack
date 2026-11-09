# Step-by-step membership attack experiments

Work in `experiments_2_overfit.ipynb`. The original-model notebook is
`experiments_1.ipynb`. This is a plan only: no settings or experiments were run
or changed when this file was created.

## Objective and fixed assumptions

Improve the attack's ability to distinguish members from non-members, rather
than simply increasing how often it predicts "member." Keep the current overfit
target fixed throughout this sequence. Do not increase its overfitting again to
obtain a desired attack score. The earlier approximately 60.34% result is an
exploratory reference, not a guaranteed reproducible improvement target.

Assume architecture-unknown black-box access to income probabilities and known
candidate income labels. Keep the five diverse Adult shadows. Do not match
shadows to hidden target architecture or tune using ACS membership labels.

Use Adult validation ROC-AUC as the primary configuration-selection criterion;
use validation balanced accuracy as a secondary diagnostic. Both prediction
thresholds continue to be chosen on Adult validation. After selecting a final
protocol, inspect Adult test and ACS MIA performance. Since ACS results have
already been examined, describe further experiments as exploratory. A future
confirmatory evaluation requires a prospectively reserved evaluation sample or
independently specified target runs; reusing the same target set is not a fresh
confirmation.

## Step 1: Establish the restored baseline

Run the restored attack once only if its outputs are unavailable. Reuse fitted
target and shadows if still loaded. Otherwise rerun their original fixed settings.
Kernel restarts lose these in-memory objects because saving was removed.

| Setting | Baseline |
|---|---|
| Target | Existing overfit MLP, unchanged |
| Shadows | XGBoost, random forest, MLP, decision tree, logistic regression |
| Shadow feature records | 2,000 members + 2,000 non-members per shadow |
| Blind walk | 20 steps, 2 walks, categorical change probability .05 |
| Attack | RF: 300 trees, depth 12, minimum leaf 20, max_features='sqrt' |
| Features | Class-scaled logit, blind walk, entropy, cross-entropy |
| Seed | 42 |
| Target MIA records | Existing 10,000 members + 10,000 non-members |

Record Adult validation AUC, balanced accuracy, TPR@1% FPR, query count, and
walk_censored_fraction. Preserve raw predictions and row IDs in memory for
comparisons. Keep the source dataset, candidate IDs, shadow models, and splits
unchanged unless a step explicitly varies them.

Establish one master Adult record-ID assignment for attack train/validation/test
before changing sample size. Apply it to every configuration using
`split_record_ids` in `train_membership_classifier`. Repeated observations of an
Adult record under different shadows must stay in the same attack partition.
Rerunning GroupShuffleSplit on a changed record pool can change assignments,
even with the same seed; seed equality alone is insufficient.

## Step 2: Diagnose the baseline before expensive queries

Inspect the Adult validation distributions of each of the four features by
membership. Look for constant features, overlapping distributions, saturated
probabilities, and walk censoring. Inspect target distributions only as descriptive
diagnostics, not to select settings.

- High censoring means many walks have not observed a prediction change; they
  all receive the cap and lose distance information.
- Low Adult validation AUC suggests weak shadow signal or an unsuitable attack.
- Strong Adult validation but weak ACS AUC suggests poor transfer.
- AUC appreciably above .5 with accuracy near .5 suggests cutoff transfer issues.

If censoring is high, proceed to Step 5 after the cheap RF checks. If it is low,
do not assume increasing the walk budget is useful.

## Step 3: Vary RF depth only (no model queries)

Reuse the SAME extracted features, record assignments, and thresholds-selection
procedure. Try max_depth 8, 12, and 20. Keep leaf size 20, 300 trees, and
max_features='sqrt'. Use Adult validation to select a depth; do not query the
target for each candidate. Existing `tune_random_forest_attack` can search a grid
with only depth varying. It selects by validation AUC. With the same input table,
its seeded record grouping stays fixed.

## Step 4: Vary RF leaf size only (no model queries)

Freeze the depth selected in Step 3. Try min_samples_leaf 5, 20, and 50, with
all remaining settings fixed. Select on Adult validation. Smaller leaves can
overfit shadow-specific behavior; higher training accuracy is not sufficient.
Keep the baseline RF if validation results do not support a change.

Optional later check: compare max_features='sqrt' with None while holding the
selected depth and leaf size fixed. Do not expand to a large search tomorrow.

## Step 5: Vary the walk length only

On the same sampled shadow records, compare max_steps=20 and max_steps=50,
holding repeats=2 and categorical_change_probability=.05. Keep RF settings fixed.
Inspect censoring reduction and validation performance versus query cost.
The original paper does not fully specify this tabular walk; these are our
explicit experimental settings.

For a clean paired comparison, use the same random directions/destinations for
both budgets. Current extraction RNG is deterministic for a complete run but
is not keyed by record ID; changing execution order or settings can change later
draws. Exact paired-walk experiments require a small collector extension or
cached per-record walk paths before claiming that only length changed.

## Step 6: Vary the walk repetitions only

Freeze the selected walk length. Compare repeats=2 and repeats=5, keeping the
same sampled records and other parameters fixed. More walks may reduce random
variation; they do not necessarily improve membership separation. Inspect Adult
validation results, censoring, and query cost. Use per-record deterministic paths
or repeat the measurement under several walk seeds to assess noise fairly.

## Step 7: Increase shadow records gradually

Freeze all selected settings. First try 5,000 members and 5,000 non-members per
shadow, rather than jumping to all records. Use nested samples that retain the
baseline records and the master attack record assignments from Step 1. Preserve
the already measured features for old records; query only new records if possible.

The current collector re-extracts an entire sample and does not guarantee nested
samples or per-record identical walks across sample sizes. Those controls require
a notebook/collector extension before running a strict sample-size ablation.
Use all available records only if the incremental query budget and validation
results justify it. Keep the diverse shadow algorithms; no matched MLP pool.

## Step 8: Confirm stability and evaluate the selected attack

Freeze the final protocol before target evaluation. Repeat attack fitting/walk
extraction with seeds 42, 43, and 44 on fixed record assignments to assess attack
randomness. This does not measure variability across target training runs;
independent target seeds would be a separate, more costly experiment.

Report every seed and mean/standard deviation, not the best seed. Evaluate the
fixed protocol against the unchanged ACS MIA records with no target-label tuning.
Report accuracy, balanced accuracy, ROC-AUC, TPR@1% FPR, and actual TPR/FPR at
the Adult-selected cutoff. Compare paired candidate outcomes where possible;
record-ID grouping is needed for uncertainty estimates on repeated Adult rows.

The earlier 60% result had many false positives. A higher accuracy with worse
low-FPR behavior is not automatically a stronger practical attack. If the final
protocol does not improve transfer, retain and report the simpler baseline.

## Query-budget guide

Upper bound = number of queried records * (1 + max_steps * repeats).
Early prediction changes reduce actual cost. This counts individual record
predictions, not batched Python/API calls. Five diverse shadows are assumed.

| Feature collection | Shadow records | Max shadow queries |
|---|---:|---:|
| 2,000/class, 20 steps, 2 walks | 20,000 | 820,000 |
| 2,000/class, 50 steps, 2 walks | 20,000 | 2,020,000 |
| 2,000/class, 50 steps, 5 walks | 20,000 | 5,020,000 |
| 5,000/class, 20 steps, 2 walks | 50,000 | 2,050,000 |

Query the 20,000 target candidates only after selection. Their corresponding
budget is 820,000, 2,020,000, or 5,020,000 predictions for the three walk settings.

## Log template

| Run | Only factor changed | Adult validation AUC | Validation balanced accuracy | Walk censored fraction | Queries | Decision |
|---|---|---|---|---|---|---|
| B0 | Restored baseline | | | | | |
| D1 | RF depth | | | unchanged | 0 additional | |
| L1 | RF leaf size | | | unchanged | 0 additional | |
| W1 | Walk length | | | | | |
| R1 | Walk repetitions | | | | | |
| N1 | Shadow sample size | | | | | |

Tomorrow's first priority: baseline diagnostics and RF depth/leaf checks using
cached features. Budget extensions come afterward. Nothing here guarantees a
particular target accuracy, and exact paired-walk/sample-size controls noted above
are planned extensions rather than changes already implemented in the notebooks.
