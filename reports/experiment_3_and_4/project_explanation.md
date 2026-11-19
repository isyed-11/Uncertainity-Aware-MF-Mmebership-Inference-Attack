# Project explanation: MIA and differential privacy
Syed Khaja Ikramuddin | ECE750T, University of Waterloo | 18 November 2026

## Problem and connection to my proposal

My proposal asks whether uncertainty in reconstructed missing features improves membership inference, and how differential privacy changes the privacy–utility trade-off. Hiding an attribute may not hide whether a record was used for training: related fields and model responses may still reveal membership. I want to test whether variation across plausible completions adds information beyond one completion or its average score.

The experiments below use California ACS 2023 for the target and ACS 2014 as auxiliary data. The task is nominal income > $50,000 from 48 fields. This adapts the idea in my proposal, which originally named Adult and GSS. I separate the reasoning behind each method from the pipeline actually implemented in the notebooks.

## Black-box setting and assumptions

• API access: the attacker knows how to call the target API, its 48-field input schema/codes and class order. It receives both income-class probabilities. Training records, model weights, gradients and private preprocessing statistics are unavailable to the attacker. The experiment simulates this interface with a local probability-query wrapper.

• Candidate knowledge: observed fields, the names of the hidden fields and the true income class are known. The hidden values and training membership are unknown. Income labels are used for attack features such as loss, not to condition reconstruction.

• Auxiliary access: the attacker has 2014 data and trains its own shadow models, so it knows their IN/OUT membership labels. It does not need the target architecture or exact training settings. The researcher knows target membership only for evaluation; those labels do not fit the attack.

• Query allowance: the API accepts completed and perturbed records repeatedly. Fees, rate limits, output rounding and plausibility filters are not simulated. The target still expects all 48 fields, so the attacker must fill missing inputs before querying it.

## Data splits and what the metrics mean

| Use | Records |
| --- | --- |
| Target training | 100,000 |
| Target validation / utility test | 10,000 / 10,000 |
| Target membership evaluation | 10,000 train-subset members + 10,000 separate nonmembers |

MIA accuracy is the fraction of correct member/nonmember decisions. ROC-AUC measures score ranking across cutoffs. TPR@1%FPR measures member detection while allowing at most 1% empirical false positives on the evaluation ROC. These are attack metrics; income-prediction test accuracy measures target utility.

The balanced target membership set has 50% chance accuracy. AUC 0.5 is random ranking. The results are saved observations, not new runs. Source notebooks: exp_3_expanded_missing_feature_baseline.ipynb and exp_4_differential_privacy_accuracy.ipynb.

# 1. MIA baseline: complete features
Experiment 3 · the reference attack before any features are hidden

## Methodology: why membership may leave a signal

A model may behave differently on records it trained on, for example through lower loss or more stable predictions. Shadow models provide examples of these differences because their training membership is known. A separate membership classifier learns the relationship between response features and IN/OUT labels, then transfers it to the target. This is a hypothesis to test; high confidence alone does not prove membership.

## Pipeline in the experiment

Target: 2023 ACS → fixed target partitions → Fit train-only preprocessing + MLP 512/256/128; 100 fixed epochs → Query 20,000 target candidates and extract four attack signals
Auxiliary: 2014 ACS → 16 seeded shadow splits 40% train / 20% val / 40% OUT → Fit 6 RF + 5 gradient boosting + 5 XGBoost shadows; train-only preprocessing → Score 2k IN + 2k OUT per shadow; fit RF attack; select validation cutoffs
Apply the frozen attack to target features → membership scores and final decisions

Repeated auxiliary IDs stay in one grouped attack train/validation/test role. The four features are the true-class log-probability ratio, entropy, cross-entropy loss and mean blind-walk step at which the prediction first changes. The target is evaluated at its final epoch; validation does not select its weights in this Exp 3 configuration.

## Illustrative example — these are not measured record scores

Suppose the known income class is >$50,000 and the target returns [0.12, 0.88]. The attack computes log(0.88/0.12), entropy and loss −log(0.88). If two blind walks first change the predicted class at steps 4 and 7, the walk feature is 5.5. The RF turns these features into a separate membership score. For example, a hypothetical score of 0.63 exceeds the saved accuracy cutoff 0.507089, producing a “member” decision. The income probability 0.88 is not itself a membership probability.

## Saved results

| Evaluation | MIA accuracy % | AUC | TPR@1%FPR % |
| --- | --- | --- | --- |
| Auxiliary attack test | 51.107 | 0.521 | 1.25 |
| Target MIA: complete features | 62.625 | 0.609 | 0.17 |

Target income accuracy: 99.849% train and 79.52% test. The target MIA accuracy cutoff gives TPR 98.01% but FPR 72.76%. Its ROC TPR at 1% FPR is only 0.17%; overall accuracy therefore does not describe a strong low-false-alarm attack.

# 2. Missing-feature PFV baseline
Experiment 3 · one unconditional fill, followed by the same complete-feature attack

## Methodology: fill hidden fields from auxiliary frequencies

PFV uses the empirical 2014 distribution of each hidden field. It draws one observed value for each field independently, without using the visible fields to choose that value. This is a simple reconstruction baseline: it allows a complete input to be queried, but does not model whether that fill is plausible for this particular person.

## Pipeline in the experiment

Choose one of seven predeclared feature masks → Remove the true values of the hidden fields → Draw one 2014 marginal value for each hidden field → Query the fixed target with all 48 fields filled → Extract the same four signals, including walks → Apply the same frozen RF attack and cutoffs

The target and attack classifier are not retrained for a mask. Observed values remain unchanged, naturally missing auxiliary donor values are excluded, and the sampling does not use income or membership labels. The same 20,000 candidates are used for every mask.

## Illustrative example

Suppose a candidate’s age is 35 and hours worked per week (WKHP) are hidden. PFV might sample 20 hours from the overall 2014 frequency distribution, regardless of that candidate’s age, education or job. The completed record goes through the target and frozen membership attack. The actual hidden WKHP value is never used. With two missing fields, PFV draws each independently.

## Saved results for all seven masks

| Hidden fields | MIA accuracy % | AUC | TPR@1%FPR % |
| --- | --- | --- | --- |
| complete | 62.625 | 0.609 | 0.17 |
| MAR | 61.870 | 0.603 | 0.29 |
| WKHP | 59.590 | 0.586 | 0.44 |
| MAR+COW | 60.925 | 0.594 | 0.33 |
| COW+SEX | 60.445 | 0.594 | 0.32 |
| MAR+WKHP+SEX | 58.815 | 0.580 | 0.45 |
| WKHP+COW+SEX | 58.070 | 0.574 | 0.63 |
| MAR+WKHP+COW+SEX | 57.805 | 0.572 | 0.44 |

MAR = marital status; WKHP = hours worked/week; COW = worker class; SEX = sex code. “complete” is the unmasked baseline. These are single-run results, not averages over seeds.

All seven masks reduce accuracy and AUC relative to complete features in this run. Hiding all four fields still leaves 57.805% MIA accuracy, so masking does not eliminate the observed signal. Some low-FPR figures increase, which shows why accuracy and low-false-alarm performance need to be read separately.

# 3. Uncertainty-aware MF-MIA: study protocol
Experiment 3 · test the information in score variability, using identical conditional draws

## Methodology: compare mean score with mean and variability

Conditional reconstruction uses visible fields to generate several plausible fills. For each candidate I obtain K=6 membership scores and compute their mean μ and population standard deviation σ. The proposal’s hypothesis is that σ may add information about membership beyond μ. A matched RF(μ) control is needed to test the added value of RF(μ, σ); high σ does not automatically mean “member”.

## Pipeline in the experiment

Fit conditional forests on reserved 2014 records → Hide selected fields; sample six completed records → Query each completion; get six frozen-base-RF scores → Summarize first score, mean μ and SD σ → Fit on auxiliary meta IDs; calibrate each method separately → Evaluate auxiliary test, then the fixed target candidates

Reconstruction IDs exclude every attack-candidate ID. Base-attack validation IDs are split into meta-fit and calibration groups; auxiliary test IDs remain held out. The same draws support all four controls. No target membership labels choose thresholds or fit a model.

## Illustrative example: same mean, different uncertainty

For MAR+COW, the chain samples MAR from visible fields, then COW from visible fields plus that sampled MAR; it repeats this procedure six times. The hypothetical score sets below show why averaging and uncertainty-aware inference can make different decisions:

| Candidate | Six membership scores (illustrative) | Mean μ | SD σ |
| --- | --- | --- | --- |
| A | 0.48, 0.49, 0.50, 0.50, 0.51, 0.52 | 0.500 | 0.013 |
| B | 0.20, 0.20, 0.20, 0.80, 0.80, 0.80 | 0.500 | 0.300 |

An average-score rule treats A and B equally. RF(μ, σ) could distinguish them only if the shadow data support an association between variability and membership. These example scores and SDs are explanatory calculations, not experimental observations.

## Saved target results: equal-weight averages across seven masks

| Decision method | Input | Accuracy % | AUC | TPR@1%FPR % |
| --- | --- | --- | --- | --- |
| One conditional draw | First score | 59.759 | 0.593 | 0.39 |
| Average score | Mean μ | 60.857 | 0.610 | 0.34 |
| Mean-only RF | μ | 51.596 | 0.525 | 0.74 |
| Uncertainty RF | [μ, σ] | 51.239 | 0.508 | 0.51 |

Averaging improves over a single conditional draw. The uncertainty RF does not show an overall gain over the matched mean RF, and both learned RFs transfer poorly compared with averaging. This is a negative result for the proposed improvement so far; it does not imply that uncertainty must help or that higher σ has a fixed membership meaning.

# 4. DP analysis: utility and membership leakage
Experiment 4 · exp_4_differential_privacy_accuracy.ipynb

## Methodology: change training privacy while fixing the evaluation

DP training clips each record’s gradient contribution and adds Gaussian noise. An accountant tracks a record-level (ε, δ) bound. I compare private and non-private MLPs on the same data, varying ε with δ=10⁻⁶ fixed. Smaller ε requires stronger privacy protection; the observed attack need not change monotonically. The implementation uses clipped, noisy Adam updates (DP-Adam), with Poisson sampling and RDP accounting.

## Pipeline in the experiment

Freeze target partitions and public preprocessing → Fit 16 mixed shadows; freeze RF attack and cutoffs → Train fresh non-private and seven private target MLPs → Select weights using public validation only → Evaluate target utility and the same frozen MIA → Report every ε with accuracy, AUC and low-FPR TPR

Public 2014 preprocessing uses 10,000 rows; the other 173,941 rows supply shadows. The 16 non-private shadows are four each RF, gradient boosting, XGBoost and generic MLP, with own 40/20/40 splits and train-only preprocessing. The same 100,000 target train records and balanced 20,000 MIA candidates are used across all conditions.

Targets use 512/256/128 hidden layers, dropout 0.1 and weight decay 0.0001, with a 30-epoch maximum and patience 5. Public-validation accuracy selects the returned weights, using loss to break ties. Noise is calibrated before training for the fixed maximum schedule; stopped-time ε is a diagnostic, not the claimed bound.

## Saved results and a concrete interpretation

| Requested ε | Test accuracy % | MIA accuracy % | MIA AUC | TPR@1%FPR % |
| --- | --- | --- | --- | --- |
| Non-private (∞) | 81.76 | 51.315 | 0.520 | 0.81 |
| 0.5 | 79.11 | 49.825 | 0.499 | 1.09 |
| 1 | 80.02 | 49.705 | 0.499 | 0.83 |
| 2 | 80.41 | 49.990 | 0.501 | 0.96 |
| 5 | 80.80 | 49.775 | 0.500 | 1.13 |
| 10 | 80.93 | 50.025 | 0.503 | 1.18 |
| 25 | 81.26 | 50.590 | 0.508 | 1.02 |
| 50 | 81.11 | 50.055 | 0.504 | 1.03 |

For ε=1, the target correctly predicts income for 80.02% of test records; the transferred attack gets 49.705% membership accuracy and AUC 0.499. This describes the tested attack on a balanced sample. It does not mean ε=1 gives a 1% chance of leakage or that all possible attacks are ineffective.

The same complete-feature attack is frozen across budgets; there is no missing-feature reconstruction in this DP notebook. ROC TPR@1%FPR uses target labels only to summarize the evaluation ROC. The auxiliary-selected cutoff is not recalibrated on target labels.

# 5. Limitations and future work
Interpreting the current evidence in relation to my proposal

## Limitations

1. Weak attack transfer. Auxiliary attack test AUC is 0.521 in Exp 3 and 0.513 in Exp 4. Shadows differ from the target in model family and overfitting. Near-chance DP attack results cannot establish resistance to stronger or DP-adapted attackers. Exp 3’s 62.625% accuracy also includes many false positives.

2. Low-FPR calibration. The ROC TPR@1%FPR is an evaluation summary. At the auxiliary low-FPR cutoff, actual target FPR in the private runs is 7.18–11.68%, so those cutoff TPRs must not be described as detection at 1% target FPR.

3. Data and masking assumptions. The 2014/2023 income distributions and category codes differ, and no inflation adjustment is used. Masks are artificial and related proxy fields remain visible. Year-qualified row-ID separation does not establish person-level independence. The local API simulation also omits real service restrictions.

4. Reconstruction and walks. Forest donor sampling is an approximate distribution, not a calibrated posterior; six draws give a noisy SD estimate. SD may include finite-walk effects. Exp 3 retains a commuting-field mismatch: JWMNP/JWRIP use categorical shadow perturbations but numeric target perturbations. Exp 4 uses consistent rules.

5. Repeatability and random-number settings. These are exploratory runs with one overall seed, not repeated whole experiments. The shadows have separate seeds, and private noise is independently sampled. Opacus secure_mode=False uses its research randomness setting; the setup is not hardened for deployment.

6. Compute cost. DP training and repeated reconstruction queries are expensive on a laptop. Each completion can require up to 41 record predictions under the walk settings, multiplied by six draws and seven masks. This limits the number of repeat runs and alternative settings I can investigate.

7. Privacy assumptions. For Experiment 4, target preprocessing data and target validation are assumed public and outside protected training records. The accountant guarantee is per-model and record-level, conditional on the fixed public data/partition. Releasing multiple models needs composition; the reported evaluation statistics are not DP-sanitized.

## Future work

Evaluate the combined uncertainty-aware MF-MIA against DP target models. For each privacy budget, hide candidate fields, generate conditional completions and compare single-draw, averaging, RF(mean) and RF(mean, SD) decisions against a matched non-private target. This would test whether DP changes the benefit of reconstruction variability. The combined experiment has not yet been run.

Based on my proposal, “Uncertainty-Aware Membership Inference and Differential Privacy” (Q1, Q3, Q5, Q8), and existing saved results from the two named notebooks. Illustrative examples are explicitly labelled and are not reported as measured results. No notebook cells or experiments were rerun to prepare this document.
