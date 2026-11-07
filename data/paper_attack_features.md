# Random forest attack with paper-inspired features

The overfit notebook now uses a random forest attack with four feature families
from Section 5.3 of the supplied MF-MIA paper. Target and shadows are unchanged;
only attack feature collection and the attack classifier change. This is still
an adaptation: ACS income target, five shadows, and explicitly chosen walk/RF
settings differ from the paper's GSS gender experiment and nine selected shadows.

## Features

- Class-scaled logit: log(max(p_true, 1e-12)) minus
  log(max(p_other, 1e-12)). For binary softmax outputs this is the true-class
  versus alternative logit margin, computed using probability access only.
  Probability saturation means original extreme logits cannot be recovered exactly.
- Blind walk: average first prediction-change step over independent walks.
  The comparison is with the original prediction, including for misclassified
  records. A walk that does not flip within the budget gets max_steps + 1.
  The diagnostics report its censored fraction; censored distances are lower
  bounds, not measured decision-boundary distances.
- Entropy: minus sum(p * log(max(p, 1e-12))).
- Cross-entropy: minus log(max(p_true, 1e-12)).

True income labels are assumed available. Membership labels and IDs are not
attack features. No target weights, gradients, or hidden training records are
used by feature extraction; the fitted pipeline is encapsulated inside a
probability-query function simulating the native-input API.

## Tabular blind-walk choices

The supplied paper describes random-noise additions but does not specify a full
tabular perturbation protocol. Its cited MICO example uses fixed Gaussian
directions in model-input space. Here, numeric fields follow a fixed Gaussian
direction per walk, with integer rounding and public domain clipping. Age uses
one-year step standard deviation; weekly hours two hours; Adult education one
category; Adult capital gains/losses $100. Each categorical field switches once
to a sampled observed category at a geometric waiting time (probability .05).
The vocabulary comes only from queried candidate records, without membership
labels. There is no Gaussian noise on one-hot blocks or raw occupation codes.
Field-valid perturbations can still create implausible combinations of fields.

Numeric names and fields differ between Adult and ACS. Thus the same procedure
does not guarantee equivalent neighborhood geometry across datasets. This
limitation should be reported, not interpreted as exact reproduction.

Defaults: 20 steps, two independent walks, batch size 512, seed 42. These are
chosen query-budget settings, not values reported by the supplied paper.
All target candidates are used. By default each shadow contributes 2,000 sampled
members and 2,000 sampled nonmembers; sampling is stratified by income class.
With five shadows and 20,000 target candidates, the maximum combined query
count is 1,640,000 individual record predictions. Batching reduces call overhead;
early flips reduce actual queries. Set per_class=None to use every shadow record
at substantially higher cost. Inspect censoring before interpreting results.

## Attack and evaluation

RF: 300 trees, depth 12, minimum leaf size 20, square-root feature subsampling,
two CPU workers, seed 42. The paper does not specify these RF settings.
Adult attack rows use grouped 60/20/20 record-ID splits. Validation alone selects
the balanced-accuracy and low-FPR cutoffs. Evaluation reports the selected
threshold, ROC-AUC, balanced accuracy, confusion counts, ROC TPR@1% FPR, and
TPR/FPR at the shadow-selected low-FPR cutoff. Repeated target evaluation is
exploratory; never select attack settings by maximizing target-MIA accuracy.

## Sources

- Supplied paper, Section 5.3 (PDF pages 5–6).
- Paper's cited MICO implementation: https://www.anshumansuri.com/blog/2023/mico/
- Carlini et al.: https://arxiv.org/abs/2112.03570
