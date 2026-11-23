"""Score frozen missing-feature attacks against one inference API at a time.

Importing this module does not train or query models or write any files.
"""
import pandas as pd

from .expanded_data import EXPANDED_FEATURES
from .membership_attack import evaluate_attack
from .missing_feature_pfv import complete_pfv
from .paper_attack_features import extract_paper_features
from .uncertainty_mf_mia import completion_scores, summarize_scores, evaluate_aggregation


def score_missing_feature_target(query, mia_partitions, auxiliary, masks,
                                reconstructors, aggregators, attack_result, *,
                                k, reconstruction_seed, pfv_seed, walk_config,
                                numeric_protocol, progress=None):
    """Apply PFV and four fixed conditional controls to identical target IDs.

    No target membership labels enter completion, feature extraction, fitting or
    threshold selection. Labels are attached only after inference. Conditional
    single, average, learned mean and uncertainty all reuse the same K scores.
    The caller supplies fitted auxiliary-only reconstructors/aggregators and a
    probability-query API, never target weights. Nothing is saved to disk.
    """
    if not isinstance(k, int) or k < 2:
        raise ValueError('Use at least two completion draws.')
    if not masks or set(masks) != set(reconstructors) or set(masks) != set(aggregators):
        raise ValueError('Masks, reconstructors and aggregators must match.')
    members, nonmembers = (mia_partitions[name] for name in ('mia_members', 'mia_nonmembers'))
    if members.empty or len(members) != len(nonmembers):
        raise ValueError('Use nonempty, balanced target MIA partitions.')
    candidates = pd.concat([members, nonmembers]).sort_index()
    if not candidates.index.is_unique:
        raise ValueError('Member/nonmember IDs must be unique and disjoint.')
    # Only predictors and the separately known income class enter completion.
    # Ignore any membership annotations attached by an external caller.
    candidates = candidates[EXPANDED_FEATURES + ['income_gt_50k']].copy()
    if set(candidates.index) & set(auxiliary.index):
        raise ValueError('Target candidate and auxiliary record IDs must be disjoint.')
    member_ids = set(members.index)
    # Match the existing PFV/collect_target_paper_features query ordering.
    pfv_order = pd.concat([members, nonmembers]).sample(frac=1, random_state=pfv_seed).index
    labels = candidates.index.isin(member_ids).astype(int)
    rows, audits = [], {}
    for mask_name, hidden in masks.items():
        reconstructor = reconstructors[mask_name]
        if list(reconstructor.hidden) != list(hidden):
            raise ValueError(f'Reconstruction fields do not match mask {mask_name}.')
        if progress:
            progress(f'{mask_name}: PFV single completion')
        completed = complete_pfv(candidates[EXPANDED_FEATURES], auxiliary[EXPANDED_FEATURES],
                                 hidden, seed=pfv_seed).loc[pfv_order]
        pfv_features, pfv_audit = extract_paper_features(query, completed,
            candidates.loc[pfv_order, 'income_gt_50k'].to_numpy(),
            config=walk_config, include_blind_walk=True, numeric_protocol=numeric_protocol)
        pfv_features['membership'] = pfv_order.isin(member_ids).astype(int)
        metrics, _ = evaluate_attack(attack_result['classifier'], pfv_features,
            threshold=attack_result['accuracy_threshold'],
            low_fpr_threshold=attack_result['low_fpr_threshold'])
        rows.append(dict(mask=mask_name, n_missing=len(hidden), method='pfv', **metrics))
        if progress:
            progress(f'{mask_name}: conditional completion scores, K={k}')
        scores, conditional_audit = completion_scores(query, candidates, reconstructor,
            attack_result['classifier'], k=k, seed=reconstruction_seed,
            walk_config=walk_config, include_blind_walk=True, numeric_protocol=numeric_protocol)
        table = summarize_scores(scores)
        table['membership'] = labels  # Evaluation only, after all K scores exist.
        learned, thresholds = aggregators[mask_name]
        conditional_metrics, _ = evaluate_aggregation(table, learned, thresholds)
        for metrics in conditional_metrics.to_dict('records'):
            rows.append(dict(mask=mask_name, n_missing=len(hidden), **metrics))
        audits[mask_name] = dict(pfv=pfv_audit, conditional=conditional_audit)
        del completed, pfv_features, scores, table
    return rows, audits


def uncertainty_contrasts(results):
    """Within-target, within-mask contrast for the additional std feature.

    Differences are descriptive, not independent repetitions or significance
    tests. The same draws and held-out calibration are used by both methods.
    """
    keys = ['condition', 'mask', 'n_missing']
    metrics = ['accuracy', 'roc_auc', 'tpr_at_1pct_fpr_roc']
    paired = results.loc[results['method'].isin(['learned_mean', 'uncertainty'])]
    if paired.duplicated(keys + ['method']).any():
        raise ValueError('Expected one row per condition/mask/method.')
    mean = paired.loc[paired['method'] == 'learned_mean', keys + metrics]
    uncertainty = paired.loc[paired['method'] == 'uncertainty', keys + metrics]
    joined = mean.merge(uncertainty, on=keys, suffixes=('_mean', '_uncertainty'),
                        how='outer', validate='one_to_one', indicator=True)
    if not joined['_merge'].eq('both').all():
        raise ValueError('Every mean-only result needs its matched uncertainty result.')
    for metric in metrics:
        joined[f'{metric}_change'] = joined[f'{metric}_uncertainty'] - joined[f'{metric}_mean']
    return joined.drop(columns='_merge')
