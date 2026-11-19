"""Disjoint same-year matched MLP attacks and offline LiRA diagnostics.

Nothing fits, queries or writes on import. Targets and the existing cross-year
experiments are never retrained or modified. Offline LiRA uses OUT distributions
only; it is not an online IN/OUT likelihood ratio or a privacy certificate.
"""
import math

import numpy as np
import pandas as pd
import torch
from scipy import sparse
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import GroupShuffleSplit, train_test_split

from .data import TARGET
from .membership_attack import split_attack_data, threshold_for_fpr
from .multiyear_baseline import (
    BASELINE_FEATURES, build_baseline_preprocessor, train_baseline_shadow,
)
from .paper_attack_features import collect_shadow_paper_features
from .notebook_output import quiet_output


RICH_FEATURES = ['p_income_le_50k', 'p_income_gt_50k', 'true_class_probability',
    'confidence', 'entropy', 'cross_entropy', 'correct', 'class_scaled_logit',
    'modified_entropy']


def split_same_year_shadow(pool, *, seed=42, early_stopping=True):
    """Independent stratified 75/10/15 or fixed-schedule 80/0/20 splits.

    Integer rounding: ceil(test fraction*N), ceil(validation fraction*N),
    remainder to training. Nonmembers are the untouched shadow TEST partition.
    """
    if type(early_stopping) is not bool or not pool.index.is_unique:
        raise ValueError('Require a boolean early-stopping flag and unique pool IDs.')
    test_size = math.ceil(len(pool) * (.15 if early_stopping else .20))
    remaining, test = train_test_split(pool, test_size=test_size,
        stratify=pool[TARGET], random_state=seed)
    if early_stopping:
        train, validation = train_test_split(remaining, test_size=math.ceil(.10*len(pool)),
            stratify=remaining[TARGET], random_state=seed+1)
        parts = dict(train=train, validation=validation, nonmembers=test)
    else:
        parts = dict(train=remaining, nonmembers=test)
    return {name: frame.sort_index().copy() for name, frame in parts.items()}


def assign_attack_record_roles(pool_ids, *, seed=42):
    """Freeze grouped 60/20/20 attack roles BEFORE balancing predictions."""
    ids = pd.Index(pool_ids)
    if not ids.is_unique:
        raise ValueError('Attack record roles require unique IDs.')
    train, remaining = next(GroupShuffleSplit(n_splits=1, test_size=.4,
        random_state=seed).split(ids, groups=ids))
    validation, test = next(GroupShuffleSplit(n_splits=1, test_size=.5,
        random_state=seed+1).split(ids[remaining], groups=ids[remaining]))
    return dict(train=set(ids[train]), validation=set(ids[remaining[validation]]),
        test=set(ids[remaining[test]]))


def balanced_shadow_attack_ids(partitions, role_ids, *, seed=42):
    """Keep ALL test OUT predictions; sample equal IN counts within each role.

    Membership balance holds per shadow and per attack role, without splitting
    repeated record IDs across roles. Income-stratify IN subsampling when feasible;
    tiny slices use uniform sampling instead. No validation records are eligible.
    """
    train, test = partitions['train'], partitions['nonmembers']
    if set(role_ids) != {'train', 'validation', 'test'}:
        raise ValueError('Expected all three attack record roles.')
    seen = set()
    for ids in role_ids.values():
        if seen & set(ids):
            raise ValueError('Attack record roles overlap.')
        seen.update(ids)
    if not set(train.index).union(test.index) <= seen:
        raise ValueError('Attack roles do not cover shadow train/test records.')
    selected_train, selected_test, audit = [], [], []
    for offset, (role, ids) in enumerate(role_ids.items()):
        in_rows = train.loc[train.index.isin(ids)]
        out_rows = test.loc[test.index.isin(ids)]
        count = len(out_rows)
        if len(in_rows) < count:
            raise ValueError(f'Insufficient shadow IN records in attack role {role} to match all OUT records.')
        if count and count < len(in_rows):
            counts = in_rows[TARGET].value_counts()
            stratify = in_rows[TARGET] if (counts.min() >= 2 and count >= len(counts)
                and len(in_rows)-count >= len(counts)) else None
            _, chosen = train_test_split(in_rows, test_size=count, stratify=stratify,
                random_state=seed+100+offset)
        else:
            chosen = in_rows.iloc[:count]
        selected_train.extend(chosen.index)
        selected_test.extend(out_rows.index)
        audit.append(dict(attack_role=role, available_in=len(in_rows), available_out=count,
            selected_in=len(chosen), selected_out=count))
    if len(selected_train) != len(selected_test) or set(selected_test) != set(test.index):
        raise RuntimeError('Balanced selection must retain every OUT test record exactly once.')
    return dict(train=np.asarray(selected_train), nonmembers=np.asarray(selected_test)), pd.DataFrame(audit)


def disjoint_same_year_pool(source, target_partitions):
    """Reserve only unused source rows; exclude ALL target train/evaluation IDs."""
    if not source.index.is_unique:
        raise ValueError('Same-year source IDs must be unique.')
    required = {'train', 'validation', 'test', 'mia_members', 'mia_nonmembers'}
    if not required <= set(target_partitions):
        raise ValueError('All five target partitions are required for the disjointness audit.')
    protected = set()
    for frame in target_partitions.values():
        if not frame.index.is_unique or not set(frame.index) <= set(source.index):
            raise ValueError('Target partitions must contain unique IDs from the same-year source.')
        protected.update(frame.index)
    pool = source.loc[~source.index.isin(protected)].sort_index().copy()
    if pool.empty or set(pool[TARGET].unique()) != {0, 1}:
        raise ValueError('Insufficient unused same-year records with both income classes.')
    audit = []
    for name, frame in target_partitions.items():
        overlap = set(pool.index) & set(frame.index)
        if overlap:
            raise ValueError(f'Shadow pool overlaps target {name}.')
        audit.append(dict(target_partition=name, target_records=len(frame),
            shadow_pool_records=len(pool), overlapping_record_ids=len(overlap)))
    return pool, pd.DataFrame(audit)


def output_features_from_logits(logits, income_labels, record_ids):
    """Full binary probability vector and stable label-aware confidence features."""
    logits = np.asarray(logits, dtype=np.float64)
    labels = np.asarray(income_labels, dtype=int)
    if (logits.shape != (len(labels), 2) or len(record_ids) != len(labels)
            or not np.isfinite(logits).all() or not np.isin(labels, [0, 1]).all()):
        raise ValueError('Expected finite binary logits with aligned income labels/IDs.')
    difference = logits[:, 1] - logits[:, 0]
    log_p0, log_p1 = -np.logaddexp(0., difference), -np.logaddexp(0., -difference)
    p0, p1 = np.exp(log_p0), np.exp(log_p1)
    margin = (2 * labels - 1) * difference
    log_true = np.where(labels == 1, log_p1, log_p0)
    true_p = np.where(labels == 1, p1, p0)
    # Binary modified entropy: -(1-p_y)log(p_y)-p_other log(1-p_other).
    # log(1-p_other) == log(p_y); no cancellation at probabilities near one.
    modified_entropy = -2 * (1 - true_p) * log_true
    return pd.DataFrame(dict(p_income_le_50k=p0, p_income_gt_50k=p1,
        true_class_probability=true_p, confidence=np.maximum(p0, p1),
        entropy=-(p0 * log_p0 + p1 * log_p1),
        cross_entropy=np.logaddexp(0., -margin), correct=(logits.argmax(1) == labels).astype(int),
        class_scaled_logit=margin, modified_entropy=modified_entropy),
        index=pd.Index(record_ids, name='record_id'))


def query_output_features(model, preprocessor, frame, *, batch_size=1024):
    """Read unaugmented logits in batches; neither membership labels nor fitting."""
    if frame.empty or not frame.index.is_unique or batch_size < 1:
        raise ValueError('Queries require nonempty unique records and a positive batch size.')
    device = next(model.parameters()).device
    model.eval()
    outputs = []
    with torch.no_grad():
        for start in range(0, len(frame), batch_size):
            rows = frame.iloc[start:start + batch_size]
            values = preprocessor.transform(rows[BASELINE_FEATURES])
            if sparse.issparse(values):
                values = values.toarray()
            logits = model(torch.as_tensor(np.asarray(values), dtype=torch.float32, device=device))
            outputs.append(logits.detach().cpu().numpy().astype(np.float64))
    return output_features_from_logits(np.concatenate(outputs), frame[TARGET], frame.index)


def score_metrics(labels, scores, *, threshold, low_fpr_threshold=None):
    """Fixed-cutoff accuracy plus class-aware metrics; single-class AUC is undefined."""
    labels, scores = np.asarray(labels, dtype=int), np.asarray(scores, dtype=float)
    if (labels.ndim != 1 or scores.shape != labels.shape or not len(labels)
            or not np.isin(labels, [0, 1]).all() or not np.isfinite(scores).all()):
        raise ValueError('Expected nonempty aligned binary membership labels and finite scores.')
    predicted = scores >= threshold
    members = labels == 1
    n_in, n_out = int(members.sum()), int((~members).sum())
    tp, tn = int(predicted[members].sum()), int((~predicted[~members]).sum())
    accuracy = (tp + tn) / len(labels)
    # Wilson interval is descriptive; repeated-shadow dependence is not removed.
    z, n = 1.959963984540054, len(labels)
    center = (accuracy + z*z/(2*n)) / (1 + z*z/n)
    half = z * math.sqrt(accuracy*(1-accuracy)/n + z*z/(4*n*n)) / (1 + z*z/n)
    result = dict(records=n, members=n_in, nonmembers=n_out, member_fraction=n_in/n,
        majority_class_accuracy=max(n_in, n_out)/n, accuracy=accuracy,
        accuracy_ci95_low=center-half, accuracy_ci95_high=center+half,
        threshold=float(threshold), tp=tp, tn=tn, fp=n_out-tn, fn=n_in-tp,
        balanced_accuracy=float('nan'), roc_auc=float('nan'),
        tpr_at_1pct_fpr_roc=float('nan'), minimum_nonzero_fpr=1/n_out if n_out else float('nan'),
        metric_status='both_classes' if n_in and n_out else 'single_membership_class_auc_undefined')
    if n_in and n_out:
        result['balanced_accuracy'] = .5 * (tp/n_in + tn/n_out)
        result['roc_auc'] = float(roc_auc_score(labels, scores))
        fpr, tpr, _ = roc_curve(labels, scores, drop_intermediate=False)
        result['tpr_at_1pct_fpr_roc'] = float(tpr[fpr <= .01].max())
    if low_fpr_threshold is not None:
        low = scores >= low_fpr_threshold
        result.update(shadow_selected_threshold=float(low_fpr_threshold),
            tpr_at_shadow_threshold=float(low[members].mean()) if n_in else float('nan'),
            fpr_at_shadow_threshold=float(low[~members].mean()) if n_out else float('nan'))
    return result


def validation_thresholds(frame, column):
    if set(frame['membership'].unique()) != {0, 1}:
        raise ValueError('Threshold calibration needs both membership classes in shadow validation.')
    labels, scores = frame['membership'], frame[column]
    fpr, tpr, thresholds = roc_curve(labels, scores)
    return float(thresholds[np.argmax(tpr-fpr)]), threshold_for_fpr(labels, scores)


def fit_rich_attack(shadow_rows, features, *, seed=42, role_ids=None):
    """Existing RF recipe and grouped split, with explicitly richer output signals."""
    if role_ids is None:
        splits = split_attack_data(shadow_rows, seed=seed, features=features)
    else:
        splits = {role: shadow_rows.loc[shadow_rows.record_id.isin(ids)].copy()
            for role, ids in role_ids.items()}
        if set(splits) != {'train', 'validation', 'test'}:
            raise ValueError('Expected fixed train/validation/test attack roles.')
        seen = set()
        for role, frame in splits.items():
            counts = frame.membership.value_counts()
            if set(counts.index) != {0, 1} or counts[0] != counts[1] or seen & set(frame.record_id):
                raise ValueError(f'Attack role {role} must be disjoint and exactly membership-balanced.')
            seen.update(frame.record_id)
        if sum(len(frame) for frame in splits.values()) != len(shadow_rows):
            raise ValueError('Attack roles must cover every scored row exactly once.')
    classifier = RandomForestClassifier(n_estimators=300, max_depth=12,
        min_samples_leaf=20, max_features='sqrt', random_state=seed, n_jobs=2)
    classifier.fit(splits['train'][features], splits['train']['membership'])
    scored = {}
    for role, frame in splits.items():
        frame = frame.copy()
        frame['rich_rf_score'] = classifier.predict_proba(frame[features])[:, 1]
        scored[role] = frame
    threshold, low = validation_thresholds(scored['validation'], 'rich_rf_score')
    metrics = {role: score_metrics(frame.membership, frame.rich_rf_score,
        threshold=threshold, low_fpr_threshold=low) for role, frame in scored.items()}
    return dict(classifier=classifier, splits=scored, metrics=metrics,
        accuracy_threshold=threshold, low_fpr_threshold=low, features=list(features))


def gaussian_out_score(values, centers, scale):
    """Offline LiRA: negative OUT Gaussian log likelihood; higher means member.

    Median per-record OUT locations and one pooled fixed variance follow the
    reference offline implementation (no augmentations or candidate IN models):
    tensorflow/privacy/research/mi_lira_2021/plot.py, generate_ours_offline.
    """
    values, centers = np.asarray(values, dtype=float), np.asarray(centers, dtype=float)
    if values.shape != centers.shape or not np.isfinite(values).all() or not np.isfinite(centers).all():
        raise ValueError('Offline scores need aligned finite values and centers.')
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError('OUT scale must be positive and finite.')
    return .5*((values-centers)/scale)**2 + math.log(scale) + .5*math.log(2*math.pi)


def fit_offline_calibration(auxiliary_margins, pure_out_masks, shadow_rows,
                            role_ids, pool_ids, *, minimum_out=3, scale_floor=1e-6):
    """Leave query model out; fit variance on attack-TRAIN IDs and cutoffs on VAL.

    pure_out_masks excludes shadow validation rows as well as training rows:
    validation can influence early stopping. Each pseudo-target score excludes
    the query model itself, even when that candidate was its nonmember.
    """
    values = np.asarray(auxiliary_margins, dtype=float)
    out = np.asarray(pure_out_masks, dtype=bool)
    pool_ids = pd.Index(pool_ids)
    if (values.ndim != 2 or values.shape != out.shape or values.shape[1] != len(pool_ids)
            or not pool_ids.is_unique or not np.isfinite(values).all()
            or not isinstance(minimum_out, int) or minimum_out < 2
            or not math.isfinite(scale_floor) or scale_floor <= 0):
        raise ValueError('Invalid reference scores, masks, IDs or calibration settings.')
    if set(role_ids) != {'train', 'validation', 'test'}:
        raise ValueError('Expected fixed attack train/validation/test record roles.')
    seen = set()
    for ids in role_ids.values():
        ids = set(ids)
        if seen & ids or not ids <= set(pool_ids):
            raise ValueError('Calibration roles must be disjoint IDs within the shadow pool.')
        seen.update(ids)
    if not set(shadow_rows.record_id) <= seen:
        raise ValueError('Every shadow attack row must have a fixed record role.')
    train_positions = np.flatnonzero(pool_ids.isin(role_ids['train']))
    train_out = values[:, train_positions][out[:, train_positions]]
    if len(train_out) < 2:
        raise ValueError('Insufficient attack-training OUT scores for fixed-variance fitting.')
    scale = max(float(np.std(train_out)), scale_floor)
    parts = []
    skipped = 0
    for query_model, rows in shadow_rows.groupby('reference_index', sort=True):
        query_model = int(query_model)
        if not 0 <= query_model < len(values):
            raise ValueError('Unknown query model index.')
        positions = pool_ids.get_indexer(rows.record_id)
        references = out[:, positions].copy()
        references[query_model] = False
        counts = references.sum(axis=0)
        usable = counts >= minimum_out
        skipped += int((~usable).sum())
        if not usable.any():
            continue
        positions, references = positions[usable], references[:, usable]
        centers = np.nanmedian(np.where(references, values[:, positions], np.nan), axis=0)
        kept = rows.loc[usable, ['record_id', 'membership', 'reference_index']].copy()
        kept['out_reference_count'] = counts[usable]
        kept['out_center'] = centers
        kept['offline_lira_score'] = gaussian_out_score(values[query_model, positions], centers, scale)
        parts.append(kept)
    if not parts:
        raise ValueError('No usable leave-one-model-out calibration rows; increase reference count.')
    scored = pd.concat(parts, ignore_index=True)
    splits = {role: scored.loc[scored.record_id.isin(ids)].copy() for role, ids in role_ids.items()}
    threshold, low = validation_thresholds(splits['validation'], 'offline_lira_score')
    metrics = {role: score_metrics(frame.membership, frame.offline_lira_score,
        threshold=threshold, low_fpr_threshold=low) for role, frame in splits.items()}
    return dict(scale=scale, minimum_out=minimum_out, skipped_rows=skipped,
        kept_rows=len(scored), out_reference_count_summary=dict(
            minimum=int(scored.out_reference_count.min()),
            median=float(scored.out_reference_count.median()),
            maximum=int(scored.out_reference_count.max())),
        accuracy_threshold=threshold, low_fpr_threshold=low, splits=splits, metrics=metrics)


def exact_outlier_mask(anomaly_scores, *, fraction=.05):
    scores = np.asarray(anomaly_scores, dtype=float)
    if scores.ndim != 1 or not len(scores) or not np.isfinite(scores).all() or not 0 < fraction < 1:
        raise ValueError('Expected finite anomaly scores and a fraction between zero and one.')
    count = max(1, int(np.ceil(fraction*len(scores))))
    mask = np.zeros(len(scores), dtype=bool)
    mask[np.argsort(-scores, kind='stable')[:count]] = True
    return mask


def fit_outlier_slices(shadow_pool, candidates, *, seed=42):
    """Fit Isolation Forest on disjoint shadow data; rank candidates without labels."""
    processor = build_baseline_preprocessor()
    fit_x = sparse.csc_matrix(processor.fit_transform(shadow_pool[BASELINE_FEATURES]), dtype=np.float32)
    detector = IsolationForest(n_estimators=100, max_samples=256, contamination='auto',
        random_state=seed, n_jobs=2)
    detector.fit(fit_x)
    candidate_x = sparse.csr_matrix(processor.transform(candidates[BASELINE_FEATURES]), dtype=np.float32)
    scores = -detector.score_samples(candidate_x)
    mask = exact_outlier_mask(scores)
    return scores, mask


def subgroup_metrics(scored, candidates, outlier_mask, *, score_column, threshold, low_fpr_threshold):
    """Predeclared slices, frozen global cutoffs, and explicit membership imbalance."""
    if not scored.index.is_unique or not scored.index.equals(candidates.index):
        raise ValueError('Candidate inputs and scored rows must have exactly aligned unique IDs.')
    outlier_mask = np.asarray(outlier_mask, dtype=bool)
    if outlier_mask.shape != (len(candidates),):
        raise ValueError('Outlier mask is misaligned.')
    slices = [('all', 'all', np.ones(len(candidates), dtype=bool)),
        ('outlier', 'top_5_percent', outlier_mask), ('outlier', 'remaining_95_percent', ~outlier_mask)]
    for field in [TARGET, 'SEX', 'RAC1P']:
        for value in sorted(candidates[field].dropna().unique()):
            slices.append((field, str(value), candidates[field].eq(value).to_numpy()))
        if candidates[field].isna().any():
            slices.append((field, 'missing', candidates[field].isna().to_numpy()))
    rows = []
    for group_type, group, mask in slices:
        part = scored.loc[mask]
        if part.empty:
            continue
        rows.append(dict(group_type=group_type, group=group, **score_metrics(
            part.membership, part[score_column], threshold=threshold, low_fpr_threshold=low_fpr_threshold)))
    return pd.DataFrame(rows)


def generalization_row(metrics):
    """Positive validation-minus-training loss is the reported loss gap."""
    validation = metrics.get('validation', dict(accuracy=float('nan'), log_loss=float('nan')))
    result = dict(train_accuracy=metrics['train']['accuracy'], validation_accuracy=validation['accuracy'],
        train_log_loss=metrics['train']['log_loss'], validation_log_loss=validation['log_loss'],
        validation_minus_train_log_loss=validation['log_loss']-metrics['train']['log_loss'],
        train_minus_validation_accuracy=metrics['train']['accuracy']-validation['accuracy'],
        validation_available='validation' in metrics)
    holdout = 'test' if 'test' in metrics else 'nonmembers'
    result.update(holdout_partition=holdout, holdout_accuracy=metrics[holdout]['accuracy'],
        holdout_log_loss=metrics[holdout]['log_loss'],
        holdout_minus_train_log_loss=metrics[holdout]['log_loss']-metrics['train']['log_loss'],
        train_minus_holdout_accuracy=metrics['train']['accuracy']-metrics[holdout]['accuracy'])
    return result


def verify_target_recipe(target, config):
    """Reject stale in-memory targets trained under a different notebook config."""
    if list(target['preprocessor'].feature_names_in_) != BASELINE_FEATURES:
        raise ValueError('Target must use the common 47-feature preprocessing recipe.')
    metadata = target['training']
    for key, value in config.items():
        field = 'epochs_requested' if key == 'epochs' else key
        if key == 'seed':
            continue  # References intentionally use independent seeded splits/initializations.
        expected = list(value) if isinstance(value, tuple) else value
        actual = metadata.get(field)
        if isinstance(actual, tuple):
            actual = list(actual)
        if actual != expected:
            raise ValueError(f'Target/config mismatch for {key}: {actual!r} versus {expected!r}. '
                'Use the target trained by this notebook, not a cached target from another study.')


def run_same_year_study(source, target, *, target_year, count, train_config, walk_config,
                       numeric_protocol, include_blind_walk=True,
                       seed=42, progress=None):
    """Explicit expensive entry point: fit same-year shadows and their rich RF.

    Offline calibration, slicing and exporting are separate notebook cells.
    Models are released after querying; no checkpoints/logs/files are created.
    """
    if not isinstance(count, int) or isinstance(count, bool) or count < 4:
        raise ValueError('At least four reference models are required; study default is 16.')
    verify_target_recipe(target, train_config)
    early_stopping = train_config.get('early_stopping', True)
    if not early_stopping and train_config.get('restore_best', True):
        raise ValueError('The 80/20 no-validation policy requires restore_best=False as well as early_stopping=False.')
    pool, audit = disjoint_same_year_pool(source, target['partitions'])
    attack_role_ids = assign_attack_record_roles(pool.index, seed=seed)
    candidates = pd.concat([target['partitions']['mia_members'].assign(membership=1),
        target['partitions']['mia_nonmembers'].assign(membership=0)]).sort_index()
    target_outputs = query_output_features(target['model'], target['preprocessor'], candidates)
    target_rows = target_outputs.copy()
    target_rows['membership'] = candidates['membership']
    if include_blind_walk:
        existing = target['attack_features'].set_index('record_id')
        target_rows['blind_walk'] = existing.loc[candidates.index, 'blind_walk']
    features = RICH_FEATURES + (['blind_walk'] if include_blind_walk else [])
    shadow_rows, margins, pure_out_masks, target_margins = [], [], [], []
    histories, model_protocols, diagnostics, balance_audits = {}, {}, [], []
    pool_positions = pd.Index(pool.index)
    for index in range(count):
        name, model_seed = f'same_year_mlp_{index+1:02d}', seed+10*index
        if progress:
            progress(f'{target_year}: matched MLP shadow {index+1}/{count}, seed={model_seed}')
        config = dict(train_config, seed=model_seed)
        partitions = split_same_year_shadow(pool, seed=model_seed, early_stopping=early_stopping)
        selected_ids, balance_audit = balanced_shadow_attack_ids(partitions, attack_role_ids, seed=model_seed)
        balance_audits.append(balance_audit.assign(shadow_model=name))
        with quiet_output():
            shadow = train_baseline_shadow(pool, 'mlp', seed=model_seed, parameters=config, partitions=partitions)
            if list(shadow['preprocessor'].feature_names_in_) != BASELINE_FEATURES:
                raise RuntimeError('Shadow preprocessing schema does not match the target recipe.')
            for role, ids in shadow['partition_ids'].items():
                if not set(ids) <= set(pool.index):
                    raise RuntimeError(f'{name}/{role} contains non-pool records.')
            full_outputs = query_output_features(shadow['model'], shadow['preprocessor'], pool)
            target_reference = query_output_features(shadow['model'], shadow['preprocessor'], candidates)
            attack_view = dict(shadow, partition_ids=selected_ids)
            queried, _ = collect_shadow_paper_features({name: attack_view}, pool,
                per_class=None, config=walk_config, seed=seed,
                include_blind_walk=include_blind_walk, numeric_protocol=numeric_protocol)
        enriched = full_outputs.loc[queried.record_id, RICH_FEATURES].reset_index()
        enriched['membership'] = queried['membership'].to_numpy()
        enriched['reference_index'], enriched['shadow_model'] = index, name
        if include_blind_walk:
            enriched['blind_walk'] = queried['blind_walk'].to_numpy()
        shadow_rows.append(enriched)
        margins.append(full_outputs.class_scaled_logit.to_numpy())
        # Validation records can influence epoch selection, so only strict held-out
        # nonmembers qualify as OUT references for pseudo-target calibration.
        pure_out_masks.append(pool_positions.isin(shadow['partition_ids']['nonmembers']))
        target_margins.append(target_reference.class_scaled_logit.to_numpy())
        histories[name], model_protocols[name] = shadow['history'], dict(
            seed=model_seed, training=shadow['metadata'],
            partition_sizes={role: len(ids) for role, ids in shadow['partition_ids'].items()})
        diagnostics.append(dict(shadow_model=name, **generalization_row(shadow['metrics'])))
        del shadow, attack_view, full_outputs, target_reference, queried, enriched, partitions
    rows = pd.concat(shadow_rows, ignore_index=True)
    if progress:
        progress(f'{target_year}: fitting rich RF attack on same-year shadow outputs')
    with quiet_output():
        attack = fit_rich_attack(rows, features, seed=seed, role_ids=attack_role_ids)
    target_rows['rich_rf_score'] = attack['classifier'].predict_proba(target_rows[features])[:, 1]
    return dict(target_year=target_year, pool=pool, disjointness_audit=audit,
        candidates=candidates, target_rows=target_rows, shadow_rows=rows, rich_attack=attack,
        auxiliary_margins=np.asarray(margins), pure_out_masks=np.asarray(pure_out_masks),
        target_reference_margins=np.asarray(target_margins),
        histories=histories, model_protocols=model_protocols,
        attack_balance_audit=pd.concat(balance_audits, ignore_index=True),
        attack_role_ids=attack_role_ids,
        shadow_split_policy='75/10/15' if early_stopping else '80/0/20',
        shadow_generalization=pd.DataFrame(diagnostics), features=features,
        target_generalization=generalization_row(target['income_metrics']))
