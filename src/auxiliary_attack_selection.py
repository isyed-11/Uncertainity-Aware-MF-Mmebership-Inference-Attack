"""Auxiliary-only attack comparison on fixed record groups; no import-time work."""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .membership_attack import PAPER_FEATURES, evaluate_attack, split_attack_data, threshold_for_fpr


POOLS = ('pooled', 'random_forest', 'gradient_boosting', 'xgboost')
FEATURE_SETS = {
    'paper': list(PAPER_FEATURES),
    'paper_no_walk': [name for name in PAPER_FEATURES if name != 'blind_walk'],
}


def master_record_roles(frame, *, seed=42):
    """Call once on the full pool, never recompute after filtering families."""
    return {name: set(part['record_id']) for name, part in
        split_attack_data(frame, seed=seed, features=PAPER_FEATURES).items()}


def pool_frame(frame, pool):
    if pool not in POOLS:
        raise ValueError(f'Unknown pool: {pool}')
    result = frame if pool == 'pooled' else frame.loc[frame['shadow_model'].str.startswith(pool + '_')]
    if result.empty:
        raise ValueError(f'Empty auxiliary pool: {pool}')
    return result


def fixed_pool_splits(frame, roles):
    """Use one master assignment even when some IDs are absent from a subpool."""
    if set(roles) != {'train', 'validation', 'test'}:
        raise ValueError('Expected train, validation and test record roles.')
    seen = set()
    for ids in roles.values():
        ids = set(ids)
        if seen & ids:
            raise ValueError('Record roles overlap.')
        seen.update(ids)
    required = set(PAPER_FEATURES + ['record_id', 'membership', 'shadow_model'])
    if required - set(frame):
        raise ValueError(f'Missing columns: {sorted(required - set(frame))}')
    if frame['record_id'].isna().any() or not set(frame['record_id']) <= seen:
        raise ValueError('Every auxiliary record must have a master role.')
    if not np.isfinite(frame[PAPER_FEATURES].to_numpy(dtype=float)).all():
        raise ValueError('Attack features must be finite.')
    result = {name: frame.loc[frame['record_id'].isin(ids)].copy() for name, ids in roles.items()}
    if sum(map(len, result.values())) != len(frame):
        raise ValueError('Master roles must cover every row exactly once.')
    for name, part in result.items():
        if set(part['membership'].unique()) != {0, 1}:
            raise ValueError(f'{name} requires both membership classes.')
    return result


class LossThresholdScore:
    """Fixed score exp(-loss): larger is more member-like, not calibrated MIA probability."""
    feature_names_in_ = np.asarray(['cross_entropy'])

    def predict_proba(self, frame):
        loss = frame['cross_entropy'].to_numpy(dtype=float)
        if not np.isfinite(loss).all() or (loss < 0).any():
            raise ValueError('Cross entropy must be finite and nonnegative.')
        score = np.exp(-loss)
        return np.column_stack([1 - score, score])


def candidate_specs():
    """84 predeclared candidates: 4 loss + 8 logistic + 72 RF."""
    result = []
    for pool in POOLS:
        result.append(dict(candidate=f'{pool}/loss_threshold', pool=pool,
            feature_set='loss_only', algorithm='loss_threshold', max_depth=None, min_samples_leaf=None))
        for feature_set in FEATURE_SETS:
            result.append(dict(candidate=f'{pool}/{feature_set}/logistic', pool=pool,
                feature_set=feature_set, algorithm='logistic_regression',
                max_depth=None, min_samples_leaf=None))
            for depth in (4, 8, 12):
                for leaf in (20, 50, 100):
                    result.append(dict(candidate=f'{pool}/{feature_set}/rf_d{depth}_leaf{leaf}',
                        pool=pool, feature_set=feature_set, algorithm='random_forest',
                        max_depth=depth, min_samples_leaf=leaf))
    return result


def _calibrate(classifier, validation):
    scores = classifier.predict_proba(validation[list(classifier.feature_names_in_)])[:, 1]
    labels = validation['membership'].to_numpy()
    fpr, tpr, cutoffs = roc_curve(labels, scores)
    return dict(accuracy_threshold=float(cutoffs[np.argmax(tpr - fpr)]),
        low_fpr_threshold=threshold_for_fpr(labels, scores))


def select_auxiliary_attack(frame, roles, *, seed=42, progress=None):
    """Fit on auxiliary train; calibrate/select on validation; never score test.

    Retain only the overall winner and the unchanged pooled paper-feature RF
    reference. Sorting: validation AUC, validation balanced accuracy, original
    candidate order. Test and target metrics cannot influence this selection.
    """
    splits = {pool: fixed_pool_splits(pool_frame(frame, pool), roles) for pool in POOLS}
    specifications = candidate_specs()
    rows, winner, reference, best_key = [], None, None, None
    for index, spec in enumerate(specifications):
        if progress:
            progress(f'Attack comparison {index + 1}/{len(specifications)}: {spec["candidate"]}')
        parts = splits[spec['pool']]
        if spec['algorithm'] == 'loss_threshold':
            classifier = LossThresholdScore()
        else:
            features = FEATURE_SETS[spec['feature_set']]
            if spec['algorithm'] == 'logistic_regression':
                classifier = make_pipeline(StandardScaler(), LogisticRegression(
                    C=1., solver='lbfgs', max_iter=1000, random_state=seed))
            else:
                classifier = RandomForestClassifier(n_estimators=300,
                    max_depth=spec['max_depth'], min_samples_leaf=spec['min_samples_leaf'],
                    max_features='sqrt', random_state=seed, n_jobs=2)
            classifier.fit(parts['train'][features], parts['train']['membership'])
        cutoffs = _calibrate(classifier, parts['validation'])
        metrics = {name: evaluate_attack(classifier, parts[name],
            threshold=cutoffs['accuracy_threshold'], low_fpr_threshold=cutoffs['low_fpr_threshold'])[0]
            for name in ('train', 'validation')}
        row = dict(spec, candidate_order=index,
            shadow_models=pool_frame(frame, spec['pool'])['shadow_model'].nunique(),
            training_records=len(parts['train']), validation_records=len(parts['validation']),
            **cutoffs)
        for name, values in metrics.items():
            for metric in ('accuracy', 'balanced_accuracy', 'roc_auc', 'tpr_at_1pct_fpr_roc'):
                row[f'{name}_{metric}'] = values[metric]
        rows.append(row)
        key = (row['validation_roc_auc'], row['validation_balanced_accuracy'], -index)
        if best_key is None or key > best_key:
            best_key = key
            winner = dict(classifier=classifier, spec=dict(spec), **cutoffs)
        if spec['candidate'] == 'pooled/paper/rf_d12_leaf20':
            reference = dict(classifier=classifier, spec=dict(spec), **cutoffs)
    table = pd.DataFrame(rows).sort_values(
        ['validation_roc_auc', 'validation_balanced_accuracy', 'candidate_order'],
        ascending=[False, False, True], kind='stable').reset_index(drop=True)
    if winner is None or reference is None:
        raise RuntimeError('Selection or the original reference is missing.')
    return dict(selected=winner, reference=reference, validation_table=table)


def evaluate_frozen_attack(result, frame):
    """Use the already selected classifier and cutoffs without any calibration."""
    return evaluate_attack(result['classifier'], frame,
        threshold=result['accuracy_threshold'], low_fpr_threshold=result['low_fpr_threshold'])


def validation_feature_diagnostics(frame, roles):
    """Descriptive membership distributions and predeclared score directions."""
    directions = dict(class_scaled_logit=1, blind_walk=1, entropy=-1, cross_entropy=-1)
    distributions, univariate = [], []
    for pool in POOLS:
        validation = fixed_pool_splits(pool_frame(frame, pool), roles)['validation']
        for feature, direction in directions.items():
            univariate.append(dict(pool=pool, feature=feature, member_score_direction=direction,
                validation_roc_auc=roc_auc_score(validation['membership'], direction * validation[feature])))
            for membership, part in validation.groupby('membership'):
                values = part[feature]
                distributions.append(dict(pool=pool, feature=feature, membership=int(membership),
                    records=len(values), mean=values.mean(), std=values.std(),
                    q10=values.quantile(.1), median=values.median(), q90=values.quantile(.9)))
    return pd.DataFrame(distributions), pd.DataFrame(univariate)
