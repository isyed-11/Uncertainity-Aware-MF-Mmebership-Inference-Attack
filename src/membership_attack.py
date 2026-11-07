"""Output-based membership classifier; importing this module runs nothing."""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, confusion_matrix,
                             roc_auc_score, roc_curve)
from sklearn.model_selection import GroupShuffleSplit, ParameterGrid
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .shadow_models import membership_features
from .training import predict_probabilities

ATTACK_FEATURES = ['p_income_le_50k', 'p_income_gt_50k', 'true_class_probability',
                   'confidence', 'entropy', 'loss', 'correct']
PROBABILITY_FEATURES = ['p_income_le_50k', 'p_income_gt_50k', 'confidence', 'entropy']
PAPER_FEATURES = ['class_scaled_logit', 'blind_walk', 'entropy', 'cross_entropy']


def split_attack_data(frame, *, seed=42, features=None):
    """60/20/20 by Adult record ID; repeated shadow observations stay together."""
    required = set((ATTACK_FEATURES if features is None else list(features)) + ['membership', 'record_id'])
    if required - set(frame):
        raise ValueError(f'Missing columns: {sorted(required - set(frame))}')
    if frame['record_id'].isna().any() or frame['membership'].isna().any():
        raise ValueError('Missing record IDs or membership labels.')
    first = GroupShuffleSplit(n_splits=1, test_size=0.4, random_state=seed)
    train_idx, remaining_idx = next(first.split(frame, groups=frame['record_id']))
    remaining = frame.iloc[remaining_idx]
    second = GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=seed + 1)
    validation_idx, test_idx = next(second.split(remaining, groups=remaining['record_id']))
    splits = {'train': frame.iloc[train_idx].copy(),
              'validation': remaining.iloc[validation_idx].copy(),
              'test': remaining.iloc[test_idx].copy()}
    for name, part in splits.items():
        if set(part['membership'].unique()) != {0, 1}:
            raise ValueError(f'{name} needs both membership classes.')
    return splits


def threshold_for_fpr(labels, scores, *, max_fpr=0.01):
    """Pick a threshold on shadow validation only; ties stay conservative."""
    if not 0 < max_fpr < 1:
        raise ValueError('max_fpr must be between zero and one.')
    fpr, tpr, thresholds = roc_curve(labels, scores, drop_intermediate=False)
    eligible = np.flatnonzero(fpr <= max_fpr)
    best = eligible[np.argmax(tpr[eligible])]
    return float(thresholds[best])


def evaluate_attack(classifier, frame, *, threshold=0.5, low_fpr_threshold=None):
    """Membership metrics, including ROC-derived and fixed-threshold low-FPR TPR.

    ROC TPR@1%FPR is an evaluation summary, not a deployable threshold chosen
    without labels. The separate fixed-threshold metrics use shadow validation.
    """
    labels = frame['membership'].to_numpy(dtype=int)
    if set(np.unique(labels)) != {0, 1}:
        raise ValueError('Evaluation requires members and non-members.')
    features = list(classifier.feature_names_in_)
    scores = classifier.predict_proba(frame[features])[:, 1]
    predictions = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    fpr, tpr, _ = roc_curve(labels, scores, drop_intermediate=False)
    metrics = {
        'records': len(labels), 'members': int(labels.sum()),
        'accuracy': float(accuracy_score(labels, predictions)),
        'balanced_accuracy': float(balanced_accuracy_score(labels, predictions)),
        'roc_auc': float(roc_auc_score(labels, scores)),
        'tpr_at_1pct_fpr_roc': float(tpr[fpr <= 0.01].max()),
        'threshold': threshold, 'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp),
    }
    if low_fpr_threshold is not None:
        low_predictions = scores >= low_fpr_threshold
        metrics['shadow_selected_threshold'] = low_fpr_threshold
        metrics['tpr_at_shadow_threshold'] = float(low_predictions[labels == 1].mean())
        metrics['fpr_at_shadow_threshold'] = float(low_predictions[labels == 0].mean())
    return metrics, scores


def train_membership_classifier(shadow_attack_data, *, seed=42,
                                algorithm='gradient_boosting', feature_set='all',
                                split_record_ids=None):
    """Fit a nonlinear attack on shadow outputs only, then select its threshold.

    Fixed tree settings, no internal early stopping: this preserves the explicit
    record-group split instead of making an internal row-wise validation split.
    """
    feature_sets = {'all': ATTACK_FEATURES, 'probability_only': PROBABILITY_FEATURES,
                    'paper': PAPER_FEATURES}
    if feature_set not in feature_sets:
        raise ValueError('feature_set must be all, probability_only, or paper.')
    features = feature_sets[feature_set]
    if split_record_ids is None:
        splits = split_attack_data(shadow_attack_data, seed=seed, features=features)
    else:
        splits = {name: shadow_attack_data.loc[shadow_attack_data['record_id'].isin(ids)].copy()
                  for name, ids in split_record_ids.items()}
        if set(splits) != {'train', 'validation', 'test'}:
            raise ValueError('Expected train, validation, and test IDs.')
        seen = set()
        for name, part in splits.items():
            ids = set(part['record_id'])
            if seen & ids or set(part['membership'].unique()) != {0, 1}:
                raise ValueError(f'Overlapping IDs or missing membership class in {name}.')
            seen.update(ids)
        if sum(map(len, splits.values())) != len(shadow_attack_data):
            raise ValueError('Record assignments must cover every attack row exactly once.')
    if algorithm == 'gradient_boosting':
        classifier = HistGradientBoostingClassifier(
            max_iter=150, learning_rate=0.05, max_leaf_nodes=15,
            min_samples_leaf=50, l2_regularization=1.0,
            early_stopping=False, random_state=seed)
    elif algorithm == 'random_forest':
        classifier = RandomForestClassifier(n_estimators=300, max_depth=12,
            min_samples_leaf=20, max_features='sqrt', random_state=seed, n_jobs=2)
    elif algorithm == 'logistic_regression':
        classifier = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=seed))
    elif algorithm == 'mlp':
        classifier = make_pipeline(StandardScaler(), MLPClassifier(
            hidden_layer_sizes=(32, 16), max_iter=100, alpha=0.001,
            early_stopping=False, random_state=seed))
    else:
        raise ValueError('Unknown attack algorithm.')
    classifier.fit(splits['train'][features], splits['train']['membership'])
    validation_scores = classifier.predict_proba(splits['validation'][features])[:, 1]
    low_fpr_threshold = threshold_for_fpr(splits['validation']['membership'], validation_scores)
    fpr, tpr, thresholds = roc_curve(splits['validation']['membership'], validation_scores)
    accuracy_threshold = float(thresholds[np.argmax(tpr - fpr)])
    metrics = {name: evaluate_attack(classifier, part,
               threshold=accuracy_threshold, low_fpr_threshold=low_fpr_threshold)[0]
               for name, part in splits.items()}
    return {'classifier': classifier, 'splits': splits, 'metrics': metrics,
            'low_fpr_threshold': low_fpr_threshold, 'seed': seed,
            'accuracy_threshold': accuracy_threshold, 'algorithm': algorithm,
            'feature_set': feature_set}


def compare_membership_attacks(mixed_shadows, mlp_shadows, *, seed=42):
    """Compare pools, classifiers, and feature sets on fixed Adult record groups.

    Winner is selected solely by validation ROC-AUC; test metrics never select it.
    Reported accuracy uses a validation-selected balanced-accuracy threshold.
    """
    def pool(results):
        return pd.concat([result['attack_data'] for result in results.values()], ignore_index=True)
    mixed, mlps = pool(mixed_shadows), pool(mlp_shadows)
    combined = pd.concat([mixed, mlps], ignore_index=True)
    groups = {name: part['record_id'].unique()
              for name, part in split_attack_data(combined, seed=seed).items()}
    pools = {'mixed': mixed, 'five_mlps': mlps}
    pools.update({f'only_{name}': result['attack_data'] for name, result in mixed_shadows.items()})
    results, rows = {}, []
    for pool_name, frame in pools.items():
        for algorithm in ['gradient_boosting', 'logistic_regression', 'mlp']:
            for feature_set in ['all', 'probability_only']:
                name = f'{pool_name}/{algorithm}/{feature_set}'
                print(f'Fitting attack: {name}', flush=True)
                result = train_membership_classifier(frame, seed=seed, algorithm=algorithm,
                    feature_set=feature_set, split_record_ids=groups)
                results[name] = result
                rows.append({'candidate': name,
                    **{f'validation_{key}': value for key, value in result['metrics']['validation'].items()}})
    summary = pd.DataFrame(rows).set_index('candidate').sort_values(
        'validation_roc_auc', ascending=False, kind='stable')
    best_name = summary.index[0]
    return {'results': results, 'validation_summary': summary,
            'best_name': best_name, 'best_result': results[best_name]}


def tune_random_forest_attack(shadow_attack_data, *, seed=42, parameter_grid=None):
    """Select an RF by grouped Adult validation AUC, then evaluate once.

    Uses the four paper features. Neither shadow test nor target data selects
    hyperparameters or thresholds. Defaults evaluate eight RF configurations.
    """
    splits = split_attack_data(shadow_attack_data, seed=seed, features=PAPER_FEATURES)
    if parameter_grid is None:
        parameter_grid = {'max_depth': [12, None], 'min_samples_leaf': [5, 20],
                          'max_features': ['sqrt', None]}
    best_auc, best_classifier, best_parameters = -np.inf, None, None
    rows = []
    configurations = list(ParameterGrid(parameter_grid))
    if not configurations:
        raise ValueError('At least one RF configuration is required.')
    for index, parameters in enumerate(configurations):
        print(f'RF validation search {index + 1}/{len(configurations)}: {parameters}', flush=True)
        settings = dict(n_estimators=300, random_state=seed, n_jobs=2)
        classifier = RandomForestClassifier(**(settings | parameters))
        classifier.fit(splits['train'][PAPER_FEATURES], splits['train']['membership'])
        scores = classifier.predict_proba(splits['validation'][PAPER_FEATURES])[:, 1]
        auc = float(roc_auc_score(splits['validation']['membership'], scores))
        rows.append({**parameters, 'validation_roc_auc': auc})
        if auc > best_auc:
            best_auc, best_classifier, best_parameters = auc, classifier, dict(parameters)
    scores = best_classifier.predict_proba(splits['validation'][PAPER_FEATURES])[:, 1]
    fpr, tpr, thresholds = roc_curve(splits['validation']['membership'], scores)
    accuracy_threshold = float(thresholds[np.argmax(tpr - fpr)])
    low_threshold = threshold_for_fpr(splits['validation']['membership'], scores)
    metrics = {name: evaluate_attack(best_classifier, part, threshold=accuracy_threshold,
               low_fpr_threshold=low_threshold)[0] for name, part in splits.items()}
    return {'classifier': best_classifier, 'splits': splits, 'metrics': metrics,
            'low_fpr_threshold': low_threshold, 'accuracy_threshold': accuracy_threshold,
            'seed': seed, 'algorithm': 'random_forest', 'feature_set': 'paper',
            'best_parameters': best_parameters,
            'validation_search': pd.DataFrame(rows).sort_values('validation_roc_auc',
                ascending=False, kind='stable')}


def build_target_attack_data(model, arrays):
    """Query ACS MIA partitions; income labels are assumed known to the attacker."""
    members, nonmembers = arrays['mia_members'], arrays['mia_nonmembers']
    if set(members['record_ids']) & set(nonmembers['record_ids']):
        raise ValueError('Target member and non-member record IDs overlap.')
    rows = []
    for values, membership in [(members, 1), (nonmembers, 0)]:
        probabilities = predict_probabilities(model, values['X'])
        frame = membership_features(probabilities, values['y'])
        frame['record_id'] = values['record_ids']
        frame['income_label'] = values['y']
        frame['membership'] = membership
        rows.append(frame)
    return pd.concat(rows, ignore_index=True)
