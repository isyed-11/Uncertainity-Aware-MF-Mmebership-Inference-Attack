"""Loss-only membership diagnostic: lower cross-entropy indicates membership."""
import numpy as np
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, confusion_matrix,
                             roc_auc_score, roc_curve)
from .membership_attack import threshold_for_fpr


def _inputs(frame, loss_column):
    losses = frame[loss_column].to_numpy(dtype=float)
    labels = frame['membership'].to_numpy()
    if not np.isfinite(losses).all() or (losses < 0).any():
        raise ValueError('Losses must be finite and nonnegative.')
    if set(np.unique(labels)) != {0, 1}:
        raise ValueError('Expected both binary membership classes.')
    return labels.astype(int), -losses


def fit_loss_only_attack(validation, *, loss_column='cross_entropy', max_fpr=0.01):
    """Select cutoffs on auxiliary validation ONLY. No classifier is fitted.

    Scores are negative loss, not calibrated membership probabilities. Choose
    the balanced-accuracy cutoff by maximizing TPR-FPR on validation. Direction
    is fixed in advance: low loss means member, never reversed using target data.
    """
    labels, scores = _inputs(validation, loss_column)
    fpr, tpr, thresholds = roc_curve(labels, scores, drop_intermediate=False)
    threshold = float(thresholds[np.argmax(tpr - fpr)])
    low_fpr_threshold = threshold_for_fpr(labels, scores, max_fpr=max_fpr)
    return {'loss_column': loss_column, 'score_threshold': threshold,
            'loss_threshold': -threshold,
            'low_fpr_score_threshold': low_fpr_threshold,
            'low_fpr_loss_threshold': -low_fpr_threshold,
            'max_fpr': max_fpr}


def evaluate_loss_only_attack(attack, frame):
    """Evaluate frozen cutoffs; never optimize a target classification cutoff."""
    labels, scores = _inputs(frame, attack['loss_column'])
    predictions = scores >= attack['score_threshold']
    low_predictions = scores >= attack['low_fpr_score_threshold']
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    fpr, tpr, _ = roc_curve(labels, scores, drop_intermediate=False)
    metrics = {
        'records': len(labels), 'members': int(labels.sum()),
        'accuracy': float(accuracy_score(labels, predictions)),
        'balanced_accuracy': float(balanced_accuracy_score(labels, predictions)),
        'roc_auc': float(roc_auc_score(labels, scores)),
        'tpr_at_1pct_fpr_roc': float(tpr[fpr <= 0.01].max()),
        'loss_threshold': attack['loss_threshold'],
        'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp),
        'tpr_at_loss_threshold': float(predictions[labels == 1].mean()),
        'fpr_at_loss_threshold': float(predictions[labels == 0].mean()),
        'shadow_selected_low_fpr_loss_threshold': attack['low_fpr_loss_threshold'],
        'tpr_at_shadow_threshold': float(low_predictions[labels == 1].mean()),
        'fpr_at_shadow_threshold': float(low_predictions[labels == 0].mean()),
    }
    return metrics, scores
