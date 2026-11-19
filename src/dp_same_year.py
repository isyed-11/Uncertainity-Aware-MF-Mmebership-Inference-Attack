"""Plotting support for the disjoint same-year DP study; no import-time work."""
from pathlib import Path

import numpy as np

from .dp_experiment import categorical_budget_rows


def plot_low_fpr_budget(results, epsilon_budgets, *, output_dir, delta):
    """Separate evaluation ROC TPR from a frozen shadow-validation cutoff.

    The latter does not guarantee 1% FPR on targets; plot its measured target
    FPR alongside TPR. No scores, labels or thresholds are fitted here.
    """
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    ordered, labels = categorical_budget_rows(results, epsilon_budgets)
    control = results.loc[~results['private']]
    columns = ['mia_tpr_at_1pct_fpr_roc', 'mia_tpr_at_shadow_threshold', 'mia_fpr_at_shadow_threshold']
    if len(control) != 1:
        raise ValueError('Exactly one non-private control is required.')
    values = results[columns].to_numpy(dtype=float)
    if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
        raise ValueError('Low-FPR metrics must be finite probabilities.')
    if ordered['delta'].nunique() != 1 or not ordered['delta'].eq(delta).all():
        raise ValueError('All private rows must use the fixed declared delta.')

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    axes[0].plot(labels, ordered[columns[0]], marker='o', label='DP target')
    axes[0].axhline(float(control.iloc[0][columns[0]]), color='tab:orange',
                    linestyle='--', label='Non-private control (epsilon = infinity)')
    axes[0].axhline(.01, color='0.45', linestyle=':', label='Random-ranking reference (1%)')
    axes[0].set(title='Evaluation ROC: TPR at FPR <= 1%', ylabel='True positive rate')
    axes[1].plot(labels, ordered[columns[1]], marker='o', label='DP target TPR')
    axes[1].plot(labels, ordered[columns[2]], marker='s', label='DP target actual FPR')
    axes[1].axhline(float(control.iloc[0][columns[1]]), color='tab:orange',
                    linestyle='--', label='Non-private TPR')
    axes[1].axhline(float(control.iloc[0][columns[2]]), color='tab:green',
                    linestyle='--', label='Non-private actual FPR')
    axes[1].axhline(.01, color='0.45', linestyle=':', label='Auxiliary FPR target (1%)')
    axes[1].set(title='Frozen auxiliary-selected threshold', ylabel='Measured target rate')
    for ax in axes:
        ax.set(xlabel='Requested epsilon (equally spaced categories)', ylim=(0, 1))
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=1))
        ax.grid(axis='y', alpha=.25)
        ax.legend(fontsize=8)
    fig.suptitle(f'Membership leakage across privacy budgets; constant delta = {delta:g}')
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    for suffix in ['png', 'svg']:
        fig.savefig(destination / f'mia_low_fpr_vs_epsilon.{suffix}', dpi=200)
    return fig, axes
