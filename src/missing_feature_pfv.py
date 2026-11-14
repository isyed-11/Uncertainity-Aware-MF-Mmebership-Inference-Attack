"""Single-draw, auxiliary-marginal PFV completion; no experiments on import."""
import numpy as np


def complete_pfv(frame, auxiliary, hidden, *, seed=42):
    """Replace whole native fields using ACS 2014 empirical nonmissing values.

    Hidden candidate values and income/membership labels are never inspected.
    This is unconditional single imputation, not uncertainty-aware inference.
    """
    hidden = list(hidden)
    if not hidden or len(set(hidden)) != len(hidden):
        raise ValueError('Provide distinct hidden predictor fields.')
    if set(hidden) - set(frame) or set(hidden) - set(auxiliary):
        raise ValueError('Hidden fields must occur in both schemas.')
    if set(hidden) & {'income_gt_50k', 'membership', 'record_id', 'PINCP'}:
        raise ValueError('Only predictor fields may be reconstructed.')
    # Drop the hidden values before constructing any completed queries.
    completed = frame.drop(columns=hidden).copy()
    rng = np.random.default_rng(seed)
    for column in hidden:
        pool = auxiliary[column].dropna().to_numpy()
        if not len(pool):
            raise ValueError(f'No observed auxiliary values for {column}.')
        completed[column] = rng.choice(pool, size=len(frame), replace=True)
    return completed.loc[:, frame.columns]
