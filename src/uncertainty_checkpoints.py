"""Validated, atomic checkpoints for unchanged uncertainty-MIA computations.

No fitting or target queries occur on import. Older unverified artifacts are
left untouched; this module writes to a separate resumable directory.
"""
import hashlib
import inspect
import os
from pathlib import Path
from uuid import uuid4

import joblib
import numpy as np
import pandas as pd
import sklearn

from .uncertainty_mf_mia import ConditionalDonorChain
from .expanded_data import EXPANDED_FEATURES


def frame_signature(frame):
    digest = hashlib.sha256()
    digest.update(repr([(str(c), str(dtype)) for c, dtype in frame.dtypes.items()]).encode())
    digest.update(pd.util.hash_pandas_object(frame, index=True).to_numpy().tobytes())
    return digest.hexdigest()


def model_context_signature(model, preprocessor):
    """Fingerprint the actual fitted oracle, including torch weights if present."""
    if hasattr(model, 'state_dict'):
        state = {name: value.detach().cpu().numpy() for name, value in model.state_dict().items()}
        fitted = (type(model).__module__, type(model).__qualname__, repr(model), state)
    else:
        fitted = model
    return joblib.hash((fitted, preprocessor))


def atomic_save(path, signature, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'{path.name}.{uuid4().hex}.part')
    try:
        joblib.dump(dict(signature=signature, value=value), temporary)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def load_verified(path, signature):
    path = Path(path)
    if not path.exists():
        return None
    try:
        saved = joblib.load(path)
    except Exception as error:
        raise ValueError(f'Unreadable checkpoint: {path}. It was not overwritten.') from error
    if not isinstance(saved, dict) or saved.get('signature') != signature or 'value' not in saved:
        raise ValueError(f'Checkpoint settings/data mismatch: {path}. Use a different checkpoint directory.')
    return saved['value']


def cached_reconstructor(training, hidden, *, seed, directory, n_estimators=100):
    """Save a fitted model immediately, before starting its reconstruction audit."""
    signature = joblib.hash(dict(data=frame_signature(training[EXPANDED_FEATURES]),
        hidden=list(hidden), seed=seed, trees=n_estimators,
        code=inspect.getsource(ConditionalDonorChain),
        versions=(sklearn.__version__, np.__version__, pd.__version__)))
    path = Path(directory) / ('+'.join(hidden) + '_reconstructor.joblib')
    model = load_verified(path, signature)
    if model is None:
        print(f'  No verified checkpoint: fitting {"+".join(hidden)}', flush=True)
        model = ConditionalDonorChain(hidden, seed=seed, n_estimators=n_estimators).fit(training)
        model.checkpoint_signature_ = signature
        atomic_save(path, signature, model)
        print('  Fitted reconstruction checkpoint saved before audit.', flush=True)
    else:
        print(f'  Loaded verified reconstruction: {"+".join(hidden)}', flush=True)
    return model


def cached_audit_draws(model, frame, *, k, seed, directory):
    """Resume audit draws with identical seeds and no target-model queries."""
    if not hasattr(model, 'checkpoint_signature_'):
        raise ValueError('Use a verified reconstruction checkpoint for resumable draws.')
    base = dict(model=model.checkpoint_signature_, frame=frame_signature(frame.drop(columns=model.hidden)),
                sampling_code=inspect.getsource(ConditionalDonorChain))
    signatures = [joblib.hash(dict(base, seed=seed + draw)) for draw in range(k)]
    paths = [Path(directory) / f'audit_draw_{draw + 1:03d}.joblib' for draw in range(k)]
    cached = [load_verified(path, signature) for path, signature in zip(paths, signatures)]
    missing = [draw for draw, value in enumerate(cached) if value is None]
    generated = model.iter_samples(frame, seeds=[seed + draw for draw in missing])
    for draw in range(k):
        if cached[draw] is not None:
            print(f'  Audit draw {draw + 1}/{k}: loaded', flush=True)
            yield cached[draw]
        else:
            print(f'  Audit draw {draw + 1}/{k}: sampling', flush=True)
            sample = next(generated)[model.hidden]
            atomic_save(paths[draw], signatures[draw], sample)
            yield sample
