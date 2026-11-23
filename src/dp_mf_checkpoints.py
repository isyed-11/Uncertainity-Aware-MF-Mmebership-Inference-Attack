"""Atomic, signature-checked internal research checkpoints; no work on import."""
import hashlib
from pathlib import Path

import joblib
import torch

from .uncertainty_checkpoints import atomic_save, load_verified


class CheckpointStore:
    def __init__(self, directory, signature):
        if not signature:
            raise ValueError('A data/settings signature is required for checkpoints.')
        self.directory = Path(directory)
        self.signature = signature

    def path(self, name):
        name = Path(name)
        if name.is_absolute() or '..' in name.parts:
            raise ValueError('Checkpoint names must stay inside their directory.')
        return self.directory / name

    def cached(self, name, compute):
        path = self.path(name).with_suffix('.joblib')
        signature = joblib.hash((self.signature, name))
        value = load_verified(path, signature)
        if value is None:
            value = compute()
            if value is None:
                raise ValueError('Checkpoint computations must return a value.')
            atomic_save(path, signature, value)
        return value


def run_fingerprint(settings, frames, source_paths):
    from .uncertainty_checkpoints import frame_signature
    data = {name: frame_signature(frame) for name, frame in frames.items()}
    code = {str(path): hashlib.sha256(Path(path).read_bytes()).hexdigest() for path in source_paths}
    return joblib.hash(dict(settings=settings, frames=data, code=code))


def capture_rng(sampling_generator, noise_generator=None):
    import random
    import numpy as np
    return dict(python=random.getstate(), numpy=np.random.get_state(),
        torch=torch.get_rng_state(), cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        sampling=sampling_generator.get_state(),
        noise=noise_generator.get_state() if noise_generator is not None else None)


def restore_rng(state, sampling_generator, noise_generator=None):
    import random
    import numpy as np
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])
    if state['cuda'] is not None:
        torch.cuda.set_rng_state_all(state['cuda'])
    sampling_generator.set_state(state['sampling'])
    if noise_generator is not None:
        noise_generator.set_state(state['noise'])
