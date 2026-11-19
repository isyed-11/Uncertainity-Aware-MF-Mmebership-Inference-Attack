"""Shared-schema ACS helpers for the approved four-year baseline study.

No data loading, fitting, querying, logging or checkpointing occurs on import.
The historical 48-feature helpers and their saved-model signatures are unchanged.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score

from .expanded_data import (
    ADDITIONAL_GROUPS, EXPANDED_FEATURES, NUMERIC, build_expanded_preprocessor,
)
from .shadow_models import (
    LABEL, PAPER_SHADOW_ALGORITHMS, MLP_SHADOW_CONFIG, build_shadow_estimator, membership_features,
    split_shadow_data,
)
from .training import train_target_model, predict_probabilities


# Explicitly approved common schema: VPS is absent from the 2024 source file.
BASELINE_FEATURES = [name for name in EXPANDED_FEATURES if name != 'VPS']


def baseline_raw_path(year, data_root='data/raw/folktables'):
    filename = 'ss14pca.csv' if year == 2014 else 'psam_p06.csv'
    return Path(data_root) / str(year) / '1-Year' / filename


def load_baseline_year(year, *, data_root='data/raw/folktables'):
    """Use original eligibility, nominal >$50k labels and year-qualified row IDs."""
    raw = pd.read_csv(baseline_raw_path(year, data_root),
        usecols=BASELINE_FEATURES + ['PINCP', 'PWGTP'],
        low_memory=False, skipinitialspace=True)
    for column in raw:
        raw[column] = pd.to_numeric(raw[column], errors='raise')
    eligible = ((raw.AGEP > 16) & (raw.PINCP > 100)
                & (raw.WKHP > 0) & (raw.PWGTP >= 1))
    frame = raw.loc[eligible, BASELINE_FEATURES].astype('float32').copy()
    frame[LABEL] = (raw.loc[eligible, 'PINCP'] > 50_000).astype('int64')
    frame.index = pd.Index([f'acs_{year}_{row}' for row in frame.index], name='record_id')
    frame.attrs['income_adjustment'] = dict(enabled=False,
        label='PINCP > 50000', adjustment='None')
    if frame.empty or set(frame[LABEL].unique()) != {0, 1}:
        raise ValueError(f'ACS {year} must contain both eligible income classes.')
    if np.isinf(frame[BASELINE_FEATURES].to_numpy()).any():
        raise ValueError(f'ACS {year} contains infinite predictor values.')
    return frame


def build_baseline_preprocessor():
    """Retain the existing transformations, with VPS excluded before fitting."""
    preprocessor = build_expanded_preprocessor()
    preprocessor.set_params(transformers=[
        (name, transformer, [column for column in columns if column in BASELINE_FEATURES])
        for name, transformer, columns in preprocessor.transformers
    ])
    return preprocessor


def preprocess_baseline_partitions(partitions):
    preprocessor = build_baseline_preprocessor()
    preprocessor.fit(partitions['train'][BASELINE_FEATURES])
    arrays = {name: dict(
        X=sparse.csr_matrix(preprocessor.transform(frame[BASELINE_FEATURES]), dtype=np.float32),
        y=frame[LABEL].to_numpy(dtype=np.int64, copy=True),
        record_ids=frame.index.to_numpy(copy=True),
    ) for name, frame in partitions.items()}
    for name, values in arrays.items():
        if not np.isfinite(values['X'].data).all():
            raise ValueError(f'Non-finite transformed values in {name}.')
    return preprocessor, arrays


def baseline_feature_inventory(training):
    groups = {column: name for name, columns in ADDITIONAL_GROUPS.items() for column in columns}
    return pd.DataFrame([dict(feature=column,
        group=groups.get(column, 'original_ACSIncome'),
        type='numeric' if column in NUMERIC else 'categorical',
        unique_nonmissing=int(training[column].nunique(dropna=True)),
        missing_fraction=float(training[column].isna().mean()),
    ) for column in BASELINE_FEATURES]).set_index('feature')


def train_baseline_shadow(auxiliary, algorithm, *, seed=42, parameters=None, partitions=None):
    """Tree or generic MLP shadows with 40/20/40 splits and 47 common inputs.

    MLP epochs are selected using shadow validation loss only. Nonmember rows
    never participate in fitting or model selection. Existing trees are unchanged.
    """
    if algorithm not in (*PAPER_SHADOW_ALGORITHMS, 'mlp'):
        raise ValueError(f'Unsupported baseline shadow algorithm: {algorithm}')
    if parameters is not None and algorithm != 'mlp':
        raise ValueError('Custom parameters are supported only for the added MLP shadows.')
    if partitions is None:
        partitions = split_shadow_data(auxiliary, seed=seed)
    else:
        if algorithm != 'mlp':
            raise ValueError('Explicit splits are supported only for the same-year MLP study.')
        if set(partitions) not in ({'train', 'validation', 'nonmembers'}, {'train', 'nonmembers'}):
            raise ValueError('Expected train/nonmembers and an optional validation partition.')
        seen = set()
        for name, part in partitions.items():
            ids = set(part.index)
            if (part.empty or not part.index.is_unique or seen & ids
                    or not ids <= set(auxiliary.index) or set(part[LABEL].unique()) != {0, 1}):
                raise ValueError(f'Invalid or overlapping same-year {name} records.')
            pd.testing.assert_frame_equal(part, auxiliary.loc[part.index])
            seen.update(ids)
        if seen != set(auxiliary.index):
            raise ValueError('Same-year shadow partitions must cover the entire pool.')
    preprocessor = build_baseline_preprocessor()
    preprocessor.fit(partitions['train'][BASELINE_FEATURES])
    arrays = {name: dict(
        X=preprocessor.transform(frame[BASELINE_FEATURES]).astype(np.float32),
        y=frame[LABEL].to_numpy(dtype=np.int64, copy=True),
    ) for name, frame in partitions.items()}
    history = []
    if algorithm == 'mlp':
        config = dict(MLP_SHADOW_CONFIG, seed=seed)
        config.update(parameters or {})
        model, history, metadata = train_target_model(
            arrays['train']['X'], arrays['train']['y'],
            arrays['validation']['X'] if 'validation' in arrays else None,
            arrays['validation']['y'] if 'validation' in arrays else None, **config)
    else:
        model = build_shadow_estimator(algorithm, seed=seed)
        model.fit(arrays['train']['X'], arrays['train']['y'])
        metadata = model.get_params()
    metrics, attack_rows = {}, []
    order = [list(model.classes_).index(label) for label in [0, 1]] if algorithm != 'mlp' else None
    for name, values in arrays.items():
        probabilities = (predict_probabilities(model, values['X']) if algorithm == 'mlp'
                         else model.predict_proba(values['X'])[:, order])
        metrics[name] = dict(records=len(values['y']),
            accuracy=float(accuracy_score(values['y'], probabilities.argmax(axis=1))),
            roc_auc=float(roc_auc_score(values['y'], probabilities[:, 1])),
            log_loss=float(log_loss(values['y'], probabilities, labels=[0, 1])))
        if name != 'validation':
            rows = membership_features(probabilities, values['y'])
            rows['record_id'] = partitions[name].index.to_numpy()
            rows['income_label'] = values['y']
            rows['membership'] = int(name == 'train')
            rows['shadow_model'] = algorithm
            attack_rows.append(rows)
    return dict(algorithm=algorithm, seed=seed, schema='expanded_acs',
        model=model, preprocessor=preprocessor, metrics=metrics,
        partition_ids={name: frame.index.to_numpy(copy=True) for name, frame in partitions.items()},
        attack_data=pd.concat(attack_rows, ignore_index=True),
        history=history, metadata=metadata)
