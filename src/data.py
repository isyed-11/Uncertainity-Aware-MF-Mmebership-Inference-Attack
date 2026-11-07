"""Load and partition ACSIncome; fit preprocessing on training records only."""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

FEATURES = ['AGEP', 'COW', 'SCHL', 'MAR', 'OCCP', 'POBP', 'RELP', 'WKHP', 'SEX', 'RAC1P']
NUMERIC = ['AGEP', 'WKHP']
CATEGORICAL = [column for column in FEATURES if column not in NUMERIC]
TARGET = 'income_gt_50k'


def load_acs_income(path='data/processed/acs_income_ca_2018.csv'):
    """Load the exported ACSIncome CSV. Index is its zero-based source row ID."""
    frame = pd.read_csv(Path(path))
    missing = set(FEATURES + [TARGET]) - set(frame.columns)
    if missing:
        raise ValueError(f'Missing columns: {sorted(missing)}')
    frame = frame[FEATURES + [TARGET]].copy()
    if frame.empty or frame[TARGET].isna().any() or set(frame[TARGET].unique()) != {0, 1}:
        raise ValueError('Expected nonempty data with both binary income labels.')
    if not all(pd.api.types.is_numeric_dtype(frame[column]) for column in FEATURES):
        raise ValueError('ACS feature columns must contain numeric Census codes.')
    if np.isinf(frame[FEATURES].to_numpy(dtype=float)).any():
        raise ValueError('Features contain infinite values.')
    frame[TARGET] = frame[TARGET].astype('int64')
    frame.index.name = 'record_id'
    return frame


def split_acs_data(frame, *, validation_size=10_000, test_size=10_000,
                   mia_nonmember_size=10_000, mia_member_size=10_000, seed=42):
    """Make disjoint stratified partitions and select MIA members within training.

    Record IDs identify CSV rows, not unique people. Identical feature vectors
    can occur naturally; they are not removed or treated as person identifiers.
    MIA members remain in training; MIA non-members never enter training.
    """
    sizes = [validation_size, test_size, mia_nonmember_size, mia_member_size]
    if any(not isinstance(size, int) or size <= 0 for size in sizes):
        raise ValueError('Partition sizes must be positive integers.')
    if not frame.index.is_unique:
        raise ValueError('Source record IDs must be unique.')
    train_size = len(frame) - sum(sizes[:3])
    if train_size < mia_member_size:
        raise ValueError('Not enough training records for the requested MIA members.')
    remaining = frame
    partitions = {}
    for offset, (name, size) in enumerate([
        ('validation', validation_size), ('test', test_size),
        ('mia_nonmembers', mia_nonmember_size),
    ]):
        remaining, partitions[name] = train_test_split(
            remaining, test_size=size, random_state=seed + offset,
            stratify=remaining[TARGET],
        )
    partitions['train'] = remaining
    if mia_member_size == len(remaining):
        partitions['mia_members'] = remaining.copy()
    else:
        _, partitions['mia_members'] = train_test_split(
            remaining, test_size=mia_member_size, random_state=seed + 3,
            stratify=remaining[TARGET],
        )
    return {name: part.sort_index().copy() for name, part in partitions.items()}


def subset_target_training(partitions, *, train_size=20_000, seed=42):
    """Reduce training only, preserving baseline MIA members and held-out IDs.

    Retain every existing MIA member; sample additional records stratified by
    income from the remaining training pool. Unselected training records are
    unused, not moved into any evaluation set. Fit a fresh preprocessor afterward.
    """
    train = partitions['train']
    members = partitions['mia_members']
    if not isinstance(train_size, int) or not len(members) <= train_size <= len(train):
        raise ValueError('train_size must be between the MIA member count and training count.')
    if not set(members.index).issubset(train.index):
        raise ValueError('Every designated MIA member must belong to training.')
    for name in ['validation', 'test', 'mia_nonmembers']:
        if set(partitions[name].index) & set(train.index):
            raise ValueError(f'{name} overlaps target training.')
    extra_pool = train.drop(index=members.index)
    extra_size = train_size - len(members)
    if extra_size == 0:
        extra = extra_pool.iloc[:0]
    elif extra_size == len(extra_pool):
        extra = extra_pool
    else:
        _, extra = train_test_split(extra_pool, test_size=extra_size,
                                    stratify=extra_pool[TARGET], random_state=seed)
    result = {name: part.copy() for name, part in partitions.items()}
    result['train'] = pd.concat([train.loc[members.index], extra]).sort_index()
    return result


def build_preprocessor():
    """Treat Census codes as categories; unseen categories map to all-zero blocks."""
    return ColumnTransformer([
        ('numeric', Pipeline([
            ('impute', SimpleImputer(strategy='median', keep_empty_features=True)),
            ('scale', StandardScaler()),
        ]), NUMERIC),
        ('categorical', Pipeline([
            ('impute', SimpleImputer(strategy='most_frequent', keep_empty_features=True)),
            ('encode', OneHotEncoder(handle_unknown='ignore', sparse_output=False,
                                     dtype=np.float32)),
        ]), CATEGORICAL),
    ], remainder='drop', verbose_feature_names_out=True)


def preprocess_partitions(partitions):
    """Fit once on train; transform validation, test and MIA partitions unchanged."""
    preprocessor = build_preprocessor()
    preprocessor.fit(partitions['train'][FEATURES])
    arrays = {}
    for name, part in partitions.items():
        arrays[name] = {
            'X': np.asarray(preprocessor.transform(part[FEATURES]), dtype=np.float32),
            'y': part[TARGET].to_numpy(dtype=np.int64),
            'record_ids': part.index.to_numpy(copy=True),
        }
    return preprocessor, arrays


def transform_external_data(frame, preprocessor):
    """Transform external records without fitting; report unseen category rates.

    Coverage does not establish semantic equivalence of category codes across
    years. In particular, 2017 OCCP has not been mapped to 2018 OCCP here.
    """
    categorical = preprocessor.named_transformers_['categorical']
    imputed = categorical.named_steps['impute'].transform(frame[CATEGORICAL])
    encoder = categorical.named_steps['encode']
    coverage = []
    for index, (column, known) in enumerate(zip(CATEGORICAL, encoder.categories_)):
        unknown = ~np.isin(imputed[:, index], known)
        coverage.append({'feature': column, 'unseen_records': int(unknown.sum()),
                         'unseen_fraction': float(unknown.mean())})
    arrays = {
        'X': np.asarray(preprocessor.transform(frame[FEATURES]), dtype=np.float32),
        'y': frame[TARGET].to_numpy(dtype=np.int64, copy=True),
        'record_ids': frame.index.to_numpy(copy=True),
    }
    return arrays, pd.DataFrame(coverage).set_index('feature')


def save_partition_ids(partitions, output_dir):
    """Persist exact row assignments for reproducible membership evaluation."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, part in partitions.items():
        pd.DataFrame({'record_id': part.index, TARGET: part[TARGET].to_numpy()}).to_csv(
            output_dir / f'{name}_ids.csv', index=False)
