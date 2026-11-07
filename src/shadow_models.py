"""Adult shadow models and output-based membership features. No import-time work."""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

from .training import train_target_model, predict_probabilities

LABEL = 'income_gt_50k'
# fnlwgt is a survey weight, not a personal characteristic; education_num
# represents education, so the redundant text education field is omitted.
NUMERIC = ['age', 'education_num', 'capital_gain', 'capital_loss', 'hours_per_week']
CATEGORICAL = ['workclass', 'marital_status', 'occupation', 'relationship',
               'race', 'sex', 'native_country']
FEATURES = NUMERIC + CATEGORICAL
ALGORITHMS = ('xgboost', 'random_forest', 'mlp', 'decision_tree', 'logistic_regression')


def load_adult_data(directory='data/processed'):
    """Pool the original Adult partitions as auxiliary data, preserving row IDs."""
    frames = []
    for partition in ['train', 'test']:
        frame = pd.read_csv(Path(directory) / f'adult_{partition}.csv')
        missing = set(FEATURES + [LABEL]) - set(frame.columns)
        if missing:
            raise ValueError(f'Missing Adult columns: {sorted(missing)}')
        frame.index = pd.Index([f'adult_{partition}_{i}' for i in range(len(frame))],
                               name='record_id')
        frames.append(frame[FEATURES + [LABEL]].copy())
    adult = pd.concat(frames)
    if adult[LABEL].isna().any() or set(adult[LABEL].unique()) != {0, 1}:
        raise ValueError('Adult must have both binary income labels and no missing labels.')
    adult[LABEL] = adult[LABEL].astype('int64')
    return adult


def split_shadow_data(adult, *, seed=42, validation_fraction=0.2):
    """40% members, 20% validation, 40% non-members by default.

    Within a shadow all partitions are disjoint. Across shadows they can overlap;
    a record can have different membership labels for different shadow models.
    """
    if not 0 < validation_fraction < 1:
        raise ValueError('validation_fraction must be between zero and one.')
    if not adult.index.is_unique:
        raise ValueError('Adult record IDs must be unique.')
    remaining, validation = train_test_split(
        adult, test_size=validation_fraction, stratify=adult[LABEL], random_state=seed)
    train, nonmembers = train_test_split(
        remaining, test_size=0.5, stratify=remaining[LABEL], random_state=seed + 1)
    return {'train': train.sort_index(), 'validation': validation.sort_index(),
            'nonmembers': nonmembers.sort_index()}


def build_adult_preprocessor():
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
    ], remainder='drop')


def build_shadow_estimator(algorithm, *, seed=42, parameters=None):
    """Construct a CPU estimator. MLP is handled by train_target_model instead."""
    parameters = parameters or {}
    if algorithm == 'xgboost':
        from xgboost import XGBClassifier
        defaults = dict(n_estimators=300, max_depth=5, learning_rate=0.05,
                        subsample=0.8, colsample_bytree=0.8, tree_method='hist',
                        objective='binary:logistic', eval_metric='logloss',
                        random_state=seed, n_jobs=2)
        cls = XGBClassifier
    elif algorithm == 'random_forest':
        defaults = dict(n_estimators=200, min_samples_leaf=2, random_state=seed, n_jobs=2)
        cls = RandomForestClassifier
    elif algorithm == 'decision_tree':
        defaults = dict(max_depth=15, min_samples_leaf=5, random_state=seed)
        cls = DecisionTreeClassifier
    elif algorithm == 'logistic_regression':
        defaults = dict(C=1.0, max_iter=1000, random_state=seed)
        cls = LogisticRegression
    else:
        raise ValueError(f'Unsupported estimator: {algorithm}')
    return cls(**(defaults | parameters))


def membership_features(probabilities, income_labels):
    """Common output features for both Adult shadows and the ACS target.

    Assumes the adversary knows the candidate's true income class. Labels here
    are income labels; membership labels are assigned separately by the caller.
    """
    probabilities = np.asarray(probabilities, dtype=float)
    labels = np.asarray(income_labels, dtype=int)
    if probabilities.shape != (len(labels), 2) or not np.isin(labels, [0, 1]).all():
        raise ValueError('Expected binary labels and an N x 2 probability matrix.')
    if not np.isfinite(probabilities).all() or (probabilities < 0).any() or (probabilities > 1).any():
        raise ValueError('Invalid prediction probabilities.')
    if not np.allclose(probabilities.sum(axis=1), 1, atol=1e-5):
        raise ValueError('Prediction probabilities must sum to one.')
    clipped = np.clip(probabilities, 1e-12, 1)
    true_probability = probabilities[np.arange(len(labels)), labels]
    return pd.DataFrame({
        'p_income_le_50k': probabilities[:, 0],
        'p_income_gt_50k': probabilities[:, 1],
        'true_class_probability': true_probability,
        'confidence': probabilities.max(axis=1),
        'entropy': -(probabilities * np.log(clipped)).sum(axis=1),
        'loss': -np.log(np.clip(true_probability, 1e-12, 1)),
        'correct': (probabilities.argmax(axis=1) == labels).astype(int),
    })


def train_shadow_model(adult, algorithm, *, seed=42, parameters=None):
    """Fit one shadow; return model, preprocessor, IDs, metrics, and attack rows.

    Trees and logistic regression use fixed settings; MLP selects its epoch using
    validation loss. Non-members never participate in fitting or model selection.
    """
    if algorithm not in ALGORITHMS:
        raise ValueError(f'Choose an algorithm from {ALGORITHMS}')
    partitions = split_shadow_data(adult, seed=seed)
    preprocessor = build_adult_preprocessor()
    preprocessor.fit(partitions['train'][FEATURES])
    arrays = {name: {
        'X': np.asarray(preprocessor.transform(frame[FEATURES]), dtype=np.float32),
        'y': frame[LABEL].to_numpy(dtype=np.int64, copy=True),
    } for name, frame in partitions.items()}
    history, metadata = [], {}
    if algorithm == 'mlp':
        config = dict(hidden_dims=(128, 64), dropout=0.0, epochs=50,
                      batch_size=256, patience=5, seed=seed, device='cpu')
        config.update(parameters or {})
        model, history, metadata = train_target_model(
            arrays['train']['X'], arrays['train']['y'],
            arrays['validation']['X'], arrays['validation']['y'], **config)
    else:
        model = build_shadow_estimator(algorithm, seed=seed, parameters=parameters)
        model.fit(arrays['train']['X'], arrays['train']['y'])
        metadata = model.get_params()
    metrics, attack_rows = {}, []
    for name, values in arrays.items():
        if algorithm == 'mlp':
            probabilities = predict_probabilities(model, values['X'])
        else:
            # Make output order explicit: class 0, then class 1.
            probabilities = model.predict_proba(values['X'])[:,
                [list(model.classes_).index(label) for label in [0, 1]]]
        metrics[name] = {
            'records': len(values['y']),
            'accuracy': float(accuracy_score(values['y'], probabilities.argmax(axis=1))),
            'roc_auc': float(roc_auc_score(values['y'], probabilities[:, 1])),
            'log_loss': float(log_loss(values['y'], probabilities, labels=[0, 1])),
        }
        if name != 'validation':
            rows = membership_features(probabilities, values['y'])
            rows['record_id'] = partitions[name].index.to_numpy()
            rows['income_label'] = values['y']
            rows['membership'] = int(name == 'train')
            rows['shadow_model'] = algorithm
            attack_rows.append(rows)
    return {'algorithm': algorithm, 'seed': seed, 'model': model,
            'preprocessor': preprocessor, 'metrics': metrics,
            'partition_ids': {name: frame.index.to_numpy(copy=True)
                              for name, frame in partitions.items()},
            'attack_data': pd.concat(attack_rows, ignore_index=True),
            'history': history, 'metadata': metadata}


def train_shadow_models(adult, *, seed=42, model_parameters=None):
    """Train the five algorithms sequentially with different deterministic splits."""
    results = {}
    model_parameters = model_parameters or {}
    for index, algorithm in enumerate(ALGORITHMS):
        print(f'Training shadow {index + 1}/{len(ALGORITHMS)}: {algorithm}', flush=True)
        results[algorithm] = train_shadow_model(
            adult, algorithm, seed=seed + 10 * index,
            parameters=model_parameters.get(algorithm))
    return results


def summarize_shadow_models(results):
    return pd.DataFrame([
        {'shadow_model': name, 'partition': partition, **metrics}
        for name, result in results.items()
        for partition, metrics in result['metrics'].items()
    ]).set_index(['shadow_model', 'partition'])


def train_mlp_shadows(adult, *, count=5, seed=42, parameters=None):
    """Train MLP shadows with separate seeded splits and weight initialization."""
    if not isinstance(count, int) or count < 1:
        raise ValueError('count must be a positive integer.')
    results = {}
    for index in range(count):
        name = f'mlp_seed_{seed + 10 * index}'
        print(f'Training {name} ({index + 1}/{count})', flush=True)
        result = train_shadow_model(adult, 'mlp', seed=seed + 10 * index,
                                    parameters=parameters)
        result['attack_data']['shadow_model'] = name
        results[name] = result
    return results
