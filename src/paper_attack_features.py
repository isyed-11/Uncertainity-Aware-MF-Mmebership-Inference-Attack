"""Paper-inspired black-box features with an explicit tabular walk protocol.

See data/paper_attack_features.md for formulas, query budgets, and differences
from the paper. No experiments run on import. Only probability queries are used.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .data import FEATURES as ACS_FEATURES
from .shadow_models import FEATURES as ADULT_FEATURES, membership_features
from .training import predict_probabilities

WALK_DEFAULTS = {'max_steps': 20, 'repeats': 2, 'categorical_change_probability': 0.05,
                 'batch_size': 512, 'seed': 42}
# Public domain choices, not statistics inferred from hidden target training.
# name: (step standard deviation in native units, lower bound, upper bound)
NUMERIC_PROTOCOL = {
    'AGEP': (1.0, 17, 99), 'WKHP': (2.0, 1, 99),
    'age': (1.0, 17, 90), 'hours_per_week': (2.0, 1, 99),
    'education_num': (1.0, 1, 16),
    'capital_gain': (100.0, 0, None), 'capital_loss': (100.0, 0, None),
}


def make_probability_oracle(model, preprocessor, *, schema):
    """Simulate an API accepting native fields and returning class-ordered probs.

    The preprocessing stays inside this oracle. Feature extraction receives a
    query function, not access to target weights, gradients, or scaling statistics.
    """
    if schema not in ('adult', 'acs'):
        raise ValueError('schema must be adult or acs.')
    columns = ADULT_FEATURES if schema == 'adult' else ACS_FEATURES

    def query(frame):
        X = np.asarray(preprocessor.transform(frame[columns]), dtype=np.float32)
        if hasattr(model, 'predict_proba'):
            order = [list(model.classes_).index(label) for label in [0, 1]]
            return model.predict_proba(X)[:, order]
        return predict_probabilities(model, X)
    return query


def extract_paper_features(query, frame, income_labels, *, config=None):
    """Return four attack features and query/censoring diagnostics.

    Blind walk measures first change from ORIGINAL PREDICTION, not true label.
    Use fixed Gaussian numeric directions with integer rounding/domain clipping.
    Each categorical column switches once to a sampled observed category at a
    geometrically sampled step. Categories come from candidate inputs only;
    membership labels are neither accepted nor used here. No fractional one-hot
    vectors or fractional occupation codes are queried.
    """
    settings = WALK_DEFAULTS | (config or {})
    steps, repeats = settings['max_steps'], settings['repeats']
    batch_size = settings['batch_size']
    change_probability = settings['categorical_change_probability']
    if any(not isinstance(value, int) or value < 1 for value in [steps, repeats, batch_size]):
        raise ValueError('max_steps, repeats, batch_size must be positive integers.')
    if not 0 < change_probability <= 1:
        raise ValueError('categorical_change_probability must be in (0,1].')
    labels = np.asarray(income_labels)
    if len(frame) != len(labels) or frame.empty:
        raise ValueError('Expected nonempty, row-aligned inputs and income labels.')
    rng = np.random.default_rng(settings['seed'])
    numeric = [column for column in frame if column in NUMERIC_PROTOCOL]
    categorical = [column for column in frame if column not in numeric]
    domains = {column: frame[column].dropna().unique() for column in categorical}
    features, censored_values, query_count = [], [], 0
    for start in range(0, len(frame), batch_size):
        base = frame.iloc[start:start + batch_size].reset_index(drop=True).copy()
        batch_labels = labels[start:start + len(base)]
        probabilities = np.asarray(query(base), dtype=float)
        basic = membership_features(probabilities, batch_labels)
        original_prediction = probabilities.argmax(axis=1)
        true_p = probabilities[np.arange(len(base)), batch_labels.astype(int)]
        other_p = probabilities[np.arange(len(base)), 1 - batch_labels.astype(int)]
        logits = np.log(np.clip(true_p, 1e-12, 1)) - np.log(np.clip(other_p, 1e-12, 1))
        query_count += len(base)
        times = np.full((repeats, len(base)), steps + 1, dtype=float)
        for repeat in range(repeats):
            directions = {column: rng.normal(0, NUMERIC_PROTOCOL[column][0], len(base))
                          for column in numeric}
            destinations = {column: rng.choice(domain, len(base)) for column, domain in domains.items()
                            if len(domain)}
            switch_steps = {column: rng.geometric(change_probability, len(base))
                            for column in destinations}
            active = np.arange(len(base))
            for step in range(1, steps + 1):
                if not len(active):
                    break
                candidate = base.iloc[active].copy()
                for column in numeric:
                    _, lower, upper = NUMERIC_PROTOCOL[column]
                    values = np.rint(base[column].to_numpy(dtype=float)[active]
                                     + directions[column][active] * step)
                    candidate[column] = np.clip(values, lower, np.inf if upper is None else upper)
                for column, destination in destinations.items():
                    switch = switch_steps[column][active] <= step
                    candidate.loc[candidate.index[switch], column] = destination[active][switch]
                changed = np.asarray(query(candidate)).argmax(axis=1) != original_prediction[active]
                query_count += len(active)
                times[repeat, active[changed]] = step
                active = active[~changed]
        features.append(pd.DataFrame({
            'class_scaled_logit': logits, 'blind_walk': times.mean(axis=0),
            'entropy': basic['entropy'].to_numpy(), 'cross_entropy': basic['loss'].to_numpy(),
        }))
        censored_values.extend((times > steps).mean(axis=0).tolist())
        print(f'Attack feature extraction: {min(start + batch_size, len(frame)):,}/{len(frame):,}', flush=True)
    diagnostics = {'record_queries': query_count,
                   'max_record_queries': len(frame) * (1 + repeats * steps),
                   'walk_censored_fraction': float(np.mean(censored_values)), 'config': settings}
    return pd.concat(features, ignore_index=True), diagnostics


def collect_shadow_paper_features(results, adult, *, per_class=2000, config=None, seed=42):
    """Query existing shadows; sample balanced known members/nonmembers per model.

    Adult row IDs come from load_adult_data. No shadows are retrained here.
    per_class=None uses all available rows and can be expensive.
    """
    frames, diagnostics = [], {}
    for name, result in results.items():
        selected = []
        for partition, membership in [('train', 1), ('nonmembers', 0)]:
            ids = result['partition_ids'][partition]
            if per_class is not None:
                if not isinstance(per_class, int) or per_class < 2 or per_class > len(ids):
                    raise ValueError('per_class must be between 2 and the available partition size.')
                if per_class < len(ids):
                    _, ids = train_test_split(ids, test_size=per_class, random_state=seed,
                        stratify=adult.loc[ids, 'income_gt_50k'])
            part = adult.loc[ids].copy()
            part['membership'] = membership
            selected.append(part)
        candidates = pd.concat(selected).sample(frac=1, random_state=seed)
        query = make_probability_oracle(result['model'], result['preprocessor'], schema='adult')
        features, audit = extract_paper_features(query, candidates[ADULT_FEATURES],
            candidates['income_gt_50k'].to_numpy(), config=config)
        features['record_id'] = candidates.index.to_numpy()
        features['membership'] = candidates['membership'].to_numpy()
        features['shadow_model'] = name
        frames.append(features)
        diagnostics[name] = audit
    return pd.concat(frames, ignore_index=True), diagnostics


def collect_target_paper_features(model, preprocessor, partitions, *, config=None, seed=42):
    """Query the actual target MIA partitions, retaining all candidate records."""
    if set(partitions['mia_members'].index) & set(partitions['mia_nonmembers'].index):
        raise ValueError('Target members and nonmembers overlap.')
    candidates = pd.concat([
        partitions['mia_members'].assign(membership=1),
        partitions['mia_nonmembers'].assign(membership=0),
    ]).sample(frac=1, random_state=seed)
    query = make_probability_oracle(model, preprocessor, schema='acs')
    features, audit = extract_paper_features(query, candidates[ACS_FEATURES],
        candidates['income_gt_50k'].to_numpy(), config=config)
    features['record_id'] = candidates.index.to_numpy()
    features['membership'] = candidates['membership'].to_numpy()
    return features, audit
