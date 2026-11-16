"""Conditional multiple imputation and held-out MF-MIA score aggregation.

Importing this module does not fit models, query models, or run experiments.
"""
import time
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .expanded_data import EXPANDED_FEATURES, NUMERIC
from .paper_attack_features import extract_paper_features
from .membership_attack import threshold_for_fpr


class ConditionalDonorChain:
    """Sample real 2014 values from random-forest leaves, in fixed field order.

    Each stage conditions on visible fields and preceding hidden fields. Random
    trees select donor neighborhoods; values are sampled from training donors
    in that leaf. This empirical predictive distribution is approximate and is
    not claimed to provide calibrated posterior uncertainty.
    """

    def __init__(self, hidden, *, seed=42, n_estimators=100):
        self.hidden = list(hidden)
        self.seed = seed
        self.n_estimators = n_estimators
        if (not self.hidden or len(set(self.hidden)) != len(self.hidden)
                or set(self.hidden) - {'MAR', 'WKHP', 'COW', 'SEX'}):
            raise ValueError('Use distinct supported stable-code mask fields.')

    def fit(self, auxiliary):
        visible = [c for c in EXPANDED_FEATURES if c not in self.hidden]
        self.stages = []
        for offset, field in enumerate(self.hidden):
            started = time.monotonic()
            print(f'  Reconstruction field {offset + 1}/{len(self.hidden)}: {field}; preparing donors', flush=True)
            columns = visible + self.hidden[:offset]
            train = auxiliary.loc[auxiliary[field].notna(), columns + [field]]
            if len(train) < 100:
                raise ValueError(f'Insufficient auxiliary reconstruction donors for {field}.')
            numeric = [c for c in columns if c in NUMERIC]
            categorical = [c for c in columns if c not in NUMERIC]
            transform = ColumnTransformer([
                ('numeric', SimpleImputer(strategy='median', keep_empty_features=True), numeric),
                ('categorical', Pipeline([
                    ('impute', SimpleImputer(strategy='constant', fill_value=-1, keep_empty_features=True)),
                    ('encode', OneHotEncoder(handle_unknown='ignore', sparse_output=True)),
                ]), categorical),
            ], sparse_threshold=1.0)
            x = sparse.csr_matrix(transform.fit_transform(train[columns]), dtype=np.float32)
            factory = RandomForestRegressor if field in NUMERIC else RandomForestClassifier
            model = factory(n_estimators=self.n_estimators, max_depth=12,
                min_samples_leaf=20, random_state=self.seed + offset, n_jobs=2)
            print(f'  Fitting {self.n_estimators} trees for {field} on {len(train):,} auxiliary records', flush=True)
            model.fit(x, train[field].to_numpy())
            print(f'  {field}: trees fitted; collecting leaf donors', flush=True)
            leaves = model.apply(x)
            # Losslessly encode values once per stage. The supported fields
            # have few distinct values, so leaf pools normally need one byte
            # per donor rather than the four bytes of ACS float32 values.
            # Every donor occurrence stays in the same leaf and row order;
            # repeated values retain exactly their empirical sampling weights.
            values, codes = np.unique(train[field].to_numpy(), return_inverse=True)
            codes = codes.astype(np.min_scalar_type(len(values) - 1), copy=False)
            donors = []
            for tree in range(leaves.shape[1]):
                order = np.argsort(leaves[:, tree], kind='stable')
                ids, starts = np.unique(leaves[order, tree], return_index=True)
                groups = np.split(order, starts[1:])
                donors.append({int(leaf): codes[group] for leaf, group in zip(ids, groups)})
            self.stages.append((field, columns, transform, model, donors, values))
            print(f'  {field}: reconstruction fitted in {time.monotonic() - started:.1f}s', flush=True)
        return self

    def sample(self, frame, *, seed):
        return self._sample(frame, seed=seed)

    def iter_samples(self, frame, *, seeds):
        """Reuse the first stage's unchanged transform across independent draws.

        Later stages are transformed afresh because preceding sampled fields
        change. Random-number calls and row ordering match sample() exactly.
        """
        if not getattr(self, 'stages', None):
            raise ValueError('Fit reconstruction on auxiliary data first.')
        _, columns, transform, _, _ = self.stages[0][:5]
        first_x = sparse.csr_matrix(transform.transform(frame[columns]), dtype=np.float32)
        for seed in seeds:
            yield self._sample(frame, seed=seed, first_x=first_x)

    def _sample(self, frame, *, seed, first_x=None):
        if not hasattr(self, 'stages'):
            raise ValueError('Fit reconstruction on auxiliary data first.')
        # Drop all hidden values before any stage; never condition on true labels.
        completed = frame[EXPANDED_FEATURES].drop(columns=self.hidden).copy()
        rng = np.random.default_rng(seed)
        for stage_index, stage in enumerate(self.stages):
            field, columns, transform, model, donors = stage[:5]
            # Five-element stages from older saved chains contain values
            # directly. New stages contain compact codes into a value codebook.
            donor_values = stage[5] if len(stage) > 5 else None
            x = (first_x if stage_index == 0 and first_x is not None else
                 sparse.csr_matrix(transform.transform(completed[columns]), dtype=np.float32))
            tree_choices = rng.integers(len(model.estimators_), size=len(completed))
            values = np.empty(len(completed), dtype=float)
            for tree in np.unique(tree_choices):
                rows = np.flatnonzero(tree_choices == tree)
                leaf_ids = model.estimators_[tree].apply(x[rows])
                for leaf in np.unique(leaf_ids):
                    selected = rows[leaf_ids == leaf]
                    sampled = rng.choice(donors[tree][int(leaf)], size=len(selected))
                    values[selected] = sampled if donor_values is None else donor_values[sampled]
            completed[field] = values
        return completed[EXPANDED_FEATURES]


def auxiliary_roles(auxiliary, attack_result, *, seed=42):
    """Reserve disjoint record IDs for reconstruction, meta fit, calibration/test.

    The frozen base attack used its training IDs. Its validation IDs are split
    into meta-training and final calibration. Base thresholds are unused here.
    Its untouched auxiliary test IDs supply the final auxiliary evaluation.
    """
    splits = attack_result['splits']
    all_attack_ids = set().union(*(set(p['record_id']) for p in splits.values()))
    validation = np.asarray(sorted(set(splits['validation']['record_id'])))
    if len(validation) < 20:
        raise ValueError('Too few auxiliary validation IDs for meta/calibration split.')
    meta, calibration = train_test_split(validation, test_size=0.5, random_state=seed)
    roles = dict(reconstruction=set(auxiliary.index) - all_attack_ids,
        base=set(splits['train']['record_id']), meta=set(meta),
        calibration=set(calibration), test=set(splits['test']['record_id']))
    names = list(roles)
    for i, name in enumerate(names):
        for other in names[i + 1:]:
            if roles[name] & roles[other]:
                raise ValueError('Auxiliary roles overlap.')
    if len(roles['reconstruction']) < 100:
        raise ValueError('Not enough unused 2014 records; do not silently reuse attack records.')
    return roles


def completion_scores(query, candidates, reconstructor, base_classifier, *, k,
                      seed, walk_config, numeric_protocol=None, include_blind_walk=True,
                      checkpoint_dir=None, context_signature=None):
    """Return K scores per candidate; optionally use common blind-walk randomness.

    Candidate ordering must be deterministic. Membership labels are never passed
    here. Each completed record uses the same selected attack features as the
    fitted base classifier (three probability features, optionally blind walk).
    """
    if k < 2:
        raise ValueError('At least two completions are needed for variability.')
    from pathlib import Path
    import inspect
    import joblib
    from .uncertainty_checkpoints import atomic_save, frame_signature, load_verified

    cached = [None] * k
    paths, signatures = [], []
    if checkpoint_dir is not None:
        if context_signature is None or not hasattr(reconstructor, 'checkpoint_signature_'):
            raise ValueError('Resumable scoring requires fitted-oracle and reconstruction signatures.')
        visible = candidates[EXPANDED_FEATURES + ['income_gt_50k']].drop(columns=reconstructor.hidden)
        source_dir = Path(__file__).parent
        feature_code = [(source_dir / name).read_bytes()
                        for name in ['paper_attack_features.py', 'shadow_models.py']]
        settings = dict(context=context_signature, base_attack=joblib.hash(base_classifier),
            reconstruction=reconstructor.checkpoint_signature_, candidates=frame_signature(visible),
            walk_config=walk_config, numeric_protocol=numeric_protocol,
            include_blind_walk=include_blind_walk,
            feature_code=feature_code, completion_code=inspect.getsource(completion_scores))
        signatures = [joblib.hash(dict(settings, seed=seed + draw)) for draw in range(k)]
        paths = [Path(checkpoint_dir) / f'score_draw_{draw + 1:03d}.joblib' for draw in range(k)]
        cached = [load_verified(path, signature) for path, signature in zip(paths, signatures)]
    missing = [draw for draw, value in enumerate(cached) if value is None]
    generated = reconstructor.iter_samples(candidates[EXPANDED_FEATURES],
        seeds=[seed + draw for draw in missing])
    scores, audits = [], []
    for draw in range(k):
        if cached[draw] is not None:
            print(f'  Completion {draw + 1}/{k}: loaded saved attack scores', flush=True)
            values, audit = cached[draw]
            scores.append(values)
            audits.append(audit)
            continue
        print(f'  Completion {draw + 1}/{k}: sampling and extracting attack features', flush=True)
        completion = next(generated)
        features, audit = extract_paper_features(query, completion,
            candidates['income_gt_50k'].to_numpy(), config=walk_config,
            numeric_protocol=numeric_protocol, include_blind_walk=include_blind_walk)
        values = base_classifier.predict_proba(features[list(base_classifier.feature_names_in_)])[:, 1]
        if checkpoint_dir is not None:
            atomic_save(paths[draw], signatures[draw], (values, audit))
        scores.append(values)
        audits.append(audit)
    return np.column_stack(scores), audits


def summarize_scores(scores):
    scores = np.asarray(scores, dtype=float)
    if scores.ndim != 2 or scores.shape[1] < 2 or not np.isfinite(scores).all():
        raise ValueError('Expected finite candidate-by-completion scores with K >= 2.')
    return pd.DataFrame(dict(single=scores[:, 0], mean=scores.mean(axis=1),
                             std=scores.std(axis=1, ddof=0)))


def fit_aggregation(meta, calibration, *, seed=42, include_logistic=False):
    """Fit matched RFs and optional L2 logistic controls; calibrate separately.

    Logistic models use the same draws/roles as RF, fixed C=1, and a scaler fit
    only on meta-training records. Calibration labels select thresholds only.
    """
    for frame in [meta, calibration]:
        if set(frame['membership'].unique()) != {0, 1}:
            raise ValueError('Meta fit and calibration each require both classes.')
    learned = {}
    for name, columns in [('learned_mean', ['mean']), ('uncertainty', ['mean', 'std'])]:
        model = RandomForestClassifier(n_estimators=300, max_depth=12,
            min_samples_leaf=20, max_features=1.0, random_state=seed, n_jobs=2)
        model.fit(meta[columns], meta['membership'])
        learned[name] = model
    if include_logistic:
        for name, columns in [('logistic_mean', ['mean']),
                              ('logistic_uncertainty', ['mean', 'std'])]:
            model = Pipeline([
                ('scale', StandardScaler()),
                ('classifier', LogisticRegression(C=1.0, solver='lbfgs',
                    max_iter=1000, random_state=seed)),
            ])
            model.fit(meta[columns], meta['membership'])
            learned[name] = model
    scores = aggregation_predictions(calibration, learned)
    thresholds = {}
    for name, values in scores.items():
        fpr, tpr, cutoffs = roc_curve(calibration['membership'], values)
        thresholds[name] = dict(accuracy=float(cutoffs[np.argmax(tpr - fpr)]),
            low_fpr=threshold_for_fpr(calibration['membership'], values))
    return learned, thresholds


def aggregation_predictions(frame, learned):
    result = dict(single=frame['single'].to_numpy(), average=frame['mean'].to_numpy())
    for name, model in learned.items():
        result[name] = model.predict_proba(frame[list(model.feature_names_in_)])[:, 1]
    return result


def evaluate_aggregation(frame, learned, thresholds):
    labels = frame['membership'].to_numpy(dtype=int)
    if set(np.unique(labels)) != {0, 1}:
        raise ValueError('Evaluation requires both membership classes.')
    rows, predictions = [], aggregation_predictions(frame, learned)
    for method, scores in predictions.items():
        cutoffs = thresholds[method]
        predicted = scores >= cutoffs['accuracy']
        tn, fp, fn, tp = confusion_matrix(labels, predicted, labels=[0, 1]).ravel()
        fpr, tpr, _ = roc_curve(labels, scores, drop_intermediate=False)
        rows.append(dict(method=method, records=len(labels),
            accuracy=float(accuracy_score(labels, predicted)),
            balanced_accuracy=float(balanced_accuracy_score(labels, predicted)),
            roc_auc=float(roc_auc_score(labels, scores)),
            tpr_at_1pct_fpr_roc=float(tpr[fpr <= 0.01].max()),
            tpr_at_shadow_threshold=float((scores[labels == 1] >= cutoffs['low_fpr']).mean()),
            fpr_at_shadow_threshold=float((scores[labels == 0] >= cutoffs['low_fpr']).mean()),
            threshold=cutoffs['accuracy'], shadow_selected_threshold=cutoffs['low_fpr'],
            tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp)))
    return pd.DataFrame(rows), predictions
