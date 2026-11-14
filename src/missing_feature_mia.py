"""Ten-field ACS income baseline and auxiliary-only masked autoencoder."""
from pathlib import Path
import hashlib
import itertools
import json
import joblib
import numpy as np
import pandas as pd
import torch
from torch import nn
from scipy import sparse
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss
from .expanded_data import load_expanded_year
from .shadow_models import (split_shadow_data, build_shadow_estimator,
                           membership_features, PAPER_SHADOW_ALGORITHMS)
from .exp3_checkpoints import save
from .paper_attack_features import extract_paper_features, make_probability_oracle
from .membership_attack import evaluate_attack

FEATURES = ['AGEP', 'COW', 'SCHL', 'MAR', 'OCCP', 'POBP', 'WKHP', 'SEX', 'RAC1P', 'HISP']
NUMERIC = ['AGEP', 'WKHP']
CATEGORICAL = [c for c in FEATURES if c not in NUMERIC]
WITHHELD = ['COW', 'WKHP', 'MAR', 'SCHL']
LABEL = 'income_gt_50k'


def load_year(year):
    return load_expanded_year(year)[FEATURES + [LABEL]].copy()


def preprocessor():
    return ColumnTransformer([
        ('numeric', Pipeline([('impute', SimpleImputer(strategy='median', add_indicator=True,
            keep_empty_features=True)), ('scale', StandardScaler())]), NUMERIC),
        ('categorical', Pipeline([('impute', SimpleImputer(strategy='constant', fill_value=-1,
            keep_empty_features=True)), ('encode', OneHotEncoder(handle_unknown='ignore',
            sparse_output=True, dtype=np.float32))]), CATEGORICAL)], sparse_threshold=1.0)


def prepare(partitions):
    pp = preprocessor().fit(partitions['train'][FEATURES])
    arrays = {name: dict(X=sparse.csr_matrix(pp.transform(frame[FEATURES]), dtype=np.float32),
        y=frame[LABEL].to_numpy(dtype=np.int64), record_ids=frame.index.to_numpy())
        for name, frame in partitions.items()}
    return pp, arrays


def train_shadows(auxiliary, run_dir, count=16, seed=42):
    signature = hashlib.sha256(pd.util.hash_pandas_object(auxiliary, index=True).values.tobytes()
        + Path(__file__).read_bytes()).hexdigest()
    results = {}
    for i in range(count):
        algorithm = PAPER_SHADOW_ALGORITHMS[i % 3]
        name = f'{algorithm}_{i+1:02d}'
        path = Path(run_dir) / f'checkpoints/{name}.joblib'
        if path.exists():
            cached = joblib.load(path)
            if cached['signature'] != signature:
                raise ValueError('Shadow configuration mismatch.')
            results[name] = cached['result']
            print('Loaded shadow:', name, flush=True)
            continue
        print(f'Training shadow {i+1}/{count}: {name}', flush=True)
        parts = split_shadow_data(auxiliary, seed=seed+10*i)
        pp, arrays = prepare(parts)
        model = build_shadow_estimator(algorithm, seed=seed+10*i)
        model.fit(arrays['train']['X'], arrays['train']['y'])
        metrics, rows = {}, []
        for partition, values in arrays.items():
            probabilities = model.predict_proba(values['X'])
            y = values['y']
            metrics[partition] = dict(records=len(y), accuracy=float(accuracy_score(y, probabilities.argmax(1))),
                roc_auc=float(roc_auc_score(y, probabilities[:, 1])),
                log_loss=float(log_loss(y, probabilities, labels=[0, 1])))
            if partition != 'validation':
                row = membership_features(probabilities, y)
                row['record_id'] = values['record_ids']
                row['membership'] = int(partition == 'train')
                row['shadow_model'] = name
                rows.append(row)
        result = dict(algorithm=algorithm, seed=seed+10*i, schema='acs', model=model,
            preprocessor=pp, partition_ids={k:v.index.to_numpy() for k,v in parts.items()},
            metrics=metrics, attack_data=pd.concat(rows, ignore_index=True))
        save(path, dict(signature=signature, result=result))
        results[name] = result
        print('Saved shadow:', name, flush=True)
    return results


class MaskedAutoencoder:
    """Paper architecture; random whole-field masks and sparse minibatch inputs.

    Uniformly sample k in 1..4, then k distinct original fields per training row.
    Reconstruct all encoded coordinates using MSE. Decode categories by argmax;
    numeric outputs are clipped to auxiliary train ranges and rounded to integers.
    These choices are explicit adaptations where the paper omits details.
    """
    def fit(self, auxiliary, seed=42, epochs=50):
        torch.manual_seed(seed)
        rng = np.random.default_rng(seed)
        train, validation = train_test_split(auxiliary, test_size=.2, random_state=seed,
                                             stratify=auxiliary[LABEL])
        self.train_ids = train.index.to_numpy()
        self.validation_ids = validation.index.to_numpy()
        self.numeric = Pipeline([('impute', SimpleImputer(strategy='median')), ('scale', MinMaxScaler())])
        self.numeric.fit(train[NUMERIC])
        self.encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=True, dtype=np.float32)
        self.encoder.fit(train[CATEGORICAL].fillna(-1))
        self.groups = {c: np.array([i]) for i,c in enumerate(NUMERIC)}
        offset = len(NUMERIC)
        for c, cats in zip(CATEGORICAL, self.encoder.categories_):
            self.groups[c] = np.arange(offset, offset+len(cats))
            offset += len(cats)
        self.model = nn.Sequential(nn.Linear(offset,128), nn.ReLU(), nn.Linear(128,64), nn.ReLU(),
            nn.Linear(64,64), nn.ReLU(), nn.Linear(64,128), nn.ReLU(), nn.Linear(128,offset))
        optimizer = torch.optim.Adam(self.model.parameters(), lr=.001)
        X = self.encode(train)
        self.history = []
        for epoch in range(epochs):
            order = rng.permutation(len(train))
            total = 0.
            self.model.train()
            for start in range(0, len(order), 256):
                clean = X[order[start:start+256]].toarray().astype(np.float32)
                masked = clean.copy()
                # Mask original fields, never individual one-hot coordinates alone.
                counts = rng.integers(1,5,len(clean))
                ranks = rng.random((len(clean),len(FEATURES))).argsort(axis=1)
                for j, c in enumerate(FEATURES):
                    rows = np.flatnonzero(np.any((ranks == j) & (np.arange(len(FEATURES))[None,:] < counts[:,None]), axis=1))
                    masked[np.ix_(rows, self.groups[c])] = -1
                y = torch.from_numpy(clean)
                optimizer.zero_grad()
                loss = nn.functional.mse_loss(self.model(torch.from_numpy(masked)), y)
                loss.backward()
                optimizer.step()
                total += loss.item()*len(clean)
            self.history.append(dict(epoch=epoch+1, reconstruction_loss=total/len(train)))
            print(f'Autoencoder epoch {epoch+1:02d}/{epochs}: loss={total/len(train):.6f}', flush=True)
        self.model.eval()
        return self

    def encode(self, frame):
        return sparse.hstack([sparse.csr_matrix(self.numeric.transform(frame[NUMERIC]).astype(np.float32)),
            self.encoder.transform(frame[CATEGORICAL].fillna(-1))], format='csr')

    def complete(self, partial, missing):
        if not partial[list(missing)].isna().all().all():
            raise ValueError('Hidden values must be removed before completion.')
        completed = partial.copy()
        encoded = self.encode(partial)
        outputs = []
        self.model.eval()
        with torch.no_grad():
            for start in range(0,len(partial),512):
                x = encoded[start:start+512].toarray().astype(np.float32)
                for c in missing:
                    x[:,self.groups[c]] = -1
                outputs.append(self.model(torch.from_numpy(x)).numpy())
        prediction = np.concatenate(outputs)
        numeric = self.numeric.named_steps['scale'].inverse_transform(np.clip(prediction[:,:2],0,1))
        for c in missing:
            if c in NUMERIC:
                completed[c] = np.rint(numeric[:,NUMERIC.index(c)])
            else:
                cats = self.encoder.categories_[CATEGORICAL.index(c)]
                completed[c] = cats[prediction[:,self.groups[c]].argmax(axis=1)]
                completed[c] = completed[c].replace(-1,np.nan)
        observed = [c for c in FEATURES if c not in missing]
        pd.testing.assert_frame_equal(completed[observed], partial[observed])
        return completed


def run_mask_evaluations(completer, target, pp, candidates, attack, baseline, run_dir, walk):
    rows = [dict(method='full_features', missing_count=0, missing_fields='', **baseline)]
    query = make_probability_oracle(target, pp, schema='acs')
    for k in range(1,5):
        for fields in itertools.combinations(WITHHELD,k):
            key = '__'.join(fields)
            print('Missing-feature evaluation:', key, flush=True)
            # Completion receives neither hidden ground truth nor membership labels.
            partial = candidates[FEATURES].copy()
            partial.loc[:,list(fields)] = np.nan
            completed = completer.complete(partial, fields)
            features, audit = extract_paper_features(query, completed, candidates[LABEL].to_numpy(), config=walk)
            features['record_id'] = candidates.index.to_numpy()
            features['membership'] = candidates['membership'].to_numpy()
            metrics, scores = evaluate_attack(attack['classifier'], features,
                threshold=attack['accuracy_threshold'], low_fpr_threshold=attack['low_fpr_threshold'])
            quality = {}
            encoded_true = completer.encode(candidates[FEATURES])
            encoded_guess = completer.encode(completed)
            block = np.concatenate([completer.groups[c] for c in fields])
            mse = float((encoded_true[:,block] - encoded_guess[:,block]).power(2).mean())
            for c in fields:
                valid = candidates[c].notna()
                if c in NUMERIC:
                    quality[c+'_mae'] = float(np.abs(candidates.loc[valid,c]-completed.loc[valid,c]).mean())
                else:
                    quality[c+'_accuracy'] = float((candidates.loc[valid,c] == completed.loc[valid,c]).mean())
            rows.append(dict(method='masked_autoencoder', missing_count=k, missing_fields=','.join(fields),
                             imputation_encoded_mse=mse, **metrics))
            save(Path(run_dir)/f'masks/{key}.joblib', dict(features=features,scores=scores,metrics=metrics,
                imputation_quality=quality, imputation_encoded_mse=mse, query_diagnostics=audit))
            pd.DataFrame(rows).to_csv(Path(run_dir)/'missing_feature_metrics.csv', index=False)
    return pd.DataFrame(rows)
