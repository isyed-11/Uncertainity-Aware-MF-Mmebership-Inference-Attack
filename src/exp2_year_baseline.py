"""Native-year ACS adapters for the restored Experiment 2 baseline.

Retain its ten predictor concepts, replacing the 2018 relationship column RELP
with native RELSHIPP for 2023. Each model fits and queries its own native schema;
category code equivalence across years is neither assumed nor fabricated.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss
from .data import FEATURES as ORIGINAL_FEATURES, TARGET
from .shadow_models import ALGORITHMS, split_shadow_data, build_shadow_estimator, membership_features
from .training import train_target_model, predict_probabilities

NUMERIC = ['AGEP', 'WKHP']


def features_for_year(year):
    return ['RELSHIPP' if c == 'RELP' and year >= 2019 else c for c in ORIGINAL_FEATURES]


def load_baseline_year(year):
    features = features_for_year(year)
    filename = 'ss14pca.csv' if year == 2014 else 'psam_p06.csv'
    path = Path(f'data/raw/folktables/{year}/1-Year') / filename
    raw = pd.read_csv(path,usecols=features+['PINCP','PWGTP'],skipinitialspace=True,low_memory=False)
    for c in raw:
        raw[c] = pd.to_numeric(raw[c],errors='raise')
    keep = (raw.AGEP > 16) & (raw.PINCP > 100) & (raw.WKHP > 0) & (raw.PWGTP >= 1)
    # Match the original Folktables baseline export's missing-value-to-zero rule.
    frame = raw.loc[keep,features].fillna(0).copy()
    frame[TARGET] = (raw.loc[keep,'PINCP'] > 50000).astype('int64')
    frame.index = pd.Index([f'acs_{year}_{i}' for i in frame.index],name='record_id')
    if frame.empty or set(frame[TARGET].unique()) != {0,1}:
        raise ValueError('Expected both income classes.')
    frame.attrs['year'] = year
    return frame


def build_preprocessor(features):
    categorical = [c for c in features if c not in NUMERIC]
    return ColumnTransformer([
        ('numeric',Pipeline([
            ('impute',SimpleImputer(strategy='median',keep_empty_features=True)),
            ('scale',StandardScaler())]),NUMERIC),
        ('categorical',Pipeline([
            ('impute',SimpleImputer(strategy='most_frequent',keep_empty_features=True)),
            ('encode',OneHotEncoder(handle_unknown='ignore',sparse_output=False,dtype=np.float32))]),categorical)
    ],remainder='drop',verbose_feature_names_out=True)


def preprocess_partitions(partitions,features):
    pp = build_preprocessor(features).fit(partitions['train'][features])
    arrays = {name:dict(X=np.asarray(pp.transform(frame[features]),dtype=np.float32),
        y=frame[TARGET].to_numpy(dtype=np.int64),record_ids=frame.index.to_numpy(copy=True))
        for name,frame in partitions.items()}
    return pp,arrays


def train_acs_shadows(auxiliary,features,seed=42,n_models=5):
    """Cycle the original algorithms with unique seeds and model identifiers."""
    if not isinstance(n_models,int) or isinstance(n_models,bool) or n_models<1:
        raise ValueError('n_models must be a positive integer.')
    results = {}
    for i in range(n_models):
        algorithm = ALGORITHMS[i % len(ALGORITHMS)]
        model_name = algorithm if n_models<=len(ALGORITHMS) else f'{algorithm}_{i+1:02d}'
        shadow_seed = seed+10*i
        print(f'Training ACS shadow {i+1}/{n_models}: {model_name}',flush=True)
        parts = split_shadow_data(auxiliary,seed=shadow_seed)
        pp,arrays = preprocess_partitions(parts,features)
        history,metadata = [],{}
        if algorithm == 'mlp':
            model,history,metadata = train_target_model(arrays['train']['X'],arrays['train']['y'],
                arrays['validation']['X'],arrays['validation']['y'],hidden_dims=(128,64),dropout=0.0,
                epochs=50,batch_size=256,patience=5,seed=shadow_seed,device='cpu')
        else:
            model = build_shadow_estimator(algorithm,seed=shadow_seed)
            model.fit(arrays['train']['X'],arrays['train']['y'])
            metadata = model.get_params()
        metrics,rows = {},[]
        for name,values in arrays.items():
            probabilities = (predict_probabilities(model,values['X']) if algorithm == 'mlp'
                else model.predict_proba(values['X'])[:,[list(model.classes_).index(k) for k in [0,1]]])
            y = values['y']
            metrics[name] = dict(records=len(y),accuracy=float(accuracy_score(y,probabilities.argmax(1))),
                roc_auc=float(roc_auc_score(y,probabilities[:,1])),
                log_loss=float(log_loss(y,probabilities,labels=[0,1])))
            if name != 'validation':
                row = membership_features(probabilities,y)
                row['record_id'] = values['record_ids']
                row['income_label'] = y
                row['membership'] = int(name == 'train')
                row['shadow_model'] = model_name
                rows.append(row)
        results[model_name] = dict(algorithm=algorithm,seed=shadow_seed,schema='acs',model=model,
            preprocessor=pp,metrics=metrics,partition_ids={k:v.index.to_numpy() for k,v in parts.items()},
            attack_data=pd.concat(rows,ignore_index=True),history=history,metadata=metadata)
    return results
