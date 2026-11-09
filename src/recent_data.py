"""Year-specific Experiment 1 ACS inputs, retaining native relationship codes."""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .data import TARGET

FEATURES = ['AGEP', 'COW', 'SCHL', 'MAR', 'OCCP', 'POBP', 'RELSHIPP', 'WKHP', 'SEX', 'RAC1P']
NUMERIC = ['AGEP', 'WKHP']
CATEGORICAL = [c for c in FEATURES if c not in NUMERIC]


def load_recent_acs(year, *, align_relationship_to_2023=False):
    """Apply ACSIncome eligibility and the nominal >$50,000 income label.

    RELSHIPP replaces the unavailable historical RELP field. Its native codes
    are retained, not renamed or claimed equivalent to the historical codes.
    Missing predictors follow Experiment 1/Folktables zero-fill behavior.
    """
    if year not in (2014, 2018, 2023, 2024):
        raise ValueError('Supported experiment years are 2014, 2018, 2023 and 2024.')
    filename = 'ss14pca.csv' if year == 2014 else 'psam_p06.csv'
    feature_columns = ['RELP' if c == 'RELSHIPP' and year in (2014, 2018) else c for c in FEATURES]
    path = Path(f'data/raw/folktables/{year}/1-Year') / filename
    extra = ['SERIALNO'] if year == 2018 and align_relationship_to_2023 else []
    raw = pd.read_csv(path, usecols=feature_columns + ['PINCP', 'PWGTP'] + extra,
                      low_memory=False, skipinitialspace=True)
    for column in raw.columns.difference(['SERIALNO']):
        raw[column] = pd.to_numeric(raw[column], errors='raise')
    relationship_status = None
    if year == 2018 and align_relationship_to_2023:
        raw['RELSHIPP'], relationship_status = map_2018_relationship(raw)
        feature_columns = FEATURES
    keep = (raw.AGEP > 16) & (raw.PINCP > 100) & (raw.WKHP > 0) & (raw.PWGTP >= 1)
    frame = raw.loc[keep, feature_columns].fillna(0).astype('float32').copy()
    frame[TARGET] = (raw.loc[keep, 'PINCP'] > 50_000).astype('int64')
    frame.index = pd.Index([f'acs_{year}_{i}' for i in frame.index], name='record_id')
    if frame.empty or frame[TARGET].nunique() != 2:
        raise ValueError('Expected nonempty data with both income classes.')
    if not np.isfinite(frame[feature_columns].to_numpy()).all():
        raise ValueError('Nonfinite ACS predictors.')
    if relationship_status is not None:
        frame.attrs['relationship_mapping_audit'] = relationship_status.loc[keep].value_counts().to_dict()
    return frame


def preprocess_recent_partitions(partitions):
    """Use the same dense train-only standardization/one-hot setup as Experiment 1."""
    feature_columns = [c for c in partitions['train'] if c != TARGET]
    categorical_columns = [c for c in feature_columns if c not in NUMERIC]
    preprocessor = ColumnTransformer([
        ('numeric', Pipeline([
            ('impute', SimpleImputer(strategy='median', keep_empty_features=True)),
            ('scale', StandardScaler()),
        ]), NUMERIC),
        ('categorical', Pipeline([
            ('impute', SimpleImputer(strategy='most_frequent', keep_empty_features=True)),
            ('encode', OneHotEncoder(handle_unknown='ignore', sparse_output=False, dtype=np.float32)),
        ]), categorical_columns),
    ], remainder='drop', verbose_feature_names_out=True)
    preprocessor.fit(partitions['train'][feature_columns])
    arrays = {name: {'X': np.asarray(preprocessor.transform(part[feature_columns]), dtype=np.float32),
                     'y': part[TARGET].to_numpy(dtype=np.int64, copy=True),
                     'record_ids': part.index.to_numpy(copy=True)}
              for name, part in partitions.items()}
    return preprocessor, arrays


def transform_recent_external(frame, preprocessor):
    """Transform external-year records without fitting; audit unseen categories."""
    feature_columns = list(preprocessor.feature_names_in_)
    categorical_columns = [c for c in feature_columns if c not in NUMERIC]
    categorical = preprocessor.named_transformers_['categorical']
    imputed = categorical.named_steps['impute'].transform(frame[categorical_columns])
    coverage = pd.DataFrame([
        {'feature': column, 'unseen_records': int((~np.isin(imputed[:, i], known)).sum()),
         'unseen_fraction': float((~np.isin(imputed[:, i], known)).mean())}
        for i, (column, known) in enumerate(zip(categorical_columns, categorical.named_steps['encode'].categories_))
    ]).set_index('feature')
    arrays = {'X': np.asarray(preprocessor.transform(frame[feature_columns]), dtype=np.float32),
              'y': frame[TARGET].to_numpy(dtype=np.int64, copy=True),
              'record_ids': frame.index.to_numpy(copy=True)}
    return arrays, coverage


def map_2018_relationship(raw):
    """Convert RELP by Census 2018/2023 category definitions, without model fitting.

    Spouse/partner categories split using the person's and reference person's
    recorded SEX, joined within SERIALNO. This is a derived approximation across
    survey coding regimes, not a claim of equivalence to a later survey response.
    Missing reference-person sex or unexpected codes become -1 (unseen category).
    SERIALNO and reference-person SEX are used only for this conversion, never
    added to model predictors. Roomer/boarder merges into other nonrelative.
    Sources: Census PUMS_Data_Dictionary_2018.pdf and _2023.pdf.
    """
    direct = {0:20, 2:25, 3:26, 4:27, 5:28, 6:29, 7:30, 8:31,
              9:32, 10:33, 11:36, 12:34, 14:35, 15:36, 16:37, 17:38}
    relationship = raw['RELP'].map(direct)
    status = pd.Series('direct_category_mapping', index=raw.index)
    status.loc[raw['RELP'].eq(11)] = 'roomer_boarder_merged_other_nonrelative'
    reference = raw.loc[raw['RELP'].eq(0), ['SERIALNO', 'SEX']].set_index('SERIALNO')['SEX']
    if not reference.index.is_unique:
        raise ValueError('Multiple reference persons in a 2018 household.')
    reference_sex = raw['SERIALNO'].map(reference)
    known_sex = raw['SEX'].isin([1,2]) & reference_sex.isin([1,2])
    same_sex = raw['SEX'].eq(reference_sex)
    for old_code, opposite_code, same_code in [(1,21,23),(13,22,24)]:
        selected = raw['RELP'].eq(old_code) & known_sex
        relationship.loc[selected] = np.where(same_sex.loc[selected], same_code, opposite_code)
        status.loc[selected] = 'spouse_partner_inferred_from_reference_sex'
    status.loc[relationship.isna()] = 'unresolved_encoded_as_unseen'
    return relationship.fillna(-1).astype('float32'), status


def evaluate_saved_target_external(year, run_dir, *, sample_size=25_000, seed=42):
    """Evaluate the saved 2023 baseline on a stratified external-year sample.

    No fitting or target/shadow retraining occurs. Writes separate external-year
    metrics, exact source IDs, and unseen-category coverage to the run directory.
    Remaining native category codes are retained; unseen-category coverage is an
    audit, not proof of full semantic harmonization across survey years.
    """
    import json
    import joblib
    import torch
    from sklearn.model_selection import train_test_split
    from .training import build_mlp, evaluate_model

    if year not in (2018, 2024):
        raise ValueError('External evaluation years are 2018 and 2024.')
    run_dir = Path(run_dir)
    summary = json.loads((run_dir / 'summary.json').read_text(encoding='utf-8'))
    if summary['target_year'] != 2023:
        raise ValueError('Expected the saved 2023 target model.')
    checkpoint = torch.load(run_dir / 'target_model.pt', map_location='cpu', weights_only=True)
    metadata = checkpoint['metadata']
    model = build_mlp(metadata['input_dim'], metadata['hidden_dims'],
                      metadata['dropout'], metadata['num_classes'])
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    preprocessor = joblib.load(run_dir / 'preprocessor.joblib')
    source = load_recent_acs(year, align_relationship_to_2023=(year == 2018))
    if not isinstance(sample_size, int) or not 1 < sample_size < len(source):
        raise ValueError('Sample size must be an integer smaller than the eligible source size.')
    _, sample = train_test_split(source, test_size=sample_size,
                                 random_state=seed, stratify=source[TARGET])
    sample = sample.sort_index()
    arrays, coverage = transform_recent_external(sample, preprocessor)
    metrics = evaluate_model(model, arrays['X'], arrays['y'])
    result = dict(target_year=2023, evaluation_year=year, records=len(sample),
                  eligible_source_records=len(source), seed=seed,
                  sampling='stratified by income label, without replacement',
                  positive_rate=float(sample[TARGET].mean()), **metrics,
                  relationship_mapping_audit=source.attrs.get('relationship_mapping_audit'),
                  relationship_audit_scope='all eligible source records',
                  notes='Nominal >$50,000 income; no inflation adjustment. Native codes retained except audited 2018 relationship conversion. No full cross-year classification harmonization claimed.')
    pd.Series(sample.index, name='record_id').to_csv(run_dir / f'external_{year}_ids.csv', index=False)
    coverage.to_csv(run_dir / f'external_{year}_category_coverage.csv')
    (run_dir / f'external_{year}_metrics.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    return result, coverage


def external_evaluation_cells():
    """Return exactly two unexecuted notebook cells for the 2023 baseline."""
    cells = []
    for year in (2024, 2018):
        source = f"""# Test the saved 2023 target on 25,000 ACS {year} records (no retraining).
# Fixed stratified sample; fitted target preprocessor is loaded unchanged.
# 2018 RELP is converted to RELSHIPP with a reported mapping audit.
# These are income-prediction metrics; keep them out of model selection.
import importlib
from pathlib import Path
import pandas as pd
from IPython.display import display
import src.recent_data
importlib.reload(src.recent_data)
from src.recent_data import evaluate_saved_target_external

external_{year}_metrics, external_{year}_coverage = evaluate_saved_target_external(
    {year}, Path('artifacts/experiment1_2023_aux2014_16shadows'), sample_size=25_000, seed=42)
display(pd.DataFrame([external_{year}_metrics]).set_index('evaluation_year'))
display(external_{year}_coverage)
"""
        cells.append(dict(cell_type='code', id=f'external-{year}-25k', metadata={},
                          execution_count=None, outputs=[], source=source.splitlines(keepends=True)))
    return cells
