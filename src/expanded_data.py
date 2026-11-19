"""Expanded California ACS inputs and training-only sparse preprocessing."""
from pathlib import Path
import ast
import json

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from scipy import sparse

from .data import FEATURES as BASE_FEATURES, TARGET, load_acs_income

ADDITIONAL_GROUPS = {
    'citizenship': ['CIT', 'CITWP'],
    'employment': ['ESR', 'WKL', 'WKW', 'WRK'],
    'industry': ['INDP'],
    'disability': ['DIS', 'DEAR', 'DEYE', 'DREM', 'DPHY', 'DDRS', 'DOUT'],
    'health_insurance': ['HICOV', 'HINS1', 'HINS2', 'HINS3', 'HINS4', 'HINS5', 'HINS6', 'HINS7'],
    'language': ['LANX', 'LANP', 'ENG'],
    'commuting': ['JWTR', 'JWMNP', 'JWRIP', 'JWAP', 'JWDP', 'POWSP'],
    'education_field': ['FOD1P', 'FOD2P', 'SCH'],
    'demographics': ['HISP', 'ANC1P', 'ANC2P'],
    'military': ['MIL', 'VPS'],
    'marriage_history': ['MARHT', 'MARHYP'],
}
# These legacy fields changed or disappeared by 2023; use the common schema.
EXPANDED_FEATURES = [c for c in BASE_FEATURES + [c for group in ADDITIONAL_GROUPS.values() for c in group]
                     if c not in {'RELP', 'WKW', 'JWTR'}]
NUMERIC = ['AGEP', 'WKHP', 'JWMNP', 'JWRIP']
CATEGORICAL = [c for c in EXPANDED_FEATURES if c not in NUMERIC]
EXCLUDED = ['ADJINC', 'PINCP', 'PERNP', 'WAGP', 'SEMP', 'INTP', 'OIP', 'PAP', 'RETP',
            'SSIP', 'SSP', 'POVPIP', 'SERIALNO', 'SPORDER', 'PUMA', 'PWGTP']


def load_expanded_acs(raw_path='data/raw/folktables/2018/1-Year/psam_p06.csv',
                      baseline_path='data/processed/acs_income_ca_2018.csv'):
    """Preserve ACSIncome eligibility, label, row ordering, and existing record IDs.

    No additional complete-case filtering is performed. Original ten features
    retain Folktables missing-value-to-zero behavior. Added columns retain NaN
    for later training-only preprocessing. PINCP is used for filtering/labeling
    only and is never returned as a predictor.
    """
    if len(set(EXPANDED_FEATURES)) != len(EXPANDED_FEATURES):
        raise ValueError('Duplicate expanded features.')
    if set(EXPANDED_FEATURES) & set(EXCLUDED):
        raise ValueError('Forbidden income or identifier feature.')
    legacy_features = list(dict.fromkeys(EXPANDED_FEATURES + BASE_FEATURES))
    raw = pd.read_csv(Path(raw_path), usecols=legacy_features + ['PINCP', 'PWGTP'],
                      low_memory=False)
    keep = ((raw['AGEP'] > 16) & (raw['PINCP'] > 100)
            & (raw['WKHP'] > 0) & (raw['PWGTP'] >= 1))
    frame = raw.loc[keep, legacy_features].reset_index(drop=True).copy()
    frame[BASE_FEATURES] = frame[BASE_FEATURES].fillna(0)
    labels = (raw.loc[keep, 'PINCP'] > 50_000).astype('int64').to_numpy()
    baseline = load_acs_income(baseline_path)
    if len(frame) != len(baseline):
        raise ValueError('Raw eligibility does not match the baseline record count.')
    np.testing.assert_allclose(frame[BASE_FEATURES].to_numpy(dtype=float),
                               baseline[BASE_FEATURES].to_numpy(dtype=float), rtol=0, atol=0)
    np.testing.assert_array_equal(labels, baseline[TARGET].to_numpy())
    frame = frame[EXPANDED_FEATURES].astype('float32')
    frame[TARGET] = labels
    frame.index.name = 'record_id'
    if np.isinf(frame[EXPANDED_FEATURES].to_numpy()).any():
        raise ValueError('Infinite values in expanded data.')
    return frame


def preprocess_expanded_partitions(partitions):
    """Fit train only; encode category codes categorically and keep missing codes.

    Missing categories use -1 instead of the most common category, preserving
    structural N/A (e.g. degree field for non-graduates). Numeric imputation is
    training median with missingness indicators; then standardize.
    """
    preprocessor = build_expanded_preprocessor()
    preprocessor.fit(partitions['train'][EXPANDED_FEATURES])
    arrays = {name: {
        'X': sparse.csr_matrix(preprocessor.transform(part[EXPANDED_FEATURES]), dtype=np.float32),
        'y': part[TARGET].to_numpy(dtype=np.int64, copy=True),
        'record_ids': part.index.to_numpy(copy=True),
    } for name, part in partitions.items()}
    for name, values in arrays.items():
        if not np.isfinite(values['X'].data).all():
            raise ValueError(f'Invalid transformed values in {name}.')
    return preprocessor, arrays


def reference_split_config(path=None):
    """Read fixed reference seed/sizes, independently of notebook filenames.

    The shared JSON preserves the historical splits. An explicit notebook path
    remains supported for callers comparing another reference configuration;
    its code is parsed as literals and is never executed.
    """
    reference = (Path(__file__).resolve().parents[1] / 'data/reference_split_config.json'
                 if path is None else Path(path))
    saved = json.loads(reference.read_text(encoding='utf-8'))
    if reference.suffix.lower() == '.json':
        required = {'validation_size', 'test_size', 'mia_nonmember_size', 'mia_member_size', 'seed'}
        if not isinstance(saved, dict) or set(saved) != required:
            raise ValueError(f'Expected the five reference split settings in {reference}.')
        if any(type(value) is not int or value < (0 if key == 'seed' else 1)
               for key, value in saved.items()):
            raise ValueError('Split sizes must be positive integers; seed must be a nonnegative integer.')
        return saved
    notebook = saved
    known = {}
    for cell in notebook['cells']:
        if cell.get('id') != 'overfit-imports':
            continue
        for node in ast.parse(''.join(cell['source'])).body:
            if not isinstance(node, ast.Assign) or len(node.targets) != 1:
                continue
            name = getattr(node.targets[0], 'id', None)
            if name == 'OVERFIT_SEED':
                known[name] = ast.literal_eval(node.value)
            if name == 'SPLIT_CONFIG':
                value = node.value
                if not isinstance(value, ast.Call) or getattr(value.func, 'id', None) != 'dict':
                    raise ValueError('Expected literal dict(...) split configuration in reference notebook.')
                return {item.arg: known[item.value.id] if isinstance(item.value, ast.Name)
                        else ast.literal_eval(item.value) for item in value.keywords}
    raise ValueError(f'Could not read reference split configuration from {reference}.')


def expanded_feature_inventory(training):
    """Counts describe actual training records, not target evaluation records."""
    group_for = {c: name for name, cols in ADDITIONAL_GROUPS.items() for c in cols}
    return pd.DataFrame([{
        'feature': c, 'group': group_for.get(c, 'original_ACSIncome'),
        'type': 'numeric' if c in NUMERIC else 'categorical',
        'unique_nonmissing': int(training[c].nunique(dropna=True)),
        'missing_fraction': float(training[c].isna().mean()),
    } for c in EXPANDED_FEATURES]).set_index('feature')


def build_expanded_preprocessor():
    return ColumnTransformer([
        ('numeric', Pipeline([
            ('impute', SimpleImputer(strategy='median', add_indicator=True, keep_empty_features=True)),
            ('scale', StandardScaler()),
        ]), NUMERIC),
        ('categorical', Pipeline([
            ('impute', SimpleImputer(strategy='constant', fill_value=-1, keep_empty_features=True)),
            ('encode', OneHotEncoder(handle_unknown='ignore', sparse_output=True, dtype=np.float32)),
        ]), CATEGORICAL),
    ], remainder='drop', verbose_feature_names_out=True, sparse_threshold=1.0)


def load_expanded_year(year, *, income_base_year=None):
    """Load native-year expanded ACS; IDs are namespaced by survey year.

    Each year retains its own categorical code system. Separate preprocessors
    avoid assuming occupation/industry codes mean the same thing across years.
    By default the income threshold is nominal $50,000. income_base_year=2023
    applies ADJINC and CPI-U before labeling; neither adjustment is a predictor.
    """
    filename = 'ss14pca.csv' if year == 2014 else 'psam_p06.csv'
    path = Path(f'data/raw/folktables/{year}/1-Year') / filename
    label_columns = ['ADJINC'] if income_base_year is not None else []
    raw = pd.read_csv(path, usecols=EXPANDED_FEATURES + ['PINCP', 'PWGTP'] + label_columns,
                      low_memory=False, skipinitialspace=True)
    # Older PUMS files use whitespace-only cells for structural N/A.
    for column in raw:
        raw[column] = pd.to_numeric(raw[column], errors='raise')
    keep = (raw.AGEP > 16) & (raw.PINCP > 100) & (raw.WKHP > 0) & (raw.PWGTP >= 1)
    frame = raw.loc[keep, EXPANDED_FEATURES].astype('float32').copy()
    label_income = raw.loc[keep, 'PINCP'].to_numpy(dtype=np.float64)
    nominal_labels = label_income > 50_000
    if income_base_year is not None:
        from .income_adjustment import income_in_base_year, CPI_U_ANNUAL, CPI_SOURCE, ACS_SOURCE
        label_income = income_in_base_year(label_income, raw.loc[keep, 'ADJINC'], year, income_base_year)
        frame.attrs['income_adjustment'] = dict(base_year=income_base_year, survey_year=year,
            label_threshold=50_000, adjinc_divisor=1_000_000, cpi_series='CPI-U US city average all items, annual NSA',
            cpi_values=CPI_U_ANNUAL, cpi_ratio=CPI_U_ANNUAL[income_base_year]/CPI_U_ANNUAL[year],
            cpi_source=CPI_SOURCE, acs_source=ACS_SOURCE, eligibility='Original nominal ACSIncome eligibility retained',
            labels_changed_from_nominal=int(np.count_nonzero((label_income > 50_000) != nominal_labels)))
    frame[TARGET] = (label_income > 50_000).astype('int64')
    frame.index = pd.Index([f'acs_{year}_{i}' for i in frame.index], name='record_id')
    if frame.empty or frame[TARGET].nunique() != 2:
        raise ValueError('Expected nonempty ACS data with both income classes.')
    if np.isinf(frame[EXPANDED_FEATURES].to_numpy()).any():
        raise ValueError('Infinite predictor value.')
    return frame
