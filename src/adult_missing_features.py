"""Adult-only completion of shared ACS fields with an explicit coarse mapping.

Target remains the ten-input ACS model. No ACS records fit this completion model.
Observed ACS occupation, birthplace and Hispanic origin are preserved, but are
not passed to the Adult completion model because no reliable mapping is supplied.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from scipy import sparse
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.model_selection import train_test_split

SHARED = ['AGEP', 'COW', 'SCHL', 'MAR', 'WKHP', 'SEX', 'RAC1P']
NUMERIC = ['AGEP', 'WKHP']
CATEGORICAL = [c for c in SHARED if c not in NUMERIC]
LABEL = 'income_gt_50k'
WORKCLASS = {'Private':1, 'Self-emp-not-inc':6, 'Self-emp-inc':7,
    'Federal-gov':5, 'Local-gov':3, 'State-gov':4, 'Without-pay':8, 'Never-worked':9}
MARITAL = {'Married-civ-spouse':1, 'Married-AF-spouse':1, 'Married-spouse-absent':1,
    'Divorced':3, 'Never-married':5, 'Separated':4, 'Widowed':2}
EDUCATION = {'Preschool':2, '1st-4th':7, '5th-6th':9, '7th-8th':11, '9th':12,
    '10th':13, '11th':14, '12th':15, 'HS-grad':16, 'Some-college':19,
    'Assoc-voc':20, 'Assoc-acdm':20, 'Bachelors':21, 'Masters':22,
    'Prof-school':23, 'Doctorate':24}
RACE = {'White':1, 'Black':2, 'Amer-Indian-Eskimo':3, 'Asian-Pac-Islander':6, 'Other':8}
ACS_EDUCATION = {**{v:2 for v in [1,2,3]}, **{v:7 for v in [4,5,6,7]},
    8:9,9:9,10:11,11:11,12:12,13:13,14:14,15:15,16:16,17:16,18:19,19:19,
    20:20,21:21,22:22,23:23,24:24}
ACS_RACE = {1:1,2:2,3:3,4:3,5:3,6:6,7:6,8:8,9:8}


def _map(series, mapping, column):
    known = series.notna() & series.ne('?')
    unknown = set(series[known].unique()) - set(mapping)
    if unknown:
        raise ValueError(f'Unmapped {column} categories: {sorted(unknown)}')
    return series.map(mapping).astype('float32')


def load_aligned_adult(directory='data/raw/uci_adult'):
    """All 48,842 records, source strings mapped rather than encoded ID guesses."""
    columns = ['age','workclass','fnlwgt','education','education_num','marital_status',
        'occupation','relationship','race','sex','capital_gain','capital_loss',
        'hours_per_week','native_country','income']
    frames = []
    for partition, filename in [('train','adult.data'),('test','adult.test')]:
        raw = pd.read_csv(Path(directory)/filename, header=None,names=columns,
            comment='|',skipinitialspace=True)
        frame = pd.DataFrame({'AGEP':raw.age.astype('float32'),
            'COW':_map(raw.workclass,WORKCLASS,'workclass'),
            'SCHL':_map(raw.education,EDUCATION,'education'),
            'MAR':_map(raw.marital_status,MARITAL,'marital_status'),
            'WKHP':raw.hours_per_week.astype('float32'),
            'SEX':_map(raw.sex,{'Male':1,'Female':2},'sex'),
            'RAC1P':_map(raw.race,RACE,'race'),
            LABEL:raw.income.str.strip().str.rstrip('.').eq('>50K').astype('int64')})
        frame.index = pd.Index([f'adult_{partition}_{i}' for i in range(len(frame))],name='record_id')
        frames.append(frame)
    combined = pd.concat(frames)
    if len(combined)!=48842 or not combined.index.is_unique:
        raise ValueError('Expected complete Adult dataset with unique namespaced IDs.')
    return combined


def align_acs_observed(frame):
    """Coarsen only observed shared fields; hidden NaNs remain NaN."""
    aligned = frame[SHARED].copy()
    aligned['COW'] = aligned.COW.replace({2:1})
    aligned['SCHL'] = _map(aligned.SCHL,ACS_EDUCATION,'ACS SCHL')
    aligned['RAC1P'] = _map(aligned.RAC1P,ACS_RACE,'ACS RAC1P')
    return aligned


class AdultCompletion:
    """Same AE hidden layers, epochs, optimizer and masking rule; seven inputs.

    Mapping is deliberately approximate. In particular Adult Private cannot
    recover nonprofit COW=2, and HS-grad cannot distinguish ACS GED from diploma.
    Predicted categories use the declared representative ACS codes.
    """
    def fit(self, auxiliary, seed=42, epochs=50):
        torch.manual_seed(seed)
        rng=np.random.default_rng(seed)
        train, validation=train_test_split(auxiliary,test_size=.2,random_state=seed,
                                           stratify=auxiliary[LABEL])
        self.train_ids=train.index.to_numpy()
        self.validation_ids=validation.index.to_numpy()
        self.numeric=Pipeline([('impute',SimpleImputer(strategy='median')),('scale',MinMaxScaler())])
        self.numeric.fit(train[NUMERIC])
        self.encoder=OneHotEncoder(handle_unknown='ignore',sparse_output=True,dtype=np.float32)
        self.encoder.fit(train[CATEGORICAL].fillna(-1))
        self.groups={c:np.array([i]) for i,c in enumerate(NUMERIC)}
        offset=2
        for c,categories in zip(CATEGORICAL,self.encoder.categories_):
            self.groups[c]=np.arange(offset,offset+len(categories))
            offset+=len(categories)
        self.model=nn.Sequential(nn.Linear(offset,128),nn.ReLU(),nn.Linear(128,64),nn.ReLU(),
            nn.Linear(64,64),nn.ReLU(),nn.Linear(64,128),nn.ReLU(),nn.Linear(128,offset))
        optimizer=torch.optim.Adam(self.model.parameters(),lr=.001)
        X=self._encode_aligned(train)
        self.history=[]
        for epoch in range(epochs):
            total=0.
            self.model.train()
            order=rng.permutation(len(train))
            for start in range(0,len(order),256):
                clean=X[order[start:start+256]].toarray().astype(np.float32)
                masked=clean.copy()
                counts=rng.integers(1,5,len(clean))
                ranks=rng.random((len(clean),len(SHARED))).argsort(axis=1)
                for j,c in enumerate(SHARED):
                    rows=np.flatnonzero(np.any((ranks==j)&(np.arange(len(SHARED))[None,:]<counts[:,None]),axis=1))
                    masked[np.ix_(rows,self.groups[c])]=-1
                optimizer.zero_grad()
                loss=nn.functional.mse_loss(self.model(torch.from_numpy(masked)),torch.from_numpy(clean))
                loss.backward()
                optimizer.step()
                total+=loss.item()*len(clean)
            self.history.append(dict(epoch=epoch+1,reconstruction_loss=total/len(train)))
            print(f'Adult AE epoch {epoch+1:02d}/{epochs}: loss={total/len(train):.6f}',flush=True)
        self.model.eval()
        return self

    def _encode_aligned(self,frame):
        return sparse.hstack([sparse.csr_matrix(self.numeric.transform(frame[NUMERIC]).astype(np.float32)),
            self.encoder.transform(frame[CATEGORICAL].fillna(-1))],format='csr')

    def encode(self,acs_frame):
        # Used only for reporting imputation error in the common representation.
        return self._encode_aligned(align_acs_observed(acs_frame))

    def complete(self,partial,missing):
        if not set(missing)<=set(SHARED) or not partial[list(missing)].isna().all().all():
            raise ValueError('Only shared fields may be reconstructed; hidden values must be removed.')
        X=self._encode_aligned(align_acs_observed(partial))
        predictions=[]
        self.model.eval()
        with torch.no_grad():
            for start in range(0,len(partial),512):
                x=X[start:start+512].toarray().astype(np.float32)
                for c in missing:
                    x[:,self.groups[c]]=-1
                predictions.append(self.model(torch.from_numpy(x)).numpy())
        prediction=np.concatenate(predictions)
        numeric=self.numeric.named_steps['scale'].inverse_transform(np.clip(prediction[:,:2],0,1))
        completed=partial.copy()
        for c in missing:
            if c in NUMERIC:
                completed[c]=np.rint(numeric[:,NUMERIC.index(c)])
            else:
                categories=self.encoder.categories_[CATEGORICAL.index(c)]
                completed[c]=categories[prediction[:,self.groups[c]].argmax(axis=1)]
                completed[c]=completed[c].replace(-1,np.nan)
        observed=[c for c in partial if c not in missing]
        pd.testing.assert_frame_equal(completed[observed],partial[observed])
        return completed
