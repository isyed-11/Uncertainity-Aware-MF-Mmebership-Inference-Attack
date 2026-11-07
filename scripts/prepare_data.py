"""Export ACSIncome and Adult CSVs without splitting or fitting preprocessors."""
import argparse
import json
from pathlib import Path
import pandas as pd
from folktables import ACSIncome

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument('--year', type=int, default=2018, choices=[2017, 2018])
args = parser.parse_args()
out = ROOT / 'data/processed'
out.mkdir(parents=True, exist_ok=True)
raw = pd.read_csv(ROOT / f'data/raw/folktables/{args.year}/1-Year/psam_p06.csv', low_memory=False)
features, labels, groups = ACSIncome.df_to_numpy(raw)
acs = pd.DataFrame(features, columns=ACSIncome.features)
acs['income_gt_50k'] = labels.astype(int)
assert len(acs) > 0 and not acs.isna().any().any()
assert set(acs['income_gt_50k'].unique()) == {0, 1}
acs.to_csv(out / f'acs_income_ca_{args.year}.csv', index=False)
if args.year != 2018:
    counts = {'year': args.year, 'state': 'CA', 'acs_raw_rows': len(raw),
              'acs_income_rows': len(acs), 'acs_features': ACSIncome.features,
              'positive_labels': int(acs['income_gt_50k'].sum())}
    (ROOT / f'data/dataset_summary_{args.year}.json').write_text(json.dumps(counts, indent=2) + '\n')
    print(json.dumps(counts, indent=2))
    raise SystemExit(0)
columns = ['age', 'workclass', 'fnlwgt', 'education', 'education_num',
           'marital_status', 'occupation', 'relationship', 'race', 'sex',
           'capital_gain', 'capital_loss', 'hours_per_week', 'native_country', 'income']
counts = {'acs_raw_rows': len(raw), 'acs_income_rows': len(acs), 'acs_features': ACSIncome.features}
for source, name in [('adult.data', 'train'), ('adult.test', 'test')]:
    adult = pd.read_csv(ROOT / 'data/raw/uci_adult' / source, names=columns,
                        skipinitialspace=True, comment='|', na_values='?')
    adult['income_gt_50k'] = adult.pop('income').str.rstrip('.').map({'<=50K': 0, '>50K': 1})
    assert adult['income_gt_50k'].notna().all()
    adult.to_csv(out / f'adult_{name}.csv', index=False)
    counts[f'adult_{name}_rows'] = len(adult)
    counts[f'adult_{name}_rows_with_missing_features'] = int(adult.isna().any(axis=1).sum())
(ROOT / 'data/dataset_summary.json').write_text(json.dumps(counts, indent=2) + '\n')
print(json.dumps(counts, indent=2))
