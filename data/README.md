# Project datasets

Target data: California ACS 2018, 1-Year person PUMS, using the Folktables
ACSIncome task. Auxiliary adversary data: UCI Adult (1994 census data).
The current user choice replaces the GSS target described in the proposal.
California ACS 2017 is kept separately for generalization across years.

## Files

- `raw/folktables/2018/1-Year/`: original Census ZIP and extracted person CSV.
- `raw/folktables/2017/1-Year/`: original 2017 Census ZIP and person CSV.
- `processed/acs_income_ca_2017.csv`: 2017 ACSIncome using the same task
  filter, feature columns, and binary label as 2018.
- `dataset_summary_2017.json`: 2017 record counts and feature names.
- `raw/uci_adult/`: original `adult.data`, `adult.test`, and `adult.names`.
- `processed/acs_income_ca_2018.csv`: standard ACSIncome features and binary
  `income_gt_50k` label. Folktables filters age >16, income >100, usual weekly
  work hours >0, and person weight >=1; label is personal income >$50,000.
- `processed/adult_train.csv` and `processed/adult_test.csv`: original UCI
  partitions with headers, whitespace trimmed, missing `?` values represented
  as empty cells, and binary `income_gt_50k` labels. Test label periods removed.
- `download_manifest.json`: source URLs, sizes, and SHA-256 checksums.
- `dataset_summary.json`: verified record counts and feature names.

ACSIncome uses AGEP, COW, SCHL, MAR, OCCP, POBP, RELP, WKHP, SEX, RAC1P.
RAC1P is included among the features; the task's group variable is also RAC1P.
Categorical codes remain unencoded. No scaling, imputation, model fitting,
or experimental train/test/MIA splitting has been performed.
Adult's original partitions are both auxiliary data, not target test data.
ACS and Adult feature codes differ and require explicit harmonization before
shadow models can share the target model's input schema.
Stored CSVs retain their original occupation codes. The notebook's 2017 test
section now maps a copy of 2017 OCCP to 2018 codes using Census TP78 E1/F1
references in `data/reference/`. Ambiguous splits use the highest conversion
rate and are reported in an audit; they are approximations. The 2018 data,
preprocessor, and model are unchanged. No inflation adjustment is applied;
the income label follows Folktables' nominal $50,000 threshold in each year.
Use the held-out 2018 partition for the main MIA non-member evaluation;
2017 is a separate generalization evaluation set.

## Reproduce (from project root, PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-data.txt
python scripts/download_data.py --state CA
.\.venv\Scripts\python.exe scripts/prepare_data.py
python scripts/download_data.py --state CA --year 2017
.\.venv\Scripts\python.exe scripts/prepare_data.py --year 2017
```

## Sources

- Folktables: https://github.com/socialfoundations/folktables
- Census source: https://www2.census.gov/programs-surveys/acs/data/pums/2018/1-Year/csv_pca.zip
- Census 2017 source: https://www2.census.gov/programs-surveys/acs/data/pums/2017/1-Year/csv_pca.zip
- UCI Adult: https://archive.ics.uci.edu/dataset/2/adult

Raw and processed datasets are excluded from Git; metadata and scripts are retained.

