# Project datasets

Experiment 3 uses California ACS 2023 1-Year person PUMS for the target and
California ACS 2014 exclusively for auxiliary shadow training (16 models).
The 2017 raw/processed datasets and manifest entry have been deleted.
2018 and Adult data remain available for the earlier experiments.

Experiment 3 uses 48 common native fields: demographics, education, employment,
occupation, industry, citizenship, disability, health insurance, language,
commuting, degree field, ancestry, military and marriage history.
RELP, WKW and JWTR are omitted because those legacy fields are absent in 2023.
Income amounts, poverty ratios, identifiers and weights are excluded as predictors.
PINCP defines eligibility and the nominal >$50,000 label; PWGTP defines eligibility.
No inflation adjustment is applied. Native categorical codes are encoded separately
for each model, without assuming cross-year semantic equivalence.

Raw CSV paths:
- `raw/folktables/2023/1-Year/psam_p06.csv`
- `raw/folktables/2014/1-Year/ss14pca.csv`

Preprocessing fits training records only; target and auxiliary row IDs include
the survey year. Attack splits group repeated auxiliary record IDs.
Downloads and SHA-256 checksums are recorded in `download_manifest.json`.
Future cell and experiment execution requires asking the user first.

## Reproduce downloads

```powershell
.\.venv\Scripts\python.exe scripts/download_data.py --state CA --year 2023
.\.venv\Scripts\python.exe scripts/download_data.py --state CA --year 2014
```

Sources: https://www2.census.gov/programs-surveys/acs/data/pums/2023/1-Year/csv_pca.zip
and https://www2.census.gov/programs-surveys/acs/data/pums/2014/1-Year/csv_pca.zip

Raw and processed datasets are excluded from Git.
