# Experiment 3 expanded features

The 2023 target and 2014 auxiliary models use 48 common fields, defined in
`src/expanded_data.py` as `EXPANDED_FEATURES`. The notebook displays and saves
the full inventory with types, missingness and training cardinality.

Base fields: age, class of worker, education level, marital status, occupation,
place of birth, usual weekly work hours, sex and race.
Additional fields cover citizenship, employment, industry, disability, health
insurance, language, commuting, degree fields, ancestry, military service and
marriage history. Legacy RELP, WKW and JWTR are absent in 2023 and excluded.

AGEP, WKHP, JWMNP and JWRIP are numeric. All other fields are categorical.
Numeric missing values use training medians and missingness indicators;
categorical missing values use a dedicated -1 category. Unknown categories
are ignored by the fitted one-hot encoder. Each shadow fits its own transform.
Category codes remain native to their survey year, including occupation codes.

PINCP is used only for eligibility and the nominal >$50,000 label. PWGTP is
used only for eligibility. Income-derived fields, IDs, weights and allocation
flags are excluded from predictors. No inflation adjustment is applied.

The target retains 50,000 training records and trains for 100 fixed epochs.
Sixteen diverse shadow models use only 2014 records. Four output-based MIA
features feed the random forest attack: class-scaled logit, blind walk,
entropy and cross-entropy. Auxiliary record IDs stay grouped across attack
partitions; target membership labels are used only for final evaluation.
