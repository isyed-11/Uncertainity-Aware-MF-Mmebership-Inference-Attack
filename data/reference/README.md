# Census occupation conversion references

Source: https://www.census.gov/data/tables/time-series/demo/industry-occupation/acs-tp78.html

- `table-e1_e3.xlsx`: https://www2.census.gov/programs-surveys/demo/tables/industry-occupation/time-series/tp78/table-e1_e3.xlsx
  Table E1 gives total 2010-to-2018 conversion rates. E2/E3 are not used.
- `table-f1.xlsx`: https://www2.census.gov/programs-surveys/demo/tables/industry-occupation/time-series/tp78/table-f1.xlsx
  Direct code changes, including merged occupations.

ACS 2017 used the 2010 classification. `src/occupation_mapping.py` uses these
change tables to recode only 2017 OCCP into the 2018 classification. Direct
matches use F1; E1 splits take the maximum-rate destination, with the lower code
breaking ties. Codes absent from both change tables retain their original value.
The notebook reports ambiguous assignments and categories unseen by the fitted
2018 encoder. Conversion rates describe populations, not a known individual's
occupation: modal mapping is an approximation and can distort category shares.
No income adjustments or changes to 2018 data/preprocessing/model are made.
