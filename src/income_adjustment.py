"""Pinned real-income label conversion for the 2014/2023 ACS experiment."""
import numpy as np

CPI_U_ANNUAL = {2014: 236.736, 2023: 304.702}
CPI_SOURCE = 'https://www.bls.gov/cpi/tables/supplemental-files/historical-cpi-u-202401.pdf'
ACS_SOURCE = 'https://www2.census.gov/programs-surveys/acs/tech_docs/pums/accuracy/2023AccuracyPUMS.pdf'


def income_in_base_year(pincp, adjinc, survey_year, base_year=2023):
    """ADJINC first makes survey-year dollars; annual CPI-U changes dollar year.

    ADJINC has six implied decimals. The original nominal ACSIncome eligibility
    filter is retained separately; only the >$50,000 label is changed.
    """
    if survey_year not in CPI_U_ANNUAL or base_year not in CPI_U_ANNUAL:
        raise ValueError('No pinned CPI-U for the requested year.')
    income = np.asarray(pincp, dtype=np.float64)
    adjustment = np.asarray(adjinc, dtype=np.float64)
    if not np.isfinite(income).all() or not np.isfinite(adjustment).all() or (adjustment<=0).any():
        raise ValueError('Eligible income and ADJINC must be finite; ADJINC must be positive.')
    return income*(adjustment/1_000_000)*(CPI_U_ANNUAL[base_year]/CPI_U_ANNUAL[survey_year])
