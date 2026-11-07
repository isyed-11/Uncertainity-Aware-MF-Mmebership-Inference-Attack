"""2017-only occupation conversion using official Census TP78 tables."""
from pathlib import Path
import zipfile
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

NS = {'m': 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}


def read_reference_sheets(path):
    """Read Census XLSX values without adding an Excel engine dependency."""
    with zipfile.ZipFile(path) as archive:
        strings = [''.join(node.itertext()) for node in
                   ET.fromstring(archive.read('xl/sharedStrings.xml')).findall('m:si', NS)]
        sheets = {}
        for name in archive.namelist():
            if name.startswith('xl/worksheets/sheet') and name.endswith('.xml'):
                rows = []
                for row in ET.fromstring(archive.read(name)).findall('m:sheetData/m:row', NS):
                    values = {}
                    for cell in row:
                        value = cell.findtext('m:v', default='', namespaces=NS)
                        if cell.attrib.get('t') == 's':
                            value = strings[int(value)]
                        column = ''.join(char for char in cell.attrib['r'] if char.isalpha())
                        values[column] = value
                    rows.append(values)
                sheets[name] = rows
        return sheets


def load_occupation_crosswalk(directory='data/reference'):
    """Combine E1 conversion rates with F1 direct matches (2010 -> 2018).

    The 2010 classification was used in ACS 2017. Unlisted codes retain their
    identity because these tables document changes, not the complete code lists.
    One-to-many changes use the most likely destination, not an exact recovery.
    """
    directory = Path(directory)
    rates = read_reference_sheets(directory / 'table-e1_e3.xlsx')['xl/worksheets/sheet1.xml']
    direct = read_reference_sheets(directory / 'table-f1.xlsx')['xl/worksheets/sheet1.xml']
    candidates = {}
    old = None
    for row in rates:
        if row.get('A', '').strip().isdigit():
            old = int(row['A'])
        if old is not None and row.get('C', '').strip().isdigit() and row.get('E', ''):
            candidates.setdefault(old, []).append((int(row['C']), float(row['E'])))
    mapping = {}
    destination = None
    for row in direct:
        if row.get('C', '').strip().isdigit():
            destination = int(row['C'])
        if row.get('A', '').strip().isdigit() and destination is not None:
            mapping[int(row['A'])] = {
                'destination': destination, 'status': 'direct', 'conversion_rate': 1.0,
                'candidate_count': 1,
            }
    for old, options in candidates.items():
        total = sum(rate for _, rate in options)
        if any(rate < 0 or rate > 1 for _, rate in options) or not np.isclose(total, 1, atol=0.002):
            raise ValueError(f'Invalid Census conversion rates for code {old}: {options}')
        destination, rate = sorted(options, key=lambda pair: (-pair[1], pair[0]))[0]
        mapping[old] = {
            'destination': destination,
            'status': 'ambiguous_modal' if len(options) > 1 else 'direct',
            'conversion_rate': rate, 'candidate_count': len(options),
        }
    if not candidates or not mapping:
        raise ValueError('Census reference tables did not contain expected conversion rows.')
    return mapping


def map_2017_occupations(frame, *, reference_directory='data/reference'):
    """Return a COPY of 2017 records with OCCP converted, plus a mapping audit.

    Do not call on 2018 records. All other columns, row order, IDs, and nominal
    income labels are retained. No model, encoder, or data file is modified.
    """
    crosswalk = load_occupation_crosswalk(reference_directory)
    codes = pd.to_numeric(frame['OCCP'], errors='raise')
    if codes.isna().any() or not np.isfinite(codes).all() or (codes != codes.astype(int)).any():
        raise ValueError('Expected finite, nonmissing integer occupation codes.')
    audit = []
    replacements = {}
    for code, count in codes.astype(int).value_counts().sort_index().items():
        entry = crosswalk.get(code, {
            'destination': code, 'status': 'unchanged_not_listed',
            'conversion_rate': 1.0, 'candidate_count': 1,
        })
        replacements[code] = entry['destination']
        audit.append({'occp_2017': code, 'occp_2018': entry['destination'],
                      'records': int(count), **{key: value for key, value in entry.items()
                                               if key != 'destination'}})
    mapped = frame.copy()
    mapped['OCCP'] = codes.astype(int).map(replacements)
    return mapped, pd.DataFrame(audit)
