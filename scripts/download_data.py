"""Download original public datasets; no experimental splits are made."""
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]

def fetch(url, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        temporary = path.with_suffix(path.suffix + '.part')
        print(f'Downloading {url}', flush=True)
        with urllib.request.urlopen(url, timeout=180) as response, temporary.open('wb') as output:
            while chunk := response.read(1024 * 1024):
                output.write(chunk)
        temporary.replace(path)
    return {'url': url, 'path': str(path.relative_to(ROOT)),
            'bytes': path.stat().st_size,
            'sha256': hashlib.file_digest(path.open('rb'), 'sha256').hexdigest()}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--state', help='ACS state abbreviation, e.g. CA')
    parser.add_argument('--year', type=int, default=2018, choices=[2017, 2018])
    args = parser.parse_args()
    records = []
    adult_dir = ROOT / 'data/raw/uci_adult'
    for name in ['adult.data', 'adult.test', 'adult.names']:
        records.append(fetch('https://archive.ics.uci.edu/ml/machine-learning-databases/adult/' + name, adult_dir / name))
    for name, expected in [('adult.data', 32561), ('adult.test', 16281)]:
        rows = [line for line in (adult_dir / name).read_text().splitlines() if line.strip() and not line.startswith('|')]
        assert len(rows) == expected, (name, len(rows))
        assert all(len(line.split(',')) == 15 for line in rows)
        print(f'{name}: {len(rows):,} records, 14 features + income label', flush=True)
    if args.state:
        state = args.state.lower()
        assert len(state) == 2 and state.isalpha()
        folder = ROOT / f'data/raw/folktables/{args.year}/1-Year'
        archive = folder / f'csv_p{state}.zip'
        records.append(fetch(f'https://www2.census.gov/programs-surveys/acs/data/pums/{args.year}/1-Year/csv_p{state}.zip', archive))
        with zipfile.ZipFile(archive) as zipped:
            assert zipped.testzip() is None
            for member in zipped.infolist():
                if member.filename.lower().endswith('.csv'):
                    target = folder / Path(member.filename).name
                    if not target.exists():
                        with zipped.open(member) as source, target.open('wb') as output:
                            while chunk := source.read(1024 * 1024):
                                output.write(chunk)
                    print(f'Extracted {target.name}', flush=True)
    manifest = ROOT / 'data/download_manifest.json'
    old = json.loads(manifest.read_text()) if manifest.exists() else []
    merged = {item['path']: item for item in old + records}
    manifest.write_text(json.dumps(list(merged.values()), indent=2) + '\n')

if __name__ == '__main__':
    main()
