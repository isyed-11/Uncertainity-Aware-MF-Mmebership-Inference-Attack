"""Copy Experiment 1 for specified ACS years without executing any cells."""
import argparse
import copy
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def build(year, auxiliary, n_shadows):
    reference = json.loads((ROOT / 'experiments_1.ipynb').read_text(encoding='utf-8'))
    cells = copy.deepcopy(reference['cells'][:10] + reference['cells'][13:17])
    for cell in cells:
        source = ''.join(cell['source']).replace('2018', str(year))
        cell['source'] = source.splitlines(keepends=True)
        if cell['cell_type'] == 'code':
            cell['execution_count'] = None
            cell['outputs'] = []
    cells[0]['source'] = [f'# Experiment 1 baseline: California ACS {year}\n',
        'Same standard MLP, 10 predictors, split sizes, seed, early stopping,\n',
        'four attack features and random forest attack as Experiment 1.\n',
        'The relationship field uses native-year codes (RELP in 2014; RELSHIPP in 2023).\n',
        'Income eligibility and nominal >$50,000 label follow ACSIncome; no inflation adjustment.\n',
        f'Auxiliary source: {auxiliary}; {n_shadows} diverse shadows.\n',
        'Target and auxiliary preprocessing fit their own training records only.\n',
        'The user authorized this run; ask before any future experiment run.\n',
        'Census documentation: https://www.census.gov/programs-surveys/acs/microdata/documentation.html\n']
    source = ''.join(cells[1]['source'])
    source = source.replace('import pandas as pd', 'import pandas as pd\nimport numpy as np\nimport torch\ntorch.set_num_threads(4)')
    source = source.replace('from src.data import load_acs_income, split_acs_data, preprocess_partitions, save_partition_ids',
        'from src.data import split_acs_data, save_partition_ids\nfrom src.recent_data import load_recent_acs, preprocess_recent_partitions')
    source += f"\nTARGET_YEAR = {year}\nAUXILIARY_SOURCE = {auxiliary!r}\nN_SHADOW_MODELS = {n_shadows}\nRUN_DIR = Path('artifacts/experiment1_{year}_aux{auxiliary}_{n_shadows}shadows')\n"
    cells[1]['source'] = source.splitlines(keepends=True)
    cells[3]['source'] = ''.join(cells[3]['source']).replace('load_acs_income(DATA_PATH)', 'load_recent_acs(TARGET_YEAR)').splitlines(keepends=True)
    cells[5]['source'] = ''.join(cells[5]['source']).replace('preprocess_partitions(partitions)', 'preprocess_recent_partitions(partitions)').splitlines(keepends=True)
    if auxiliary == '2014':
        source = ''.join(cells[11]['source']).replace('adult_auxiliary = load_adult_data()', 'adult_auxiliary = load_recent_acs(2014)')
        source = source.replace('train_shadow_models(adult_auxiliary, seed=42)', "train_shadow_models(adult_auxiliary, seed=42, schema='acs')")
        cells[11]['source'] = source.splitlines(keepends=True)
        cells[12]['source'] = ''.join(cells[12]['source']).replace('adult_for_rf_queries = load_adult_data()', 'adult_for_rf_queries = adult_auxiliary').splitlines(keepends=True)
        for cell in cells[10:]:
            cell['source'] = [line.replace('Adult', 'ACS 2014 auxiliary') for line in cell['source']]
    checkpoint = """
RUN_DIR.mkdir(parents=True, exist_ok=True)
save_target_model(model, metadata, RUN_DIR / 'target_model.pt')
joblib.dump(preprocessor, RUN_DIR / 'preprocessor.joblib')
save_partition_ids(partitions, RUN_DIR)
"""
    cells[7]['source'] = (''.join(cells[7]['source']) + checkpoint).splitlines(keepends=True)
    cells[11]['source'] = ''.join(cells[11]['source']).replace('train_shadow_models(adult_auxiliary, seed=42)', 'train_shadow_models(adult_auxiliary, seed=42, n_models=N_SHADOW_MODELS)').replace("seed=42, schema='acs')", "seed=42, schema='acs', n_models=N_SHADOW_MODELS)").splitlines(keepends=True)
    for cell in cells[10:]:
        cell['source'] = [line.replace('five', str(n_shadows)) for line in cell['source']]
    # Audit each fitted shadow before expensive feature extraction.
    source = ''.join(cells[11]['source'])
    source += """
from src.paper_attack_features import make_probability_oracle
for name, result in shadow_results.items():
    rows = result['attack_data'].iloc[:32]
    query = make_probability_oracle(result['model'], result['preprocessor'], schema=result['schema'])
    np.testing.assert_allclose(query(adult_auxiliary.loc[rows['record_id']]),
        rows[['p_income_le_50k', 'p_income_gt_50k']].to_numpy(), rtol=1e-5, atol=1e-6)
print('Probability consistency verified for all', N_SHADOW_MODELS, 'shadows.')
"""
    cells[11]['source'] = source.splitlines(keepends=True)
    final = """
RUN_DIR.mkdir(parents=True, exist_ok=True)
pd.DataFrame(history).to_csv(RUN_DIR / 'target_history.csv', index=False)
summarize_shadow_models(shadow_results).to_csv(RUN_DIR / 'shadow_metrics.csv')
joblib.dump(attack_result['classifier'], RUN_DIR / 'attack_classifier.joblib')
summary = dict(target_year=TARGET_YEAR, auxiliary_source=AUXILIARY_SOURCE,
    n_shadow_models=len(shadow_results), original_features=list(preprocessor.feature_names_in_),
    encoded_features=arrays['train']['X'].shape[1], split_config=SPLIT_CONFIG,
    partition_counts={name:len(part) for name,part in partitions.items()}, training=metadata,
    target_metrics=metrics, target_mia_metrics=target_attack_metrics,
    shadow_attack_metrics=attack_result['metrics'], target_query_diagnostics=rf_target_walk_diagnostics,
    shadow_query_diagnostics=rf_shadow_walk_diagnostics,
    oracle_consistency_verified=True,
    source_sha256=hashlib.file_digest(Path(f'data/raw/folktables/{TARGET_YEAR}/1-Year/csv_pca.zip').open('rb'), 'sha256').hexdigest())
(RUN_DIR / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
print('Results saved to', RUN_DIR)
"""
    cells.append(dict(cell_type='code', id=f'baseline-{year}-save', metadata={},
        source=final.splitlines(keepends=True), execution_count=None, outputs=[]))
    if year == 2023 and auxiliary == '2014' and n_shadows == 16:
        import sys
        sys.path.insert(0, str(ROOT))
        from src.recent_data import external_evaluation_cells
        cells.extend(external_evaluation_cells())
    reference['cells'] = cells
    path = ROOT / f'experiments_1_{year}.ipynb'
    path.write_text(json.dumps(reference, indent=1) + '\n', encoding='utf-8')
    print(path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--years', nargs='+', type=int, choices=[2014, 2023], required=True)
    parser.add_argument('--auxiliary', choices=['adult', '2014'], default='adult')
    parser.add_argument('--shadows', type=int, default=5)
    args = parser.parse_args()
    for year in args.years:
        build(year, args.auxiliary, args.shadows)
