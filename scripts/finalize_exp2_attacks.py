"""Finish the authorized Experiment 2 run from saved margins after a numerical fix."""
from pathlib import Path
import copy
import os
import nbformat
from nbclient import NotebookClient
ROOT=Path(__file__).resolve().parents[1]
os.chdir(ROOT)
path=ROOT/'experiments_2_overfit.ipynb'
notebook=nbformat.read(path,as_version=4)
final=next(c for c in notebook.cells if c.id=='exp2-attack-comparison')
setup=nbformat.v4.new_code_cell("""
from pathlib import Path
import json
import pandas as pd
from IPython.display import display
from src.calibrated_attack import refit_saved_reference_attacks
REPAIR_DIR=Path('artifacts/exp2_attack_repair')
saved=json.loads((REPAIR_DIR/'summary.json').read_text())
overfit_metrics=saved['target_metrics']
overfit_metadata=saved['target_training']
OVERFIT_TRAIN_SIZE=saved['target_train_size']
target_attack_metrics=saved['baseline_target_mia']
attack_result={'metrics':saved['baseline_auxiliary_metrics']}
reference_results=refit_saved_reference_attacks(REPAIR_DIR/'references')
print('Refitted attacks with stable losses; target and references unchanged.')
""")
execution=nbformat.v4.new_notebook(cells=[setup,copy.deepcopy(final)],metadata=copy.deepcopy(notebook.metadata))
client=NotebookClient(execution,timeout=1200,kernel_name='python3',resources={'metadata':{'path':str(ROOT)}})
client.execute()
final.outputs=execution.cells[1].outputs
final.execution_count=execution.cells[1].execution_count
nbformat.write(notebook,path)
print('Updated comparison cell and summary from saved features.')
