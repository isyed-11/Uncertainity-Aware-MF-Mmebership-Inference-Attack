"""Run only the two external-year evaluation cells after explicit user approval."""
import copy
import os
from pathlib import Path
import time

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / 'experiments_1_2023.ipynb'
os.chdir(ROOT)
notebook = nbformat.read(PATH, as_version=4)
ids = ['external-2024-25k', 'external-2018-25k']
selected = [next(c for c in notebook.cells if c.id == cell_id) for cell_id in ids]
execution = nbformat.v4.new_notebook(cells=copy.deepcopy(selected), metadata=copy.deepcopy(notebook.metadata))
started = time.monotonic()


def save_outputs():
    for original, executed in zip(selected, execution.cells):
        original.outputs = executed.outputs
        original.execution_count = executed.execution_count
        original.metadata = executed.metadata
    nbformat.write(notebook, PATH)


def cell_finished(cell, cell_index, **kwargs):
    save_outputs()
    print(f'Completed {cell.id}; elapsed {time.monotonic() - started:.1f}s', flush=True)


client = NotebookClient(execution, timeout=600, kernel_name='python3',
    resources={'metadata': {'path': str(ROOT)}}, on_cell_executed=cell_finished)
try:
    client.execute()
finally:
    save_outputs()
print('Both external-year evaluation cells completed; target and shadows were not retrained.', flush=True)
