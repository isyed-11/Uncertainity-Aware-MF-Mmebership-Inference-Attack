"""Execute Experiment 3 only; running this script requires explicit authorization."""
import argparse
import os
from pathlib import Path
import time

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument('--notebook', required=True)
args = parser.parse_args()
PATH = (ROOT / args.notebook).resolve()
if PATH.parent != ROOT or PATH.suffix != '.ipynb':
    raise ValueError('Expected a notebook in the project root.')
os.chdir(ROOT)
notebook = nbformat.read(PATH, as_version=4)
started = time.monotonic()


class LoggingClient(NotebookClient):
    def process_message(self, msg, cell, cell_index):
        if msg['header']['msg_type'] == 'stream':
            print(msg['content']['text'], end='', flush=True)
        return super().process_message(msg, cell, cell_index)


def cell_start(cell, cell_index, **kwargs):
    if cell.cell_type == 'code':
        print(f'\n--- Executing cell {cell_index}: {cell.source.splitlines()[0]} ---', flush=True)


def cell_finished(cell, cell_index, **kwargs):
    nbformat.write(notebook, PATH)
    if cell.cell_type == 'code':
        print(f'Cell {cell_index} saved. Elapsed {time.monotonic() - started:.1f}s', flush=True)


client = LoggingClient(notebook, timeout=7200, kernel_name='python3',
                       resources={'metadata': {'path': str(ROOT)}},
                       on_cell_start=cell_start, on_cell_executed=cell_finished)
try:
    client.execute()
finally:
    nbformat.write(notebook, PATH)
print(f'Completed all cells in {time.monotonic() - started:.1f}s', flush=True)
