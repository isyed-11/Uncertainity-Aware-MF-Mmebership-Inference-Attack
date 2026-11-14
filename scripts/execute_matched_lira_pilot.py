"""Execute only the explicitly authorized matched-size LiRA pilot notebook cell."""
import copy
import os
from pathlib import Path
import nbformat
from nbclient import NotebookClient
ROOT=Path(__file__).resolve().parents[1]
os.chdir(ROOT)
PATH=ROOT/'experiments_2_overfit.ipynb'
notebook=nbformat.read(PATH,as_version=4)
original=next(c for c in notebook.cells if c.id=='exp2-matched-lira-pilot')
execution=nbformat.v4.new_notebook(cells=[copy.deepcopy(original)],metadata=copy.deepcopy(notebook.metadata))
class LoggingClient(NotebookClient):
    def process_message(self,msg,cell,cell_index):
        if msg['header']['msg_type']=='stream':
            print(msg['content']['text'],end='',flush=True)
        return super().process_message(msg,cell,cell_index)
client=LoggingClient(execution,timeout=14400,kernel_name='python3',resources={'metadata':{'path':str(ROOT)}})
try:
    client.execute()
finally:
    original.outputs=execution.cells[0].outputs
    original.execution_count=execution.cells[0].execution_count
    original.metadata=execution.cells[0].metadata
    nbformat.write(notebook,PATH)
print('Matched-size offline LiRA pilot completed and notebook outputs saved.')
