"""Run the four-year, complete-feature baseline notebook after explicit approval.

Interrupt an older busy session of this same notebook through local Jupyter,
then use the normal executor, which saves outputs after each cell. No process
is killed and no experimental settings are changed by this launcher.
"""
import ast
import json
from pathlib import Path
import runpy
import sys
import time
from urllib.error import URLError
from urllib.parse import urljoin, urlparse
from urllib.request import Request, urlopen

from jupyter_server.serverapp import list_running_servers

ROOT = Path(__file__).resolve().parents[1]
NAME = 'exp_3_expanded dataset.ipynb'


def request(server, endpoint, *, method='GET'):
    base = server['url']
    parsed = urlparse(base)
    if parsed.hostname not in {'localhost', '127.0.0.1', '::1', '0.0.0.0'}:
        raise ValueError('Only a local Jupyter server can be contacted.')
    if parsed.hostname == '0.0.0.0':
        base = base.replace('0.0.0.0', '127.0.0.1', 1)
    token = server.get('token', '')
    headers = {'Authorization': f'token {token}'} if token else {}
    req = Request(urljoin(base, endpoint), headers=headers, method=method,
                  data=b'' if method == 'POST' else None)
    with urlopen(req, timeout=10) as response:
        body = response.read()
        return json.loads(body) if body else None


def preflight():
    notebook = json.loads((ROOT / NAME).read_text(encoding='utf-8'))
    values = {}
    for cell in notebook['cells']:
        if cell['cell_type'] != 'code':
            continue
        for node in ast.parse(''.join(cell['source'])).body:
            if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                try:
                    values[node.targets[0].id] = ast.literal_eval(node.value)
                except (ValueError, TypeError):
                    pass
    assert values['N_SHADOW_MODELS'] == 16
    assert values['TARGET_YEARS'] == (2023, 2024)
    assert values['AUXILIARY_YEARS'] == (2014, 2018)
    assert values['INCLUDE_BLIND_WALK'] is True
    assert 'UNCERTAINTY_K' not in values and 'MISSING_MASKS' not in values
    print('Confirmed: targets 2023/2024, auxiliary pools 2014/2018, 16 shadows per pool; baseline MIA only.', flush=True)


def interrupt_previous_session():
    reached = False
    for server in list_running_servers():
        try:
            sessions = request(server, 'api/sessions')
        except (URLError, OSError):
            continue
        reached = True
        for session in sessions:
            if Path(session.get('path', '')).name != NAME:
                continue
            kernel = session['kernel']
            print(f'Existing matching notebook kernel: {kernel["execution_state"]}', flush=True)
            if kernel['execution_state'] != 'busy':
                continue
            endpoint = f'api/kernels/{kernel["id"]}'
            request(server, endpoint + '/interrupt', method='POST')
            deadline = time.monotonic() + 60
            while time.monotonic() < deadline:
                if request(server, endpoint)['execution_state'] != 'busy':
                    print('Older notebook computation interrupted; kernel and checkpoints retained.', flush=True)
                    break
                time.sleep(1)
            else:
                raise RuntimeError('The older notebook did not become idle; reduced run was not launched.')
    if not reached:
        raise RuntimeError('Could not inspect local Jupyter sessions; refusing to start a competing run.')


if __name__ == '__main__':
    preflight()
    interrupt_previous_session()
    sys.argv = ['execute_expanded_notebook.py', '--notebook', NAME]
    runpy.run_path(str(ROOT / 'scripts/execute_expanded_notebook.py'), run_name='__main__')
