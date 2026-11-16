"""Archive verbose saved streams and compact display; never execute notebook code."""
import json
from pathlib import Path
import re
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / 'exp_3_expanded_missing_feature_baseline.ipynb'
RUN = Path('artifacts/exp_3_2023_2014_100000_nominal_paper_algorithms')
WARNING_START = re.compile(r'(?m)(?=^[^\n]+:\d+: \w*Warning:)')
WARNING = re.compile(r'\b\w*Warning:')


def compact_saved_outputs(notebook):
    archive_dir = RUN / 'logs' / f'saved_output_archive_{uuid4().hex[:8]}'
    seen = set()
    before = after = 0
    for index, cell in enumerate(notebook['cells']):
        streams = [o for o in cell.get('outputs', []) if o.get('output_type') == 'stream']
        if not streams:
            continue
        cell_before = sum(len(''.join(o.get('text', []))) for o in streams)
        archive = ROOT / archive_dir / f'cell_{index}.log'
        archive.parent.mkdir(parents=True, exist_ok=True)
        archive.write_text(''.join(f"[{o.get('name', 'stdout')}]\n{''.join(o.get('text', []))}\n"
                                   for o in streams), encoding='utf-8')
        for output in streams:
            text = ''.join(output.get('text', []))
            before += len(text)
            if output.get('name') == 'stderr':
                keep = []
                for block in WARNING_START.split(text):
                    if WARNING.search(block):
                        key = block.strip()
                        if key in seen:
                            continue
                        seen.add(key)
                    keep.append(block)
                text = ''.join(keep)
            else:
                text = ''.join(line for line in text.splitlines(keepends=True)
                    if not (line.lstrip().startswith(('Attack feature extraction:', 'Reconstruction field ',
                                                     'Fitting ', 'Completion ', 'Audit draw '))
                            or 'reconstruction audit draw ' in line
                            or 'trees fitted; collecting leaf donors' in line
                            or 'reconstruction fitted in ' in line))
            after += len(text)
            output['text'] = text.splitlines(keepends=True)
        cell['outputs'] = [o for o in cell['outputs']
                           if o.get('output_type') != 'stream' or o.get('text')]
        cell_after = sum(len(''.join(o.get('text', []))) for o in streams)
        if cell_after < cell_before:
            cell['outputs'].append(dict(output_type='stream', name='stdout',
                text=[f'Full original progress and warnings archived in {archive_dir / archive.name}\n']))
    return before, after, archive_dir


if __name__ == '__main__':
    notebook = json.loads(NOTEBOOK.read_text(encoding='utf-8'))
    before, after, directory = compact_saved_outputs(notebook)
    NOTEBOOK.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + '\n', encoding='utf-8')
    print(f'Saved stream output compacted: {before:,} -> {after:,} characters. Archive: {directory}')
