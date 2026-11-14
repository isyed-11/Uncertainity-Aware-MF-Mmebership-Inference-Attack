"""Completed-stage checkpoints for authorized Experiment 3 runs."""
import hashlib
import json
from pathlib import Path
import joblib
import pandas as pd
from .training import train_target_model
from .shadow_models import ALGORITHMS, train_shadow_model


def code_signature():
    h=hashlib.sha256()
    for name in ['training.py','expanded_data.py','shadow_models.py','income_adjustment.py']:
        h.update((Path(__file__).parent/name).read_bytes())
    return h.hexdigest()


def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.part')
    joblib.dump(value,temporary)
    temporary.replace(path)


def cached_target_train(arrays,config,run_dir):
    h=hashlib.sha256(json.dumps(dict(config=config,code=code_signature()),sort_keys=True).encode())
    for name in ['train','validation']:
        values=arrays[name]
        for a in [values['X'].data,values['X'].indices,values['X'].indptr,values['y']]:
            h.update(a.tobytes())
        h.update(str(values['X'].shape).encode())
        h.update(json.dumps(values['record_ids'].tolist()).encode())
    signature=h.hexdigest();path=Path(run_dir)/'checkpoints/target.joblib'
    if path.exists():
        result=joblib.load(path)
        if result['signature']!=signature:raise ValueError('Target checkpoint configuration mismatch.')
        print('Loaded completed target checkpoint.',flush=True)
        return result['model'],result['history'],result['metadata']
    model,history,metadata=train_target_model(arrays['train']['X'],arrays['train']['y'],
        arrays['validation']['X'],arrays['validation']['y'],**config)
    save(path,dict(signature=signature,model=model,history=history,metadata=metadata))
    print('Completed target checkpoint saved.',flush=True)
    return model,history,metadata


def cached_expanded_shadows(auxiliary,*,seed,n_models,run_dir,
                            algorithms=ALGORITHMS,schema='expanded_acs'):
    if not algorithms or n_models < 1:
        raise ValueError('Provide shadow algorithms and a positive model count.')
    frame_hash=hashlib.sha256(pd.util.hash_pandas_object(auxiliary,index=True).to_numpy().tobytes()).hexdigest()
    results={}
    for index in range(n_models):
        algorithm=algorithms[index%len(algorithms)]
        name=f'{algorithm}_{index+1:02d}'
        signature=json.dumps(dict(data=frame_hash,code=code_signature(),algorithm=algorithm,seed=seed+10*index,schema=schema),sort_keys=True)
        path=Path(run_dir)/f'checkpoints/{name}.joblib'
        if path.exists():
            cached=joblib.load(path)
            if cached['signature']!=signature:raise ValueError(f'Shadow checkpoint mismatch: {name}')
            result=cached['result'];print(f'Loaded shadow {index+1}/{n_models}: {name}',flush=True)
        else:
            print(f'Training shadow {index+1}/{n_models}: {name}',flush=True)
            result=train_shadow_model(auxiliary,algorithm,seed=seed+10*index,schema=schema)
            result['attack_data']['shadow_model']=name
            save(path,dict(signature=signature,result=result))
            print('Completed shadow checkpoint saved:',name,flush=True)
        results[name]=result
    return results
