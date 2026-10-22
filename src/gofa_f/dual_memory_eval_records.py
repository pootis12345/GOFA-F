"""Atomic, independently resumable evaluation outputs with strict input binding."""
from pathlib import Path
import time

import numpy as np

from gofa_f.dual_memory_eval_data import read,seal
from gofa_f.dual_memory_policy import digest
from gofa_f.generative_data import Query,sha
from gofa_f.prefix_cache import atomic_json


class PredictionStore:
    def __init__(self,root):
        self.root=Path(root);self.manifest=read(self.root/'manifest.json');self.binding=sha(self.root/'manifest.json')
        self.keys={Query(**q).key for q in self.manifest['queries']}

    def identity(self,row,item,model):
        if Query(**row).key not in self.keys:raise ValueError('Prediction query is not sealed')
        deployment=self.manifest['deployment']
        model_hash=deployment['weights_sha256'] if model=='native' else deployment['expert']['checkpoint_sha256'] if model=='enhanced' else \
            deployment['gate']['sha256'] if model=='fusion' else deployment['ufg'][f"{row['length']}-{row['horizon']}"]['checkpoint_sha256']
        return dict(binding=self.binding,query=row,input_sha256=item.identity,model=model,model_sha256=model_hash,
            deployment_sha256=digest(deployment),sampling=self.manifest['configuration']['sampling'] if model=='native' else None,
            execution=self.manifest['prediction_execution'])

    def path(self,row,model):return self.root/'predictions'/model/(digest(Query(**row).key)+'.json')

    def get(self,row,item,model):
        path=self.path(row,model)
        if not path.exists():return None
        value=read(path)
        if value['identity']!=self.identity(row,item,model):raise ValueError('Prediction input/model binding changed')
        p=np.asarray(value['prediction'],np.float64)
        if p.shape!=(row['horizon'],) or not np.isfinite(p).all():raise ValueError('Invalid saved prediction')
        return value

    def save(self,row,item,model,prediction,seconds,**fields):
        p=np.asarray(prediction,np.float64)
        if p.shape!=(row['horizon'],) or not np.isfinite(p).all():raise ValueError('Incomplete/nonfinite forecast cannot be saved')
        if self.get(row,item,model) is not None:raise ValueError('A completed prediction cannot be replaced')
        value=dict(identity=self.identity(row,item,model),prediction=p.tolist(),seconds=float(seconds),
            real_nodes=len(item.node_ids),node_ids=list(item.node_ids),sampling_audit=item.audit,**fields)
        path=self.path(row,model);path.parent.mkdir(parents=True,exist_ok=True);atomic_json(path,value)
        return value

    def trajectory(self,row,item,sample,generate):
        from gofa_f.short_sampling_diagnostic import sample_seed
        identity=self.identity(row,item,'native');seed=sample_seed(row,sample)
        path=self.root/'trajectories'/(digest(Query(**row).key)+f'-{sample}.json')
        if path.exists():
            value=read(path)
            if value['identity']!=identity or value['seed']!=seed:raise ValueError('Native trajectory binding changed')
        else:
            started=time.monotonic();result=generate(seed)
            value=dict(identity=identity,seed=seed,sample=sample,result=result,seconds=time.monotonic()-started)
            path.parent.mkdir(parents=True,exist_ok=True);atomic_json(path,value)
        result=value['result'];p=np.asarray(result['prediction'])
        if len(result['numeric_ids'])!=row['horizon'] or p.shape!=(row['horizon'],) or not np.isfinite(p).all():
            raise ValueError('Incomplete Native number slots or inverse transform')
        return value,path

    def complete(self,model,inputs):
        shards={}
        for row in self.manifest['queries']:
            item,_=inputs.get(row);record=self.get(row,item,model)
            if record is None:raise ValueError('Cannot seal an incomplete model group')
            path=self.path(row,model);shards[str(path.relative_to(self.root))]=sha(path)
            if model=='native':
                paths=[self.root/'trajectories'/(digest(Query(**row).key)+f'-{i}.json') for i in range(8)]
                values=[self.trajectory(row,item,i,lambda _:(_ for _ in ()).throw(ValueError('Missing saved Native trajectory')))[0] for i in range(8)]
                median=np.median(np.asarray([v['result']['prediction'] for v in values],np.float64),axis=0)
                if not np.array_equal(median,record['prediction']):raise ValueError('Native median no longer matches eight saved paths')
                for path in paths:shards[str(path.relative_to(self.root))]=sha(path)
        value=dict(binding=self.binding,complete=True,queries=len(self.manifest['queries']),model=model,
            input_index_sha256=sha(self.root/'inputs/index.json'),shards=shards)
        path=self.root/f'{model}_complete.json';seal(path,value);return path
