"""Shared source inputs/labels; UFG consumes identical serialized real rosters."""
from collections import OrderedDict
from dataclasses import asdict
import json
from pathlib import Path
import time

import numpy as np
import torch

from gofa_f.dual_memory_data import GROUPS, memory_input, all_node_labels
from gofa_f.dual_memory_policy import digest
from gofa_f.generative_data import Query, Series, sha
from gofa_f.prefix_cache import atomic_json


def atomic_torch(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    part=path.with_suffix(path.suffix+'.part');torch.save(value,part);part.replace(path)


class SourceStore:
    def __init__(self,manifest,cache,held_group=None,confirmation=False):
        self.manifest=manifest;self.held_group=held_group;self.confirmation=confirmation
        self.out_of_fit_unlocked=False
        self.directory=Path(cache);self.directory.mkdir(parents=True,exist_ok=True)
        self.sources={};self.binding=digest(dict(source_hashes=manifest['source_hashes'],
            metadata_hashes=manifest['metadata_hashes'],partitions=manifest['partitions'],sampler=manifest['sampler']))
        self.index={};self.shards=OrderedDict();self.shard_hashes={}
        self.allowed={split:{Query(**q).key for d in manifest['queries'].values() for v in d.values() for q in v[split]}
                      for split in ('train','development','confirmation')}
        index=self.directory/'index.json'
        if index.exists():
            saved=json.loads(index.read_text())
            if saved['binding']!=self.binding:raise ValueError('Source input cache binding changed')
            self.index=saved['index'];self.shard_hashes=saved['shard_sha256']

    def source(self,name):
        if name not in self.sources:
            path=Path(self.manifest['source_directory'])/(name+'.npz')
            if sha(path)!=self.manifest['source_hashes'][name] or sha(path.with_suffix('.json'))!=self.manifest['metadata_hashes'][name]:
                raise ValueError('Audited source changed')
            value=Series(path)
            if name.startswith('pems04'):value.external=np.maximum(value.external,value.external.T)
            self.sources[name]=value
        return self.sources[name]

    def _verify(self,row,split,role):
        q=Query(**row)
        if q.key not in self.allowed[split]:raise ValueError('Query outside sealed source list')
        if split=='confirmation' and not self.confirmation:raise ValueError('Source confirmation remains locked')
        if role=='out_of_fit' and (split!='development' or not self.out_of_fit_unlocked or GROUPS[q.source]!=self.held_group):
            raise ValueError('Out-of-fit scoring requires a locked training-source best')
        if self.held_group is not None and GROUPS[q.source]==self.held_group and role!='out_of_fit':
            raise ValueError('Held source labels cannot enter training or checkpoint selection')
        return q

    def permit_out_of_fit(self,selection_path):
        value=json.loads(Path(selection_path).read_text())
        if value['excluded_group']!=self.held_group or value['held_source_used_for_checkpoint_selection'] or value['target_labels_used']:
            raise ValueError('Invalid source-group selection seal')
        self.out_of_fit_unlocked=True

    def get(self,row,split,role='fit'):
        q=self._verify(row,split,role)
        if q.key in self.index:
            name,offset=self.index[q.key]
            if name not in self.shards:
                if sha(self.directory/name)!=self.shard_hashes[name]:raise ValueError('Serialized source inputs changed')
                value=torch.load(self.directory/name,map_location='cpu',weights_only=False)
                if value['binding']!=self.binding:raise ValueError('Serialized source roster binding changed')
                self.shards[name]=value['records']
                while len(self.shards)>128:self.shards.popitem(last=False)
            records=self.shards.pop(name);self.shards[name]=records
            record=records[offset]
            if record['query']!=row:raise ValueError('Serialized source query mapping changed')
            return record['item'],record['target'],record['observed']
        data=self.source(q.source)
        partition={'train':'train','development':'validation','confirmation':'confirmation'}[split]
        item=memory_input(data,q,self.manifest['partitions'][q.source][partition])
        target,observed=all_node_labels(data,q,item)
        return item,target,observed&item.present[:,None]

    def build(self,check=lambda:None,progress=lambda **_:None):
        if self.held_group is not None or self.confirmation:raise ValueError('Only source train/development cache can be prepared')
        rows=[(split,q) for split in ('train','development') for name in GROUPS
            for shape,values in self.manifest['queries'][name].items() for q in values[split]]
        index={};hashes={};start=time.monotonic()
        for shard,start_row in enumerate(range(0,len(rows),128)):
            check();name=f'inputs-{shard:04d}.pt';path=self.directory/name
            own=rows[start_row:start_row+128]
            if path.exists():
                seal=json.loads(path.with_suffix('.sha.json').read_text())
                if seal['binding']!=self.binding or seal['sha256']!=sha(path):raise ValueError('Partial source input shard changed')
                value=torch.load(path,map_location='cpu',weights_only=False)
                if value['binding']!=self.binding or [r['query'] for r in value['records']]!=[q for _,q in own]:
                    raise ValueError('Shared source input shard changed')
                records=value['records']
            else:
                records=[]
                for split,q in own:
                    item,y,mask=self.get(q,split)
                    records.append(dict(query=q,item=item,target=y,observed=mask,input_sha256=item.identity))
                atomic_torch(path,dict(binding=self.binding,records=records))
                atomic_json(path.with_suffix('.sha.json'),dict(binding=self.binding,sha256=sha(path)))
            for offset,record in enumerate(records):index[Query(**record['query']).key]=[name,offset]
            hashes[name]=sha(path)
            progress(completed=min(start_row+128,len(rows)),total=len(rows))
        atomic_json(self.directory/'index.json',dict(binding=self.binding,index=index,complete=True,
            shard_sha256=hashes,records=len(rows),cpu_seconds=time.monotonic()-start,confirmation_labels_opened=False))
        self.index=index;self.shard_hashes=hashes
        return index


class HistoryCache:
    """Bounded CPU/disk cache of detached pre-graph history states only."""
    def __init__(self,directory,ram_bytes=2*2**30,disk_bytes=8*2**30):
        self.directory=Path(directory);self.directory.mkdir(parents=True,exist_ok=True)
        self.maximum_ram=ram_bytes;self.maximum_disk=disk_bytes;self.ram=OrderedDict();self.bytes=0
        self.hits=0;self.misses=0

    @staticmethod
    def size(value):return sum(x.numel()*x.element_size() for x in (value.hidden,value.mask,value.ends))

    def get(self,model,tokens):
        key=model.input_binding(tokens);device=tokens.input_ids.device;path=self.directory/(key+'.pt')
        if key in self.ram:
            self.hits+=1;value=self.ram.pop(key);self.ram[key]=value;return value.to(device)
        if path.exists():
            seal=json.loads(path.with_suffix('.sha.json').read_text())
            if seal['binding']!=key or seal['sha256']!=sha(path):raise ValueError('Frozen history cache content changed')
            self.hits+=1;value=torch.load(path,map_location='cpu',weights_only=False)
            if value.binding!=key:raise ValueError('Frozen prefix cache identity changed')
            path.touch()
        else:
            self.misses+=1;value=model.frozen_history(tokens).to('cpu')
            atomic_torch(path,value)
            atomic_json(path.with_suffix('.sha.json'),dict(binding=key,sha256=sha(path)))
            paths=sorted(self.directory.glob('*.pt'),key=lambda p:p.stat().st_mtime)
            total=sum(p.stat().st_size for p in paths)
            for old in paths:
                if total<=self.maximum_disk:break
                if old==path:continue
                if not old.resolve().is_relative_to(self.directory.resolve()):raise ValueError('Cache pruning escaped its directory')
                total-=old.stat().st_size;old.unlink();old.with_suffix('.sha.json').unlink(missing_ok=True)
        if self.size(value)<=self.maximum_ram:
            self.ram[key]=value;self.bytes+=self.size(value)
            while self.bytes>self.maximum_ram:
                _,old=self.ram.popitem(last=False);self.bytes-=self.size(old)
        return value.to(device)
