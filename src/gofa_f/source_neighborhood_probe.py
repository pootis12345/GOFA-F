"""Stage38: isolate physical neighbor selection with the existing source probe."""
import argparse
import copy
import os
from pathlib import Path
import resource
import signal
import time

import numpy as np
import torch
import yaml

from gofa_f import source_graph_probe as p
from gofa_f.graph_probe_data import CONDITIONS, VIEWS, node_roster, pack
from gofa_f.generative_data import sha
from gofa_f.prefix_cache import atomic_json

PROTOCOL = 'stage38-source-neighborhood-probe-v1'
PARENT_REPORT = 'd4c7264b44281a7dd6618d17799e0d652e93701f74b566d9baa64084c22c3280'
SEMANTICS = dict(physical='physical_BFS_roster_dense_attention',
    self='same_physical_roster_no_cross_messages',
    rewired='uniform_partition_roster_matched_count_dense_attention')


def stage_number(cfg):
    return {'stage38-source-neighborhood-probe-v1':38,'stage40-source-normalized-probe-v1':40,
        'stage41-source-time-robust-probe-v1':41,'stage43-source-state-attention-probe-v1':43,'stage45-source-scale-encoding-probe-v1':45,'stage47-source-distance-prior-probe-v1':47,'stage49-source-innovation-probe-v1':49,'stage50-source-convergence-probe-v1':50,'stage51-source-sparse-normalized-v1':51}[cfg['protocol']]


def configuration(path):
    cfg=yaml.safe_load(Path(path).read_text().replace('${GOFA_WORKSPACE}',os.environ['GOFA_WORKSPACE']))
    base=p.configuration(Path(__file__).parents[2]/'configs/stage37_source_road_probe.yaml')
    if set(cfg)!={'protocol','parent_output','parent_report_sha256','model','training','seeds','conditions','gate','policy'} \
            or cfg['protocol']!=PROTOCOL or cfg['parent_report_sha256']!=PARENT_REPORT \
            or any(cfg[k]!=base[k] for k in ('model','training','seeds','conditions','gate','policy')):
        raise ValueError('Single registered neighborhood experiment required')
    return cfg


def code_hashes():
    return dict(p.code_hashes(),source_neighborhood_probe=sha(Path(__file__)))


def prepare(cfg,output):
    parent=Path(cfg['parent_output'])
    old=p.read(parent/'manifest.json')
    if sha(parent/'report.json')!=PARENT_REPORT or p.read(parent/'activity.json')['worker'] is not None \
            or not p.read(parent/'final_verification.json')['passed'] \
            or (parent/'source_confirmation.json').exists():
        raise ValueError('Stopped verified parent and unopened confirmation required')
    for name in ('lag_graph_probe.py','source_graph_probe.py','generative_data.py'):
        if old['code'][name]!=p.code_hashes()[name]:
            raise ValueError('Numerical model/loss/selection/source reader changed; full acceptance required')
    proof=p.read(parent/'cpu_preflight.json')
    if not proof['passed'] or proof['failures'] or proof['errors'] \
            or proof['manifest_sha256']!=sha(parent/'manifest.json'):
        raise ValueError('Verified full regression baseline required')
    m={k:copy.deepcopy(old[k]) for k in ('source_directory','source_hashes','metadata_hashes',
        'graph_contract','partition','queries','exclusion_sha256')}
    m.update(configuration=cfg,code=code_hashes(),parent_manifest_sha256=sha(parent/'manifest.json'),
        baseline_full_proof_sha256=sha(parent/'cpu_preflight.json'),condition_semantics=SEMANTICS,
        source_validation_previously_seen=True,target_data_opened=False)
    p.source_series(m)
    output.mkdir(parents=True,exist_ok=True)
    p.immutable(output/'manifest.json',m)
    return m


def neighborhood_pack(data,record,allowed,random=False,labels=True,split='train'):
    target=record['query']['variable']
    physical=node_roster(data.external,target,allowed,data.ids)
    indices=physical
    if random:
        # Fixed by target/partition, shared across channels, dates and training seeds.
        rng=np.random.default_rng(np.random.SeedSequence([42,38,target,*sorted(allowed)]))
        candidates=sorted(set(allowed)-{target},key=lambda i:str(data.ids[i]))
        others=rng.choice(candidates,len(physical)-1,replace=False).tolist()
        indices=[target]+sorted(others,key=lambda i:str(data.ids[i]))
    a=pack(data,record,allowed,labels=labels,split=split,indices=indices)
    active=a['nodes']>=0
    a['adjacency']=active[:,None]&active[None,:]&~np.eye(9,dtype=bool)
    return a


def caches(output,m,charge):
    data=p.source_series(m)
    result={c:{s:{} for s in ('train','validation','confirmation')} for c in CONDITIONS}
    for c in ('physical','rewired'):
        for split in result[c]:
            for name in VIEWS:
                path=output/'cache'/c/f'{split}-{name}.npz'
                binding=p.digest(dict(manifest=sha(output/'manifest.json'),condition=c,split=split,view=name))
                if path.exists():
                    proof=p.read(path.with_suffix('.json'))
                    if proof['binding']!=binding or proof['sha256']!=sha(path):
                        raise ValueError('Bound neighborhood cache differs')
                else:
                    rows=[neighborhood_pack(data[name],r,m['partition'][split],random=c=='rewired',
                        labels=split!='confirmation',split=split) for r in m['queries'][split][name]]
                    arrays={k:np.stack([r[k] for r in rows]) for k in rows[0]}
                    path.parent.mkdir(parents=True,exist_ok=True)
                    temporary=path.with_suffix('.part')
                    with temporary.open('wb') as stream:np.savez_compressed(stream,**arrays)
                    temporary.replace(path)
                    p.immutable(path.with_suffix('.json'),dict(binding=binding,sha256=sha(path)))
                with np.load(path,allow_pickle=False) as stream:a={k:stream[k] for k in stream.files}
                if split=='confirmation' and {'y','observed','raw_target'}&set(a):
                    raise ValueError('Confirmation labels must remain closed')
                if not set(a['nodes'][a['nodes']>=0])<=set(m['partition'][split]):
                    raise ValueError('Cross-partition neighbor detected')
                result[c][split][name]=a
                charge()
    result['self']=result['physical']
    return result


def confirm(cfg,output,m,packed,selections,charge):
    if not p.read(output/'repeated_validation_gate.json',{}).get('passed'):
        raise ValueError('Repeated source gate required before confirmation labels')
    lock=p.read(output/'source_selection.json')
    if not lock or lock['manifest_sha256']!=sha(output/'manifest.json'):
        raise ValueError('Checkpoint lock required')
    for s in cfg['seeds']:
        for c in CONDITIONS:
            if sha(output/'models'/f'seed-{s}'/c/'best.pt')!=lock['checkpoints'][str(s)][c]:
                raise ValueError('Locked checkpoint changed')
    data=p.source_series(m);results={s:{} for s in cfg['seeds']}
    for s in cfg['seeds']:
        for c in CONDITIONS:
            model=p.model_for(cfg,s)
            model.load_state_dict(torch.load(output/'models'/f'seed-{s}'/c/'best.pt',
                map_location='cpu',weights_only=False)['model'])
            scores={}
            for name in VIEWS:
                a=dict(packed[c]['confirmation'][name]);rows=m['queries']['confirmation'][name]
                truth=np.stack([data[name].values[r['query']['origin']:r['query']['origin']+12,
                    r['query']['variable']] for r in rows]).astype(np.float64)
                observed=np.stack([data[name].mask[r['query']['origin']:r['query']['origin']+12,
                    r['query']['variable']] for r in rows]).astype(bool)&np.isfinite(truth)
                a.update(raw_target=np.where(observed,truth,0),observed=observed)
                prediction=p.forecast(model,a,rows,c);scores[name]=p.metrics(prediction,a,rows)
                p.immutable(output/'confirmation'/f'{s}-{c}-{name}.json',dict(prediction=prediction.tolist(),
                    target=a['raw_target'].tolist(),observed=observed.tolist(),metrics=scores[name]))
                charge()
            results[s][c]=dict(validation=scores)
    value=p.gate(results,cfg,repeated=True)
    p.immutable(output/'source_confirmation.json',value)
    return value


def run(cfg,output,charge):
    m=p.read(output/'manifest.json')
    proof=p.read(output/'cpu_preflight.json')
    if m['configuration']!=cfg or m['code']!=code_hashes() or not proof['passed'] \
            or proof['manifest_sha256']!=sha(output/'manifest.json'):
        raise ValueError('Current code, manifest and CPU proof required')
    if (output/'report.json').exists():return p.read(output/'report.json')
    packed=caches(output,m,charge);speed=p.profile(cfg,output);selections={}
    initial=repeat=confirmed=None
    for s in cfg['seeds']:
        selections[s]={}
        for c in CONDITIONS:
            selected=p.fit(cfg,output,m,packed[c],s,c,charge,speed)
            if selected is None or p.STOP[0]:return None
            selections[s][c]=selected
        if s==42:
            initial=p.gate(selections,cfg);p.immutable(output/'initial_validation_gate.json',initial)
            if not initial['passed']:break
    if initial['passed']:
        repeat=p.gate(selections,cfg,repeated=True)
        p.immutable(output/'repeated_validation_gate.json',repeat)
        if repeat['passed']:
            p.immutable(output/'source_selection.json',dict(manifest_sha256=sha(output/'manifest.json'),
                checkpoints={str(s):{c:selections[s][c]['checkpoint_sha256'] for c in CONDITIONS} for s in selections}))
            confirmed=confirm(cfg,output,m,packed,selections,charge)
    passed=bool(confirmed and confirmed['passed'])
    report=dict(stage=stage_number(cfg),protocol=PROTOCOL,complete=True,passed=passed,condition_semantics=SEMANTICS,
        initial_validation=initial,repeated_validation=repeat,source_confirmation=confirmed,
        selections={str(s):v for s,v in selections.items()},gpu_seconds=0.,target_forecasts=0,chattime_updates=0,
        limitations=['exposed single-source validation; no whole-dataset zero-shot proof',
            'small diagnostic continuous-output GNN, not a ChatTime numerical head',
            'control semantics fixed in condition_semantics; label budgets across stages may differ'])
    p.immutable(output/'report.json',report)
    atomic_json(output/'status.json',dict(state='source_probe_passed' if passed else 'stopped_at_source_gate',stage=stage_number(cfg)))
    atomic_json(output/'progress.json',dict(state='complete',eta_seconds=None))
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True,type=Path);parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--mode',choices=('prepare','run','progress'),required=True)
    args=parser.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='' or torch.cuda.is_initialized():
        raise ValueError('CUDA-disabled CPU execution required')
    torch.set_num_threads(2);cfg=configuration(args.config);args.output.mkdir(parents=True,exist_ok=True)
    if args.mode=='progress':
        print(p.read(args.output/'progress.json'));return
    import fcntl
    with (args.output/'run.lock').open('a') as lease:
        fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if args.mode=='prepare':prepare(cfg,args.output);return
        for sig in (signal.SIGTERM,signal.SIGINT,signal.SIGUSR1):
            signal.signal(sig,lambda *_:p.STOP.__setitem__(0,True))
        activity=p.read(args.output/'activity.json',dict(cpu_active_wall_seconds=0.,gpu_seconds=0.))
        anchor=[time.monotonic()]
        def charge():
            now=time.monotonic();activity['cpu_active_wall_seconds']+=now-anchor[0];anchor[0]=now
            activity.update(worker=dict(pid=os.getpid(),start_ticks=Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19]),
                cuda_initialized=torch.cuda.is_initialized(),peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)
            if activity['cuda_initialized']:raise RuntimeError('Unexpected CUDA initialization')
            atomic_json(args.output/'activity.json',activity)
        charge()
        try:run(cfg,args.output,charge)
        except Exception as exc:
            p.immutable(args.output/f'failure-{time.time_ns()}.json',dict(error=repr(exc)))
            atomic_json(args.output/'status.json',dict(state='failed',error=repr(exc)));raise
        finally:
            charge();activity['worker']=None;atomic_json(args.output/'activity.json',activity)


if __name__=='__main__':main()
