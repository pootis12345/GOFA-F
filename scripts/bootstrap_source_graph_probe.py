"""One-shot CPU acceptance/controller; scientific entry handles exact resume."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import xml.etree.ElementTree as ET


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--workspace',required=True,type=Path)
    p.add_argument('--snapshot',required=True,type=Path)
    p.add_argument('--output',required=True,type=Path)
    p.add_argument('--config',type=Path)
    p.add_argument('--module',default='gofa_f.source_graph_probe',
        choices=('gofa_f.source_graph_probe','gofa_f.source_neighborhood_probe','gofa_f.source_normalized_probe',
            'gofa_f.source_time_robust_probe'))
    p.add_argument('--related-baseline',type=Path)
    args=p.parse_args()
    python=args.workspace/'.venv-chat/bin/python'
    env=os.environ.copy()
    env.update(GOFA_WORKSPACE=str(args.workspace),CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',
        MKL_NUM_THREADS='2',PYTHONPATH=str(args.snapshot/'src'),TOKENIZERS_PARALLELISM='false')
    args.output.mkdir(parents=True,exist_ok=True)
    # Advisory lock also rejects a duplicate controller during the test phase.
    import fcntl
    with (args.output/'bootstrap.lock').open('a') as lease:
        fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
        base=[str(python),'-m',args.module,'--config',
            str(args.config or args.snapshot/'configs/stage36_source_graph_probe.yaml'),'--output',str(args.output)]
        start=time.monotonic()
        timing=dict(started_at_unix=time.time(),phases=[],scope='controller_wall_excludes_development_and_transfer')
        def write(name,value):
            path=args.output/name;temporary=path.with_suffix(path.suffix+'.tmp')
            temporary.write_text(json.dumps(value,indent=2)+'\n');temporary.replace(path)
        def phase(name,action):
            before=time.monotonic();item=dict(phase=name,started_at_unix=time.time(),passed=False)
            try:
                value=action();item['passed']=True;return value
            finally:
                item['wall_seconds']=time.monotonic()-before;timing['phases'].append(item)
                timing['elapsed_wall_seconds']=time.monotonic()-start
                write('controller_timing.json',timing)
        try:
            phase('prepare',lambda:subprocess.run(base+['--mode','prepare'],env=env,cwd=args.snapshot,check=True))
            proof=args.output/'cpu_preflight.json'
            binding=hashlib.sha256((args.output/'manifest.json').read_bytes()).hexdigest()
            if proof.exists():
                old=json.loads(proof.read_text())
                if not old['passed'] or old['manifest_sha256']!=binding:
                    raise ValueError('Existing CPU proof not reusable')
            else:
                tests=[str(args.snapshot/'tests')]
                baseline=None
                if args.related_baseline:
                    if args.module not in ('gofa_f.source_neighborhood_probe','gofa_f.source_normalized_probe',
                            'gofa_f.source_time_robust_probe'):
                        raise ValueError('Related proof reuse requires a registered source probe')
                    baseline=args.related_baseline/'cpu_preflight.json'
                    manifest=json.loads((args.output/'manifest.json').read_text())
                    if hashlib.sha256(baseline.read_bytes()).hexdigest()!=manifest['baseline_full_proof_sha256']:
                        raise ValueError('Full baseline proof binding changed')
                    files=['test_source_graph_probe.py','test_source_neighborhood_probe.py']
                    if args.module in ('gofa_f.source_normalized_probe','gofa_f.source_time_robust_probe'):
                        files.append('test_source_normalized_probe.py')
                    if args.module=='gofa_f.source_time_robust_probe':files.append('test_source_time_robust_probe.py')
                    tests=[str(args.snapshot/'tests'/n) for n in files]
                with (args.output/'cpu_regression.log').open('w') as log:
                    code=phase('related_tests' if baseline else 'full_tests',lambda:subprocess.run([str(python),'-m','pytest',*tests,'-q',
                        '--disable-warnings','--junitxml='+str(args.output/'cpu_tests.xml')],
                        env=env,cwd=args.snapshot,stdout=log,stderr=subprocess.STDOUT).returncode)
                suites=list(ET.parse(args.output/'cpu_tests.xml').getroot().iter('testsuite'))
                fields={k:sum(int(s.get(k,0)) for s in suites) for k in ('tests','failures','errors','skipped')}
                write('cpu_preflight.json',dict(passed=code==0,**fields,manifest_sha256=binding,
                    cpu_wall_seconds=time.monotonic()-start,cuda_visible_devices='',omp_threads=2,mkl_threads=2))
                if baseline:
                    value=json.loads(proof.read_text());value.update(scope='inherited_full_plus_related',
                        inherited_full_proof_sha256=hashlib.sha256(baseline.read_bytes()).hexdigest(),
                        related_test_files=[Path(t).name for t in tests]);write('cpu_preflight.json',value)
                if code:raise RuntimeError('CPU acceptance failed; no fitting')
            phase('source_worker',lambda:subprocess.run(base+['--mode','run'],env=env,cwd=args.snapshot,check=True))
        except Exception as error:
            write('bootstrap_failure.json',dict(error=repr(error),time=time.time()))
            raise
        finally:
            write('bootstrap_completion.json',dict(cpu_wall_seconds=time.monotonic()-start,
                scientific_report_present=(args.output/'report.json').exists()))


if __name__=='__main__':main()
