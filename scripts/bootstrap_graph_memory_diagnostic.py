"""Bind the completed isolated CPU proof, then start the cap-only pilot."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
import xml.etree.ElementTree as ET


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--workspace',required=True,type=Path)
    parser.add_argument('--snapshot',required=True,type=Path);parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--cpu-proof-directory',required=True,type=Path);args=parser.parse_args()
    w,s,o,previous=args.workspace,args.snapshot,args.output,args.cpu_proof_directory
    cfg=s/'configs/stage57_graph_memory_diagnostic.yaml';python=w/'.venv-chat/bin/python'
    env=dict(os.environ,GOFA_WORKSPACE=str(w),PYTHONPATH=str(s/'src'),CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',
        HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',PYTHONDONTWRITEBYTECODE='1')
    started=time.monotonic()
    import fcntl
    o.mkdir(parents=True,exist_ok=True)
    with (o/'bootstrap.lock').open('a') as lease:
        fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
        subprocess.run([str(python),'-m','gofa_f.graph_memory_diagnostic','--config',str(cfg),'--output',str(o),'--prepare'],env=env,cwd=s,check=True)
        if sha(previous/'manifest.json')!=sha(o/'manifest.json'):raise ValueError('Prepared input/code binding differs from completed CPU proof')
        tested=previous.parent
        checks={}
        for folder in ('src/gofa_f','tests'):
            a={str(p.relative_to(tested)):sha(p) for p in (tested/folder).rglob('*.py')}
            b={str(p.relative_to(s)):sha(p) for p in (s/folder).rglob('*.py')}
            if a!=b:raise ValueError('Source/tests changed after CPU acceptance')
            checks.update(a)
        if sha(cfg)!=sha(tested/'configs/stage57_graph_memory_diagnostic.yaml'):raise ValueError('Fixed configuration changed after CPU acceptance')
        suites=list(ET.parse(previous/'cpu_tests.xml').getroot().iter('testsuite'))
        counts={k:sum(int(v.get(k,0)) for v in suites) for k in ('tests','failures','errors','skipped')}
        if counts!={'tests':6,'failures':0,'errors':0,'skipped':0}:raise ValueError('Completed related CPU acceptance required')
        for name in ('cpu_tests.xml','cpu_regression.log'):shutil.copyfile(previous/name,o/name)
        proof=dict(passed=True,**counts,manifest_sha256=sha(o/'manifest.json'),cuda_disabled=True,omp_threads=2,mkl_threads=2,
            cpu_test_wall_seconds=sum(float(v.get('time',0)) for v in suites),reused_exact_isolated_proof=True,
            original_cpu_tests_sha256=sha(previous/'cpu_tests.xml'),all_source_and_tests_sha256=checks)
        (o/'cpu_preflight.json').write_text(json.dumps(proof,indent=2)+'\n')
        command=[str(python),'-m','gofa_f.short_launcher','--output',str(o),'--config',str(cfg),
            '--snapshot',str(s),'--worker-mode','graph_memory_diagnostic']
        with o.with_suffix('.log').open('w') as log:
            child=subprocess.Popen(command,env=env,cwd=s,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        ticks=Path(f'/proc/{child.pid}/stat').read_text().rsplit(')',1)[1].split()[19]
        launch=dict(pid=child.pid,start_ticks=ticks,command=command,snapshot=str(s),started_at_unix=time.time())
        o.with_suffix('.launch.json').write_text(json.dumps(launch,indent=2)+'\n')
        (o/'bootstrap_completion.json').write_text(json.dumps(dict(passed=True,wall_seconds=time.monotonic()-started,
            reused_cpu_proof=True,supervisor=launch),indent=2)+'\n')
        print(json.dumps(dict(cpu_passed=True,tests=counts['tests'],supervisor=launch)))


if __name__=='__main__':main()
