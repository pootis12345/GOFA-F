"""One related CPU proof then the existing no-deadline cap-only supervisor."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import xml.etree.ElementTree as ET


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--workspace',required=True,type=Path)
    parser.add_argument('--snapshot',required=True,type=Path);parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args();w=args.workspace;s=args.snapshot;o=args.output;o.mkdir(parents=True,exist_ok=True)
    cfg=s/'configs/stage56_graph_memory.yaml';python=w/'.venv-chat/bin/python'
    env=dict(os.environ,GOFA_WORKSPACE=str(w),PYTHONPATH=str(s/'src'),CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',
        HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',PYTHONDONTWRITEBYTECODE='1')
    started=time.monotonic()
    import fcntl
    with (o/'bootstrap.lock').open('a') as lease:
        fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
        subprocess.run([str(python),'-m','gofa_f.graph_memory_stage','--config',str(cfg),'--output',str(o),'--prepare'],env=env,cwd=s,check=True)
        begin=time.monotonic()
        tests=['test_graph_memory.py','test_short_training.py']
        with (o/'cpu_regression.log').open('w') as log:
            code=subprocess.run([str(python),'-m','pytest',*[str(s/'tests'/name) for name in tests],'-q','--disable-warnings',
                '--junitxml='+str(o/'cpu_tests.xml')],env=env,cwd=s,stdout=log,stderr=subprocess.STDOUT).returncode
        suites=list(ET.parse(o/'cpu_tests.xml').getroot().iter('testsuite'))
        counts={k:sum(int(s.get(k,0)) for s in suites) for k in ('tests','failures','errors','skipped')}
        proof=dict(passed=code==0,**counts,manifest_sha256=hashlib.sha256((o/'manifest.json').read_bytes()).hexdigest(),
            test_files=tests,cuda_disabled=True,omp_threads=2,mkl_threads=2,cpu_test_wall_seconds=time.monotonic()-begin)
        (o/'cpu_preflight.json').write_text(json.dumps(proof,indent=2)+'\n')
        if code:
            print((o/'cpu_regression.log').read_text());raise RuntimeError('Related CPU acceptance failed; GPU not launched')
        command=[str(python),'-m','gofa_f.short_launcher','--output',str(o),'--config',str(cfg),'--snapshot',str(s),'--worker-mode','graph_memory_mechanisms']
        with o.with_suffix('.log').open('w') as log:
            child=subprocess.Popen(command,env=env,cwd=s,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        ticks=Path(f'/proc/{child.pid}/stat').read_text().rsplit(')',1)[1].split()[19]
        launch=dict(pid=child.pid,start_ticks=ticks,command=command,snapshot=str(s),started_at_unix=time.time())
        o.with_suffix('.launch.json').write_text(json.dumps(launch,indent=2)+'\n')
        (o/'bootstrap_completion.json').write_text(json.dumps(dict(passed=True,cpu_wall_seconds=time.monotonic()-started,supervisor=launch),indent=2)+'\n')
        print(json.dumps(dict(cpu_proof=proof,supervisor=launch)))


if __name__=='__main__':main()
