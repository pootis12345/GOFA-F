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
        base=[str(python),'-m','gofa_f.source_graph_probe','--config',
            str(args.snapshot/'configs/stage36_source_graph_probe.yaml'),'--output',str(args.output)]
        start=time.monotonic()
        def write(name,value):
            path=args.output/name;temporary=path.with_suffix(path.suffix+'.tmp')
            temporary.write_text(json.dumps(value,indent=2)+'\n');temporary.replace(path)
        try:
            subprocess.run(base+['--mode','prepare'],env=env,cwd=args.snapshot,check=True)
            proof=args.output/'cpu_preflight.json'
            binding=hashlib.sha256((args.output/'manifest.json').read_bytes()).hexdigest()
            if proof.exists():
                old=json.loads(proof.read_text())
                if not old['passed'] or old['manifest_sha256']!=binding:
                    raise ValueError('Existing CPU proof not reusable')
            else:
                with (args.output/'cpu_regression.log').open('w') as log:
                    code=subprocess.run([str(python),'-m','pytest',str(args.snapshot/'tests'),'-q',
                        '--disable-warnings','--junitxml='+str(args.output/'cpu_tests.xml')],
                        env=env,cwd=args.snapshot,stdout=log,stderr=subprocess.STDOUT).returncode
                suites=list(ET.parse(args.output/'cpu_tests.xml').getroot().iter('testsuite'))
                fields={k:sum(int(s.get(k,0)) for s in suites) for k in ('tests','failures','errors','skipped')}
                write('cpu_preflight.json',dict(passed=code==0,**fields,manifest_sha256=binding,
                    cpu_wall_seconds=time.monotonic()-start,cuda_visible_devices='',omp_threads=2,mkl_threads=2))
                if code:raise RuntimeError('CPU acceptance failed; no fitting')
            subprocess.run(base+['--mode','run'],env=env,cwd=args.snapshot,check=True)
        except Exception as error:
            write('bootstrap_failure.json',dict(error=repr(error),time=time.time()))
            raise
        finally:
            write('bootstrap_completion.json',dict(cpu_wall_seconds=time.monotonic()-start,
                scientific_report_present=(args.output/'report.json').exists()))


if __name__=='__main__':main()
