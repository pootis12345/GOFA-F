"""CPU acceptance/search then conditional GPU confirmation within fixed user window."""
import argparse
import fcntl
import os
from pathlib import Path
import subprocess
import sys
import time
import xml.etree.ElementTree as ET


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    from gofa_f.allnode_stage import read
    from gofa_f.scada_wind_iteration import check_time, load_coefficients, prepare
    from gofa_f.scada_wind_iteration_imports import import_confirmation
    from gofa_f.generative_data import sha
    from gofa_f.idle_launcher import process_identity
    from gofa_f.prefix_cache import atomic_json
    from gofa_f.source_retraining import configuration
    from gofa_f.six_calibration_cpu import no_worker
    output=args.output
    output.mkdir(parents=True,exist_ok=True)
    lock=output.with_suffix('.bootstrap.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    no_worker(output)
    identity=read(output.with_suffix('.launch.json'),{})
    if identity.get('pid') and process_identity(identity['pid'])==str(identity.get('start_ticks')):
        raise RuntimeError('Existing Stage33 supervisor; no duplicate')
    if read(output/'status.json',{}).get('state')=='complete':
        raise RuntimeError('Terminal Stage33 remains sealed')
    check_time()
    snapshot=Path(__file__).resolve().parents[1]
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',
        PYTHONPATH=str(snapshot/'src'),PYTHONDONTWRITEBYTECODE='1',TOKENIZERS_PARALLELISM='false',
        HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
    cfg=configuration(args.config)
    manifest=prepare(cfg,output)
    if not (output/'cpu_preflight.json').exists():
        result=subprocess.run([sys.executable,'-m','pytest','-q','--disable-warnings',
            '--junitxml='+str(output/'cpu_tests.xml')],cwd=snapshot,env=env)
        suites=ET.parse(output/'cpu_tests.xml').getroot()
        cases=list(suites.iter('testcase'))
        import torch
        value=dict(passed=result.returncode==0 and not torch.cuda.is_initialized(),
            cuda_disabled=True,manifest_sha256=sha(output/'manifest.json'),cases=len(cases),
            skipped=sum(c.find('skipped') is not None for c in cases),
            inherited_frozen_core_proof_sha256=manifest['inherited_cpu_proof_sha256'],
            cuda_initialized=torch.cuda.is_initialized())
        atomic_json(output/'cpu_preflight.json',value)
    proof=read(output/'cpu_preflight.json')
    if not proof['passed'] or proof['manifest_sha256']!=sha(output/'manifest.json'):
        raise ValueError('Complete bound CPU proof failed')
    if not (output/'coefficient_lock.json').exists():
        result=subprocess.run([sys.executable,'-u','-m','gofa_f.scada_wind_iteration',
            '--config',args.config,'--output',str(output),'--phase','run-cpu'],cwd=snapshot,env=env)
        if result.returncode:
            return result.returncode
    if read(output/'status.json',{}).get('state')=='complete':
        return 0
    manifest,_,_=load_coefficients(cfg,output)
    import_confirmation(cfg,output,manifest)
    check_time()
    command=[sys.executable,'-u','-m','gofa_f.scada_wind_iteration_launcher',
        '--config',args.config,'--output',str(output),'--snapshot',str(snapshot),
        '--worker-mode','scada_wind_iteration']
    log=output.with_suffix('.log').open('ab',buffering=0)
    process=subprocess.Popen(command,cwd=snapshot,env=env,stdout=log,stderr=subprocess.STDOUT,
        start_new_session=True,stdin=subprocess.DEVNULL)
    tick=process_identity(process.pid)
    if not tick:
        raise RuntimeError('Supervisor exited before identity was recorded')
    atomic_json(output.with_suffix('.launch.json'),dict(pid=process.pid,start_ticks=tick,
        command=command,snapshot=str(snapshot),deployed_at=time.time()))
    return 0


if __name__=='__main__':
    raise SystemExit(main())
