"""Read-only formal source progress, including conditional remaining evaluations."""
import argparse
import json
from pathlib import Path
import time


def read(path):return json.loads(path.read_text()) if path.exists() else {}


def duration(value):
    if value is None:return 'unknown'
    return f'{value/3600:.2f}h' if value>=3600 else f'{value/60:.1f}min'


def report(output):
    status=read(output/'status.json');ufg=read(output/'ufg_status.json');activity=read(output/'activity.json')
    fits=len(list(output.glob('depth*/hold-*/selection.json')))+int((output/'final-source-model/selection.json').exists())
    prep=read(output/'data_preparation.json')
    if status.get('state') in ('failed','complete'):
        eta=None
    else:
        current=status.get('eta_current_fit_seconds');rate=status.get('estimated_seconds_per_update')
        eta=None if current is None or rate is None else current+max(0,6-fits)*2000*rate
        native=[read(p) for p in (output/'native/development').glob('*.json')]
        if eta is not None and native:
            per_query=sum(v.get('generation_seconds',0.) for v in native)/len(native)
            # Remaining source Native development, source confirmation, six128
            # targets, plus enhanced confirmation/target inference allowance.
            eta+=(max(0,256-len(native))+256+768)*per_query
            parent=read(output/'manifest.json').get('formal_configuration',{}).get('parent_output')
            if parent:
                profile=read(Path(parent)/'throughput_partial.json').get('records',[])
                if profile:eta+=(256+768)*max(v['seconds'] for v in profile)
    shown_eta='not_applicable' if status.get('state') in ('failed','complete') else duration(eta)
    line=(f"DualMemory {status.get('state',prep.get('state','not_started'))}/{status.get('phase','source_inputs')} "
        f"fit={status.get('fit','-')} depth={status.get('depth','-')} "
        f"updates={status.get('updates',0)}/{status.get('maximum',2000)} pending={status.get('pending',0)}/8 "
        f"sourceFits={fits}/7 conditionalETA={shown_eta}")
    if prep.get('state')=='preparing':line+=f" inputQueries={prep.get('completed',0)}/{prep.get('total',0)}"
    print(line)
    print(f"UFG {ufg.get('state','pending')}/{ufg.get('phase','source_training')} "
        f"updates={ufg.get('updates',0)}/{ufg.get('maximum',10000)} pending={ufg.get('pending',0)}/64 "
        f"fitETA={'not_applicable' if ufg.get('state') in ('failed','complete') else duration(ufg.get('eta_current_fit_seconds'))}")
    print(f"GPU activity={duration(activity.get('spent_seconds'))}; "
        'ETA is conditional on all stages proceeding, includes remaining Native sampling/source and target inference; '
        'excludes unknown resource pauses and may shorten with source early stop. Target precision goal is unproven.')


def main():
    p=argparse.ArgumentParser();p.add_argument('output',type=Path);p.add_argument('--watch',action='store_true');args=p.parse_args()
    while True:
        report(args.output)
        if not args.watch:break
        time.sleep(10)


if __name__=='__main__':main()
