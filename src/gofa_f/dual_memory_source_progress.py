"""Read-only source progress with per-fit identity and conditional evaluation ETA."""
import argparse
import json
from pathlib import Path
import time


def read(path):
    return json.loads(path.read_text()) if path.exists() else {}


def duration(value):
    if value is None:return 'unknown'
    return f'{value/3600:.2f}h' if value>=3600 else f'{value/60:.1f}min'


def report(output):
    status=read(output/'status.json');activity=read(output/'activity.json')
    manifest=read(output/'manifest.json');settings=manifest.get('configuration',{}).get('training',{})
    maximum=settings.get('max_updates',2000);batch=settings.get('effective_batch',8)
    fits=len(list(output.glob('depth*/hold-*/selection.json')))+int((output/'final-source-model/selection.json').exists())
    terminal=status.get('state') in ('failed','complete','incomplete','failed_source_warmup')
    eta=None
    if not terminal:
        current=status.get('eta_current_fit_seconds');rate=status.get('estimated_seconds_per_update')
        if current is not None and rate is not None:
            eta=current+max(0,6-fits)*maximum*rate
            baseline=[read(p) for p in (output/'native/development').glob('*.json')]
            reused=read(output/'native_reuse/records.json').get('records',[])
            if reused:baseline=reused
            if baseline:
                seconds=sum(v.get('generation_seconds',0.) for v in baseline)/len(baseline)
                eta+=(max(0,256-len(baseline))+256+768)*seconds
                parent=manifest.get('formal_configuration',{}).get('parent_output')
                if not parent:
                    previous=manifest.get('formal_configuration',{}).get('parent_source_output')
                    if previous:parent=read(Path(previous)/'manifest.json').get('formal_configuration',{}).get('parent_output')
                profile=read(Path(parent)/'throughput_partial.json').get('records',[]) if parent else []
                if profile:eta+=(256+768)*max(v['seconds'] for v in profile)
    print(f"DualMemory {status.get('state','not_started')}/{status.get('phase','source_inputs')} "
          f"fit={status.get('fit','-')} depth={status.get('depth','-')} "
          f"updates={status.get('updates',0)}/{status.get('maximum',maximum)} "
          f"pending={status.get('pending',0)}/{batch} sourceFits={fits}/7 "
          f"conditionalETA={'not_applicable' if terminal else duration(eta)}")
    if status.get('initialization_best_update') is not None:
        print(f"CPU initialization best={status['initialization_best_update']}; joint best="
              f"{status.get('selection',{}).get('best_update','pending')} within this fit only. "
              f"Scope={status.get('fit_scope','source_fit')}; crossfit weights are not the final all-source fit.")
    ufg=read(output/'ufg_status.json');reuse=read(output/'ufg_reuse.json')
    if reuse:
        chosen={shape:{k:read(output/f'ufg-{shape}/selection.json').get(k)
            for k in ('best_update','stopped_update')} for shape in ('12-12','48-24')}
        print(f'UFG complete/unchanged source models reused: {chosen}; new UFG training=0.')
    else:
        print(f"UFG {ufg.get('state','pending')}/{ufg.get('phase','source_training')} "
            f"updates={ufg.get('updates',0)}/{ufg.get('maximum',10000)} "
            f"fitETA={'not_applicable' if ufg.get('state') in ('failed','complete') else duration(ufg.get('eta_current_fit_seconds'))}")
    print(f"GPU cumulative activity={duration(activity.get('spent_seconds'))}; "
          'ETA includes conditional remaining training/Native sampling/confirmation/targets; '
          'excludes unknown resource pauses, may shorten by early stop. Accuracy goal remains unproven.')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('output',type=Path);parser.add_argument('--watch',action='store_true')
    args=parser.parse_args()
    while True:
        report(args.output)
        if not args.watch:break
        time.sleep(10)


if __name__=='__main__':main()
