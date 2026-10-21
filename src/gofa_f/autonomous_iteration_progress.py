"""Conditional Stage30 sampling progress; frozen finite development search."""
from gofa_f.allnode_stage import read
from gofa_f.six_target_progress import Progress as BaseProgress


class Progress(BaseProgress):
    def __init__(self, output, manifest):
        params = read(output/'coefficients.json')['coefficients']
        total = 8*sum(sum(q['horizon'] for q in manifest['queries'][n][64:])*(
            1+int(p['full']['weights'].get('gofa',0.)!=0.)) for n,p in params.items())
        super().__init__(output,dict(manifest,total_numeric_work=total))

    def tick(self, phase=None, force=False, **fields):
        return super().tick(phase,force,**dict(fields,stage=30))
