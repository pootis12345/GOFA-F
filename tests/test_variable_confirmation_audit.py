from dataclasses import asdict
import json
import numpy as np
import pytest
from gofa_f.generative_data import Query,Series,sha
from gofa_f.variable_graph_data import variable_input,length_balanced_summary
from gofa_f.variable_graph_audit import confirmation_audit


def fixture(tmp_path):
    source=tmp_path/'sources';source.mkdir();p=source/'boston2017.npz'
    values=np.stack([np.sin(np.arange(110)/7),np.cos(np.arange(110)/7)],1)
    mask=np.ones_like(values,dtype=bool);mask[92,0]=False;values[92,0]=np.nan
    np.savez(p,values=values,mask=mask,times=np.arange(110,dtype=np.int64)*3600_000_000_000,
        segments=np.zeros(110,dtype=int),external=np.ones((2,2)))
    p.with_suffix('.json').write_text(json.dumps(dict(variables=['0','1'])))
    rows=[asdict(Query('boston2017',0,90,10,1)),asdict(Query('boston2017',0,91,14,7))]
    m=dict(source_directory=str(source),source_hashes={'boston2017':sha(p)},metadata_hashes={'boston2017':sha(p.with_suffix('.json'))},
        partitions={'boston2017':dict(confirmation=[0,1])},training_shapes=[[10,1]],held_length_pairs=[[14,7]],
        training_unit_normalizers={'boston2017':1.},queries={'boston2017':{str(i):dict(train=[],development=[],confirmation=[q]) for i,q in enumerate(rows)}})
    (tmp_path/'manifest.json').write_text(json.dumps(m));binding=sha(tmp_path/'manifest.json');own=tmp_path/'expert';(own/'confirmation').mkdir(parents=True)
    data=Series(p);records=[]
    for i,q in enumerate(rows):
        y=values[q['origin']:q['origin']+q['horizon'],q['variable']];mask=data.mask[q['origin']:q['origin']+q['horizon'],q['variable']]&np.isfinite(y)
        item=variable_input(data,Query(**q),[0,1]);target=np.where(mask,y,0).tolist()
        r=dict(identity=dict(binding=binding,checkpoint_sha='fixed-model',query=q,split='confirmation'),query=q,
            prediction=(np.where(mask,y,0)+1).tolist(),target=target,observed=mask.tolist(),input_sha256=item.identity)
        (own/'confirmation'/f'{i:05d}.json').write_text(json.dumps(r));records.append(r)
    summary=dict(binding=binding,model_sha256='fixed-model',used_for_selection=False,metrics=length_balanced_summary(records,m['training_unit_normalizers']))
    (own/'confirmation.json').write_text(json.dumps(summary));return m,own


def test_confirmation_replays_held_lengths_truth_mask_and_input(tmp_path):
    m,own=fixture(tmp_path);r=confirmation_audit(tmp_path,m,'expert','fixed-model')
    assert r['records']==2 and r['held_length_pairs']==[(14,7)]
    assert r['independent_selection_score']==pytest.approx(1.) and r['used_for_selection'] is False


@pytest.mark.parametrize('field',['target','observed','input_sha256','identity'])
def test_confirmation_refuses_changed_evidence(tmp_path,field):
    m,own=fixture(tmp_path);p=own/'confirmation/00001.json';r=json.loads(p.read_text())
    if field=='target':r[field][0]+=1.
    elif field=='observed':r[field][0]=False
    elif field=='input_sha256':r[field]='wrong'
    else:r[field]['checkpoint_sha']='another-model'
    p.write_text(json.dumps(r))
    with pytest.raises(ValueError):confirmation_audit(tmp_path,m,'expert','fixed-model')


def test_confirmation_refuses_missing_length_or_selection_exposure(tmp_path):
    m,own=fixture(tmp_path);p=own/'confirmation.json';v=json.loads(p.read_text());v['used_for_selection']=True;p.write_text(json.dumps(v))
    with pytest.raises(ValueError):confirmation_audit(tmp_path,m,'expert','fixed-model')
    v['used_for_selection']=False;p.write_text(json.dumps(v));(own/'confirmation/00001.json').unlink()
    with pytest.raises(ValueError):confirmation_audit(tmp_path,m,'expert','fixed-model')
