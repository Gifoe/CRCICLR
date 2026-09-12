"""Fixed S2 held-out evaluation for the user-authorized seed-0 provisional pilot."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from evaluate_search import logits, score
from task_datasets import OUTPUTS, PROTOCOL, RawGPUCache, build_model, load_bundle, normalizer, sha256, split_reference, write_json


def main() -> int:
    amend=json.loads((PROTOCOL/'SEED0_PRELIMINARY_AMENDMENT.json').read_text(encoding='utf-8'))
    p=json.loads((PROTOCOL/'SEED0_CHECKPOINT_PROVENANCE.json').read_text(encoding='utf-8'))
    if not amend.get('heldout_evaluation_authorized') or not p.get('pass') or len(p.get('records',[]))!=20:
        raise RuntimeError('seed0 heldout protocol/provenance invalid')
    if not (OUTPUTS/'SEED0_SEARCH_REPLICATE_RESULTS.csv').is_file(): raise RuntimeError('SEARCH evaluation must precede heldout evaluation')
    search,heldout,split,_=split_reference(); by={(r['task'],int(r['fold']),r['model']):r for r in p['records']}; device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); rows=[]; comp={t:{k:0 for k in ('both_correct','both_wrong','EEGNet_only_correct','LiteBN_only_correct','trials')} for t in ('ERP','SSVEP')}
    for task in ('ERP','SSVEP'):
        source=load_bundle(task,search); held=load_bundle(task,heldout,sessions=(2,)); cache=RawGPUCache(held,device)
        for fold in split['folds']:
            mean,std,norm=normalizer(source,fold['inner_train_subjects']); models={}
            for name in ('EEGNet','LiteBN'):
                r=by[(task,int(fold['fold_id']),name)]; path=Path(r['checkpoint_path'])
                if sha256(path)!=r['checkpoint_sha256'] or r['normalizer']['mean_std_sha256']!=norm['mean_std_sha256']: raise RuntimeError('checkpoint/normalizer provenance failure')
                m=build_model(name,task).to(device); m.load_state_dict(torch.load(path,map_location=device,weights_only=False),strict=True); m.eval()
                for q in m.parameters(): q.requires_grad_(False)
                models[name]=m
            for subject in heldout:
                ix=held.indices([subject],(2,)); y=held.labels(ix); ze=logits(models['EEGNet'],cache,ix,mean,std); zl=logits(models['LiteBN'],cache,ix,mean,std)
                for name,z in (('EEGNet',ze),('LiteBN',zl),('LOGIT50',(ze+zl)/2.0)):
                    rows.append({'task':task,'subject_id':subject,'fold':int(fold['fold_id']),'seed':0,'method':name,'trials':int(len(y)),**score(task,y,z)})
                ep,lp=ze.argmax(1),zl.argmax(1); comp[task]['both_correct']+=int(((ep==y)&(lp==y)).sum()); comp[task]['both_wrong']+=int(((ep!=y)&(lp!=y)).sum()); comp[task]['EEGNet_only_correct']+=int(((ep==y)&(lp!=y)).sum()); comp[task]['LiteBN_only_correct']+=int(((lp==y)&(ep!=y)).sum()); comp[task]['trials']+=int(len(y))
            del models
            if device.type=='cuda': torch.cuda.empty_cache()
        del source,held,cache
        if device.type=='cuda': torch.cuda.empty_cache()
    out=pd.DataFrame(rows)
    if len(out)!=420 or out.duplicated(['task','subject_id','fold','method']).any(): raise RuntimeError('seed0 heldout cardinality invalid')
    OUTPUTS.mkdir(parents=True,exist_ok=True); out.to_csv(OUTPUTS/'SEED0_HELDOUT_REPLICATE_RESULTS.csv',index=False)
    for task,v in comp.items():
        for k in ('both_correct','both_wrong','EEGNet_only_correct','LiteBN_only_correct'): v[k+'_fraction']=v[k]/v['trials']
        v['complementarity_fraction']=v['EEGNet_only_correct_fraction']+v['LiteBN_only_correct_fraction']; v['description']='descriptive trial counts across five fixed fold replicates; not used to adapt fusion'
    write_json(OUTPUTS/'SEED0_HELDOUT_CARRIER_COMPLEMENTARITY.json',comp)
    print('OPENBMI_TASK_SEED0_HELDOUT_COMPLETE')
    return 0

if __name__=='__main__': raise SystemExit(main())