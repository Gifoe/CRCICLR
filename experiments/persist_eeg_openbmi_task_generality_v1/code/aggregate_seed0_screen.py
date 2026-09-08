"""Aggregate the seed-0 provisional SEARCH and held-out results."""
import json
import numpy as np
import pandas as pd
from task_datasets import OUTPUTS, write_json

def subject_table(frame, task, count):
    p=frame[frame.task==task].groupby(['subject_id','method'],as_index=False)[['BA','macro_F1','accuracy','AUROC','AUPRC']].mean()
    if p.subject_id.nunique()!=count or not (p.groupby('subject_id').size()==3).all(): raise RuntimeError('invalid cardinality')
    w=p.pivot(index='subject_id',columns='method',values=['BA','macro_F1','accuracy','AUROC','AUPRC'])
    w.columns=['%s_%s' % (method,metric) for metric,method in w.columns]
    w=w.reset_index(); w.insert(0,'task',task); w['LOGIT50_minus_EEGNet_pp']=(w.LOGIT50_BA-w.EEGNet_BA)*100; w['LOGIT50_minus_LiteBN_pp']=(w.LOGIT50_BA-w.LiteBN_BA)*100
    return w

def summary(sub, scope):
    rows=[]
    for task in ('ERP','SSVEP'):
        p=sub[sub.task==task]; d=p.LOGIT50_minus_EEGNet_pp.to_numpy(float); rng=np.random.default_rng(0); draws=rng.choice(d,size=(10000,len(d)),replace=True).mean(1)
        rows.append({'task':task,'scope':scope,'n_subjects':len(p),'EEGNet_BA':float(p.EEGNet_BA.mean()),'LiteBN_BA':float(p.LiteBN_BA.mean()),'LOGIT50_BA':float(p.LOGIT50_BA.mean()),'LOGIT50_minus_LiteBN_pp':float(p.LOGIT50_minus_LiteBN_pp.mean()),'mean_delta_pp':float(d.mean()),'median_delta_pp':float(np.median(d)),'bootstrap_ci_low_pp':float(np.quantile(draws,.025)),'bootstrap_ci_high_pp':float(np.quantile(draws,.975)),'positive_subjects':int((d>0).sum())})
    return pd.DataFrame(rows)

def main():
    search=pd.read_csv(OUTPUTS/'SEED0_SEARCH_REPLICATE_RESULTS.csv'); held=pd.read_csv(OUTPUTS/'SEED0_HELDOUT_REPLICATE_RESULTS.csv')
    if len(search)!=240 or len(held)!=420: raise RuntimeError('incomplete seed0 evaluations')
    ss=pd.concat([subject_table(search,t,40) for t in ('ERP','SSVEP')],ignore_index=True); hs=pd.concat([subject_table(held,t,14) for t in ('ERP','SSVEP')],ignore_index=True); sr=summary(ss,'SEARCH'); hr=summary(hs,'HELDOUT')
    vals=hr.set_index('task').mean_delta_pp; term='SEED0_HELDOUT_PROVISIONAL_POSITIVE_BOTH' if vals.ERP>0 and vals.SSVEP>0 else ('SEED0_HELDOUT_PROVISIONAL_PARTIAL' if (vals.ERP>0)!=(vals.SSVEP>0) else 'SEED0_HELDOUT_PROVISIONAL_NONPOSITIVE_BOTH')
    ss.to_csv(OUTPUTS/'SEED0_SEARCH_SUBJECT_RESULTS.csv',index=False); sr.to_csv(OUTPUTS/'SEED0_SEARCH_DATASET_RESULTS.csv',index=False); hs.to_csv(OUTPUTS/'SEED0_HELDOUT_SUBJECT_RESULTS.csv',index=False); hr.to_csv(OUTPUTS/'SEED0_HELDOUT_TASK_RESULTS.csv',index=False)
    write_json(OUTPUTS/'SEED0_PROVISIONAL_TERMINAL.json',{'terminal':term,'seed':0,'heldout_subjects':14,'replicates_per_subject':5,'not_final':True,'heldout_rows':hr.to_dict('records')})
    lines=['# Seed-0 provisional task result','','All five frozen folds and all 14 held-out S2 subjects were run, but only initialization seed 0. This is not the pre-registered three-seed confirmation.','', '| Scope | Task | EEGNet BA | LiteBN BA | LOGIT50 BA | Delta vs EEGNet | Median delta | 95% CI | Positive subjects |','|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for row in pd.concat([sr,hr]).itertuples(): lines.append('| %s | %s | %.4f | %.4f | %.4f | %+.3f pp | %+.3f pp | [%+.3f, %+.3f] pp | %s/%s |' % (row.scope,row.task,row.EEGNet_BA,row.LiteBN_BA,row.LOGIT50_BA,row.mean_delta_pp,row.median_delta_pp,row.bootstrap_ci_low_pp,row.bootstrap_ci_high_pp,row.positive_subjects,row.n_subjects))
    lines += ['', 'Provisional terminal: `%s`.' % term, 'Seeds 1 and 2 remain required for the locked 15-replicate subject average and formal final terminal.', '']
    (OUTPUTS/'SEED0_PROVISIONAL_DECISION.md').write_text('\n'.join(lines),encoding='utf-8'); print(term)
if __name__=='__main__': main()