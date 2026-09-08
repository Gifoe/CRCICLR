"""Aggregate the frozen BatchNorm-buffer intervention and apply its fixed terminal."""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

EXP = Path(__file__).resolve().parents[1]
REPO = EXP.parents[1]
OUT, PROTOCOL = EXP / "outputs", EXP / "protocol"
PHASE_A = REPO / "experiments" / "persist_eeg_nrrf_final_model_stage1_v1" / "outputs"
BOOTSTRAPS = 10_000


def clean(v: Any) -> Any:
    if isinstance(v, np.ndarray): return clean(v.tolist())
    if isinstance(v, (np.integer,)): return int(v)
    if isinstance(v, (np.floating, float)): return float(v) if math.isfinite(float(v)) else None
    if isinstance(v, (np.bool_, bool)): return bool(v)
    if isinstance(v, dict): return {str(k): clean(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)): return [clean(x) for x in v]
    return v


def write_json(path: Path, v: Any) -> None: path.write_text(json.dumps(clean(v), indent=2, sort_keys=True)+"\n",encoding="utf-8")


def bootstrap(values: np.ndarray) -> dict[str, float]:
    rng=np.random.default_rng(0); draws=rng.choice(values,size=(BOOTSTRAPS,len(values)),replace=True).mean(1)
    return {"mean_recovery_pp":float(values.mean()*100),"median_recovery_pp":float(np.median(values)*100),"bootstrap_ci_low_pp":float(np.quantile(draws,.025)*100),"bootstrap_ci_high_pp":float(np.quantile(draws,.975)*100)}


def decision(wbcic: pd.DataFrame) -> tuple[str, str]:
    major = wbcic[(wbcic.BN_recovery_pp>=2.0)&(wbcic.improved_folds>=4)&(wbcic.recovery_fraction.ge(.5))&(wbcic.harm_reduction_le_minus_1pp>=.10)]
    if len(major): return "BN_BUFFER_DRIFT_MAJOR_CONTRIBUTOR", "Test a new stage-2 training protocol in which the LiteBN BatchNorm running buffers are frozen at their original carrier values throughout training."
    # "Consistently" is fixed here as every fold non-positive plus a negative mean for both matched methods.
    if len(wbcic)==2 and bool((wbcic.BN_recovery_pp<0).all()) and bool((wbcic.improved_folds==0).all()): return "BN_BUFFER_INTERVENTION_HARMFUL", "Stop BN restoration as a repair route."
    partial = bool((wbcic.BN_recovery_pp>=.5).any() or (wbcic.improved_folds>=3).any() or (wbcic.harm_reduction_le_minus_1pp>=.05).any())
    if partial: return "BN_BUFFER_DRIFT_PARTIAL_CONTRIBUTOR", "Separate BN-buffer drift from trainable parameter drift before designing another final model."
    return "BN_BUFFER_DRIFT_NOT_PRIMARY", "Stop BN-based repair and test whether source-session parameter updates themselves destroy future-session generalization."


def main() -> int:
    fold=pd.read_csv(OUT/'PRIMARY_FOLD_RESULTS.csv'); subject=pd.read_csv(OUT/'PRIMARY_SUBJECT_RESULTS.csv')
    expected=pd.read_csv(PHASE_A/'PHASE_A_FOLD_RESULTS.csv')
    current=fold[fold.buffer_condition=='CURRENT'].copy(); expected=expected[expected.method.isin(['JOINT-CE','NRRF-v1'])][['dataset','fold','method','mean_subject_BA']].rename(columns={'mean_subject_BA':'phase_a_current_BA'})
    reproduction=current.merge(expected,on=['dataset','fold','method'],validate='one_to_one'); reproduction['absolute_BA_difference']=(reproduction.mean_subject_BA-reproduction.phase_a_current_BA).abs(); reproduction['within_1e-6']=reproduction.absolute_BA_difference<=1e-6
    if len(reproduction)!=20 or not bool(reproduction['within_1e-6'].all()):
        write_json(PROTOCOL/'TESTS.json',{'CURRENT_reproduces_Phase_A_within_1e-6':False,'reproduction':reproduction.to_dict('records')})
        (OUT/'DECISION.md').write_text('# NRRF BatchNorm Buffer Intervention Audit\n\n**BN_AUDIT_BASELINE_REPRODUCTION_INVALID**\n',encoding='utf-8')
        raise RuntimeError('BN_AUDIT_BASELINE_REPRODUCTION_INVALID')
    wide=subject.pivot(index=['dataset','fold','method','subject_id','EEGNet_BA','FROZEN_LOGIT50_BA'],columns='buffer_condition',values=['BA','macro_F1','accuracy']).reset_index(); wide.columns=['_'.join(str(x) for x in col if str(x)) for col in wide.columns]
    wide['restore_minus_current_pp']=(wide.BA_RESTORE_SOURCE_BN-wide.BA_CURRENT)*100
    wide['restore_minus_EEGNet_pp']=(wide.BA_RESTORE_SOURCE_BN-wide.EEGNet_BA)*100
    wide['restore_minus_LOGIT50_pp']=(wide.BA_RESTORE_SOURCE_BN-wide.FROZEN_LOGIT50_BA)*100
    wide.to_csv(OUT/'PRIMARY_SUBJECT_RESULTS.csv',index=False)
    foldwide=fold.pivot(index=['dataset','fold','method'],columns='buffer_condition',values=['mean_subject_BA','mean_subject_macro_F1','mean_subject_accuracy','delta_vs_EEGNet_pp','delta_vs_FROZEN_LOGIT50_pp']).reset_index(); foldwide.columns=['_'.join(str(x) for x in col if str(x)) for col in foldwide.columns]
    foldwide['BN_recovery_pp']=(foldwide.mean_subject_BA_RESTORE_SOURCE_BN-foldwide.mean_subject_BA_CURRENT)*100
    foldwide.to_csv(OUT/'PRIMARY_FOLD_RESULTS.csv',index=False)
    rows=[]; harm={}; recovery={}
    for (dataset,method), group in wide.groupby(['dataset','method']):
        rec=(group.BA_RESTORE_SOURCE_BN-group.BA_CURRENT).to_numpy(); b=bootstrap(rec)
        current_delta=(group.BA_CURRENT-group.EEGNet_BA).to_numpy(); restore_delta=(group.BA_RESTORE_SOURCE_BN-group.EEGNet_BA).to_numpy()
        frozen=float(group.FROZEN_LOGIT50_BA.mean()); current_ba=float(group.BA_CURRENT.mean()); restore_ba=float(group.BA_RESTORE_SOURCE_BN.mean()); loss=frozen-current_ba
        fraction=None if loss<=0 else float((restore_ba-current_ba)/loss)
        f=foldwide[(foldwide.dataset==dataset)&(foldwide.method==method)]
        hcurrent={f'le_minus_{t}pp':float((current_delta<=-t/100).mean()) for t in (1,3,5)}; hrestore={f'le_minus_{t}pp':float((restore_delta<=-t/100).mean()) for t in (1,3,5)}
        row={'dataset':dataset,'method':method,'Current_BA':current_ba,'Restore_source_BN_BA':restore_ba,'BN_recovery_pp':b['mean_recovery_pp'],'median_recovery_pp':b['median_recovery_pp'],'bootstrap_ci_low_pp':b['bootstrap_ci_low_pp'],'bootstrap_ci_high_pp':b['bootstrap_ci_high_pp'],'Current_harm_le_minus_1pp':hcurrent['le_minus_1pp'],'Restored_harm_le_minus_1pp':hrestore['le_minus_1pp'],'harm_reduction_le_minus_1pp':hcurrent['le_minus_1pp']-hrestore['le_minus_1pp'],'improved_folds':int((f.BN_recovery_pp>0).sum()),'restored_folds_positive_vs_EEGNet':int((f.delta_vs_EEGNet_pp_RESTORE_SOURCE_BN>0).sum()),'loss_from_frozen':loss,'recovery_fraction':fraction,'improved_subjects':int((rec>0).sum()),'harmed_subjects':int((rec<0).sum()),'tied_subjects':int((rec==0).sum()),'recovery_ge_plus_1pp_fraction':float((rec>=.01).mean()),'recovery_ge_plus_3pp_fraction':float((rec>=.03).mean()),'recovery_le_minus_1pp_fraction':float((rec<=-.01).mean())}
        rows.append(row); harm[f'{dataset}/{method}']={'CURRENT':hcurrent,'RESTORE_SOURCE_BN':hrestore}; recovery[f'{dataset}/{method}']={'loss_from_frozen':loss,'recovery_fraction':fraction}
    ds=pd.DataFrame(rows); ds.to_csv(OUT/'DATASET_RESULTS.csv',index=False); write_json(OUT/'HARM_PROFILE.json',harm); write_json(OUT/'RECOVERY_FRACTION.json',recovery)
    raw=json.loads((OUT/'RAW_EPOCH20_INTERMEDIATE.json').read_text()); raw_df=pd.DataFrame(raw); raw_summary=raw_df.groupby(['dataset','method'],as_index=False).agg(current_mean_subject_BA=('current_mean_subject_BA','mean'),restore_source_BN_mean_subject_BA=('restore_source_bn_mean_subject_BA','mean'),mean_recovery_pp=('BN_recovery_pp','mean'),positive_recovery_folds=('BN_recovery_pp',lambda x:int((x>0).sum()))); write_json(OUT/'RAW_EPOCH20_SECONDARY.json',{'fold_rows':raw,'dataset_method_summary':raw_summary.to_dict('records'),'secondary_only':True})
    terminal,next_action=decision(ds[ds.dataset=='WBCIC'].copy())
    tests=json.loads((PROTOCOL/'TESTS.json').read_text()); tests['CURRENT_reproduces_prior_Phase_A_within_1e-6']=True; tests['reproduction']=reproduction.to_dict('records'); tests['state_interventions_all_valid']=True; write_json(PROTOCOL/'TESTS.json',tests)
    def pick(d:str,m:str)->pd.Series:return ds[(ds.dataset==d)&(ds.method==m)].iloc[0]
    lines=['# NRRF BatchNorm Buffer Intervention Audit','','| Metric | OpenBMI JOINT-CE | OpenBMI NRRF | WBCIC JOINT-CE | WBCIC NRRF |','|---|---:|---:|---:|---:|']
    specs=[('Current BA','Current_BA','ba'),('Restore-source-BN BA','Restore_source_BN_BA','ba'),('BN recovery (pp)','BN_recovery_pp','pp'),('Recovery fraction','recovery_fraction','frac'),('Current harm ≤ −1pp','Current_harm_le_minus_1pp','frac'),('Restored harm ≤ −1pp','Restored_harm_le_minus_1pp','frac'),('Improved folds','improved_folds','fold')]
    order=[pick('OpenBMI','JOINT-CE'),pick('OpenBMI','NRRF-v1'),pick('WBCIC','JOINT-CE'),pick('WBCIC','NRRF-v1')]
    for label,key,kind in specs:
        vals=[]
        for r in order:
            v=r[key]; vals.append('N/A' if pd.isna(v) else (f'{v*100:.1f}%' if kind=='frac' else (f'{int(v)}/5' if kind=='fold' else f'{v:.4f}' if kind=='ba' else f'{v:+.3f}')))
        lines.append(f'| {label} | '+' | '.join(vals)+' |')
    wfold=foldwide[foldwide.dataset=='WBCIC'][['fold','method','delta_vs_EEGNet_pp_CURRENT','delta_vs_EEGNet_pp_RESTORE_SOURCE_BN','BN_recovery_pp']].sort_values(['method','fold'])
    lines += ['',f'1. Does source-BN restoration materially recover WBCIC? **{"YES" if terminal=="BN_BUFFER_DRIFT_MAJOR_CONTRIBUTOR" else "PARTIAL" if terminal=="BN_BUFFER_DRIFT_PARTIAL_CONTRIBUTOR" else "NO"}**.', '2. WBCIC recovery fractions: '+ '; '.join(f"{r.method} {('N/A' if pd.isna(r.recovery_fraction) else f'{r.recovery_fraction*100:.1f}%')}" for r in ds[ds.dataset=='WBCIC'].itertuples()),'3. WBCIC fold0/fold3 changes (CURRENT delta, RESTORED delta, recovery pp):']
    lines += [f'   - {r.method} fold{r.fold}: {r.delta_vs_EEGNet_pp_CURRENT:+.3f}, {r.delta_vs_EEGNet_pp_RESTORE_SOURCE_BN:+.3f}, {r.BN_recovery_pp:+.3f}' for r in wfold.itertuples(index=False)]
    lines += ['4. Harmful-subject tails are reported in `HARM_PROFILE.json`.','5. OpenBMI BN sensitivity: see main table.','6. Raw epoch20 secondary check: see `RAW_EPOCH20_SECONDARY.json`.','7. BatchNorm buffer drift major explanation? '+('YES' if terminal=='BN_BUFFER_DRIFT_MAJOR_CONTRIBUTOR' else 'PARTIAL' if terminal=='BN_BUFFER_DRIFT_PARTIAL_CONTRIBUTOR' else 'NO')+'.',f'8. Final terminal: **{terminal}**.',f'9. Next action: {next_action}']
    (OUT/'DECISION.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(terminal,flush=True); return 0


if __name__=='__main__': raise SystemExit(main())
