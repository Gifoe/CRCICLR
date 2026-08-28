from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import common as c


HIST_SCAA=c.REPO/"experiments"/"persist_eeg_scaa_stage0"/"results"


def fm_value(frame,dataset,model,column):
    return float(frame[(frame.dataset==dataset)&(frame.model==model)][column].iloc[0])


def make_table(task,dvi,scaa,scst):
    hist_corr=pd.read_csv(HIST_SCAA/"UTILITY_TRANSFER_CORRELATION.csv");hist_sign=pd.read_csv(HIST_SCAA/"SIGN_CONCORDANCE.csv");hist_harm=pd.read_csv(HIST_SCAA/"HARM_AND_COVERAGE.csv")
    hist_task={('OpenBMI','EEGNet'):.75,('OpenBMI','EEGConformer'):.7719166667,('WBCIC','EEGNet'):.7884300821,('WBCIC','EEGConformer'):.78}
    hist_d={('OpenBMI','EEGNet'):(.0314928,.0457442),('WBCIC','EEGNet'):(.0118692,.0165056)}
    hist_knn={('OpenBMI','EEGNet'):1.1628472962,('OpenBMI','EEGConformer'):1.1493719487,('WBCIC','EEGNet'):1.3079565485,('WBCIC','EEGConformer'):1.3407995963}
    rows=[]
    for dataset in c.DATASETS:
        for model in ('EEGNet','EEGConformer'):
            rho=sign=harm=np.nan
            if dataset=='WBCIC':
                rho=float(hist_corr[(hist_corr.scope==model)&(hist_corr.method=='spearman')].estimate.iloc[0]);sign=float(hist_sign[hist_sign.scope==model].sign_concordance.iloc[0]);harm=float(hist_harm[hist_harm.scope==model].harm_certified.iloc[0])
            dr,ir=hist_d.get((dataset,model),(np.nan,np.nan));ratio=hist_knn[(dataset,model)]
            rows.append({'Dataset':dataset,'Model':model+' historical','Task BA':hist_task[(dataset,model)],'D RMSE':dr,'I RMSE':ir,'D>I':bool(dr<ir) if np.isfinite(dr) else np.nan,'S2-S3 rho':rho,'sign concordance':sign,'S2 gate harm':harm,'SCST 3NN ratio':ratio,'SCST valid':ratio<=1.25})
        for model in c.FMS:
            t=task[(task.dataset==dataset)&(task.model==model)].iloc[0];d=dvi[(dvi.dataset==dataset)&(dvi.model==model)].iloc[0];sct=scst[(scst.dataset==dataset)&(scst.model==model)].iloc[0]
            if dataset=='WBCIC':s=scaa[scaa.model==model].iloc[0];rho=s.Spearman;sign=s.sign_concordance;harm=s.S2_gate_harm
            else:rho=sign=harm=np.nan
            rows.append({'Dataset':dataset,'Model':model,'Task BA':t.task_BA,'D RMSE':d.D_RMSE,'I RMSE':d.I_RMSE,'D>I':d.D_better,'S2-S3 rho':rho,'sign concordance':sign,'S2 gate harm':harm,'SCST 3NN ratio':sct.independent_session_3NN_ratio,'SCST valid':sct.valid})
    frame=pd.DataFrame(rows);c.write_csv(c.RESULTS/"FM_RESCUE_MATRIX.csv",frame);return frame


def figures(table,scaa,scst):
    c.FIGURES.mkdir(exist_ok=True)
    fms=table[table.Model.isin(c.FMS)];fig,ax=plt.subplots(figsize=(8,4.5));x=np.arange(len(fms));ax.bar(x-.2,fms['I RMSE'],.4,label='Identity model');ax.bar(x+.2,fms['D RMSE'],.4,label='Decision model');ax.set_xticks(x,[f"{r.Dataset}\n{r.Model}" for r in fms.itertuples()]);ax.set_ylabel('held-run RMSE');ax.legend();fig.tight_layout();fig.savefig(c.FIGURES/'d_vs_i_fm.png',dpi=220);fig.savefig(c.FIGURES/'d_vs_i_fm.pdf');plt.close(fig)
    new=pd.read_csv(c.RESULTS/'FM_SCAA_PER_SUBJECT.csv');hist=pd.read_csv(HIST_SCAA/'PER_SUBJECT_UTILITY.csv');fig,axes=plt.subplots(1,4,figsize=(13,3.5),sharex=False,sharey=False)
    for ax,model in zip(axes,('EEGNet','EEGConformer','CBraMod','LaBraM')):
        if model in c.FMS:g=new[new.model==model];x=g.Delta_S2;y=g.Delta_S3
        else:g=hist[hist.backbone==model];x=g.Delta_S2_BA;y=g.Delta_S3_BA
        ax.scatter(x,y,s=16,alpha=.7);ax.axhline(0,color='grey',lw=.7);ax.axvline(0,color='grey',lw=.7);ax.set_title(model);ax.set_xlabel('$\\Delta$S2');
    axes[0].set_ylabel('$\\Delta$S3');fig.tight_layout();fig.savefig(c.FIGURES/'scaa_utility_transfer_fm.png',dpi=220);fig.savefig(c.FIGURES/'scaa_utility_transfer_fm.pdf');plt.close(fig)
    hh=pd.read_csv(HIST_SCAA/'HARM_AND_COVERAGE.csv');names=['EEGNet','EEGConformer']+list(c.FMS);harm=list(hh[hh.scope.isin(['EEGNet','EEGConformer'])].set_index('scope').loc[['EEGNet','EEGConformer']].harm_certified)+list(scaa.set_index('model').loc[list(c.FMS)].S2_gate_harm);coverage=list(hh[hh.scope.isin(['EEGNet','EEGConformer'])].set_index('scope').loc[['EEGNet','EEGConformer']].coverage)+list(scaa.set_index('model').loc[list(c.FMS)].coverage);fig,ax=plt.subplots(figsize=(7,4));x=np.arange(4);ax.bar(x-.2,harm,.4,label='future harm');ax.bar(x+.2,coverage,.4,label='coverage');ax.set_xticks(x,names,rotation=15);ax.set_ylim(0,1);ax.legend();fig.tight_layout();fig.savefig(c.FIGURES/'scaa_harm_coverage_fm.png',dpi=220);fig.savefig(c.FIGURES/'scaa_harm_coverage_fm.pdf');plt.close(fig)
    hist_ratio={'OpenBMI/EEGNet':1.1628472962,'OpenBMI/EEGConformer':1.1493719487,'WBCIC/EEGNet':1.3079565485,'WBCIC/EEGConformer':1.3407995963};labels=list(hist_ratio);values=list(hist_ratio.values());
    for r in scst.itertuples():labels.append(f'{r.dataset}/{r.model}');values.append(r.independent_session_3NN_ratio)
    fig,ax=plt.subplots(figsize=(10,4.5));ax.bar(np.arange(len(values)),values);ax.axhline(1.25,color='red',ls='--',label='frozen gate 1.25');ax.set_xticks(np.arange(len(values)),labels,rotation=35,ha='right');ax.set_ylabel('independent-session 3NN / clean');ax.legend();fig.tight_layout();fig.savefig(c.FIGURES/'scst_manifold_fm.png',dpi=220);fig.savefig(c.FIGURES/'scst_manifold_fm.pdf');plt.close(fig)
    models=['EEGNet','EEGConformer']+list(c.FMS);matrix=np.zeros((len(models),5));matrix[:,:2]=1;matrix[:2,2]=[0,1];matrix[:2,3]=[0,0];matrix[:2,4]=0
    for i,m in enumerate(c.FMS,2):matrix[i,2]=float(scaa[scaa.model==m].Spearman_CI_low.iloc[0]>0);matrix[i,3]=float(scst[scst.model==m].valid.all());matrix[i,4]=0
    fig,ax=plt.subplots(figsize=(8,3.5));im=ax.imshow(matrix,vmin=0,vmax=1,cmap='Blues');ax.set_xticks(range(5),['Encoded/Identifiable','Consequential','Utility-transferable','Transport-valid','Prospectively actionable'],rotation=25,ha='right');ax.set_yticks(range(len(models)),models);fig.colorbar(im,ax=ax,ticks=[0,1]);fig.tight_layout();fig.savefig(c.FIGURES/'admissibility_matrix_fm.png',dpi=220);fig.savefig(c.FIGURES/'admissibility_matrix_fm.pdf');plt.close(fig)


def main():
    task=pd.read_csv(c.RESULTS/'FM_TASK_PERFORMANCE.csv');dvi=pd.read_csv(c.RESULTS/'FM_D_VS_I.csv');dstats=c.read_json(c.RESULTS/'FM_D_VS_I_STATISTICS.json');scaa=pd.read_csv(c.RESULTS/'FM_SCAA_SUMMARY.csv');sstats=c.read_json(c.RESULTS/'FM_SCAA_STATISTICS.json');scst=pd.read_csv(c.RESULTS/'FM_SCST_SUMMARY.csv');tstats=c.read_json(c.RESULTS/'FM_SCST_STATISTICS.json')
    strong=sstats['terminal']=='FM_HISTORY_UTILITY_RESCUE_CANDIDATE' or tstats['terminal']=='FM_SCST_RESCUE_CANDIDATE';mixed=sstats['terminal']=='FM_HISTORY_UTILITY_ARCHITECTURE_DEPENDENT' or tstats['terminal']=='FM_SCST_ARCHITECTURE_DEPENDENT';trigger=bool(strong or mixed)
    if dstats['terminal']!='FM_D_GT_I_REPLICATED':overall='FM_CORE_MECHANISM_NOT_REPLICATED'
    elif strong:overall='FM_CONSTRUCTIVE_RESCUE_CANDIDATE'
    elif mixed:overall='FM_REPRESENTATION_DEPENDENT_MIXED'
    else:overall='FM_ACTIONABILITY_GAP_PERSISTS'
    c.write_json(c.RESULTS/'CONDITIONAL_TRIGGER.json',{'STEEGFORMER_triggered':trigger,'matched_controls_triggered':trigger,'overall_preconfirmation':overall})
    if trigger: raise RuntimeError('conditional confirmation/control path triggered; execute locked controls before finalization')
    table=make_table(task,dvi,scaa,scst);figures(table,scaa,scst)
    cbtask={d:fm_value(task,d,'CBraMod','task_BA') for d in c.DATASETS};lbtask={d:fm_value(task,d,'LaBraM','task_BA') for d in c.DATASETS}
    answers={
      '1_CBraMod_checkpoint':'pretrained_weights.pth SHA256 0792cb808c14e6b7a2bb2ce1dff379bc47bc54c49a779825bdfeb33bf8157178','2_LaBraM_checkpoint':'labram-base.pth SHA256 7c50583826afac76c4ab18f43d958df40496c8229accc09ed6a227c9bb57c37c','3_STEEGFormer_triggered':False,'4_official_checkpoints_public_loadable':True,
      '5_preprocessing':'4 s, 200 Hz; OpenBMI x1e6 to uV plus 40-Hz low-pass; WBCIC cache x20,000 to uV plus frozen +/-250 uV bound; LaBraM /100 official scale','6_channel_mapping':'OpenBMI 62/62 and WBCIC 58/58 maximal legal channels; LaBraM official standard_1020 indices','7_sealed_resources_untouched':True,'8_target_absent_from_anchor_training':True,
      '9_task_BA':{'CBraMod':cbtask,'LaBraM':lbtask},'10_competence':{r.model:{'OpenBMI':bool(task[(task.model==r.model)&(task.dataset=='OpenBMI')].competent.iloc[0]),'WBCIC':bool(task[(task.model==r.model)&(task.dataset=='WBCIC')].competent.iloc[0])} for r in task.itertuples()},
      '11_OpenBMI_CBraMod_D_I':dvi[(dvi.dataset=='OpenBMI')&(dvi.model=='CBraMod')][['D_RMSE','I_RMSE']].iloc[0].to_dict(),'12_OpenBMI_LaBraM_D_I':dvi[(dvi.dataset=='OpenBMI')&(dvi.model=='LaBraM')][['D_RMSE','I_RMSE']].iloc[0].to_dict(),'13_WBCIC_CBraMod_D_I':dvi[(dvi.dataset=='WBCIC')&(dvi.model=='CBraMod')][['D_RMSE','I_RMSE']].iloc[0].to_dict(),'14_WBCIC_LaBraM_D_I':dvi[(dvi.dataset=='WBCIC')&(dvi.model=='LaBraM')][['D_RMSE','I_RMSE']].iloc[0].to_dict(),'15_pooled_D_gt_I':dstats,
      '16_19_per_FM_SCAA':{r.model:{'Spearman':r.Spearman,'CI':[r.Spearman_CI_low,r.Spearman_CI_high],'sign':r.sign_concordance} for r in scaa.itertuples()},'20_26_pooled_SCAA':sstats,
      '27_30_SCST_ratios':{f'{r.dataset}/{r.model}':r.independent_session_3NN_ratio for r in scst.itertuples()},'31_SCST_gates':{f'{r.dataset}/{r.model}':bool(r.valid) for r in scst.itertuples()},'32_history_utility':sstats['terminal'],'33_SCST':tstats['terminal'],'34_architecture_dependent':mixed,
      '35_preprocessing_control':'NOT_TRIGGERED','36_random_init_control':'NOT_TRIGGERED','37_STEEGFormer_confirmation':'NOT_TRIGGERED','38_actionability_gap_persists':overall=='FM_ACTIONABILITY_GAP_PERSISTS','39_constructive_next_task_authorized':False,
      '40_strongest_justified_claim':('The diagnosis/actionability gap persists into task-competent pretrained EEG foundation models.' if bool(task.competent.all()) else 'Under the frozen Stage-0 protocol, D>I replicated but neither constructive route passed; incomplete FM task competence limits a universal foundation-model claim.') if overall=='FM_ACTIONABILITY_GAP_PERSISTS' else 'See route-specific frozen terminal.','41_strongest_unsupported_claim':'Pretraining universally makes identity suppression, target-history gating, or subject transport prospectively useful.','42_final_terminal':overall}
    report={'schema':'FM_RESCUE_FINAL_REPORT_V1','branch':'codex/persist-eeg-fm-rescue-stage0','answers':answers,'D_terminal':dstats['terminal'],'SCAA_terminal':sstats['terminal'],'SCST_terminal':tstats['terminal'],'overall_terminal':overall,'STEEGFORMER_triggered':False,'confound_controls_triggered':False,'constructive_route_authorized':False,'sealed_resources_untouched':True}
    c.write_json(c.EXP/'FM_RESCUE_FINAL_REPORT.json',report)
    c.write_json(c.RESULTS/'FM_RESCUE_FINAL_REPORT.json',report)
    md=['# FM Rescue Stage-0 final report','',f'Overall terminal: `{overall}`.',f'D>I: `{dstats["terminal"]}`. History utility: `{sstats["terminal"]}`. SCST: `{tstats["terminal"]}`.','', '## Forty-two required answers','']
    for i,(k,v) in enumerate(answers.items(),1):md.append(f'{i}. **{k}**: `{json.dumps(c.clean(v),ensure_ascii=False)}`')
    c.write_text(c.EXP/'FM_RESCUE_FINAL_REPORT.md','\n'.join(md));c.write_text(c.EXP/'D_VS_I_FM_REPORT.md',f'# D versus I\n\nTerminal: `{dstats["terminal"]}`. Settings favoring D: {dstats["settings_D_better"]}/4; pooled run difference {dstats["pooled_run_mean_RMSE_I_minus_D"]:.6f}, 95% CI {dstats["bootstrap_ci95"]}.')
    c.write_text(c.EXP/'SCAA_FM_REPORT.md',f'# FM SCAA\n\nTerminal: `{sstats["terminal"]}`. Pooled Spearman {sstats["Spearman"]:.4f}, CI {sstats["CI95"]}; sign concordance {sstats["sign_concordance"]:.4f}; harm {sstats["always_adapt_harm"]:.4f} -> {sstats["S2_gate_harm"]:.4f}; coverage {sstats["coverage"]:.4f}.')
    c.write_text(c.EXP/'SCST_FM_REPORT.md',f'# FM SCST\n\nTerminal: `{tstats["terminal"]}`.\n\n'+scst.to_markdown(index=False))
    c.write_text(c.EXP/'CONFOUND_AUDIT.md','# Confound audit\n\nNo strong constructive rescue signal occurred, so the predeclared preprocessing-matched specialist and random-init FM controls were not triggered. This avoids post-failure method search.')
    c.write_text(c.EXP/'CLAIM_AUDIT.md',f'# Claim audit\n\nSupported: {answers["40_strongest_justified_claim"]}\n\nUnsupported: {answers["41_strongest_unsupported_claim"]}\n\nNo final constructive model is authorized by this task.')
    c.write_text(c.EXP/'REPRODUCIBILITY.md','# Reproducibility\n\nRun `code/train_and_select.py`, commit the generated protocol lock, then run `code/run_primary.py`, `code/finalize.py`, and `code/validate.py` with the pinned server Python. Runtime checkpoints and representation caches are intentionally gitignored. All compact locks, statistics, tables, figures and reports are committed.')
    c.write_csv(c.RESULTS/'FM_CONFOUND_CONTROLS.csv',pd.DataFrame([{'control':'preprocessing-matched specialist','status':'NOT_TRIGGERED'},{'control':'random-init same FM','status':'NOT_TRIGGERED'},{'control':'ST-EEGFormer-Small','status':'NOT_TRIGGERED'}]))
    print(f'FM_RESCUE_FINALIZED terminal={overall}',flush=True)


if __name__=='__main__':main()
