"""Independent artifact consistency checks; does not need EEG data or CUDA."""
import argparse,csv,json,math
from pathlib import Path

def rows(p):
    with p.open(encoding='utf-8') as h:return list(csv.DictReader(h))
def verify(out):
    meta=json.loads((out/'RUN_METADATA.json').read_text());assert meta['training_cells']==20 and meta['SEED']==0 and meta['initial_equivalence_passed']==20
    eq=rows(out/'INITIAL_FUNCTION_EQUIVALENCE.csv');assert len(eq)==20 and len({(r['task'],r['fold']) for r in eq})==20
    for r in eq:
        assert r['status']=='PASS' and int(r['prediction_mismatch'])==0 and float(r['max_abs_logit_diff'])<=1e-6
        assert all(r[k]=='True' for k in ['shared_parameters_identical','U_zero','V_nonzero','rng_restored'])
    params=rows(out/'PARAMETER_COUNT.csv');assert len(params)==20
    for r in params:assert int(r['extra_parameters'])==6656==int(r['LS_parameters'])-int(r['baseline_parameters']) and r['q_dim']=='768' and r['rank']==r['bins']=='8'
    batches=rows(out/'EXACT_BATCH_AUDIT.csv');updates=rows(out/'UPDATE_AUDIT.csv');assert len(batches)==len(updates)==1200
    by={(r['task'],r['fold'],r['epoch']):r for r in batches};assert len(by)==1200
    previous={};skips=0
    for r in updates:
        b=by[(r['task'],r['fold'],r['epoch'])];assert int(r['attempts'])==int(b['batches']) and int(r['exposure'])==int(b['trial_exposure']) and b['match']=='True'
        key=(r['task'],r['fold']);step=int(r['successful_cumulative']);success=step-previous.get(key,0);assert 0<=success<=int(r['attempts']);skips+=int(r['attempts'])-success;previous[key]=step
    logs=json.loads((out/'TRAINING_LOGS.json').read_text());assert len(logs)==20
    for r in logs:
        h=r['history'];assert [v['epoch'] for v in h]==list(range(1,61))
        best=-float('inf');chosen=None
        for v in h[9:]:
            if v['inner_val_subject_BA']>best+1e-12:best=v['inner_val_subject_BA'];chosen=v['epoch']
        assert r['selected_epoch']==chosen and abs(r['best_inner_val_BA']-best)<1e-12
    pairs=rows(out/'PAIRED_SUBJECT_RESULTS.csv');assert len(pairs)==411 and len({(r['task'],r['fold'],r['population'],r['subject_id']) for r in pairs})==411
    assert sum(r['population']=='outer' for r in pairs)==151
    for r in pairs:
        for k in ['BA','macro_F1','accuracy']:assert math.isclose(float(r[f'delta_{k}_pp']),100*(float(r[f'LS_{k}'])-float(r[f'LiteBN_{k}'])),abs_tol=1e-8)
    assert len(rows(out/'OUTER_FOLD_RESULTS.csv'))==20 and len(rows(out/'INTERNAL_HELDOUT_SUBJECT_RESULTS.csv'))==52
    summary=rows(out/'OUTER_TASK_SUMMARY.csv');assert len(summary)==4
    assert math.isclose(sum(float(r['delta_BA_pp']) for r in summary)/4,meta['equal_task_mean_delta_pp'],abs_tol=1e-8)
    diag=rows(out/'LOCALSTATS_DIAGNOSTICS.csv');assert len(diag)==40
    for r in diag:
        for k in ['base_norm','stats_norm','stats_base_ratio','q_std','mu_contribution_norm','logvar_contribution_norm']:assert math.isfinite(float(r[k])) and float(r[k])>=0
    prov=json.loads((out/'BASELINE_PROVENANCE.json').read_text());assert len(prov)==40 and all(r['baseline_metric_replay_pass'] for r in prov)
    return {'status':'PASS','cells':20,'epochs':1200,'evaluation_pairs':411,'manifest_epochs':1200,'AMP_skipped_updates':skips,'terminal':meta['terminal']}
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('outputs',type=Path);a=p.parse_args();print(json.dumps(verify(a.outputs),indent=2))
