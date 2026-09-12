"""Resolve source-gate evidence; fail closed on incompatible historical sources."""
import argparse, ast, csv, hashlib, json, subprocess, zipfile
from pathlib import Path
import torch
from torch import nn
import torch.nn.functional as F
from audit_sources import sha, write_csv

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--repo',type=Path,required=True);a=ap.parse_args()
    exp=a.repo/'experiments/persist_eeg_litebn_precisebn_seed0_v1';out=exp/'outputs'
    manifest=list(csv.DictReader((out/'LITEBN_PRECISEBN_SOURCE_MANIFEST.csv').open(encoding='utf-8')))
    # Execute the actual historical class AST, not a reconstructed approximation.
    source=a.repo/'experiments/persist_eeg_carrier_dualdataset_screen_v1/code/run_carrier_screen.py'
    tree=ast.parse(source.read_text());nodes=[n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in ['norm_layer','CompactLite']]
    assert len(nodes)==2
    ns={'torch':torch,'nn':nn,'F':F};exec(compile(ast.Module(body=nodes,type_ignores=[]),str(source),'exec'),ns)
    checks=[]; replay=[]
    for r in manifest:
        historical=json.loads(r['historical_metadata']);historical.pop('history',None)
        r['historical_metadata']=json.dumps(historical)
        p=Path(r['checkpoint_path']);c=torch.load(p,map_location='cpu',weights_only=True);s=c.get('state_dict',c)
        model=ns['CompactLite'](58 if r['task']=='WBCIC_MI' else 62,'bn')
        if r['task']=='OpenBMI_SSVEP':model.head=nn.Linear(64,4)
        try:model.load_state_dict(s,strict=True);status='STRICT_STATE_LOAD_PASS';error=''
        except RuntimeError as e:status='STRICT_STATE_LOAD_FAIL';error=str(e)
        checks.append({'task':r['task'],'fold':r['fold'],'checkpoint_sha256':sha(p),'status':status,'error':error})
        r['strict_state_load']=status
        # Known local training checkpoints are trusted project artifacts. Inspect
        # their saved best_state only; no optimizer restoration or execution.
        if r['task'] in ['OpenBMI_ERP','OpenBMI_SSVEP']:
            latest=p.parent/'latest.pt';v=torch.load(latest,map_location='cpu',weights_only=False)
            best=v['best_state'];assert all(torch.equal(best[k].cpu(),s[k].cpu()) for k in s)
            checks.append({'task':r['task'],'fold':r['fold'],'checkpoint_sha256':sha(latest),'status':'LATEST_BEST_EQUALS_INCOMPATIBLE_SELECTED','error':''})
        replay.append({'task':r['task'],'fold':r['fold'],'seed':0,'historical_BA':json.loads(r['historical_metadata']).get('selected_outer_BA',json.loads(r['historical_metadata']).get('outer_development',{}).get('subject_equal_BA','')),'replayed_BA':'','delta_BA':'','status':'NOT_RUN_GLOBAL_SOURCE_GATE_BLOCKED'})
    write_csv(out/'LITEBN_PRECISEBN_SOURCE_MANIFEST.csv',manifest)
    write_csv(out/'STRICT_ARCHITECTURE_AUDIT.csv',checks)
    write_csv(out/'LITEBN_PRECISEBN_BASELINE_REPLAY.csv',replay)
    inventory=list(csv.DictReader((out/'CHECKPOINT_RECOVERY_INVENTORY.csv').open(encoding='utf-8')))
    archives=[]
    for r in inventory:
        p=Path(r['path'])
        if p.suffix=='.zip':
            try:
                with zipfile.ZipFile(p) as z: members=[x for x in z.namelist() if x.lower().endswith(('.pt','.pth'))]
                archives.append({'path':str(p),'checkpoint_members':members})
            except Exception as e:archives.append({'path':str(p),'error':str(e)})
    tracked=subprocess.check_output(['git','-C',str(a.repo),'ls-files','*.pt','*.pth'],text=True).splitlines()
    (out/'ADDITIONAL_RECOVERY_AUDIT.json').write_text(json.dumps({'zip_member_inventory':archives,'tracked_checkpoint_paths':tracked,'scope_limit':'Git bundles inventoried but not all historical objects deserialized; unrelated/large checkpoints not exhaustively loaded. No claim of global nonexistence.'},indent=2))
    failures=sum(x['status']=='STRICT_STATE_LOAD_FAIL' for x in checks)
    assert failures==10, 'Reassess gate: sources changed.'
    metadata={'status':'BLOCKED_SOURCE_ARCHITECTURE_MISMATCH','completed_four_task_diagnostic':False,'seed':0,'requested_checkpoints':20,'strict_load_pass':10,'strict_load_fail':10,'baseline_replay_completed':0,'calibrations_completed':0,'BN_BUFFERS_UPDATED':'NO','LEARNABLE_WEIGHTS_UPDATED':'NO','CALIBRATION_DATA':'INNER_TRAIN_ONLY (planned; no calibration performed)','INNER_VAL_USED_FOR_CALIBRATION':'NO','OUTER_USED_FOR_CALIBRATION':'NO','CURRENT_HELDOUT_USED_FOR_CALIBRATION':'NO','NEW_SEALED_FINAL_DATA_ACCESSED':'NO','EEG_DATA_LOADED':'NO','decision':None,'decision_reason':'Prerequisite failed, not evidence of useful or useless BN signal. Correct exact-historical ERP and SSVEP selected checkpoints plus provenance required. No substitute architecture or retraining permitted.','source_code_sha256':sha(source),'torch_version':torch.__version__}
    (out/'PRECISE_BN_METADATA.json').write_text(json.dumps(metadata,indent=2))
    # Header-only scientific outputs are explicitly NOT RUN, never zero effects.
    schemas={'STATE_AUDIT':['task','fold','max_abs_parameter_diff','non_BN_buffers_identical','status'],'LAYER_STATS':['task','fold','layer','calibration_count','old_mean_norm','new_mean_norm','mean_shift','old_variance_norm','new_variance_norm','variance_shift'],'OUTER_FOLD_RESULTS':['task','fold','original_BA','preciseBN_BA','delta_BA_pp','original_macro_F1','preciseBN_macro_F1','original_accuracy','preciseBN_accuracy'],'OUTER_TASK_SUMMARY':['task','original_BA','preciseBN_BA','delta_BA_pp','original_fold_SD','preciseBN_fold_SD','SD_change'],'HELDOUT_SUBJECT_RESULTS':['task','subject','original_BA','preciseBN_BA','delta_BA_pp'],'HELDOUT_TASK_SUMMARY':['task','original_BA','preciseBN_BA','delta_BA_pp']}
    for suffix,fields in schemas.items():write_csv(out/f'LITEBN_PRECISEBN_{suffix}.csv',[],fields)
    print(json.dumps(metadata),flush=True)
if __name__=='__main__':main()
