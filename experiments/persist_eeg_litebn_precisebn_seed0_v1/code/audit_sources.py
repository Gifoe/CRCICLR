"""Read-only checkpoint recovery gate. Never loads EEG or changes a checkpoint."""
import argparse, csv, hashlib, json, os, subprocess
from pathlib import Path
import torch

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def write_csv(p, rows, fields=None):
    with p.open('w',newline='',encoding='utf-8') as h:
        w=csv.DictWriter(h,fieldnames=fields or list(rows[0]));w.writeheader();w.writerows(rows)
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--repo',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
    a.output.mkdir(parents=True,exist_ok=True)
    source=a.repo/'experiments/persist_eeg_carrier_dualdataset_screen_v1/code/run_carrier_screen.py'
    inventory=[]; matches=[]; errors=[]; manifest=[]
    # Do not follow cache junctions. Inventory binaries, inspect small candidates;
    # large optimizer/foundation states are inventoried without deserializing.
    for directory,dirs,files in os.walk(a.root,followlinks=False):
        dirs[:]=[d for d in dirs if d not in ['.git','__pycache__'] and not os.path.islink(Path(directory)/d)]
        for name in files:
            p=Path(directory)/name
            if p.suffix.lower() not in ['.pt','.pth','.zip','.tar','.gz','.bundle']:continue
            size=p.stat().st_size
            row={'path':str(p),'bytes':size,'status':'INVENTORIED_NOT_LOADED'}
            if p.suffix.lower() in ['.pt','.pth'] and size<2_000_000:
                try:
                    c=torch.load(p,map_location='cpu',weights_only=True)
                    s=next((c[k] for k in ['state_dict','model','best_state','current_state'] if isinstance(c,dict) and k in c),c)
                    if isinstance(s,dict) and 'depth1.weight' in s:
                        shapes={k:list(v.shape) for k,v in s.items() if torch.is_tensor(v)}
                        exact=(shapes.get('depth1.weight')==[48,1,1,15] and shapes.get('point1.weight')==[64,48,1,1] and shapes.get('depth2.weight')==[64,1,1,31] and shapes.get('embedding.0.weight')==[64,512])
                        row.update(status='EXACT_SHAPE_CANDIDATE' if exact else 'DIFFERENT_ARCHITECTURE',sha256=sha(p),depth1=str(shapes['depth1.weight']),point1=str(shapes.get('point1.weight')),head=str(shapes.get('head.weight')))
                        if exact: matches.append({**row,'shapes':shapes})
                    else:row['status']='NOT_COMPACTLITE'
                except Exception as e:row['status']='LOAD_ERROR';errors.append({'path':str(p),'error':str(e)[:500]})
            inventory.append(row)
    for task in ['OpenBMI_MI','WBCIC_MI','OpenBMI_ERP','OpenBMI_SSVEP']:
        for f in range(5):
            mi=task in ['OpenBMI_MI','WBCIC_MI']
            p=(a.root/'carrier_5fold_multiseed_stability_runtime'/f"{'openbmi' if task=='OpenBMI_MI' else 'wbcic'}_fold{f}_seed0_litebn"/'selected_best.pt') if mi else (a.root/'seven_backbone_fourtask_3seed_runtime/search_cells'/task.lower()/'litebn'/f'fold{f}_seed0/selected.pt')
            r=next((x for x in inventory if x['path']==str(p)),{})
            metadata={};record=p.parent/'record.json'
            if record.exists():metadata=json.loads(record.read_text())
            if mi:
                logs=a.repo/'experiments/persist_eeg_carrier_5fold_multiseed_stability_v1/outputs/TRAINING_LOGS.json'
                rows=json.loads(logs.read_text())
                metadata=next((x for x in rows if x.get('dataset')==('OpenBMI' if task=='OpenBMI_MI' else 'WBCIC') and x.get('fold')==f and x.get('seed')==0 and x.get('model')=='LiteBN'),{})
            manifest.append({'task':task,'fold':f,'seed':0,'checkpoint_path':str(p),'checkpoint_sha256':r.get('sha256',''),'architecture_status':r.get('status','MISSING'),'selected_epoch':metadata.get('selected_epoch','UNKNOWN'),'split_sha256':metadata.get('split_sha256',sha(a.repo/'experiments/persist_eeg_carrier_5fold_multiseed_stability_v1/protocol/FIVEFOLD_SPLIT.json')),'normalization_provenance':json.dumps(metadata.get('normalizer',{'status':'NOT_YET_REPLAY_VERIFIED'})),'source_commit':'UNKNOWN_CHECKPOINT_CREATION_COMMIT','historical_metadata':json.dumps(metadata),'replay_status':'NOT_RUN_SOURCE_GATE_BLOCKED'})
    write_csv(a.output/'LITEBN_PRECISEBN_SOURCE_MANIFEST.csv',manifest)
    write_csv(a.output/'CHECKPOINT_RECOVERY_INVENTORY.csv',inventory,['path','bytes','status','sha256','depth1','point1','head'])
    (a.output/'CHECKPOINT_RECOVERY_DETAILS.json').write_text(json.dumps({'exact_shape_candidates':matches,'load_errors':errors,'canonical_source_path':str(source),'canonical_source_sha256':sha(source),'inspected_repo_commit':subprocess.check_output(['git','-C',str(a.repo),'rev-parse','HEAD'],text=True).strip(),'note':'Shape candidates do not establish task/seed/selection or weight provenance.'},indent=2))
    print(json.dumps({'inventory':len(inventory),'shape_candidates':len(matches),'errors':len(errors),'manifest_statuses':{t:[r['architecture_status'] for r in manifest if r['task']==t] for t in ['OpenBMI_MI','WBCIC_MI','OpenBMI_ERP','OpenBMI_SSVEP']} }),flush=True)
if __name__=='__main__':main()
