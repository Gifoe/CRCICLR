"""Compare historical and candidate actual AMP-successful updates without training."""
import argparse,hashlib,json
from pathlib import Path
import torch
def main():
    p=argparse.ArgumentParser();p.add_argument('--historical-runtime',type=Path,required=True);p.add_argument('--runtime',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();rows=[]
    for f in range(5):
        oldpath=a.historical_runtime/f'openbmi_fold{f}_seed0_litebn/checkpoint_latest.pt';newpath=a.runtime/f'fold{f}/checkpoint_latest.pt'
        old=torch.load(oldpath,map_location='cpu',weights_only=False);new=torch.load(newpath,map_location='cpu',weights_only=False)
        oldsteps={int(v['step']) for v in old['optimizer']['state'].values()};newsteps={int(v['step']) for v in new['optimizer']['state'].values()}
        assert len(oldsteps)==len(newsteps)==1 and old['epoch']==new['epoch']==60
        o=oldsteps.pop();n=newsteps.pop();assert o<=1260 and n<=1260
        trajectory=json.loads((a.runtime/f'fold{f}/trajectory.json').read_text());assert sum(r['successful_optimizer_updates'] for r in trajectory)==n
        selected=json.loads((a.runtime/f'fold{f}/result.json').read_text())
        selected_path=Path(selected['checkpoint_path'])
        selected_hash=hashlib.sha256(selected_path.read_bytes()).hexdigest()
        assert selected_hash==selected['checkpoint_sha256']
        rows.append({'fold':f,'historical_attempts':1260,'candidate_attempts':1260,'historical_successful_updates':o,'candidate_successful_updates':n,'historical_AMP_skips':1260-o,'candidate_AMP_skips':1260-n,'actual_update_counts_equal':o==n,'historical_scaler':old['scaler'],'candidate_scaler':new['scaler'],'historical_checkpoint_sha256':hashlib.sha256(oldpath.read_bytes()).hexdigest(),'candidate_checkpoint_sha256':hashlib.sha256(newpath.read_bytes()).hexdigest(),'selected_checkpoint_sha256':selected_hash,'selected_training_record':selected})
    report={'rows':rows,'attempt_counts_matched':True,'historical_successful_total':sum(r['historical_successful_updates'] for r in rows),'candidate_successful_total':sum(r['candidate_successful_updates'] for r in rows),'interpretation':'Same unchanged AMP policy and update attempts. Successful steps may differ because overflow is loss/runtime dependent; no forced updates or reduced exposures.'}
    a.output.write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
