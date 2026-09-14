import json,os,time,torch
from pathlib import Path
import train_tfformer as t
R=Path(os.environ.get('TFF_REPO','/root/rivermind-data/CRCICLR_TFF_WORK'));O=R/'experiments/persist_eeg_litebn_tfformer_v1/outputs';d=torch.device('cuda');m,_=t.model('OpenBMI_SSVEP',0,d);m.eval();x=torch.zeros(1,62,1000,device=d);torch.cuda.reset_peak_memory_stats();
with torch.inference_mode():
 [m(x) for _ in range(5)];torch.cuda.synchronize();s=time.perf_counter();m(x);torch.cuda.synchronize()
q={'historical_LiteBN_parameters':sum(p.numel() for p in m.base.parameters()),'TFFormer_parameters':sum(p.numel() for p in m.parameters()),'single_forward_latency_ms':(time.perf_counter()-s)*1000,'peak_cuda_bytes':torch.cuda.max_memory_allocated(),'inference_mode':'eval/no-grad/one model/no labels/no adaptation'};(O/'EFFICIENCY_REPORT.json').write_text(json.dumps(q,indent=2)+'\n')
