from pathlib import Path
import json
import numpy as np
import pandas as pd
import torch

root = Path(r'D:\nips-temp\TotalP\P1\CRCICLR_V7_FUTURE_UTILITY_META\experiments\persist_eeg_final_model_v7\outputs')
for name in ['OPENBMI_HISTORY_EA_EEGNET_FOLD_0.pt','OPENBMI_MI_SPECIFIC_FOLD_0_FEATURES.npy','OPENBMI_MI_SPECIFIC_FOLD_0_METADATA.parquet','OPENBMI_HISTORY_EA_EEGNET_FOLD_0_METADATA.parquet']:
    p = root / 'cache' / name
    print(name, 'exists', p.is_file(), 'size', p.stat().st_size if p.is_file() else None)
    if p.suffix == '.pt' and p.is_file():
        d = torch.load(p, map_location='cpu', weights_only=False)
        print('payload_keys', sorted(map(str, d.keys())))
        for k in ('configuration','fold','OUTER_TEST_USED','subject_ids','train_subjects','normalization'):
            if k in d:
                v = d[k]
                print(k, type(v).__name__, (len(v) if hasattr(v,'__len__') and not isinstance(v,(str,bytes)) else v))
    if p.suffix == '.npy' and p.is_file():
        a=np.load(p,mmap_mode='r',allow_pickle=False); print('shape',a.shape,'dtype',a.dtype)
    if p.suffix == '.parquet' and p.is_file():
        f=pd.read_parquet(p); print('columns',list(f.columns),'rows',len(f)); print('session_counts',f.groupby('session_id').size().to_dict() if 'session_id' in f else {})
