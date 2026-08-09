#!/usr/bin/env python
from __future__ import annotations
import json
from dataclasses import asdict
from pathlib import Path
import pandas as pd
from _common import project_root
from hsc_tta.v3.augmentations import nuisance_config_for
def main():
 root=project_root(".");base=root/"outputs/v3_probecert/action_search";corrected=root/"outputs/v3_probecert/action_search_mi_corrected"
 old_detail=pd.read_parquet(base/"ACTION_CONFIG_SUBJECT_RESULTS.parquet");new_detail=pd.read_parquet(corrected/"ACTION_CONFIG_SUBJECT_RESULTS.parquet")
 detail=pd.concat([old_detail[old_detail.dataset=="hmc"],new_detail],ignore_index=True);detail.to_parquet(base/"ACTION_CONFIG_SUBJECT_RESULTS.parquet",index=False)
 old_summary=pd.read_csv(base/"ACTION_CONFIG_RESULTS.csv");new_summary=pd.read_csv(corrected/"ACTION_CONFIG_RESULTS.csv");pd.concat([old_summary[old_summary.dataset=="hmc"],new_summary],ignore_index=True).to_csv(base/"ACTION_CONFIG_RESULTS.csv",index=False)
 old_selected=json.loads((base/"SELECTED_ACTION_CONFIGS.json").read_text());new_selected=json.loads((corrected/"SELECTED_ACTION_CONFIGS.json").read_text())
 selected=[x for x in old_selected if x["dataset"]=="hmc"]+new_selected
 for item in selected:item["nuisance_config"]=asdict(nuisance_config_for(item["dataset"]))
 (base/"SELECTED_ACTION_CONFIGS.json").write_text(json.dumps(selected,indent=2,sort_keys=True)+"\n")
 cv_old=pd.read_csv(base/"ACTION_CONFIG_GROUPED_CV.csv");cv_new=pd.read_csv(corrected/"ACTION_CONFIG_GROUPED_CV.csv");pd.concat([cv_old[cv_old.dataset=="hmc"],cv_new],ignore_index=True).to_csv(base/"ACTION_CONFIG_GROUPED_CV.csv",index=False)
 print({"detail_rows":len(detail),"selected":len(selected)})
if __name__=="__main__":main()
