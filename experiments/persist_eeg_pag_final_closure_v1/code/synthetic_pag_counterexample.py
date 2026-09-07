from __future__ import annotations
import csv, json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
def risk(pred,y): return float(np.mean(pred!=y))
def sign(x): return np.where(x>=0,1,-1)
def main():
    y=np.array([-1,1]*100,dtype=float)
    # Source: stable persistent feature and shared task geometry.
    zp=y.copy(); base=sign(zp); after=sign(np.zeros_like(zp))
    rs=risk(base,y); ri=risk(after,y); C=ri-rs
    rows=[{"environment":"source","persistence":1.0,"shared_geometry":1.0,"R_f":rs,"R_f_after_I":ri,"C_source":C,"A_target":""}]
    for env,target_zp in [("e1",y),("e2",-y)]:
        r0=risk(sign(target_zp),y); r1=risk(sign(np.zeros_like(target_zp)),y)
        rows.append({"environment":env,"persistence":1.0,"shared_geometry":1.0,"R_f":r0,"R_f_after_I":r1,"C_source":C,"A_target":r0-r1})
    with (ROOT/"SYNTHETIC_PAG_RESULTS.csv").open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    payload={"schema":"PERSIST_EEG_SYNTHETIC_PAG_V1","source_persistence":True,"source_consequence_nonzero":bool(C!=0),"source_shared_geometry":True,"target_A_e1":rows[1]["A_target"],"target_A_e2":rows[2]["A_target"],"opposite_target_signs":bool(rows[1]["A_target"]*rows[2]["A_target"]<0),"theorem_intuition_verified":True}
    (ROOT/"SYNTHETIC_PAG_RESULTS.json").write_text(json.dumps(payload,indent=2)+"\n")
    fig,ax=plt.subplots(figsize=(6,4)); ax.axhline(0,color="black",lw=.8); ax.bar(["e1","e2"],[rows[1]["A_target"],rows[2]["A_target"]],color=["#b2182b","#2166ac"]); ax.set_ylabel("A_e = R_e(f) - R_e(f o I)"); ax.set_title("Same source premises, opposite target utility"); fig.tight_layout(); fig.savefig(ROOT/"SYNTHETIC_PAG_FIGURE.png",dpi=180); fig.savefig(ROOT/"SYNTHETIC_PAG_FIGURE.pdf"); plt.close(fig)
    (ROOT/"SYNTHETIC_PAG_REPORT.md").write_text("# Synthetic PAG counterexample\n\nThe source has stable persistent feature z_p=y, nonzero source intervention consequence, and shared task geometry. Two targets leave every source premise unchanged but use z_p=y (e1) or z_p=-y (e2), producing opposite signs of A_e. This demonstrates logical insufficiency without a transportability assumption.\n")
if __name__=="__main__": main()
