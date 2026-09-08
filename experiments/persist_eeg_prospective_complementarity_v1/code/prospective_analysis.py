"""Frozen prospective complementarity analysis (no training or holdout access).

Predictors are computed from INNER_VAL future-session trials using the fixed
seed-0 checkpoints. Existing OUTER_DEV rows are linked only after predictor
rows are hashed. Since this protocol was frozen after the screen completed,
all linked units are explicitly preexisting/supportive, not confirmatory.
"""
from __future__ import annotations
import csv, hashlib, importlib.util, json, math, sys, time
from pathlib import Path
import numpy as np
import torch
import itertools
import datetime

ROOT = Path(__file__).resolve().parents[3]
EXP = ROOT / "experiments" / "persist_eeg_prospective_complementarity_v1"
OUT = EXP / "outputs"
CODE = EXP / "code"
SCREEN = ROOT / "experiments" / "persist_eeg_backbone_generality_seed0_v1" / "code" / "backbone_screen.py"
BACKBONES = ("EEGConformer", "CBraMod", "CodeBrain")
COMPANIONS = ("LiteBN", "EEGNet")
DATASETS = ("OpenBMI", "WBCIC")
BOOTSTRAPS = 10000

def load_screen_module():
    spec = importlib.util.spec_from_file_location("backbone_screen", SCREEN)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod

def write_csv(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields=[]
    for r in rows:
        for k in r:
            if k not in fields: fields.append(k)
    with path.open("w", newline="", encoding="utf-8") as f:
        w=csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)

def sha_row(row: dict) -> str:
    payload=json.dumps(row, sort_keys=True, separators=(",",":"), default=str).encode()
    return hashlib.sha256(payload).hexdigest()

def subject_stats(mod, bundle, idx, logits, subjects):
    vals=[]
    for s in mod.sort_subjects(subjects):
        pos=np.flatnonzero(bundle.subject[idx] == str(s))
        y=bundle.y[idx[pos]]; z=logits[pos]; pred=z.argmax(1); ok=pred == y
        ba=mod.subject_metric(y,z)[0]
        vals.append({"subject":str(s),"ba":float(ba),"ok":ok})
    return vals

def comp_stats(a, b, subjects):
    ex=[]; dis=[]; both_wrong=[]
    for s in subjects:
        aa=a[s]["ok"]; bb=b[s]["ok"]
        ex.append(float(np.mean(np.logical_xor(aa,bb))))
        dis.append(float(np.mean(aa != bb))); both_wrong.append(float(np.mean((~aa)&(~bb))))
    return float(np.mean(ex)), float(np.mean(np.concatenate([np.logical_xor(a[s]["ok"],b[s]["ok"]) for s in subjects]))), float(np.mean(dis)), float(np.mean(both_wrong))

def rankdata(x):
    x=np.asarray(x,float); order=np.argsort(x, kind="mergesort"); r=np.empty(len(x),float); i=0
    while i<len(x):
        j=i+1
        while j<len(x) and x[order[j]]==x[order[i]]: j+=1
        r[order[i:j]]=(i+j-1)/2+1; i=j
    return r

def spearman(x,y):
    if len(x)<3 or np.std(x)==0 or np.std(y)==0: return float("nan")
    return float(np.corrcoef(rankdata(x),rankdata(y))[0,1])

def pvalue_perm(x,y,n=10000):
    if len(x)<3: return None
    obs=spearman(x,y); rng=np.random.default_rng(0); c=0
    for _ in range(n):
        if abs(spearman(x,rng.permutation(y))) >= abs(obs)-1e-15: c+=1
    return (c+1)/(n+1)

def analyze(rows):
    eligible=[r for r in rows if r["prospective_status"]=="PROSPECTIVE_ELIGIBLE"]
    def arrays(rr): return np.array([float(r["C_subject_inner_val"]) for r in rr]), np.array([float(r["G_best_outer_dev"]) for r in rr])
    primary={"n":len(eligible),"rho":None,"p_two_sided":None,"bootstrap_ci":None}
    supportive={"n":len(rows)}
    if len(rows)>=3:
        x,y=arrays(rows); supportive.update({"rho":spearman(x,y),"p_two_sided_permutation":pvalue_perm(x,y)})
        blocks=sorted({(r["dataset"],r["backbone"]) for r in rows}); rng=np.random.default_rng(0); draws=[]
        for _ in range(BOOTSTRAPS):
            chosen=rng.integers(0,len(blocks),len(blocks)); rr=[]
            for i in chosen: rr.extend([r for r in rows if (r["dataset"],r["backbone"])==blocks[i]])
            xx,yy=arrays(rr); draws.append(spearman(xx,yy))
        supportive["block_bootstrap_ci"]= [float(np.nanquantile(draws,.025)),float(np.nanquantile(draws,.975))]
        centered=[]
        for _,g in itertools.groupby(sorted(rows,key=lambda r:(r["dataset"],r["backbone"])), key=lambda r:(r["dataset"],r["backbone"])):
            gg=list(g); cm=np.mean([float(r["C_subject_inner_val"]) for r in gg]); gm=np.mean([float(r["G_best_outer_dev"]) for r in gg])
            centered.extend((float(r["C_subject_inner_val"])-cm,float(r["G_best_outer_dev"])-gm) for r in gg)
        supportive["within_block_centered_rho"]=spearman([a for a,b in centered],[b for a,b in centered])
    if len(eligible)>=3:
        x,y=arrays(eligible); primary.update({"rho":spearman(x,y),"p_two_sided":pvalue_perm(x,y)})
    rank_rows=[]
    for d,b in itertools.product(DATASETS,BACKBONES):
        aa=[r for r in rows if r["dataset"]==d and r["backbone"]==b and r["companion"]=="LiteBN"]
        bb=[r for r in rows if r["dataset"]==d and r["backbone"]==b and r["companion"]=="EEGNet"]
        for a in aa:
            z=next((q for q in bb if int(q["fold"])==int(a["fold"])),None)
            if z is None: continue
            dc=float(a["C_subject_inner_val"])-float(z["C_subject_inner_val"]); dg=float(a["G_best_outer_dev"])-float(z["G_best_outer_dev"])
            tie_c=abs(dc)<1e-8; tie_g=abs(dg)<1e-8; correct=(not tie_c and not tie_g and ((dc>0)==(dg>0)))
            rank_rows.append({"dataset":d,"backbone":b,"fold":int(a["fold"]),"delta_C":dc,"delta_G":dg,"prediction":"tie" if tie_c else ("LiteBN" if dc>0 else "EEGNet"),"outcome_better":"tie" if tie_g else ("LiteBN" if dg>0 else "EEGNet"),"correct_excluding_ties":correct,"tie_C":tie_c,"tie_G":tie_g,"prospective_status":"PREEXISTING_OUTCOME"})
    write_csv(OUT/"PARTNER_RANKING_RESULTS.csv",rank_rows)
    block_rows=[]
    for d,b in itertools.product(DATASETS,BACKBONES):
        rr=[r for r in rank_rows if r["dataset"]==d and r["backbone"]==b]
        block_rows.append({"dataset":d,"backbone":b,"folds":len(rr),"mean_delta_C":float(np.mean([r["delta_C"] for r in rr])) if rr else None,"mean_delta_G":float(np.mean([r["delta_G"] for r in rr])) if rr else None,"matching_direction":(np.mean([r["delta_C"] for r in rr])*np.mean([r["delta_G"] for r in rr])>0) if rr else None,"prospective_status":"PREEXISTING_OUTCOME"})
    write_csv(OUT/"BLOCK_LEVEL_RESULTS.csv",block_rows)
    ncorrect=sum(bool(r["correct_excluding_ties"]) for r in rank_rows); ntie=sum(bool(r["tie_C"] or r["tie_G"]) for r in rank_rows)
    summary={"terminal":"PROSPECTIVE_TEST_INSUFFICIENT" if not eligible else "PROSPECTIVE_COMPLEMENTARITY_NOT_SUPPORTED","primary":primary,"supportive_posthoc":supportive,"prospective_eligible_units":len(eligible),"preexisting_units":len(rows),"partner_ranking":{"n":len(rank_rows),"correct":ncorrect,"incorrect":len(rank_rows)-ncorrect-ntie,"ties":ntie,"accuracy_excluding_ties":(ncorrect/(len(rank_rows)-ntie) if len(rank_rows)>ntie else None)},"block_matching_count":sum(bool(r["matching_direction"]) for r in block_rows),"holdout_access":False}
    (OUT/"PROSPECTIVE_SUMMARY.json").write_text(json.dumps(summary,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    decision = f"""# Prospective complementarity decision

**Terminal: {summary['terminal']}.** The protocol lock was created after the backbone screen OUTER_DEV outcomes already existed. Therefore all {len(rows)} pair-level units are `PREEXISTING_OUTCOME`; the genuinely prospective confirmatory sample is n=0. No final holdout subjects were accessed.

## Confirmatory result

No confirmatory Spearman correlation or partner-ranking claim is estimable because the lock post-dates every eligible OUTER_DEV outcome. This is an information-timing limitation, not a negative scientific finding.

## Supportive post-hoc diagnostics (not prospective evidence)

- Pair-level Spearman rho: {supportive.get('rho')}
- Permutation p-value (two-sided): {supportive.get('p_two_sided_permutation')}
- Block-bootstrap 95% CI: {supportive.get('block_bootstrap_ci')}
- Within-block centered rho: {supportive.get('within_block_centered_rho')}
- Partner ranking: {summary['partner_ranking']['correct']} correct, {summary['partner_ranking']['incorrect']} incorrect, {summary['partner_ranking']['ties']} ties; accuracy excluding ties = {summary['partner_ranking']['accuracy_excluding_ties']}
- Dataset×backbone blocks with matching mean direction: {summary['block_matching_count']}/6

These supportive values must not be presented as prospective confirmation. They do not justify metric search, architecture changes, checkpoint reselection, or seeds 1/2.
"""
    (EXP/"PROSPECTIVE_DECISION.md").write_text(decision,encoding="utf-8")
    (EXP/"protocol"/"BUGFIX_LOG.md").write_text("""# Bugfix log

1. The first analysis invocation used the wrong runtime root for existing EEGNet/LiteBN checkpoints; corrected it to the frozen carrier runtime. No predictor or outcome row was produced by the failed invocation.
2. The first analysis invocation deleted the shared search bundle inside the fold loop, causing a real `UnboundLocalError`; changed only the variable lifetime so all five folds use the same loaded SEARCH bundle.
3. Partner-ranking correctness was initially initialized to false; replaced it with the predeclared sign comparison using the fixed 1e-8 tie tolerance. This changes only an output field, not any predictor or outcome.

No final holdout data were accessed.
""",encoding="utf-8")
    return summary

def main():
    mod=load_screen_module(); search,folds,_=mod.load_split(); OUT.mkdir(parents=True,exist_ok=True)
    runtime=mod.RUNTIME; carrier_runtime=mod.CARRIER_RUNTIME; predictor=[]; timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat()
    for d in DATASETS:
        bundle=mod.load_bundle(d,search[d])
        for spec in folds[d]:
            fold=int(spec["fold_id"]); mean,std,_=mod.normalizer(bundle,spec["inner_train_subjects"]); cache=mod.GPUCache(bundle,mean,std,torch.device("cuda" if torch.cuda.is_available() else "cpu")); idx=bundle.indices(spec["inner_val_subjects"],mod.future_session(d)); subjects=mod.sort_subjects(spec["inner_val_subjects"])
            models={}
            for name in ("EEGNet","LiteBN")+BACKBONES:
                if name in BACKBONES:
                    path=runtime/d.lower()/name.lower()/f"fold-{fold}"/"seed-0"/"selected_best.pt"; m=mod.build_new_model(name,bundle.channels)
                else:
                    path=carrier_runtime/f"{d.lower()}_fold{fold}_seed0_{name.lower()}"/"selected_best.pt"; m=mod.load_carrier(name,bundle.channels,d,fold)
                m.load_state_dict(torch.load(path,map_location="cpu",weights_only=True),strict=True); models[name]=mod.logits_for(m.to(cache.device),cache,idx); del m
            stats={n:{str(x["subject"]):x for x in subject_stats(mod,bundle,idx,models[n],subjects)} for n in models}
            for b in BACKBONES:
                for c in COMPANIONS:
                    ca,ct,dis,bw=comp_stats(stats[b],stats[c],subjects); base=float(np.mean([stats[b][s]["ba"] for s in subjects])); partner=float(np.mean([stats[c][s]["ba"] for s in subjects])); gap=abs(base-partner)
                    ck1=runtime/d.lower()/b.lower()/f"fold-{fold}"/"seed-0"/"selected_best.pt"; ck2=carrier_runtime/f"{d.lower()}_fold{fold}_seed0_{c.lower()}"/"selected_best.pt"
                    row={"dataset":d,"backbone":b,"fold":fold,"companion":c,"prospective_status":"PREEXISTING_OUTCOME","inner_val_subject_count":len(subjects),"inner_val_trial_count":int(len(idx),),"inner_val_base_BA":base,"inner_val_companion_BA":partner,"inner_val_accuracy_gap":gap,"C_subject_inner_val":ca,"C_trial_inner_val":ct,"disagreement_inner_val":dis,"both_wrong_inner_val":bw,"predictor_timestamp":timestamp,"base_checkpoint_sha256":mod.sha256_file(ck1),"companion_checkpoint_sha256":mod.sha256_file(ck2)}
                    row["predictor_hash"]=sha_row(row); predictor.append(row)
            # The normalized GPU cache is fold-local; the search bundle is
            # shared by all five folds of this dataset.
            del cache
        del bundle
    write_csv(OUT/"PREDICTOR_FREEZE_LEDGER.csv",predictor)
    # Link already-existing OUTER_DEV outcomes only after every predictor row is frozen.
    def read(path):
        with path.open(encoding="utf-8") as f: return list(csv.DictReader(f))
    base_rows=read(ROOT/"experiments"/"persist_eeg_backbone_generality_seed0_v1"/"outputs"/"STANDALONE_SUBJECT_RESULTS.csv")
    fusion_rows=read(ROOT/"experiments"/"persist_eeg_backbone_generality_seed0_v1"/"outputs"/"FUSION_SUBJECT_RESULTS.csv")
    linked=[]
    for r in predictor:
        d,b,f,c=r["dataset"],r["backbone"],int(r["fold"]),r["companion"]; br=[x for x in base_rows if x["dataset"]==d and x["model"]==b and int(x["fold"])==f]; pr=[x for x in base_rows if x["dataset"]==d and x["model"]==c and int(x["fold"])==f]; fr=[x for x in fusion_rows if x["dataset"]==d and x["backbone"]==b and x["pair"]==f"{b}+{c}" and int(x["fold"])==f]; mb=float(np.mean([float(x["BA"]) for x in br])); mp=float(np.mean([float(x["BA"]) for x in pr])); mf=float(np.mean([float(x["BA"]) for x in fr])); linked.append({**r,"outer_dev_subject_count":len(br),"outer_dev_base_BA":mb,"outer_dev_companion_BA":mp,"outer_dev_fusion_BA":mf,"G_best_outer_dev":(mf-max(mb,mp))*100.0})
    rankmap={(r["dataset"],r["backbone"],int(r["fold"])):r for r in linked}
    for r in linked:
        other=[q for q in linked if q["dataset"]==r["dataset"] and q["backbone"]==r["backbone"] and int(q["fold"])==int(r["fold"]) and q["companion"]!=r["companion"]][0]
        dc=float(r["C_subject_inner_val"])-float(other["C_subject_inner_val"]); dg=float(r["G_best_outer_dev"])-float(other["G_best_outer_dev"])
        r["delta_C_partner"]=dc; r["delta_G_partner"]=dg
        r["partner_prediction_correct"]=bool(abs(dc)>=1e-8 and abs(dg)>=1e-8 and ((dc>0)==(dg>0)))
    write_csv(OUT/"INNER_VAL_COMPLEMENTARITY.csv",predictor); write_csv(OUT/"OUTER_DEV_FUSION_GAIN.csv",[{k:r[k] for k in ("dataset","backbone","fold","companion","outer_dev_subject_count","outer_dev_base_BA","outer_dev_companion_BA","outer_dev_fusion_BA","G_best_outer_dev","prospective_status")} for r in linked]); write_csv(OUT/"PROSPECTIVE_PAIR_LEVEL.csv",linked); analyze(linked)
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig,(ax,bx)=plt.subplots(1,2,figsize=(10,4)); colors={"EEGConformer":"tab:blue","CBraMod":"tab:orange","CodeBrain":"tab:green"}; marks={"LiteBN":"o","EEGNet":"s"}
        for r in linked: ax.scatter(float(r["C_subject_inner_val"]),float(r["G_best_outer_dev"]),c=colors[r["backbone"]],marker=marks[r["companion"]],alpha=.75)
        ax.set(xlabel="INNER_VAL C_subject",ylabel="OUTER_DEV G_best (pp)"); ax.axhline(0,color="k",lw=.5); ax.set_title("Post-hoc only: all outcomes preexisting")
        for r in itertools.product(DATASETS,BACKBONES):
            z=[q for q in csv.DictReader((OUT/"PARTNER_RANKING_RESULTS.csv").open(encoding="utf-8")) if q["dataset"]==r[0] and q["backbone"]==r[1]]
            if z: bx.scatter(float(np.mean([float(q["delta_C"]) for q in z])),float(np.mean([float(q["delta_G"]) for q in z])),label=f"{r[0]}-{r[1]}")
        bx.axhline(0,color="k",lw=.5); bx.axvline(0,color="k",lw=.5); bx.set(xlabel="mean Delta_C",ylabel="mean Delta_G (pp)"); bx.legend(fontsize=6)
        fig.tight_layout(); fig.savefig(OUT/"PROSPECTIVE_COMPLEMENTARITY_FIGURE.pdf"); plt.close(fig)
    except Exception as e:
        (OUT/"PROSPECTIVE_COMPLEMENTARITY_FIGURE.pdf").write_text("Figure unavailable: "+repr(e),encoding="utf-8")
    print("PROSPECTIVE_ANALYSIS_COMPLETE",flush=True)

if __name__ == "__main__": main()
