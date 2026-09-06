from __future__ import annotations

import csv
import json
import os
import re
import stat
from collections import Counter, defaultdict
from pathlib import Path

# Metadata-only inventory: no label/prediction values and no sealed paths are opened.
ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]
P1 = REPO.parent
BRANCH = "codex/persist-eeg-pag-final-closure-v1"
DATA_EXTS = {".npy", ".npz", ".pt", ".pth", ".bin", ".h5", ".hdf5", ".mat", ".parquet", ".arrow", ".feather", ".pkl", ".pickle", ".edf", ".gdf", ".cnt", ".set", ".vhdr", ".vmrk", ".eeg", ".fif", ".zarr"}
META_EXTS = {".json", ".yaml", ".yml", ".toml", ".csv", ".tsv", ".txt", ".md"}
IDENTIFIER_RE = re.compile(r"(?i)(?:^|[/_. -])(?:subject|participant|sub)[-_ ]?(\d{1,3})(?!\d)")
OPEN_SUB_RE = re.compile(r"(?i)(?:^|[/_. -])s[-_ ]?(\d{1,3})(?!\d)")
SESSION_RE = re.compile(r"(?i)(?:session|sess|ses)[-_ ]?(\d+)")
TASK_RE = re.compile(r"(?i)(?:^|[/_.-])(mi|motor[_ -]?imagery|erp|p300|oddball|ssvep)(?:[/_.-]|$)")
OUTCOME_TOKENS = ("outcome", "prediction", "performance", "metric", "score", "leaderboard")
SENSITIVE_TOKENS = ("sealed", "outer_split", "outer-subject", "outer_subject")
SKIP_DIRS = {".git", "__pycache__", "node_modules", ".venv", "venv", "dist", "build", "logs"}
SYSTEM_DIR_TOKENS = ("programdata", "appdata", "nvidia", "microsoft", "edge", "chrome")
DATA_NAMES = {"cache", "caches", "data", "datasets", "features", "tensors", "preprocessed", "data_mirror"}
ENV_KEYS = ("CACHE", "DATA_ROOT", "DATA_PATH", "OPENBMI", "WBCIC", "EEG", "BCI", "HF_HOME", "HUGGINGFACE", "TORCH_HOME")
OPENBMI_HOLDOUT = {4, 12, 13, 17, 18, 24, 25, 29, 36, 37, 39, 42, 51, 54}
OPENBMI_SEARCH = set(range(1, 55)) - OPENBMI_HOLDOUT
WBCIC_HOLDOUT = {f"sub-{x}" for x in (2, 3, 17, 19, 21, 25, 31, 33, 38, 42)}

def norm_path(p: Path | str) -> str: return str(p).replace("\\", "/")
def is_reparse(p: Path) -> bool:
    try:
        a = getattr(p.stat(), "st_file_attributes", 0)
        return bool(a & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)) or p.is_symlink()
    except OSError: return False
def dataset_for(p: Path) -> str:
    x = norm_path(p).lower()
    if "openbmi" in x or "openbmi_trials" in x: return "OpenBMI"
    if "wbcic" in x or "bcic" in x: return "WBCIC"
    return "unknown"
def normalize_subject(v: object, ds: str) -> str | None:
    if v is None: return None
    s = str(v).strip()
    if not s: return None
    m = re.search(r"(?i)(?:sub[-_ ]?|s[-_ ]?)?(\d{1,3})$", s)
    if not m: return s if ds == "WBCIC" and s.lower().startswith("sub-") else None
    return f"sub-{int(m.group(1))}" if ds == "WBCIC" else str(int(m.group(1)))
def ids_from_path(p: Path, ds: str) -> set[str]:
    x, out = norm_path(p), set()
    for m in IDENTIFIER_RE.finditer(x):
        z = normalize_subject(m.group(1), ds)
        if z: out.add(z)
    if ds == "OpenBMI": out.update(str(int(m.group(1))) for m in OPEN_SUB_RE.finditer(x))
    return out
def sessions_from_path(p: Path) -> set[str]: return {f"session-{int(m.group(1))}" for m in SESSION_RE.finditer(norm_path(p))}
def tasks_from_path(p: Path) -> set[str]:
    out = set()
    for m in TASK_RE.finditer(norm_path(p).lower()):
        z = m.group(1).lower(); out.add("MI" if z in {"mi", "motor", "motor imagery"} else ("ERP" if z in {"erp", "p300", "oddball"} else "SSVEP"))
    return out
def safe_scalar(v: object): return v if isinstance(v, (str, int, float, bool)) or v is None else None

def inspect_array(p: Path) -> dict:
    e, out = p.suffix.lower(), {"format": p.suffix.lower().lstrip(".")}
    try:
        if e == ".npy":
            import numpy as np
            a = np.load(p, mmap_mode="r", allow_pickle=False); out.update(shapes=[list(a.shape)], dtypes=[str(a.dtype)], sample_count_candidate=(int(a.shape[0]) if a.ndim else None))
        elif e == ".npz":
            import numpy as np
            with np.load(p, mmap_mode="r", allow_pickle=False) as z:
                out["members"] = list(z.files)[:100]; out["shapes"] = [{"key": k, "shape": list(z[k].shape), "dtype": str(z[k].dtype)} for k in z.files[:100]]
        elif e in {".h5", ".hdf5"}:
            import h5py
            with h5py.File(p, "r") as h:
                ds = []
                def visit(n, o):
                    if isinstance(o, h5py.Dataset) and len(ds) < 100: ds.append({"name": n, "shape": list(o.shape), "dtype": str(o.dtype)})
                h.visititems(visit); out["datasets"] = ds
                keep = {"sfreq", "sampling_rate", "fs", "n_channels", "channels", "trial_length", "window"}; out["attrs"] = {str(k): safe_scalar(v.item() if hasattr(v, "item") else v) for k, v in h.attrs.items() if str(k).lower() in keep}
        elif e == ".mat":
            from scipy.io import whosmat
            out["members"] = [{"name": n, "shape": list(s), "class": c} for n, s, c in whosmat(p)[:100]]
        elif e == ".parquet":
            import pyarrow.parquet as pq
            pf = pq.ParquetFile(p); names = list(pf.schema.names); out.update(rows=(pf.metadata.num_rows if pf.metadata else None), columns=names[:100]); ids = [x for x in names if re.search(r"subject|participant|session|task|trial_uid", x, re.I)]; out["metadata_identifier_columns"] = ids
            # Do not read parquet columns here.  Identifier columns are read
            # once, below, only for canonical metadata manifests; historical
            # feature/result copies can be thousands of rows and are schema-
            # only for this audit.
        elif e in {".arrow", ".feather"}:
            import pyarrow.feather as feather
            t = feather.read_table(p); out.update(rows=t.num_rows, columns=list(t.column_names)[:100])
        elif e in {".pt", ".pth", ".bin"}: out["metadata_note"] = "binary tensor archive; values not loaded"
        else: out["metadata_note"] = "header/content not loaded"
    except Exception as ex: out["inspection_error"] = type(ex).__name__
    return out
def inspect_text(p: Path) -> dict:
    if any(x in p.name.lower() for x in OUTCOME_TOKENS): return {"metadata_note": "outcome-like file not opened"}
    try:
        if p.suffix.lower() in {".csv", ".tsv"}:
            return {"columns": [x.strip() for x in p.open(encoding="utf-8", errors="ignore").readline().strip().split("\t" if p.suffix.lower()==".tsv" else ",")][:100]}
        if p.suffix.lower() == ".json":
            v = json.loads(p.read_text(encoding="utf-8", errors="ignore")); return {"top_level_keys": sorted(map(str, v.keys()))[:100]} if isinstance(v, dict) else {"top_level_type": type(v).__name__}
        return {"first_line": p.open(encoding="utf-8", errors="ignore").readline().strip()[:300]}
    except Exception: return {"inspection_error": "unreadable_metadata"}
def cache_group(p: Path) -> Path:
    parts = list(p.parts)
    for i in range(len(parts)-1, -1, -1):
        if parts[i].lower() in DATA_NAMES:
            j=i+1
            return Path(*parts[:j+1]) if j < len(parts) and parts[j].lower() in {"openbmi","wbcic"} else Path(*parts[:i+1])
    return p.parent

def candidate_roots():
    # P1 already contains REPO and the stage0 checkout; avoid walking those
    # trees a second time.  The explicit stage0 path is retained in the report
    # as a provenance root but is not rescanned separately.
    roots=[P1, Path(r"C:\Users\fyl412\.cache"), Path(r"C:\Users\fyl412\.cache\huggingface"), Path(r"C:\Users\fyl412\.cache\torch"), Path(r"C:\autodl-tmp"), Path(r"D:\autodl-tmp"), Path(r"C:\root\autodl-tmp"), Path(r"D:\root\autodl-tmp"), Path(r"D:\all_seeg_data")]
    env=[]
    for k,v in sorted(os.environ.items()):
        if any(t in k.upper() for t in ENV_KEYS) and not any(t in k.upper() for t in ("TOKEN","KEY","SECRET","PASSWORD")):
            env.append({"name":k,"value":v})
            roots.extend(Path(x) for x in re.split(r"[;]",v or "") if x and ("\\" in x or "/" in x or re.match(r"^[A-Za-z]:",x)))
    seen=set(); uniq=[]
    for p in roots:
        if norm_path(p).lower() not in seen: seen.add(norm_path(p).lower()); uniq.append(p)
    return uniq,env
def relevant(p: Path) -> bool:
    low,name,ext=norm_path(p).lower(),p.name.lower(),p.suffix.lower(); ds=dataset_for(p)
    if ext not in DATA_EXTS and not (ext in META_EXTS and any(x in name for x in ("metadata","manifest","schema","config","provenance","participants","split","scope","cache"))): return False
    if ds=="unknown" and not any(x in low for x in ("openbmi","wbcic","openbmi_trials","persist_eeg_stage0")): return False
    if any(x in low for x in ("predictions","leaderboard","figures")): return False
    return True
def discover(base: Path):
    if not base.exists(): return
    try:
        for root,dirs,files in os.walk(base):
            dirs[:]=[d for d in dirs if d.lower() not in SKIP_DIRS and not any(t in d.lower() for t in SYSTEM_DIR_TOKENS)]
            low=root.lower().replace("\\","/")
            if any(t in low for t in SENSITIVE_TOKENS): dirs[:]=[]; continue
            # Runtime/checkpoint/model trees are enormous.  Only descend to a
            # direct cache/data child; this still finds canonical runtime/cache
            # metadata while excluding model checkpoints and embeddings.
            if "/runtime" in low and not ("/runtime/cache" in low or low.endswith("/runtime")):
                dirs[:]=[d for d in dirs if d.lower() in {"cache", "data", "datasets"}]
            elif low.endswith("/runtime"):
                dirs[:]=[d for d in dirs if d.lower() in {"cache", "data", "datasets"}]
            if "/outputs" in low and "/cache" not in low and not low.endswith("/outputs"):
                dirs[:]=[d for d in dirs if d.lower() in {"cache", "data", "datasets", "persist_eeg_stage0", "manifests", "delivery"}]
            for n in files:
                p=Path(root)/n
                if relevant(p): yield p
    except (OSError,PermissionError): return
def historical_refs():
    out=[]; allowed={".py",".ps1",".cmd",".bat",".yaml",".yml",".toml",".json",".md",".txt"}
    for root,dirs,files in os.walk(REPO):
        dirs[:]=[d for d in dirs if d.lower() not in SKIP_DIRS and d.lower() not in {"results","outputs","figures","runtime"}]
        for n in files:
            p=Path(root)/n
            if p.suffix.lower() not in allowed: continue
            try: lines=[x.strip()[:400] for x in p.read_text(encoding="utf-8",errors="ignore").splitlines() if re.search(r"(?i)(openbmi|wbcic|cache|data_root|data_path|huggingface|hf_home)",x)]
            except OSError: continue
            if lines: out.append({"path":norm_path(p),"matching_lines":lines[:5]})
            if len(out)>=300: return out
    return out

def add(records,p):
    ds=dataset_for(p); g=cache_group(p); key=norm_path(g); r=records.setdefault(key,{"cache_path":key,"dataset":ds,"cache_format":set(),"total_file_size_bytes":0,"file_count":0,"subjects":set(),"sessions":set(),"tasks":set(),"subject_sessions":defaultdict(set),"trial_counts_by_subject_session":Counter(),"schema_signatures":Counter(),"representative_files":[],"preprocessing_metadata_paths":set(),"labels_embedded":False,"raw_eeg_present":False,"preprocessed_tensor_or_feature_present":False,"sensitive_files_not_opened":0,"reparse":is_reparse(g)})
    try: size=p.stat().st_size
    except OSError: size=None
    r["file_count"]+=1; r["total_file_size_bytes"]+=int(size or 0); e=p.suffix.lower(); r["cache_format"].add(e.lstrip(".")); subs=ids_from_path(p,ds); ses=sessions_from_path(p); tasks=tasks_from_path(p); r["subjects"].update(subs); r["sessions"].update(ses); r["tasks"].update(tasks)
    if e in {".edf",".gdf",".cnt",".set",".vhdr",".vmrk",".eeg",".fif"}: r["raw_eeg_present"]=True
    if e in {".npy",".npz",".pt",".pth",".parquet",".arrow",".feather"} or any(x in norm_path(p).lower() for x in ("feature","tensor","embedding","preprocess")): r["preprocessed_tensor_or_feature_present"]=True
    if any(x in p.name.lower() for x in ("metadata","manifest","schema","config","provenance","normaliz","filter","sampling","participants","split","scope")): r["preprocessing_metadata_paths"].add(norm_path(p))
    sens=any(t in norm_path(p).lower() for t in SENSITIVE_TOKENS); details=None
    if sens: r["sensitive_files_not_opened"]+=1
    elif e in DATA_EXTS and e not in {".edf",".gdf",".cnt",".set",".vhdr",".vmrk",".eeg",".fif",".pkl",".pickle"}: details=inspect_array(p)
    elif e in META_EXTS: details=inspect_text(p)
    if details:
        sig=json.dumps({k:details[k] for k in details if k in {"shapes","dtypes","datasets","columns","rows","members","attrs","metadata_identifier_columns"}},sort_keys=True,default=str)
        if sig: r["schema_signatures"][sig]+=1
        if re.search(r"label|target|class|codes",json.dumps(details).lower()) or "code" in p.name.lower(): r["labels_embedded"]=True
        rows=details.get("rows")
        if rows and len(subs)==1 and len(ses)==1: r["trial_counts_by_subject_session"][f"{next(iter(subs))}|{next(iter(ses))}"]+=int(rows)
        if e==".npy" and details.get("sample_count_candidate") is not None and len(subs)==1 and len(ses)==1: r["trial_counts_by_subject_session"][f"{next(iter(subs))}|{next(iter(ses))}"]+=int(details["sample_count_candidate"])
    if e==".parquet" and not sens:
        try:
            import pyarrow.parquet as pq
            pf=pq.ParquetFile(p); names=list(pf.schema.names); idc=[x for x in names if re.search(r"subject|participant|session",x,re.I)]
            if any(re.search(r"label|target|class",n,re.I) for n in names): r["labels_embedded"]=True
            # Read identifiers only from canonical cache manifests.  Never
            # read labels/outcomes, and do not repeatedly read historical
            # copies with the same schema.
            canonical = any(x in p.name.lower() for x in ("wbcic_development_mi_metadata", "wbcic_dev_keep_experts", "openbmi_trials"))
            scanned = r.setdefault("identifier_scan_done", False)
            if idc and canonical and not scanned:
                r["identifier_scan_done"] = True
                t=pq.read_table(p,columns=idc); sc=next((c for c in idc if re.search(r"subject|participant",c,re.I)),None); xc=next((c for c in idc if re.search(r"session",c,re.I)),None)
                if sc:
                    for i in range(t.num_rows):
                        s=normalize_subject(t[sc][i].as_py(),ds)
                        if not s: continue
                        r["subjects"].add(s)
                        if xc and t[xc][i].as_py() is not None:
                            x=f"session-{t[xc][i].as_py()}"; r["sessions"].add(x); r["subject_sessions"][s].add(x); r["trial_counts_by_subject_session"][f"{s}|{x}"]+=1
        except Exception: pass
    for s in subs: r["subject_sessions"][s].update(ses)
    if len(r["representative_files"])<12: r["representative_files"].append({"path":norm_path(p),"size_bytes":size,"format":e.lstrip("."),"subjects_from_path":sorted(subs),"sessions_from_path":sorted(ses),"tasks_from_path":sorted(tasks),"schema":details})

def write_csv(path,rows,fields):
    with path.open("w",newline="",encoding="utf-8") as h: w=csv.DictWriter(h,fieldnames=fields); w.writeheader(); w.writerows(rows)
def main():
    roots,env=candidate_roots(); records={}; seen=set()
    for root in roots:
        for p in discover(root) or ():
            k=norm_path(p.resolve()).lower() if p.exists() else norm_path(p).lower()
            if k not in seen: seen.add(k); add(records,p)
    norm=[]
    for rr in sorted(records.values(),key=lambda x:(x["dataset"],x["cache_path"])):
        r=dict(rr)
        for f in ("cache_format","subjects","sessions","tasks","preprocessing_metadata_paths"): r[f]=sorted(rr[f])
        r["subject_sessions"]={k:sorted(v) for k,v in sorted(rr["subject_sessions"].items())}; r["trial_counts_by_subject_session"]=dict(sorted(rr["trial_counts_by_subject_session"].items())); r["schema_signatures"]=[{"signature":k,"file_count":v} for k,v in rr["schema_signatures"].most_common(20)]; norm.append(r)
    def agg(ds):
        s=set();x=set();t=set();raw=lab=pre=False
        for r in norm:
            if r["dataset"]==ds: s.update(r["subjects"]); x.update(r["sessions"]); t.update(r["tasks"]); raw|=r["raw_eeg_present"];lab|=r["labels_embedded"];pre|=r["preprocessed_tensor_or_feature_present"]
        return s,x,t,raw,lab,pre
    osub,oses,otask,oraw,olab,opre=agg("OpenBMI"); wsub,wses,wtask,wraw,wlab,wpre=agg("WBCIC"); wnorm={normalize_subject(x,"WBCIC") for x in wsub}; wnorm.discard(None)
    # WBCIC is MI-only; historical output names may contain other task tokens.
    wtask = {"MI"} if wnorm else set()
    subject_session_map = {"OpenBMI": defaultdict(set), "WBCIC": defaultdict(set)}
    for rr in norm:
        if rr["dataset"] in subject_session_map:
            for sid, ss in rr.get("subject_sessions", {}).items():
                subject_session_map[rr["dataset"]][sid].update(ss)
    dev={f"sub-{x}" for x in (1,2,3,5,6,7,9,11,12,13,14,16,17,18,19,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,41,42,44,45,47,48,49,50)}
    # Only enumerate names in the current repository for outer/sealed evidence;
    # never recurse through external roots or open an outer split file.
    outer=[]
    try:
        for p in REPO.rglob("*"):
            if any(t in norm_path(p).lower() for t in SENSITIVE_TOKENS):
                outer.append({"path":norm_path(p),"exists":p.exists(),"is_file":p.is_file()})
            if len(outer)>=100: break
    except (OSError,PermissionError): pass
    op={int(x) for x in osub if str(x).isdigit()}; open_can=bool(op>=set(range(1,55)) and {"session-1","session-2"}.issubset(oses) and "MI" in otask and opre)
    outer_data=[x for x in outer if x.get("is_file") and Path(x["path"]).suffix.lower() in DATA_EXTS and any(t in x["path"].lower() for t in ("cache","data","raw"))]
    inv={"schema":"PERSIST_EEG_CACHE_INVENTORY_V2_METADATA_ONLY","server":"fyl412@100.94.171.11","repository":norm_path(REPO),"branch":BRANCH,"read_policy":{"training_run":False,"sealed_confirmation_run":False,"sealed_outcomes_read":False,"labels_or_predictions_values_read":False,"identifier_columns_and_shapes_only":True,"sensitive_outer_paths_opened":False},"environment_paths":env,"candidate_roots":[{"path":norm_path(p),"exists":p.exists(),"reparse":is_reparse(p)} for p in roots],"historical_cache_path_references":historical_refs(),"sensitive_outer_path_evidence":outer,"cache_records":norm,"aggregate":{"openbmi_subjects":sorted(op),"wbcic_subjects":sorted(wnorm),"openbmi_sessions":sorted(oses),"wbcic_sessions":sorted(wses),"openbmi_tasks":sorted(otask),"wbcic_tasks":sorted(wtask),"openbmi_search_present":sorted(op&OPENBMI_SEARCH),"openbmi_search_missing":sorted(OPENBMI_SEARCH-op),"openbmi_internal_holdout_present":sorted(op&OPENBMI_HOLDOUT),"openbmi_internal_holdout_missing":sorted(OPENBMI_HOLDOUT-op),"wbcic_v8_development_subjects_present":sorted(wnorm&dev),"wbcic_v8_development_subjects_missing":sorted(dev-wnorm),"wbcic_v8_holdout_present":sorted(wnorm&WBCIC_HOLDOUT),"wbcic_v8_holdout_missing":sorted(WBCIC_HOLDOUT-wnorm),"wbcic_extra_non_development_subjects":sorted(wnorm-dev),"openbmi_raw_eeg_present":oraw,"wbcic_raw_eeg_present":wraw,"openbmi_labels_embedded":olab,"wbcic_labels_embedded":wlab,"openbmi_preprocessed_or_features":opre,"wbcic_preprocessed_or_features":wpre,"true_outer_data_physical_evidence":"UNCERTAIN" if outer_data else "NO"}}
    (ROOT/"CACHE_INVENTORY.json").write_text(json.dumps(inv,indent=2,sort_keys=True,default=str)+"\n",encoding="utf-8")
    write_csv(ROOT/"OPENBMI_CACHE_COVERAGE.csv",[{"subject_id":i,"frozen_role":"V8_INTERNAL_HOLDOUT" if i in OPENBMI_HOLDOUT else "V8_SEARCH","present_in_cache":i in op,"sessions_seen":";".join(sorted(subject_session_map["OpenBMI"].get(str(i), set()))),"tasks_seen":";".join(sorted(otask)),"raw_eeg_seen":oraw,"labels_embedded":olab} for i in range(1,55)],["subject_id","frozen_role","present_in_cache","sessions_seen","tasks_seen","raw_eeg_seen","labels_embedded"])
    write_csv(ROOT/"WBCIC_CACHE_COVERAGE.csv",[{"subject_id":s,"frozen_role":"V8_INTERNAL_HOLDOUT" if s in WBCIC_HOLDOUT else ("V8_DEVELOPMENT" if s in dev else "EXTRA_NON_DEVELOPMENT"),"present_in_cache":s in wnorm,"sessions_seen":";".join(sorted(subject_session_map["WBCIC"].get(s, set()))),"tasks_seen":";".join(sorted(wtask)),"raw_eeg_seen":wraw,"labels_embedded":wlab} for s in sorted(dev|WBCIC_HOLDOUT|wnorm)],["subject_id","frozen_role","present_in_cache","sessions_seen","tasks_seen","raw_eeg_seen","labels_embedded"])
    report=f"""# Cache gap report\n\nMetadata-only audit. No training, PAG sealed confirmation, sealed outcome, label value, prediction value, or performance outcome was read. Sensitive outer/sealed paths were not opened.\n\n### OpenBMI\n\n- cache currently has: {len(op)} subjects\n- frozen search subjects present: {len(op&OPENBMI_SEARCH)}/40\n- frozen internal holdout present: {len(op&OPENBMI_HOLDOUT)}/14\n- missing internal holdout IDs: {sorted(OPENBMI_HOLDOUT-op)}\n- sessions available: {sorted(oses)}\n- tasks available: {sorted(otask)}\n- raw EEG available: {'YES' if oraw else 'NO'}\n- labels/codes embedded in cache: {'YES' if olab else 'NO'}\n- can sealed holdout be run from current cache alone: {'YES' if open_can else 'NO'} (input-ready MI cache + split manifest; not executed)\n\n### WBCIC\n\n- cache currently has: {len(wnorm)} subjects\n- V8 development subjects covered: {len(wnorm&dev)}/41\n- V8 internal holdout present: {len(wnorm&WBCIC_HOLDOUT)}/10\n- missing V8 internal holdout IDs: {sorted(WBCIC_HOLDOUT-wnorm)}\n- any extra non-development subjects found: {'YES' if wnorm-dev else 'NO'}\n- sessions available: {sorted(wses)}\n- tasks available: {sorted(wtask or {'MI'})}\n- raw EEG available: {'YES' if wraw else 'NO'}\n- any evidence true outer data are physically present: {inv['aggregate']['true_outer_data_physical_evidence']}\n- can true outer evaluation run from current cache alone: NO\n\n### Minimum missing data\n\nOpenBMI needs no additional subject files for an input-only replay, but raw source would be needed to regenerate preprocessing. WBCIC needs authorized outer-10 MI data (raw or independently verified compatible 58-channel × 1000-sample preprocessed cache), sessions 0/1/2, subject/session metadata, and separately authorized scoring labels.\n"""
    (ROOT/"CACHE_GAP_REPORT.md").write_text(report,encoding="utf-8"); md=["# Cache inventory","","Metadata-only inventory; no labels/outcomes were read.","","| Dataset | Cache path | Files | Size (bytes) | Subjects | Sessions | Tasks | Raw EEG | Labels embedded |","|---|---|---:|---:|---:|---|---|---|---|"]
    for r in norm:
        if r["dataset"] in {"OpenBMI","WBCIC"}: md.append(f"| {r['dataset']} | `{r['cache_path']}` | {r['file_count']} | {r['total_file_size_bytes']} | {len(r['subjects'])} | {', '.join(r['sessions'])} | {', '.join(r['tasks'])} | {'YES' if r['raw_eeg_present'] else 'NO'} | {'YES' if r['labels_embedded'] else 'NO'} |")
    (ROOT/"CACHE_INVENTORY.md").write_text("\n".join(md+["",report])+"\n",encoding="utf-8")
if __name__=="__main__": main()
