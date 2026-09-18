"""Map locally supplied frozen checkpoints to the Experiment 1 provenance audit."""
from __future__ import annotations

import csv, hashlib, json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
AUDIT = BASE / "CHECKPOINT_AUDIT.csv"
OUT = BASE / "outputs" / "LOCAL_CHECKPOINT_MAP.csv"
TARGET = {"EEGNet", "EEGConformer", "FBCNet"}


def sha(path: Path) -> str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda:f.read(8<<20),b""): h.update(block)
    return h.hexdigest()


def item(path: Path) -> dict:
    record=json.loads((path.parent/"record.json").read_text(encoding="utf-8"))
    return {"local_checkpoint_path":str(path),"local_record_path":str(path.parent/"record.json"),"model":record["model"],"task":record["task"],"fold":int(record["fold"]),"seed":int(record["seed"]),"local_sha256":sha(path)}


def main() -> None:
    audit={(r["model"],r["task"],int(r["fold"]),int(r["seed"])):r for r in csv.DictReader(AUDIT.open(encoding="utf-8")) if r["model"] in TARGET}
    checkpoints=[p for p in BASE.rglob("selected.pt") if (p.parent/"record.json").is_file()]
    with ThreadPoolExecutor(max_workers=4) as pool: supplied=list(pool.map(item,checkpoints))
    rows=[]; seen={}
    for value in supplied:
        key=(value["model"],value["task"],value["fold"],value["seed"])
        expected=audit.get(key)
        value["expected_sha256"]=expected["checkpoint_sha256"] if expected else ""
        value["audit_status"]=expected["status"] if expected else "NOT_IN_AUDIT"
        value["sha_match"]=bool(expected and value["local_sha256"]==expected["checkpoint_sha256"])
        value["duplicate_local_key"]=key in seen
        seen.setdefault(key,value["local_checkpoint_path"]); rows.append(value)
    for key, expected in audit.items():
        if key not in seen: rows.append({"model":key[0],"task":key[1],"fold":key[2],"seed":key[3],"expected_sha256":expected["checkpoint_sha256"],"audit_status":expected["status"],"sha_match":False,"missing_local_checkpoint":True})
    OUT.parent.mkdir(parents=True,exist_ok=True)
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with OUT.open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
    valid=[r for r in rows if not r.get("duplicate_local_key") and r.get("sha_match")]
    print(f"LOCAL_CHECKPOINT_AUDIT supplied={len(supplied)} valid_unique={len(valid)} expected={len(audit)}")


if __name__ == "__main__": main()
