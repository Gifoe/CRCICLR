from __future__ import annotations
import hashlib, json, subprocess, sys
from pathlib import Path

ROOT=Path(__file__).resolve().parent
REPO=ROOT.parents[1]
LOCK=ROOT/"PREREGISTRATION_LOCK.json"
AUDIT=ROOT/"SEALED_RESOURCE_AUDIT.json"
def sha(p):
    h=hashlib.sha256();
    with p.open("rb") as f:
        for b in iter(lambda:f.read(1024*1024),b""): h.update(b)
    return h.hexdigest()
def head(): return subprocess.check_output(["git","-C",str(REPO),"rev-parse","HEAD"],text=True).strip()
def is_ancestor(commit):
    return subprocess.run(
        ["git", "-C", str(REPO), "merge-base", "--is-ancestor", commit, "HEAD"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    ).returncode == 0
def main():
    lock=json.loads(LOCK.read_text(encoding="utf-8")); audit=json.loads(AUDIT.read_text(encoding="utf-8"))
    checks={"lock_present":True,"preregistration_commit_anchored":is_ancestor(lock["preregistration_commit"]),"sealed_outcome_not_read":audit.get("outcome_labels_read") is False,"resources_available":audit.get("resources_available") is True}
    for rel,expected in lock["file_sha256"].items(): checks["hash:"+rel]=sha(ROOT/rel)==expected
    if not all(checks.values()):
        payload={"schema":"PERSIST_EEG_PAG_SEALED_CONFIRMATION_V1","status":"SEALED_CONFIRMATION_NOT_RUN_RESOURCE_UNAVAILABLE","checks":checks,"missing_evidence_type":"independent sealed OpenBMI internal holdout and WBCIC outer subjects are not materialized in the server repository","outcome_labels_read":False,"post_unblinding_repair":False}
        (ROOT/"SEALED_CONFIRMATION_BLOCKED.json").write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n",encoding="utf-8")
        print(json.dumps(payload,indent=2,sort_keys=True)); return 3
    raise RuntimeError("A sealed runner is not authorized without a separately audited resource adapter")
if __name__=="__main__": raise SystemExit(main())
