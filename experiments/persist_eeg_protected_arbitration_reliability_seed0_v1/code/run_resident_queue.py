"""Run remaining frozen cells serially in one resident CUDA process.

The process is only started after the supervisor's normal RAM/VRAM admission
gate. It retains PyTorch's allocator cache between real cells so another
workload cannot claim released GPU allocation in the gap. It neither creates
synthetic reservations nor alters frozen protocol inputs or cell validation.
"""
import gc
import json
import os
import time

os.environ["ARBITRATION_RESIDENT"] = "1"

import torch
import run_arbitration as RA

ORDER = [(m, t, f) for m in ("EEGNet", "EEGConformer") for t in ("OpenBMI_MI", "OpenBMI_SSVEP") for f in range(5)]
STATE = RA.RUNTIME / "resident_queue_status.json"


def write(status, **extra):
    value = {"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "status": status, **extra}
    tmp = STATE.with_suffix(".part")
    tmp.write_text(json.dumps(value, indent=2), encoding="utf-8")
    os.replace(tmp, STATE)


def read(cell):
    path = RA.tpath(*cell)
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def complete(data):
    return bool(data and data.get("status") == "COMPLETE" and data.get("implementation_revision") == "protocol_repair_v2")


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("resident queue requires CUDA")
    if not (RA.PROTOCOL / "PROVENANCE.json").is_file():
        raise RuntimeError("protocol lock required")
    prior = [cell for cell in ORDER if complete(read(cell))]
    print("RESIDENT_QUEUE_START", len(prior), "of", len(ORDER), flush=True)
    write("RUNNING", completed=len(prior), total=len(ORDER), current=None, final_heldout_accessed=False)
    for cell in ORDER:
        data = read(cell)
        if complete(data):
            continue
        if data is not None:
            raise RuntimeError(f"terminal cell requires engineering review: {cell}: {data.get('status')} {data.get('reason', '')}")
        done = sum(complete(read(candidate)) for candidate in ORDER)
        write("RUNNING", completed=done, total=len(ORDER), current=list(cell), final_heldout_accessed=False)
        print("RESIDENT_CELL_START", cell, flush=True)
        RA.cell(*cell)
        gc.collect()
        data = read(cell)
        if not complete(data):
            raise RuntimeError(f"cell did not produce a valid protocol_repair_v2 completion: {cell}")
        done = sum(complete(read(candidate)) for candidate in ORDER)
        write("RUNNING", completed=done, total=len(ORDER), current=None, final_heldout_accessed=False)
        print("RESIDENT_CELL_COMPLETE", cell, done, "of", len(ORDER), flush=True)
    write("COMPLETE", completed=len(ORDER), total=len(ORDER), current=None, final_heldout_accessed=False)
    print("RESIDENT_QUEUE_COMPLETE", len(ORDER), flush=True)


if __name__ == "__main__":
    try:
        main()
    except BaseException as error:
        write("ERROR", reason=f"{type(error).__name__}: {error}", final_heldout_accessed=False)
        raise
