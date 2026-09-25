"""Bounded four-process queue for the frozen audit cells."""
import argparse
import concurrent.futures
import os
import subprocess
import sys
from pathlib import Path

CODE=Path(__file__).resolve().parent
REPO=CODE.parents[2]
TASKS=("OpenBMI_MI","OpenBMI_ERP","OpenBMI_SSVEP","WBCIC_MI")


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("stage",choices=("prepare","discovery"))
    ap.add_argument("--workers",type=int,default=4)
    args=ap.parse_args()
    if args.workers<1:ap.error("workers must be positive")
    runtime=Path(os.environ.get("CP_DIRECTION_RUNTIME",str(REPO.parent/"cp_direction_utility_v1_runtime"))).resolve()
    logs=runtime/"queue_logs"/args.stage;logs.mkdir(parents=True,exist_ok=True)
    jobs=[(task,fold) for task in TASKS for fold in range(5)]
    env=os.environ.copy()
    env.setdefault("CP_DIRECTION_CPU_THREADS","4")
    env.setdefault("OMP_NUM_THREADS","4");env.setdefault("MKL_NUM_THREADS","4")
    env.setdefault("OPENBLAS_NUM_THREADS","4");env.setdefault("NUMEXPR_NUM_THREADS","4")

    def run(job):
        task,fold=job;log=logs/f"{task.lower()}_fold{fold}.log"
        command=[sys.executable,str(CODE/"run.py"),args.stage,"--task",task,"--fold",str(fold)]
        with log.open("wb") as stream:
            proc=subprocess.run(command,cwd=REPO,env=env,stdout=stream,stderr=subprocess.STDOUT)
        return task,fold,proc.returncode,str(log)

    failures=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures=[pool.submit(run,job) for job in jobs]
        for future in concurrent.futures.as_completed(futures):
            task,fold,code,log=future.result()
            print(f"QUEUE_CELL {args.stage} {task} fold{fold} exit={code} log={log}",flush=True)
            if code:failures.append((task,fold,code,log))
    if failures:
        raise SystemExit(f"{len(failures)} cells failed: {failures}")
    print(f"QUEUE_STAGE_COMPLETE {args.stage} cells={len(jobs)} workers={args.workers}",flush=True)


if __name__=="__main__":main()
