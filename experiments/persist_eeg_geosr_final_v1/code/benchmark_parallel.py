"""Measure wall-clock overlap of two independent fold workers."""
from __future__ import annotations

import os
import subprocess
import sys
import time


def main() -> None:
    root = os.path.dirname(__file__)
    env = os.environ.copy()
    env.setdefault("PERSIST_TORCH_THREADS", "8")
    jobs = [("OpenBMI", "0"), ("WBCIC", "0")]
    t0 = time.perf_counter()
    procs = [subprocess.Popen([sys.executable, os.path.join(root, "benchmark_geosr_numeric.py"), "--dataset", d, "--fold", f],
                              cwd=os.path.dirname(root), env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
             for d, f in jobs]
    outputs = [p.communicate()[0] for p in procs]
    wall = time.perf_counter() - t0
    print("parallel_wall_sec", round(wall, 3))
    for (d, f), out in zip(jobs, outputs):
        print(f"--- {d} fold {f} ---")
        print(out)


if __name__ == "__main__":
    main()
