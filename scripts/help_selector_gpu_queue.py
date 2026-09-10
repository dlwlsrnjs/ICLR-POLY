#!/usr/bin/env python3
"""Let GPU 0 help the tail of GPU 1's queue after its own collection ends."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

import psutil


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-root", type=Path, required=True)
    a = ap.parse_args()
    status_path = a.run_root/"expanded_pipeline_status.json"
    status = json.loads(status_path.read_text())
    if status["stage"] != "collect128_30" or status["status"] != "running":
        raise ValueError("Helper is specific to the active 30-target preliminary collection")
    primary = next(j for j in status["jobs"] if j["gpu"] == 0)
    source = next(j for j in status["jobs"] if j["gpu"] == 1)
    try:
        process = psutil.Process(primary["pid"])
        birth = process.create_time()
    except psutil.NoSuchProcess:
        process = None
    while process is not None:
        try:
            if not process.is_running() or process.create_time() != birth or process.status() == psutil.STATUS_ZOMBIE:
                break
        except psutil.NoSuchProcess:
            break
        time.sleep(5)
    current = json.loads(status_path.read_text())
    if current["stage"] != "collect128_30" or current["status"] != "running":
        print("Collection already advanced; no helper needed.",flush=True)
        return
    args = source["command"]
    start = args.index("--targets")+1
    end = args.index("--cohort")
    targets = list(reversed([int(x) for x in args[start:end]]))
    cmd = [sys.executable,str(Path(__file__).with_name("run_selector_gpu_queue.py")),
           "--data",str(a.run_root/"data"),"--outdir",str(a.run_root/"collection"),
           "--gpu","0","--targets",*map(str,targets),"--cohort",str(a.run_root/"data/cohort_128.json")]
    print(json.dumps({"stage":"help_tail","targets":targets,"gpu":0}),flush=True)
    result = subprocess.run(cmd)
    raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
