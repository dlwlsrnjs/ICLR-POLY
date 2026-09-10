#!/usr/bin/env python3
"""Waits for the whole GPU chain to finish (BASELINES_DONE) then runs the CPU analyses and
writes the definitive tables to results/FINAL_REPORT_20260906.txt."""
import subprocess, time
from pathlib import Path
ROOT = Path("/home/ubuntu/342/jinkwon/poly/PolyJigsaw")
DEP = ROOT / "results/method_baselines_20260906/METHODS_DONE.marker"
OUT = ROOT / "results/FINAL_REPORT_20260906.txt"
PY = "/home/ubuntu/342/jinkwon/poly/.vllm_env/bin/python"
def run(script):
    r = subprocess.run([PY, script], cwd=ROOT, capture_output=True, text=True)
    return f"\n===== {script} =====\n" + r.stdout + (("\n[stderr]\n"+r.stderr) if r.returncode else "")
def main():
    for _ in range(1200):
        if DEP.exists(): break
        time.sleep(60)
    out = f"FINAL REPORT {time.strftime('%Y-%m-%d %H:%M')}\n"
    for s in ["scripts/final_asr_table.py", "scripts/method_table.py",
              "scripts/bandit_on_new_arms.py", "scripts/bandit_warmstart.py",
              "scripts/model_characteristics.py"]:
        out += run(s)
    OUT.write_text(out)
    print("FINALIZER DONE ->", OUT)
if __name__ == "__main__":
    main()
