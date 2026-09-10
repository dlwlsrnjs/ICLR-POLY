#!/usr/bin/env python3
"""Fetch pinned, ungated public target weights into this experiment's own cache."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import time


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-root", type=Path, required=True)
    a = ap.parse_args()
    cache = a.run_root/"additional_models"
    cache.mkdir(parents=True, exist_ok=True)
    os.environ["HF_XET_CACHE"] = str(cache/"xet")
    from huggingface_hub import HfApi, snapshot_download
    candidates = json.loads((a.run_root/"additional_candidates.json").read_text())
    began = time.time()

    def fetch(candidate):
        if candidate["status"] != "available" or candidate["gated"]:
            return {**candidate, "download_status":"not_eligible"}
        try:
            info = HfApi(token=False).model_info(candidate["model_id"], revision=candidate["revision"])
            if info.gated:
                raise RuntimeError("Access-gated repository excluded")
            allowed = [s.rfilename for s in info.siblings if "/" not in s.rfilename and
                       (s.rfilename.endswith((".safetensors", ".json", ".model", ".txt", ".jinja"))
                        or s.rfilename.startswith(("README", "LICENSE", "NOTICE")))]
            path = snapshot_download(candidate["model_id"], revision=candidate["revision"],
                                     cache_dir=str(cache), allow_patterns=allowed, token=False, max_workers=4)
            config = json.loads((Path(path)/"config.json").read_text())
            tokenizer = json.loads((Path(path)/"tokenizer_config.json").read_text())
            return {**candidate, "download_status":"completed", "path":str(Path(path).resolve()),
                    "architecture":config.get("architectures"), "model_type":config.get("model_type"),
                    "max_position_embeddings":config.get("max_position_embeddings"),
                    "chat_template_present":bool(tokenizer.get("chat_template")) or (Path(path)/"chat_template.jinja").is_file()}
        except Exception as exc:
            return {**candidate, "download_status":"failed", "error":str(exc).splitlines()[0][:300]}

    results = {}
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = {pool.submit(fetch, c):c["model_id"] for c in candidates}
        for future in as_completed(futures):
            result = future.result()
            results[result["model_id"]] = result
            ordered = [results[c["model_id"]] for c in candidates if c["model_id"] in results]
            report = {"elapsed_seconds":time.time()-began, "completed":sum(r["download_status"]=="completed" for r in ordered),
                      "finished":len(ordered), "planned":len(candidates), "targets":ordered}
            p = a.run_root/"downloads.json"
            tmp = p.with_suffix(".tmp")
            tmp.write_text(json.dumps(report,indent=2)+"\n")
            tmp.replace(p)
            print(json.dumps({"model":result["model_id"],"status":result["download_status"],
                              "completed":report["completed"],"planned":report["planned"]}),flush=True)
    if any(r["download_status"] != "completed" for r in results.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
