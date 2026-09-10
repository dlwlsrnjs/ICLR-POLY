#!/usr/bin/env python3
"""Use a trained selector checkpoint with measured calibration observations."""
import argparse
import json
import math
from pathlib import Path

import torch

from train_selector_gpu import Policy


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--observations", type=Path, required=True,
                    help='JSON {"items_per_configuration":8,"scores":{"cfg_000":0.9,"cfg_063":0.7}}')
    ap.add_argument("--remaining", type=int, default=4)
    ap.add_argument("--policy", choices=["ppo", "supervised", "supervised_continued"], default="ppo")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    if not 0 <= a.remaining <= 16:
        ap.error("remaining must be between 0 and 16")
    # Early checkpoints serialized torch.__version__ as TorchVersion (a str subclass).
    with torch.serialization.safe_globals([torch.torch_version.TorchVersion]):
        ckpt = torch.load(a.checkpoint, map_location=a.device, weights_only=True)
    weights = ckpt[a.policy]
    policy = Policy(weights["features"], weights["prior"]).to(a.device).eval()
    policy.load_state_dict(weights)
    configurations = json.loads((a.data/"configurations.json").read_text())
    features = torch.tensor([c["features"] for c in configurations], device=a.device)
    if not torch.equal(features, policy.features):
        raise ValueError("Candidate features differ from the trained checkpoint")
    indices = {c["config_id"]:i for i,c in enumerate(configurations)}
    observations = json.loads(a.observations.read_text())
    if observations["items_per_configuration"] != 8:
        raise ValueError("This checkpoint expects means over eight calibration items")
    if not {configurations[0]["config_id"], configurations[-1]["config_id"]} <= set(observations["scores"]):
        raise ValueError("Measure the two initial configurations first")
    obs = torch.zeros(1,len(configurations),device=a.device)
    mask = torch.zeros_like(obs)
    for key,value in observations["scores"].items():
        if key not in indices or type(value) not in (float,int) or not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError("Invalid configuration or measured proxy score")
        obs[0,indices[key]], mask[0,indices[key]] = value, 1
    if a.remaining > int((mask == 0).sum()):
        raise ValueError("Remaining budget exceeds available unqueried configurations")
    with torch.no_grad():
        q,r,_ = policy(obs,mask,torch.tensor([a.remaining],device=a.device))
    chosen = int(r.argmax(1))
    result = {"policy":a.policy,"device":str(next(policy.parameters()).device),
              "observed_configuration_count":int(mask.sum()),
              "calibration_item_calls_so_far":int(mask.sum())*8,
              "recommended_configuration":configurations[chosen],
              "training_reward":"benign lexical reconstruction proxy; safety ASR unverified"}
    if a.remaining:
        next_index = int(q.masked_fill(mask.bool(),-1e9).argmax(1))
        result["next_configuration_to_measure"] = configurations[next_index]
        result["next_measurement_item_calls"] = 8
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__ == "__main__":
    main()
