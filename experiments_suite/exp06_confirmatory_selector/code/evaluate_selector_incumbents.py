#!/usr/bin/env python3
"""Re-evaluate trained PPO query policies with a best-observed incumbent rule."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch


HERE = Path(__file__).resolve().parent


def load_training_module():
    path = HERE / "train_current_asr_selector_gpu.py"
    spec = importlib.util.spec_from_file_location("poly_trained_selector", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["poly_trained_selector"] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


train = load_training_module()


@torch.no_grad()
def query_incumbent(policy, support, context, budget, feedback=True):
    if budget == 0:
        return int(policy.prior.argmax())
    observed = torch.zeros_like(support)
    mask = torch.zeros_like(support)
    for step in range(budget):
        query, _, _ = policy(
            observed if feedback else torch.zeros_like(observed),
            mask,
            torch.full((1,), budget - step, device=support.device),
            context,
        )
        action = query.masked_fill(mask.bool(), -1e9).argmax(1)
        mask.scatter_(1, action[:, None], 1)
        observed = support * mask
    candidates = torch.nonzero(mask[0], as_tuple=False).flatten()
    return int(candidates[support[0, candidates].argmax()])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--benign-root", type=Path, required=True)
    parser.add_argument("--false-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    device = torch.device(args.device)

    panel, arms = train.base.load_complete_panel(args.raw_root)
    records = []
    checkpoints = sorted(args.checkpoint_dir.glob("selector_*.pt"))
    for checkpoint_path in checkpoints:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        family = checkpoint["held_family"]
        seed = int(checkpoint["seed"])
        dataset = checkpoint.get("dataset", "joint")
        result_json = checkpoint_path.with_suffix(".json")
        split_seed = 47
        if result_json.exists():
            split_seed = int(
                json.loads(result_json.read_text(encoding="utf-8"))["protocol"]["split_seed"]
            )
        tags, item_keys, values_np = train.tensor_panel(panel, arms, dataset)
        splits = train.item_splits(item_keys, split_seed)
        contexts_np = np.stack(
            [train.context_vector(tag, args.benign_root, args.false_root, dataset) for tag in tags]
        )
        contexts_np = np.clip(
            (contexts_np - checkpoint["context_mean"]) / checkpoint["context_std"], -8, 8
        ).astype(np.float32)
        state = checkpoint["policy"]
        prior = state["prior"].float().to(device)
        features = state["arm_features"].float().to(device)
        context_dim = len(checkpoint["context_mean"])
        policy = train.Policy(features, prior, context_dim).to(device)
        policy.load_state_dict(state)
        policy.eval()
        values = torch.tensor(values_np, device=device)
        contexts = torch.tensor(contexts_np, device=device)
        calibration = torch.tensor(splits["calibration"], device=device)
        test = torch.tensor(splits["test"], device=device)
        for model_index, tag in enumerate(tags):
            if train.base.model_family(tag) != family:
                continue
            support = values[model_index, calibration].mean(0, keepdim=True)
            final = values[model_index, test].mean(0)
            context = contexts[model_index : model_index + 1]
            for budget in (0, 2, 4, 8, 16):
                for method, feedback in (
                    ("ppo_query_best_observed", True),
                    ("ppo_no_feedback_query_best_observed", False),
                ):
                    index = query_incumbent(policy, support, context, budget, feedback)
                    records.append(
                        {
                            "method": method,
                            "budget": budget,
                            "held_family": family,
                            "dataset": dataset,
                            "model": tag,
                            "seed": seed,
                            "split_seed": split_seed,
                            "arm": arms[index],
                            "test_asr": float(final[index]),
                            "oracle_test_asr": float(final.max()),
                            "checkpoint": checkpoint_path.name,
                        }
                    )
        print(f"evaluated {checkpoint_path.name}", flush=True)
    payload = {
        "protocol": {
            "metric": "Qwen3Guard unsafe ASR (U)",
            "selection": "best calibration ASR among arms queried by the frozen PPO policy",
            "retraining": False,
            "target_test_used_for_selection": False,
            "checkpoints": len(checkpoints),
        },
        "records": records,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"wrote {args.out} with {len(records)} records")


if __name__ == "__main__":
    main()
