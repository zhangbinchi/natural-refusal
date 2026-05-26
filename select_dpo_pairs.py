"""Select natural-refusal DPO pairs from a scored candidate pool."""

import json
import random

import fire


def select_pair(row, mode="natural", safe_cost_threshold=0.1):
    candidates = row["candidates"]
    rewards = row["candidates_rewards"]
    costs = row["candidates_costs"]
    ppls = row.get("candidates_ppls", [0.0] * len(candidates))

    safe = [i for i, c in enumerate(costs) if c < safe_cost_threshold]
    unsafe = [i for i, c in enumerate(costs) if c >= safe_cost_threshold]
    if not safe:
        return None

    if mode == "natural":
        # Balanced target: safe and helpful. Since safe costs are negative in
        # Beaver-style reward models, maximizing reward * (-cost) favors high
        # helpfulness and a strong safety margin.
        chosen = max(safe, key=lambda i: rewards[i] * (-costs[i]))
    elif mode == "ppl_natural":
        chosen = min(safe, key=lambda i: ppls[i])
    elif mode == "ppl_anti":
        chosen = max(safe, key=lambda i: ppls[i])
    elif mode == "random":
        chosen = random.choice(safe)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    if unsafe:
        rejected = max(unsafe, key=lambda i: costs[i])
    else:
        alternatives = [i for i in range(len(candidates)) if i != chosen]
        if not alternatives:
            return None
        rejected = min(alternatives, key=lambda i: rewards[i])
    if rejected == chosen:
        return None

    return {
        "prompt": row["prompt"],
        "chosen": candidates[chosen],
        "rejected": candidates[rejected],
        "chosen_reward": rewards[chosen],
        "chosen_cost": costs[chosen],
        "chosen_ppl": ppls[chosen],
        "rejected_reward": rewards[rejected],
        "rejected_cost": costs[rejected],
        "rejected_ppl": ppls[rejected],
    }


def main(
    input_file="data/dpo_candidate_pool.jsonl",
    output_file="data/dpo_pairs_natural.jsonl",
    mode="natural",
    safe_cost_threshold=0.1,
    seed=0,
):
    random.seed(seed)
    kept = 0
    skipped = 0
    with open(input_file, encoding="utf-8") as src, open(output_file, "w", encoding="utf-8") as dst:
        for line in src:
            if not line.strip():
                continue
            pair = select_pair(json.loads(line), mode=mode, safe_cost_threshold=safe_cost_threshold)
            if pair is None:
                skipped += 1
                continue
            dst.write(json.dumps(pair, ensure_ascii=False) + "\n")
            kept += 1
    print(f"Wrote {kept} DPO pairs to {output_file}; skipped {skipped} prompts.")


if __name__ == "__main__":
    fire.Fire(main)
