"""Generate and score DPO candidates for natural-refusal preference training.

For each harmful prompt, the backbone model samples multiple candidate safety
responses. A helpfulness/reward model and a safety/cost model score each
candidate. The resulting candidate pool is consumed by ``select_dpo_pairs.py``.
"""

import json
import math
from pathlib import Path

import fire
import torch
from tqdm import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from generate_sft_labels import SYSTEM_PROMPT, format_prompts, read_prompts


def load_score_model(model_name):
    try:
        from safe_rlhf.models import AutoModelForScore

        model = AutoModelForScore.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map="auto")
    except Exception:
        from transformers import AutoModelForSequenceClassification

        model = AutoModelForSequenceClassification.from_pretrained(
            model_name, torch_dtype=torch.bfloat16, device_map="auto"
        )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()
    return model, tokenizer


@torch.no_grad()
def score_text(model, tokenizer, prompt, response):
    text = f"BEGINNING OF CONVERSATION: USER: {prompt} ASSISTANT: {response}"
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=1024).to(model.device)
    outputs = model(**inputs)
    score = getattr(outputs, "end_scores", None)
    if score is None:
        score = getattr(outputs, "scores", None)
    if score is None:
        score = outputs.logits
    return float(score.squeeze().detach().float().cpu().item())


def generation_ppl(output):
    token_count = max(1, len(output.token_ids))
    return math.exp(-float(output.cumulative_logprob) / token_count)


def main(
    backbone_model="mistralai/Mistral-7B-Instruct-v0.1",
    reward_model="PKU-Alignment/beaver-7b-v3.0-reward",
    cost_model="PKU-Alignment/beaver-7b-v3.0-cost",
    prompt_file="data/saferlhf_prompts.jsonl",
    output_file="data/dpo_candidate_pool.jsonl",
    prompt_key="prompt",
    limit=None,
    num_candidates=50,
    max_new_tokens=256,
    temperature=1.0,
    top_p=0.9,
    tensor_parallel_size=1,
    seed=0,
):
    prompts = read_prompts(prompt_file, prompt_key=prompt_key, limit=limit)
    gen_tokenizer = AutoTokenizer.from_pretrained(backbone_model)
    formatted = format_prompts(prompts, gen_tokenizer)

    llm = LLM(
        model=backbone_model,
        dtype="bfloat16",
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=0.9,
    )
    sampling = SamplingParams(
        n=num_candidates,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_new_tokens,
        logprobs=1,
        seed=seed,
    )
    reward_net, reward_tok = load_score_model(reward_model)
    cost_net, cost_tok = load_score_model(cost_model)

    outputs = llm.generate(formatted, sampling)
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        for prompt, out in tqdm(list(zip(prompts, outputs)), total=len(prompts)):
            candidates = [sample.text.strip() for sample in out.outputs]
            ppls = [generation_ppl(sample) for sample in out.outputs]
            rewards = [score_text(reward_net, reward_tok, prompt, cand) for cand in candidates]
            costs = [score_text(cost_net, cost_tok, prompt, cand) for cand in candidates]
            row = {
                "prompt": prompt,
                "candidates": candidates,
                "candidates_ppls": ppls,
                "candidates_rewards": rewards,
                "candidates_costs": costs,
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Wrote scored DPO candidate pool to {output_file}")


if __name__ == "__main__":
    fire.Fire(main)
