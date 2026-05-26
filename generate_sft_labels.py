"""Generate prompt-specific natural safety labels for SFT.

Given harmful prompts, this script asks an instruction-tuned backbone model to
produce a safety response that is grounded in the prompt rather than a fixed
refusal string. The output JSONL can be passed directly to ``train_sft.py``.
"""

import csv
import json
import random
from pathlib import Path

import fire
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


SYSTEM_PROMPT = (
    "Below is an instruction that describes a harmful task. "
    "Analyze why the instruction is unsafe, refuse to provide harmful help, "
    "and redirect toward a safe alternative when appropriate."
)


def read_prompts(path, prompt_key="prompt", limit=None):
    path = Path(path)
    prompts = []
    if path.suffix == ".csv":
        with path.open(newline="", encoding="utf-8") as f:
            for row in csv.reader(f):
                if row:
                    prompts.append(row[0])
    else:
        with path.open(encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                item = json.loads(line)
                prompts.append(item.get(prompt_key) or item.get("instruction") or item.get("question"))
    prompts = [p for p in prompts if p]
    return prompts[:limit] if limit else prompts


def format_prompts(prompts, tokenizer):
    formatted = []
    for prompt in prompts:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        if getattr(tokenizer, "chat_template", None):
            formatted.append(tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))
        else:
            formatted.append(f"{SYSTEM_PROMPT}\n\nInstruction: {prompt}\nResponse:")
    return formatted


def main(
    model_name="meta-llama/Meta-Llama-3-8B-Instruct",
    prompt_file="data/beavertail_dataset/beavertail_train_harmful_diverse.jsonl",
    output_file="data/natural_label/generated_sft_labels.jsonl",
    prompt_key="prompt",
    limit=None,
    tensor_parallel_size=1,
    max_new_tokens=192,
    temperature=0.7,
    top_p=0.9,
    seed=0,
):
    random.seed(seed)
    prompts = read_prompts(prompt_file, prompt_key=prompt_key, limit=limit)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    formatted = format_prompts(prompts, tokenizer)

    llm = LLM(
        model=model_name,
        dtype="bfloat16",
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=0.9,
    )
    sampling = SamplingParams(
        n=1,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_new_tokens,
        seed=seed,
    )
    outputs = llm.generate(formatted, sampling)

    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        for prompt, out in zip(prompts, outputs):
            answer = out.outputs[0].text.strip()
            f.write(json.dumps({"prompt": prompt, "answer": answer}, ensure_ascii=False) + "\n")
    print(f"Wrote {len(prompts)} natural SFT labels to {output_file}")


if __name__ == "__main__":
    fire.Fire(main)
