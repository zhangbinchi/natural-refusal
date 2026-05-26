"""LoRA DPO training for natural-refusal preference pairs."""

import json
from pathlib import Path

import fire
import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOConfig, DPOTrainer


SYSTEM_PROMPT = "Below is an instruction that describes a task. Write a response that appropriately completes the request."


def format_prompt(tokenizer, prompt):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return f"{SYSTEM_PROMPT}\n\nInstruction: {prompt}\nResponse:"


def load_pairs(path, tokenizer):
    prompts, chosen, rejected = [], [], []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            prompts.append(format_prompt(tokenizer, row["prompt"]))
            chosen.append(row["chosen"])
            rejected.append(row["rejected"])
    return Dataset.from_dict({"prompt": prompts, "chosen": chosen, "rejected": rejected})


def main(
    model_name="mistralai/Mistral-7B-Instruct-v0.1",
    dpo_data="data/dpo_pairs_natural.jsonl",
    output_dir="finetuned_models/natural-refusal-dpo",
    num_epochs=2,
    batch_size=4,
    gradient_accumulation_steps=4,
    lr=5e-6,
    beta=0.1,
    max_prompt_length=512,
    max_completion_length=256,
    seed=0,
):
    torch.manual_seed(seed)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map="auto")
    lora = LoraConfig(
        r=32,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    dataset = load_pairs(dpo_data, tokenizer)
    args = DPOConfig(
        output_dir=output_dir,
        beta=beta,
        loss_type="sigmoid",
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        logging_steps=10,
        save_strategy="no",
        eval_strategy="no",
        report_to="none",
        max_prompt_length=max_prompt_length,
        max_completion_length=max_completion_length,
        bf16=True,
        fp16=False,
        optim="adamw_torch_fused",
    )
    trainer = DPOTrainer(
        model=model,
        ref_model=None,
        args=args,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=lora,
    )
    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Saved DPO LoRA adapter to {output_dir}")


if __name__ == "__main__":
    fire.Fire(main)
