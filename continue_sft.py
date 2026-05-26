"""Continue fine-tuning a LoRA adapter on benign instructions."""

import json
import random
from pathlib import Path

import fire
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

from train_sft import SFTDataset, collate, read_json_or_jsonl


def main(
    model_name="meta-llama/Meta-Llama-3-8B-Instruct",
    peft_name="finetuned_models/natural-refusal-sft",
    benign_data="data/alpaca_dataset/alpaca_data_no_safety.json",
    output_dir="finetuned_models/natural-refusal-sft/continued",
    max_benign_samples=10000,
    max_length=1024,
    num_epochs=3,
    batch_size=4,
    gradient_accumulation_steps=4,
    lr=2e-5,
    seed=0,
):
    random.seed(seed)
    torch.manual_seed(seed)

    tokenizer = AutoTokenizer.from_pretrained(peft_name if Path(peft_name).exists() else model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map="auto")
    model = PeftModel.from_pretrained(base_model, peft_name, is_trainable=True)
    model.print_trainable_parameters()

    rows = read_json_or_jsonl(benign_data, max_benign_samples)
    random.shuffle(rows)
    train_dataset = SFTDataset(rows, tokenizer, max_length=max_length)

    args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=lr,
        bf16=True,
        fp16=False,
        logging_steps=10,
        save_strategy="no",
        eval_strategy="no",
        report_to="none",
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        data_collator=lambda batch: collate(batch, tokenizer.pad_token_id),
        processing_class=tokenizer,
    )
    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Saved continued LoRA adapter to {output_dir}")


if __name__ == "__main__":
    fire.Fire(main)
