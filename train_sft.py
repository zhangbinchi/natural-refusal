"""LoRA SFT for natural refusal safety alignment.

The safety file should contain JSONL rows with ``prompt`` and ``answer``. An
optional benign Alpaca-style JSON file can be mixed in to preserve utility.
"""

import json
import random
from pathlib import Path

import fire
import torch
from peft import LoraConfig, get_peft_model
from torch.utils.data import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments


IGNORE_INDEX = -100
SYSTEM_PROMPT = "Below is an instruction that describes a task. Write a response that appropriately completes the request."


def read_json_or_jsonl(path, limit=None):
    path = Path(path)
    if path.suffix == ".json":
        rows = json.load(path.open(encoding="utf-8"))
    else:
        rows = [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]
    return rows[:limit] if limit else rows


def format_user_prompt(tokenizer, instruction, input_text=""):
    content = instruction if not input_text else f"{instruction}\n{input_text}"
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": content},
    ]
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return f"{SYSTEM_PROMPT}\n\nInstruction: {content}\nResponse:"


class SFTDataset(Dataset):
    def __init__(self, rows, tokenizer, max_length=1024):
        self.rows = rows
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        prompt = row.get("prompt") or row.get("instruction") or row.get("question")
        input_text = row.get("input", "")
        answer = row.get("answer") or row.get("output") or row.get("response")
        if prompt is None or answer is None:
            raise ValueError(f"Row must contain prompt/instruction and answer/output: {row}")

        prompt_text = format_user_prompt(self.tokenizer, prompt, input_text)
        full_text = prompt_text + str(answer)
        prompt_ids = self.tokenizer.encode(prompt_text, add_special_tokens=False)
        full_ids = self.tokenizer.encode(full_text, add_special_tokens=False) + [self.tokenizer.eos_token_id]
        full_ids = full_ids[: self.max_length]
        labels = full_ids.copy()
        labels[: min(len(prompt_ids), len(labels))] = [IGNORE_INDEX] * min(len(prompt_ids), len(labels))
        return {"input_ids": full_ids, "labels": labels, "attention_mask": [1] * len(full_ids)}


def collate(batch, pad_token_id):
    max_len = max(len(x["input_ids"]) for x in batch)
    input_ids, labels, masks = [], [], []
    for x in batch:
        pad = max_len - len(x["input_ids"])
        input_ids.append(x["input_ids"] + [pad_token_id] * pad)
        labels.append(x["labels"] + [IGNORE_INDEX] * pad)
        masks.append(x["attention_mask"] + [0] * pad)
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
        "attention_mask": torch.tensor(masks, dtype=torch.long),
    }


def main(
    model_name="meta-llama/Meta-Llama-3-8B-Instruct",
    safety_data="data/natural_label/generated_sft_labels.jsonl",
    benign_data=None,
    output_dir="finetuned_models/natural-refusal-sft",
    max_safety_samples=None,
    max_benign_samples=9000,
    max_length=1024,
    num_epochs=3,
    batch_size=4,
    gradient_accumulation_steps=4,
    lr=2e-5,
    seed=0,
):
    random.seed(seed)
    torch.manual_seed(seed)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    rows = read_json_or_jsonl(safety_data, max_safety_samples)
    if benign_data:
        rows = rows + read_json_or_jsonl(benign_data, max_benign_samples)
    random.shuffle(rows)

    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map="auto")
    lora = LoraConfig(
        r=32,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

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
    print(f"Saved LoRA adapter to {output_dir}")


if __name__ == "__main__":
    fire.Fire(main)
