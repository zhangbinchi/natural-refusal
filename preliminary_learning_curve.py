"""
Preliminary target-dynamics experiment.

The experiment has two phases:

1. Learning curve: train on harmful prompts paired with a safety target plus
   benign Alpaca examples. The paper-scale setting is 1k harmful + 9k benign;
   for a sharper diagnostic, reduce --benign_train_size.
2. Forgetting curve: continue fine-tuning the same adapter on benign Alpaca data
   only and keep evaluating the original harmful target behavior. The
   paper-scale setting uses 10k benign continuation examples; shorter runs are
   useful for quick diagnostics.

This separates fixedness from prompt grounding:
    fixed_refusal       fixed direct refusal target
    random              fixed arbitrary random-string target
    fixed_natural       fixed explanatory safety target
    natural_per_prompt  diverse prompt-specific safety target

The log contains exact-match accuracy, ROUGE-L, target NLL, Self-BLEU, and
benign ROUGE-L for both the learning and forgetting phases.

Usage:
    python run_learning_curve.py --label_type fixed_refusal --run_forgetting True
    python run_learning_curve.py --label_type random --run_forgetting True
    python run_learning_curve.py --label_type fixed_natural --run_forgetting True
    python run_learning_curve.py --label_type natural_per_prompt \
        --harmful_data output/natural_label/8b_instruct_beavertail_jailbroken_proj_refuse_cleaned_filtered.jsonl \
        --run_forgetting True
"""

import os
import json
import copy
import random
import fire
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from peft import get_peft_model, LoraConfig
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    TrainerCallback,
)
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction


FIXED_LABELS = {
    "fixed_refusal": "I cannot help with it.",
    "refusal": "I cannot help with it.",
    "random": "Sky is blue.",
    "fixed_random": "Sky is blue.",
    "fixed_natural": (
        "This request could cause real-world harm and may involve illegal or unsafe activity. "
        "I cannot help carry it out, but I can discuss safer, lawful alternatives."
    ),
}

LABEL_ALIASES = {
    "original": "natural_per_prompt",
    "natural": "natural_per_prompt",
    "per_prompt_natural": "natural_per_prompt",
}


def compute_self_bleu(predictions, sample_size=200):
    if len(predictions) < 2:
        return 0.0
    tokenized = [p.split() for p in predictions if p.strip()]
    if len(tokenized) < 2:
        return 0.0
    sample = random.sample(tokenized, min(sample_size, len(tokenized)))
    smoother = SmoothingFunction().method1
    scores = []
    for i, hyp in enumerate(sample):
        refs = [r for j, r in enumerate(sample) if j != i]
        if hyp and refs:
            scores.append(sentence_bleu(refs, hyp, smoothing_function=smoother))
    return sum(scores) / len(scores) if scores else 0.0


class HarmfulDataset(Dataset):
    """BeaverTails harmful JSONL. Label is fixed string or per-sample response."""

    def __init__(self, data_path, tokenizer, label_type, partition="train",
                 size=1000, max_words=512, pad=True, val_fraction=0.5,
                 offset=0, eval_limit=None, fixed_target=None,
                 shuffle_seed=None):
        ann = []
        with open(data_path) as f:
            for line in f:
                ann.append(json.loads(line))
        if shuffle_seed is not None:
            rng = random.Random(shuffle_seed)
            rng.shuffle(ann)
        ann = ann[offset: offset + size]
        if partition == "all":
            self.ann = ann
        else:
            split = int(len(ann) * val_fraction)
            self.ann = ann[split:] if partition == "train" else ann[:split]
        if eval_limit is not None:
            self.ann = self.ann[:eval_limit]
        self.tokenizer = tokenizer
        self.label_type = LABEL_ALIASES.get(label_type, label_type)
        self.max_words = max_words
        self.pad = pad
        self.fixed_label = fixed_target if fixed_target is not None else FIXED_LABELS.get(self.label_type)

    def __len__(self):
        return len(self.ann)

    def __getitem__(self, index):
        IGNORE_INDEX = -100
        ann = self.ann[index]

        messages = [{"role": "user", "content": ann["prompt"]}]
        prompt_str = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        if self.fixed_label is not None:
            label_str = self.fixed_label
        elif self.label_type == "natural_per_prompt":
            label_str = ann.get("answer", ann.get("response", ""))
        else:
            raise ValueError(
                f"Unknown label_type={self.label_type!r}. Use one of "
                "fixed_refusal, random, fixed_natural, natural_per_prompt."
            )
        full_str = prompt_str + label_str

        prompt_ids = self.tokenizer.encode(prompt_str, add_special_tokens=False)
        full_ids = self.tokenizer.encode(full_str, add_special_tokens=False)
        full_ids.append(self.tokenizer.eos_token_id)

        example = torch.tensor(full_ids, dtype=torch.int64)
        if self.pad:
            pad_len = self.max_words - example.shape[0]
            if pad_len > 0:
                example = torch.cat([example, torch.full((pad_len,), -1, dtype=torch.int64)])
            else:
                example = example[:self.max_words]

        labels = copy.deepcopy(example)
        labels[:len(prompt_ids)] = -1
        mask = example.ge(0)
        label_mask = labels.ge(0)
        example[~mask] = 0
        labels[~label_mask] = IGNORE_INDEX

        return {
            "input_ids": example.tolist(),
            "labels": labels.tolist(),
            "attention_mask": mask.float().tolist(),
        }


class AlpacaBenignDataset(Dataset):
    """Alpaca JSON benign data for mixing into train and measuring side effects."""

    SYSTEM = "Below is an instruction that describes a task. Write a response that appropriately completes the request."

    def __init__(self, data_path, tokenizer, size=2000, max_words=512, pad=True,
                 offset=0):
        ann = json.load(open(data_path))
        self.ann = ann[offset: offset + size]
        self.tokenizer = tokenizer
        self.max_words = max_words
        self.pad = pad

    def __len__(self):
        return len(self.ann)

    def __getitem__(self, index):
        IGNORE_INDEX = -100
        ann = self.ann[index]
        content = ann["instruction"]
        if ann.get("input", ""):
            content += "\n" + ann["input"]

        messages = [
            {"role": "system", "content": self.SYSTEM},
            {"role": "user", "content": content},
        ]
        prompt_str = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        full_str = prompt_str + ann["output"]

        prompt_ids = self.tokenizer.encode(prompt_str, add_special_tokens=False)
        full_ids = self.tokenizer.encode(full_str, add_special_tokens=False)
        full_ids.append(self.tokenizer.eos_token_id)

        example = torch.tensor(full_ids, dtype=torch.int64)
        if self.pad:
            pad_len = self.max_words - example.shape[0]
            if pad_len > 0:
                example = torch.cat([example, torch.full((pad_len,), -1, dtype=torch.int64)])
            else:
                example = example[:self.max_words]

        labels = copy.deepcopy(example)
        labels[:len(prompt_ids)] = -1
        mask = example.ge(0)
        label_mask = labels.ge(0)
        example[~mask] = 0
        labels[~label_mask] = IGNORE_INDEX

        return {
            "input_ids": example.tolist(),
            "labels": labels.tolist(),
            "attention_mask": mask.float().tolist(),
        }


def batch_collator(batch):
    return {
        "input_ids": torch.stack([torch.tensor(ex["input_ids"], dtype=torch.long) for ex in batch]),
        "attention_mask": torch.stack([torch.tensor(ex["attention_mask"], dtype=torch.float) for ex in batch]),
        "labels": torch.stack([torch.tensor(ex["labels"], dtype=torch.long) for ex in batch]),
    }


class DynamicsLoggerCallback(TrainerCallback):
    def __init__(self, label_type, harmful_eval_dataset, benign_eval_dataset,
                 tokenizer, log_path, phase, step_offset=0,
                 compute_every_n_steps=50, max_new_tokens=96,
                 compute_nll=True, eval_at_step_zero=True,
                 eval_batch_size=8):
        self.label_type = label_type
        self.harmful_eval_dataset = harmful_eval_dataset
        self.benign_eval_dataset = benign_eval_dataset
        self.tokenizer = tokenizer
        self.log_path = log_path
        self.phase = phase
        self.step_offset = step_offset
        self.compute_every_n_steps = compute_every_n_steps
        self.max_new_tokens = max_new_tokens
        self.compute_nll = compute_nll
        self.eval_at_step_zero = eval_at_step_zero
        self.eval_batch_size = eval_batch_size
        self._did_step_zero = False
        os.makedirs(os.path.dirname(log_path), exist_ok=True)

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        if self.eval_at_step_zero and not self._did_step_zero:
            self._evaluate_and_log(model, state.global_step)
            self._did_step_zero = True

    def on_log(self, args, state, control, logs=None, model=None, **kwargs):
        if state.global_step % self.compute_every_n_steps != 0 or state.global_step == 0:
            return
        self._evaluate_and_log(model, state.global_step)

    def _evaluate_and_log(self, model, local_step):
        total_step = self.step_offset + local_step
        print(f"[{self.phase} step {local_step} | total {total_step}] Evaluating...")
        harmful_acc, harmful_rougel, harmful_out = self._compute_generation_metrics(
            model, self.harmful_eval_dataset
        )
        harmful_self_bleu = compute_self_bleu(harmful_out)
        harmful_nll = self._compute_target_nll(model, self.harmful_eval_dataset) if self.compute_nll else None
        _, benign_rougel, _ = self._compute_generation_metrics(model, self.benign_eval_dataset)

        print(
            f"[{self.phase} step {local_step}] harmful_acc={harmful_acc:.4f} "
            f"harmful_rougel={harmful_rougel:.4f} harmful_self_bleu={harmful_self_bleu:.4f} "
            f"harmful_nll={harmful_nll if harmful_nll is not None else 'NA'} "
            f"benign_rougel={benign_rougel:.4f}"
        )

        record = {
            "phase": self.phase,
            "step": local_step,
            "total_step": total_step,
            "label_type": self.label_type,
            "harmful_accuracy": harmful_acc,
            "harmful_rougel": harmful_rougel,
            "harmful_self_bleu": harmful_self_bleu,
            "harmful_target_nll": harmful_nll,
            "benign_rougel": benign_rougel,
        }
        with open(self.log_path, "a") as f:
            json.dump(record, f)
            f.write("\n")

    def _compute_generation_metrics(self, model, dataset):
        model.eval()
        loader = DataLoader(dataset, batch_size=self.eval_batch_size, collate_fn=batch_collator)
        scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
        preds, labels_list = [], []
        with torch.no_grad():
            for batch in loader:
                prompt_tensors = []
                label_texts = []
                for i in range(batch["input_ids"].shape[0]):
                    label_pos = (batch["labels"][i] != -100).nonzero(as_tuple=True)[0]
                    if len(label_pos) == 0:
                        continue
                    label_start = label_pos.min().item()
                    prompt_tensors.append(batch["input_ids"][i][:label_start])
                    label_ids = batch["labels"][i].clone()
                    label_ids[label_ids == -100] = self.tokenizer.pad_token_id
                    label_texts.append(self.tokenizer.decode(label_ids, skip_special_tokens=True).strip())
                if not prompt_tensors:
                    continue
                max_len = max(t.numel() for t in prompt_tensors)
                prompt_ids = torch.full(
                    (len(prompt_tensors), max_len),
                    self.tokenizer.pad_token_id,
                    dtype=torch.long,
                    device=model.device,
                )
                attention_mask = torch.zeros_like(prompt_ids)
                for i, ids in enumerate(prompt_tensors):
                    ids = ids.to(model.device)
                    prompt_ids[i, -ids.numel():] = ids
                    attention_mask[i, -ids.numel():] = 1
                gen_ids = model.generate(
                    input_ids=prompt_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=self.max_new_tokens,
                    pad_token_id=self.tokenizer.eos_token_id,
                    do_sample=False,
                )
                for generated, label_text in zip(gen_ids, label_texts):
                    pred_text = self.tokenizer.decode(generated[max_len:], skip_special_tokens=True)
                    preds.append(pred_text.strip())
                    labels_list.append(label_text)

        if not preds:
            return 0.0, 0.0, []
        rouge_scores = [scorer.score(l, p)["rougeL"].fmeasure for p, l in zip(preds, labels_list)]
        rougel = sum(rouge_scores) / len(rouge_scores)
        accuracy = sum(p == l for p, l in zip(preds, labels_list)) / len(preds)
        return accuracy, rougel, preds

    def _compute_target_nll(self, model, dataset):
        model.eval()
        loader = DataLoader(dataset, batch_size=4, collate_fn=batch_collator)
        total_loss = 0.0
        total_tokens = 0
        with torch.no_grad():
            for batch in loader:
                labels = batch["labels"].to(model.device)
                n_tokens = (labels != -100).sum().item()
                if n_tokens == 0:
                    continue
                out = model(
                    input_ids=batch["input_ids"].to(model.device),
                    attention_mask=batch["attention_mask"].to(model.device),
                    labels=labels,
                )
                total_loss += float(out.loss.detach().cpu()) * n_tokens
                total_tokens += n_tokens
        if total_tokens == 0:
            return None
        return total_loss / total_tokens


class TrainValCurveCallback(TrainerCallback):
    """Old-style harmful-only train/val learning curves.

    This intentionally logs only harmful train/eval fitting metrics. Benign
    samples may be present in the training set, but they are not included in
    the plotted preliminary curves.
    """

    def __init__(self, label_type, train_eval_dataset, val_eval_dataset,
                 tokenizer, log_path, phase="learn", step_offset=0,
                 run_id=0, target_name=None,
                 compute_every_n_steps=50, max_new_tokens=50,
                 eval_batch_size=8):
        self.label_type = label_type
        self.train_eval_dataset = train_eval_dataset
        self.val_eval_dataset = val_eval_dataset
        self.tokenizer = tokenizer
        self.log_path = log_path
        self.phase = phase
        self.step_offset = step_offset
        self.run_id = run_id
        self.target_name = target_name
        self.compute_every_n_steps = compute_every_n_steps
        self.max_new_tokens = max_new_tokens
        self.eval_batch_size = eval_batch_size
        os.makedirs(os.path.dirname(log_path), exist_ok=True)

    def on_log(self, args, state, control, logs=None, model=None, **kwargs):
        if state.global_step % self.compute_every_n_steps != 0 or state.global_step == 0:
            return
        self._evaluate_and_log(model, state.global_step)

    def _evaluate_and_log(self, model, step):
        total_step = self.step_offset + step
        print(f"[{self.phase} step {step} | total {total_step} | {self.label_type} run {self.run_id}] Evaluating train/val harmful curves...")
        train_acc, train_rougel, train_out = self._compute_generation_metrics(model, self.train_eval_dataset)
        val_acc, val_rougel, val_out = self._compute_generation_metrics(model, self.val_eval_dataset)
        record = {
            "phase": self.phase,
            "step": step,
            "total_step": total_step,
            "label_type": self.label_type,
            "run_id": self.run_id,
            "target_name": self.target_name,
            "train_accuracy": train_acc,
            "val_accuracy": val_acc,
            "train_rougel": train_rougel,
            "val_rougel": val_rougel,
            "train_self_bleu": compute_self_bleu(train_out),
            "val_self_bleu": compute_self_bleu(val_out),
        }
        print(
            f"[{self.phase} step {step}] train_rougel={train_rougel:.4f} "
            f"val_rougel={val_rougel:.4f} train_acc={train_acc:.4f} val_acc={val_acc:.4f}"
        )
        with open(self.log_path, "a") as f:
            json.dump(record, f)
            f.write("\n")

    def _compute_generation_metrics(self, model, dataset):
        model.eval()
        loader = DataLoader(dataset, batch_size=self.eval_batch_size, collate_fn=batch_collator)
        scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
        preds, labels_list = [], []
        with torch.no_grad():
            for batch in loader:
                prompt_tensors = []
                label_texts = []
                for i in range(batch["input_ids"].shape[0]):
                    label_pos = (batch["labels"][i] != -100).nonzero(as_tuple=True)[0]
                    if len(label_pos) == 0:
                        continue
                    label_start = label_pos.min().item()
                    prompt_tensors.append(batch["input_ids"][i][:label_start])
                    label_ids = batch["labels"][i].clone()
                    label_ids[label_ids == -100] = self.tokenizer.pad_token_id
                    label_texts.append(self.tokenizer.decode(label_ids, skip_special_tokens=True).strip())
                if not prompt_tensors:
                    continue
                max_len = max(t.numel() for t in prompt_tensors)
                prompt_ids = torch.full(
                    (len(prompt_tensors), max_len),
                    self.tokenizer.pad_token_id,
                    dtype=torch.long,
                    device=model.device,
                )
                attention_mask = torch.zeros_like(prompt_ids)
                for i, ids in enumerate(prompt_tensors):
                    ids = ids.to(model.device)
                    prompt_ids[i, -ids.numel():] = ids
                    attention_mask[i, -ids.numel():] = 1
                gen_ids = model.generate(
                    input_ids=prompt_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=self.max_new_tokens,
                    pad_token_id=self.tokenizer.eos_token_id,
                    do_sample=False,
                )
                for generated, label_text in zip(gen_ids, label_texts):
                    pred_text = self.tokenizer.decode(generated[max_len:], skip_special_tokens=True)
                    preds.append(pred_text.strip())
                    labels_list.append(label_text)

        if not preds:
            return 0.0, 0.0, []
        rouge_scores = [scorer.score(l, p)["rougeL"].fmeasure for p, l in zip(preds, labels_list)]
        rougel = sum(rouge_scores) / len(rouge_scores)
        accuracy = sum(p == l for p, l in zip(preds, labels_list)) / len(preds)
        return accuracy, rougel, preds


def build_tokenizer(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.chat_template is None:
        jinja_path = os.path.join(model_name, "chat_template.jinja")
        if os.path.exists(jinja_path):
            with open(jinja_path, "r", encoding="utf-8") as f:
                tokenizer.chat_template = f.read()
    return tokenizer


def build_lora_model(model_name):
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, device_map="auto"
    )
    peft_config = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        inference_mode=False,
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    return model


def main(
    model_name="meta-llama/Meta-Llama-3-8B-Instruct",
    label_type="fixed_refusal",
    harmful_data="data/beavertail_dataset/beavertail_train_harmful_diverse.jsonl",
    benign_data="data/alpaca_dataset/alpaca_data_no_safety.json",
    output_dir=None,
    log_dir="output/learning_curve/llama",
    batch_size=8,
    lr=2e-5,
    num_epochs=3,
    continue_lr=None,
    continue_num_epochs=3,
    gradient_accumulation_steps=1,
    logging_steps=50,
    run_forgetting=True,
    log_train_val_curves=False,
    harmful_train_size=1000,
    harmful_eval_size=100,
    harmful_train_offset=0,
    harmful_eval_offset=0,
    harmful_shuffle_seed=None,
    train_curve_eval_size=100,
    harmful_train_partition="all",
    harmful_eval_partition="all",
    benign_train_size=9000,
    benign_eval_size=50,
    continue_benign_size=10000,
    continue_benign_offset=9000,
    eval_max_new_tokens=96,
    eval_batch_size=8,
    compute_nll=True,
    eval_at_step_zero=False,
    fixed_target=None,
    label_group=None,
    run_id=0,
    target_name=None,
    seed=0,
    **kwargs,
):
    label_type = LABEL_ALIASES.get(label_type, label_type)
    label_group = LABEL_ALIASES.get(label_group or label_type, label_group or label_type)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    random.seed(seed)

    if output_dir is None:
        output_dir = f"finetuned_models/prelim-dynamics-{label_type}"
    continue_output_dir = os.path.join(output_dir, "continued")

    tokenizer = build_tokenizer(model_name)
    model = build_lora_model(model_name)

    dataset_harmful_train = HarmfulDataset(
        harmful_data, tokenizer, label_type, harmful_train_partition,
        size=harmful_train_size, offset=harmful_train_offset,
        fixed_target=fixed_target, shuffle_seed=harmful_shuffle_seed,
    )
    dataset_harmful_train_eval = HarmfulDataset(
        harmful_data, tokenizer, label_type, harmful_train_partition,
        size=train_curve_eval_size, offset=harmful_train_offset,
        fixed_target=fixed_target, shuffle_seed=harmful_shuffle_seed,
    )
    dataset_harmful_eval = HarmfulDataset(
        harmful_data, tokenizer, label_type, harmful_eval_partition,
        size=harmful_eval_size, offset=harmful_eval_offset,
        eval_limit=harmful_eval_size, fixed_target=fixed_target,
        shuffle_seed=harmful_shuffle_seed,
    )
    dataset_benign_train = AlpacaBenignDataset(benign_data, tokenizer, size=benign_train_size)
    dataset_benign_eval = AlpacaBenignDataset(benign_data, tokenizer, size=benign_eval_size)
    dataset_continue = AlpacaBenignDataset(
        benign_data, tokenizer, size=continue_benign_size, offset=continue_benign_offset
    )

    dataset_train = ConcatDataset([dataset_harmful_train, dataset_benign_train])
    print(
        f"Learning train size: {len(dataset_train)} "
        f"({len(dataset_harmful_train)} harmful + {len(dataset_benign_train)} benign) | "
        f"Harmful train-curve eval: {len(dataset_harmful_train_eval)} | "
        f"Harmful eval: {len(dataset_harmful_eval)} | Benign eval: {len(dataset_benign_eval)} | "
        f"Continual benign: {len(dataset_continue)} | "
        f"Harmful train offset: {harmful_train_offset} | Harmful eval offset: {harmful_eval_offset} | "
        f"Harmful shuffle seed: {harmful_shuffle_seed}"
    )

    if log_train_val_curves:
        log_path = os.path.join(log_dir, f"learning_{label_group}_run{run_id}.jsonl")
    else:
        log_path = os.path.join(log_dir, f"dynamics_{label_type}.jsonl")
    # Clear existing log for a fresh run
    if os.path.exists(log_path):
        os.remove(log_path)

    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        num_train_epochs=num_epochs,
        learning_rate=lr,
        bf16=True,
        fp16=False,
        logging_steps=logging_steps,
        logging_strategy="steps",
        eval_strategy="no",
        save_strategy="no",
        load_best_model_at_end=False,
        report_to="none",
        dataloader_num_workers=0,
    )

    if log_train_val_curves:
        learning_callback = TrainValCurveCallback(
            label_type=label_group,
            train_eval_dataset=dataset_harmful_train_eval,
            val_eval_dataset=dataset_harmful_eval,
            tokenizer=tokenizer,
            log_path=log_path,
            phase="learn",
            step_offset=0,
            run_id=run_id,
            target_name=target_name,
            compute_every_n_steps=logging_steps,
            max_new_tokens=eval_max_new_tokens,
            eval_batch_size=eval_batch_size,
        )
    else:
        learning_callback = DynamicsLoggerCallback(
            label_type=label_type,
            harmful_eval_dataset=dataset_harmful_eval,
            benign_eval_dataset=dataset_benign_eval,
            tokenizer=tokenizer,
            log_path=log_path,
            phase="learn",
            step_offset=0,
            compute_every_n_steps=logging_steps,
            max_new_tokens=eval_max_new_tokens,
            compute_nll=compute_nll,
            eval_at_step_zero=eval_at_step_zero,
            eval_batch_size=eval_batch_size,
        )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset_train,
        tokenizer=tokenizer,
        data_collator=batch_collator,
        callbacks=[learning_callback],
    )

    train_result = trainer.train()
    learning_steps = trainer.state.global_step

    if run_forgetting:
        if continue_lr is None:
            continue_lr = lr
        continue_args = TrainingArguments(
            output_dir=continue_output_dir,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            num_train_epochs=continue_num_epochs,
            learning_rate=continue_lr,
            bf16=True,
            fp16=False,
            logging_steps=logging_steps,
            logging_strategy="steps",
            eval_strategy="no",
            save_strategy="no",
            load_best_model_at_end=False,
            report_to="none",
            dataloader_num_workers=0,
        )
        if log_train_val_curves:
            forgetting_callback = TrainValCurveCallback(
                label_type=label_group,
                train_eval_dataset=dataset_harmful_train_eval,
                val_eval_dataset=dataset_harmful_eval,
                tokenizer=tokenizer,
                log_path=log_path,
                phase="forget",
                step_offset=learning_steps,
                run_id=run_id,
                target_name=target_name,
                compute_every_n_steps=logging_steps,
                max_new_tokens=eval_max_new_tokens,
                eval_batch_size=eval_batch_size,
            )
        else:
            forgetting_callback = DynamicsLoggerCallback(
                label_type=label_type,
                harmful_eval_dataset=dataset_harmful_eval,
                benign_eval_dataset=dataset_benign_eval,
                tokenizer=tokenizer,
                log_path=log_path,
                phase="forget",
                step_offset=learning_steps,
                compute_every_n_steps=logging_steps,
                max_new_tokens=eval_max_new_tokens,
                compute_nll=compute_nll,
                eval_at_step_zero=eval_at_step_zero,
                eval_batch_size=eval_batch_size,
            )
        continue_trainer = Trainer(
            model=model,
            args=continue_args,
            train_dataset=dataset_continue,
            tokenizer=tokenizer,
            data_collator=batch_collator,
            callbacks=[forgetting_callback],
        )
        continue_trainer.train()

    print(f"Done. Results saved to {log_path}")


if __name__ == "__main__":
    fire.Fire(main)
