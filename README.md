# Natural Refusal

Minimal code release for the NeurIPS submission. The repository is organized
around the three pieces needed to reproduce the paper's core methodology.

## Repository Structure

```text
.
├── preliminary_learning_curve.py   # Fixed refusal vs. fixed random string vs. natural target diagnostic
├── plot_preliminary_dynamics.py    # Plot preliminary learning curves
├── generate_sft_labels.py          # Generate prompt-specific natural refusal labels
├── train_sft.py                    # LoRA SFT on natural refusal labels, optionally mixed with benign data
├── continue_sft.py                 # Continual benign SFT for retention experiments
├── generate_dpo_candidates.py      # Sample backbone generations and score them with reward/cost models
├── select_dpo_pairs.py             # Select balanced natural DPO targets from scored candidates
├── train_dpo.py                    # LoRA DPO on selected preference pairs
├── scripts/
│   ├── run_preliminary.sh
│   ├── run_sft.sh
│   └── run_dpo.sh
├── data/README.md                  # Expected local dataset formats
└── requirements.txt
```

Large datasets, model checkpoints, W&B logs, and raw generated outputs are not
included.

## Setup

```bash
conda create -n natural-refusal python=3.12
conda activate natural-refusal
pip install -r requirements.txt
```

The experiments require access to the corresponding Hugging Face models, e.g.
Llama-3-8B-Instruct, Mistral-7B-Instruct, and Beaver reward/cost models.

## 1. Preliminary Learning-Curve Diagnostic

This experiment tests whether learning a direct refusal target behaves similarly
to learning an unrelated fixed string. It compares:

- `fixed_refusal`: three fixed refusal strings,
- `random`: three fluent but unrelated fixed strings such as "Sky is blue.",
- `natural_per_prompt`: prompt-specific natural refusal labels.

```bash
GPU=0 \
MODEL_NAME=meta-llama/Meta-Llama-3-8B-Instruct \
HARMFUL_DATA=data/natural_label/generated_sft_labels.jsonl \
BENIGN_DATA=data/alpaca_dataset/alpaca_data_no_safety.json \
scripts/run_preliminary.sh
```

The script writes JSONL logs and curve PDFs under `output/preliminary_learning_curves/`.

## 2. Natural Refusal for SFT

First, an instruction model generates prompt-specific natural refusal labels for
harmful prompts. Then LoRA SFT trains on those labels, optionally mixed with
benign Alpaca-style instruction data.

```bash
GPU=0 \
LABEL_MODEL=meta-llama/Meta-Llama-3-8B-Instruct \
HARMFUL_PROMPTS=data/beavertail_dataset/beavertail_train_harmful_diverse.jsonl \
SFT_LABELS=data/natural_label/generated_sft_labels.jsonl \
MODEL_NAME=meta-llama/Meta-Llama-3-8B-Instruct \
BENIGN_DATA=data/alpaca_dataset/alpaca_data_no_safety.json \
OUTPUT_DIR=finetuned_models/natural-refusal-sft \
scripts/run_sft.sh
```

For safety-retention experiments, continue fine-tuning the SFT adapter on benign
instructions:

```bash
python continue_sft.py \
  --model_name meta-llama/Meta-Llama-3-8B-Instruct \
  --peft_name finetuned_models/natural-refusal-sft \
  --benign_data data/alpaca_dataset/alpaca_data_no_safety.json \
  --output_dir finetuned_models/natural-refusal-sft/continued
```

## 3. Natural Refusal for DPO

The DPO pipeline has three steps:

1. sample multiple candidate responses from the backbone model,
2. score candidates with a helpfulness/reward model and a safety/cost model,
3. choose a balanced natural target and a rejected response for DPO.

```bash
GPU=0 \
BACKBONE_MODEL=mistralai/Mistral-7B-Instruct-v0.1 \
REWARD_MODEL=PKU-Alignment/beaver-7b-v3.0-reward \
COST_MODEL=PKU-Alignment/beaver-7b-v3.0-cost \
DPO_PROMPTS=data/saferlhf_prompts.jsonl \
MODEL_NAME=mistralai/Mistral-7B-Instruct-v0.1 \
OUTPUT_DIR=finetuned_models/natural-refusal-dpo \
scripts/run_dpo.sh
```

`select_dpo_pairs.py` also supports ablations:

```bash
python select_dpo_pairs.py --mode ppl_natural
python select_dpo_pairs.py --mode ppl_anti
python select_dpo_pairs.py --mode random
```

## Data Formats

See `data/README.md`. In short:

- harmful prompts: JSONL with `prompt`;
- natural SFT labels: JSONL with `prompt` and `answer`;
- benign data: Alpaca-style JSON with `instruction`, optional `input`, and `output`;
- DPO candidate pool: JSONL with `prompt`, `candidates`, `candidates_rewards`, `candidates_costs`, and `candidates_ppls`;
- DPO pairs: JSONL with `prompt`, `chosen`, and `rejected`.

## Safety Note

This code is intended for controlled safety research. It uses harmful prompts
only to study and improve refusal behavior; it should not be used to bypass
deployed safeguards.
