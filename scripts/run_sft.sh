#!/usr/bin/env bash
set -euo pipefail

if [[ -n "${GPU:-}" ]]; then
  export CUDA_VISIBLE_DEVICES="$GPU"
fi

python generate_sft_labels.py \
  --model_name "${LABEL_MODEL:-meta-llama/Meta-Llama-3-8B-Instruct}" \
  --prompt_file "${HARMFUL_PROMPTS:-data/beavertail_dataset/beavertail_train_harmful_diverse.jsonl}" \
  --output_file "${SFT_LABELS:-data/natural_label/generated_sft_labels.jsonl}" \
  --limit "${LABEL_LIMIT:-1000}" \
  --tensor_parallel_size "${TP:-1}"

python train_sft.py \
  --model_name "${MODEL_NAME:-meta-llama/Meta-Llama-3-8B-Instruct}" \
  --safety_data "${SFT_LABELS:-data/natural_label/generated_sft_labels.jsonl}" \
  --benign_data "${BENIGN_DATA:-data/alpaca_dataset/alpaca_data_no_safety.json}" \
  --output_dir "${OUTPUT_DIR:-finetuned_models/natural-refusal-sft}" \
  --max_benign_samples "${MAX_BENIGN_SAMPLES:-9000}" \
  --num_epochs "${NUM_EPOCHS:-3}" \
  --batch_size "${BATCH_SIZE:-4}" \
  --gradient_accumulation_steps "${GRAD_ACCUM:-4}" \
  --lr "${LR:-2e-5}"
