#!/usr/bin/env bash
set -euo pipefail

if [[ -n "${GPU:-}" ]]; then
  export CUDA_VISIBLE_DEVICES="$GPU"
fi

python generate_dpo_candidates.py \
  --backbone_model "${BACKBONE_MODEL:-mistralai/Mistral-7B-Instruct-v0.1}" \
  --reward_model "${REWARD_MODEL:-PKU-Alignment/beaver-7b-v3.0-reward}" \
  --cost_model "${COST_MODEL:-PKU-Alignment/beaver-7b-v3.0-cost}" \
  --prompt_file "${DPO_PROMPTS:-data/saferlhf_prompts.jsonl}" \
  --output_file "${CANDIDATE_POOL:-data/dpo_candidate_pool.jsonl}" \
  --num_candidates "${NUM_CANDIDATES:-50}" \
  --limit "${DPO_LIMIT:-1000}" \
  --tensor_parallel_size "${TP:-1}"

python select_dpo_pairs.py \
  --input_file "${CANDIDATE_POOL:-data/dpo_candidate_pool.jsonl}" \
  --output_file "${DPO_PAIRS:-data/dpo_pairs_natural.jsonl}" \
  --mode "${PAIR_MODE:-natural}"

python train_dpo.py \
  --model_name "${MODEL_NAME:-mistralai/Mistral-7B-Instruct-v0.1}" \
  --dpo_data "${DPO_PAIRS:-data/dpo_pairs_natural.jsonl}" \
  --output_dir "${OUTPUT_DIR:-finetuned_models/natural-refusal-dpo}" \
  --num_epochs "${NUM_EPOCHS:-2}" \
  --batch_size "${BATCH_SIZE:-4}" \
  --gradient_accumulation_steps "${GRAD_ACCUM:-4}" \
  --lr "${LR:-5e-6}"
