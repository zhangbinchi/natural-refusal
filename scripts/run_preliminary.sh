#!/usr/bin/env bash
set -euo pipefail

if [[ -n "${GPU:-}" ]]; then
  export CUDA_VISIBLE_DEVICES="$GPU"
fi

MODEL_NAME="${MODEL_NAME:-meta-llama/Meta-Llama-3-8B-Instruct}"
LOG_DIR="${LOG_DIR:-output/preliminary_learning_curves/llama}"
BENIGN_DATA="${BENIGN_DATA:-data/alpaca_dataset/alpaca_data_no_safety.json}"
HARMFUL_DATA="${HARMFUL_DATA:-data/natural_label/generated_sft_labels.jsonl}"
RUN_LABELS="${RUN_LABELS:-fixed_refusal random natural_per_prompt}"
FORCE_RERUN="${FORCE_RERUN:-False}"

COMMON_ARGS=(
  --model_name "$MODEL_NAME"
  --benign_data "$BENIGN_DATA"
  --harmful_data "$HARMFUL_DATA"
  --log_dir "$LOG_DIR"
  --run_forgetting "${RUN_FORGETTING:-False}"
  --log_train_val_curves True
  --harmful_train_size "${HARMFUL_TRAIN_SIZE:-500}"
  --harmful_eval_size "${HARMFUL_EVAL_SIZE:-100}"
  --harmful_eval_offset "${HARMFUL_EVAL_OFFSET:-500}"
  --train_curve_eval_size "${TRAIN_CURVE_EVAL_SIZE:-100}"
  --benign_train_size "${BENIGN_TRAIN_SIZE:-2000}"
  --num_epochs "${NUM_EPOCHS:-5}"
  --batch_size "${BATCH_SIZE:-8}"
  --lr "${LR:-2e-5}"
  --logging_steps "${LOGGING_STEPS:-50}"
  --eval_max_new_tokens "${EVAL_MAX_NEW_TOKENS:-50}"
  --eval_batch_size "${EVAL_BATCH_SIZE:-8}"
)

run_condition() {
  local group="$1"
  local label_type="$2"
  local fixed_target="$3"
  local target_name="$4"
  local run_id="$5"
  local split_seed="$6"
  local output_dir="finetuned_models/preliminary-${group}-${run_id}"
  local log_path="$LOG_DIR/learning_${group}_run${run_id}.jsonl"

  if [[ "$FORCE_RERUN" != "True" && -s "$log_path" ]]; then
    echo "Skipping $group run $run_id because $log_path already exists."
    return
  fi

  local extra_args=()
  if [[ -n "$fixed_target" ]]; then
    extra_args+=(--fixed_target "$fixed_target")
  fi

  python preliminary_learning_curve.py "${COMMON_ARGS[@]}" \
    --label_type "$label_type" \
    --label_group "$group" \
    --run_id "$run_id" \
    --target_name "$target_name" \
    --harmful_shuffle_seed "$split_seed" \
    --output_dir "$output_dir" \
    "${extra_args[@]}"
}

for label_type in $RUN_LABELS; do
  case "$label_type" in
    fixed_refusal)
      run_condition fixed_refusal fixed_refusal "Sorry, I cannot help with this." refusal 0 0
      run_condition fixed_refusal fixed_refusal "I don't know." idk 1 1
      run_condition fixed_refusal fixed_refusal "Don't ask me about this." dont 2 2
      ;;
    random)
      run_condition random random "Sky is blue." sky 0 0
      run_condition random random "You are welcome!" welcome 1 1
      run_condition random random "It is a beautiful day outside." day 2 2
      ;;
    natural_per_prompt)
      run_condition natural_per_prompt natural_per_prompt "" natural_seed0 0 0
      run_condition natural_per_prompt natural_per_prompt "" natural_seed1 1 1
      run_condition natural_per_prompt natural_per_prompt "" natural_seed2 2 2
      ;;
    *)
      echo "Unknown RUN_LABELS entry: $label_type" >&2
      exit 1
      ;;
  esac
done

python plot_preliminary_dynamics.py --log_dir "$LOG_DIR" --out_dir output/preliminary_learning_curves/figures
