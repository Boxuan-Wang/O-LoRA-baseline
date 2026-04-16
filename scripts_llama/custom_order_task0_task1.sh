#!/bin/bash
set -euo pipefail
set -x

# Example:
# DATA_ROOT=/absolute/path/to/DATA_ROOT \
# GPU_IDS=0,1,2,3,4,5,6,7 \
# bash scripts_llama/custom_order_task0_task1.sh

export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HOME/.cache/huggingface}"

GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"
DATA_ROOT="${DATA_ROOT:-DATA_ROOT}"
BASE_MODEL="${BASE_MODEL:-initial_model/llama}"
OUTPUT_ROOT="${OUTPUT_ROOT:-logs_and_outputs_llama/custom_order_task0_task1}"
DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-configs/ds_configs/stage2_llama.config}"

TRAIN_BS="${TRAIN_BS:-1}"
EVAL_BS="${EVAL_BS:-4}"
GRAD_ACC="${GRAD_ACC:-8}"
NUM_EPOCHS="${NUM_EPOCHS:-1}"
LR_TASK0="${LR_TASK0:-1e-4}"
LR_TASK1="${LR_TASK1:-1e-4}"
MAX_SOURCE_LEN="${MAX_SOURCE_LEN:-512}"
MAX_TARGET_LEN="${MAX_TARGET_LEN:-128}"

PORT="$(shuf -i25000-30000 -n1)"

TASK0_CONFIG_DIR="configs/custom_qa_order/task_0"
TASK1_CONFIG_DIR="configs/custom_qa_order/task_1"
TASK0_OUTPUT="${OUTPUT_ROOT}/outputs/1-task_0"
TASK1_OUTPUT="${OUTPUT_ROOT}/outputs/2-task_1"

mkdir -p "${OUTPUT_ROOT}/logs" "${OUTPUT_ROOT}/outputs"

CUDA_VISIBLE_DEVICES="${GPU_IDS}" deepspeed --master_port "${PORT}" src/run_uie_lora.py \
  --do_train \
  --do_predict \
  --predict_with_generate \
  --model_name_or_path "${BASE_MODEL}" \
  --data_dir "${DATA_ROOT}" \
  --task_config_dir "${TASK0_CONFIG_DIR}" \
  --instruction_strategy single \
  --output_dir "${TASK0_OUTPUT}" \
  --per_device_train_batch_size "${TRAIN_BS}" \
  --per_device_eval_batch_size "${EVAL_BS}" \
  --gradient_accumulation_steps "${GRAD_ACC}" \
  --learning_rate "${LR_TASK0}" \
  --num_train_epochs "${NUM_EPOCHS}" \
  --deepspeed "${DEEPSPEED_CONFIG}" \
  --run_name custom_order_round1_task0 \
  --max_source_length "${MAX_SOURCE_LEN}" \
  --max_target_length "${MAX_TARGET_LEN}" \
  --generation_max_length "${MAX_TARGET_LEN}" \
  --add_task_name True \
  --add_dataset_name True \
  --overwrite_output_dir \
  --overwrite_cache \
  --lr_scheduler_type constant \
  --warmup_steps 0 \
  --logging_strategy steps \
  --logging_steps 10 \
  --evaluation_strategy no \
  --save_strategy no \
  --save_steps 1500 \
  --lamda_1 0.5 \
  --lamda_2 0

sleep 5

CUDA_VISIBLE_DEVICES="${GPU_IDS}" deepspeed --master_port "${PORT}" src/run_uie_lora.py \
  --do_train \
  --do_predict \
  --predict_with_generate \
  --model_name_or_path "${TASK0_OUTPUT}/adapter" \
  --data_dir "${DATA_ROOT}" \
  --task_config_dir "${TASK1_CONFIG_DIR}" \
  --instruction_strategy single \
  --output_dir "${TASK1_OUTPUT}" \
  --per_device_train_batch_size "${TRAIN_BS}" \
  --per_device_eval_batch_size "${EVAL_BS}" \
  --gradient_accumulation_steps "${GRAD_ACC}" \
  --learning_rate "${LR_TASK1}" \
  --num_train_epochs "${NUM_EPOCHS}" \
  --deepspeed "${DEEPSPEED_CONFIG}" \
  --run_name custom_order_round2_task1 \
  --max_source_length "${MAX_SOURCE_LEN}" \
  --max_target_length "${MAX_TARGET_LEN}" \
  --generation_max_length "${MAX_TARGET_LEN}" \
  --add_task_name True \
  --add_dataset_name True \
  --overwrite_output_dir \
  --overwrite_cache \
  --lr_scheduler_type constant \
  --warmup_steps 0 \
  --logging_strategy steps \
  --logging_steps 10 \
  --evaluation_strategy no \
  --save_strategy no \
  --save_steps 1500 \
  --lamda_1 0.5 \
  --lamda_2 0
