#!/bin/bash
export CUDA_VISIBLE_DEVICE=1
INPUT_PATH=/userhomes/minsu/symr/data/proofnet_test.jsonl    #/userhomes/minsu/symr/data/proofnet_test.jsonl /userhomes/minsu/symr/data/miniF2F_test.jsonl
MODEL_PATH=Minsukim/internlm_1_8_sft_bf16    #  deepseek-ai/DeepSeek-Prover-V1.5-SFT Minsukim/internlm_1_8_sft_bf16
OUTPUT_DIR=/userhomes/minsu/symr/eval/results/proofnet/internlm_GRPO_baseline_workbook_data_32
SPLIT=test
N=32
CPU=32 #32
GPU=1
FIELD=complete
while getopts ":i:m:o:s:n:c:g:" opt; do
  case $opt in
    i) INPUT_PATH="$OPTARG"
    ;;
    m) MODEL_PATH="$OPTARG"
    ;;
    o) OUTPUT_DIR="$OPTARG"
    ;;
    s) SPLIT="$OPTARG"
    ;;
    n) N="$OPTARG"
    ;;
    c) CPU="$OPTARG"
    ;;
    g) GPU="$OPTARG"
    ;;
  esac
done
#python -m step1_inference --input_path ${INPUT_PATH}  --model_path ${MODEL_PATH}  --output_dir $OUTPUT_DIR --split $SPLIT --n $N --gpu $GPU

INPUT_FILE=${OUTPUT_DIR}/to_inference_codes.json
COMPILE_OUTPUT_PATH=${OUTPUT_DIR}/code_compilation.json
python -m step2_compile --input_path $INPUT_FILE --output_path $COMPILE_OUTPUT_PATH --cpu $CPU


SUMMARIZE_OUTPUT_PATH=${OUTPUT_DIR}/compilation_summarize.json
python -m step3_summarize_compile --input_path $COMPILE_OUTPUT_PATH --output_path $SUMMARIZE_OUTPUT_PATH --field ${FIELD}