#!/bin/bash -l
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 4
#SBATCH --mem 32G
#SBATCH --time 04:00:00
#SBATCH --gres gpu:1
#SBATCH --account cs-552
#SBATCH --qos cs-552
#SBATCH --reservation cs-552
#SBATCH --output=output/slurm/slurm-%j.out

# Load modules
module load intel/2021.6.0
module load cuda/12.1.1
module load python/3.10.4

# Activate venv
source ~/venvs/mnlp-project/bin/activate

# Source .env
if [ -f .env ]; then
  export $(grep -v '^#' .env | xargs)
fi

# Available baselines
PHI3="pretrained=microsoft/Phi-3-mini-4k-instruct"
OPENELM="pretrained=apple/OpenELM-3B-Instruct,trust_remote_code=True,add_bos_token=True,tokenizer=meta-llama/Llama-2-7b-hf"
LLAMA3="pretrained=meta-llama/Meta-Llama-3-8B-Instruct"

# Available finetunes
PHI3_OPENBOOKQA="${PHI3},peft=cs552-mlp/phi3-lora-openbookqa3"
PHI3_SCIQ="${PHI3},peft=cs552-mlp/phi3-lora-sciq3"
PHI3_ARC="${PHI3},peft=cs552-mlp/phi3-lora-arc3"
PHI3_MCQ="${PHI3},peft=cs552-mlp/phi3-lora-mcq3"

# Quantised models
PHI3_GPTQ_8b="pretrained=cs552-mlp/phi3-gptq-8bits"
PHI3_GPTQ_4b="pretrained=cs552-mlp/phi3-gptq-4bits"

# Choose model
MODEL_ID="phi3-gptq-8b"
MODEL_ARGS=${PHI3_GPTQ_8b}

# More settings
OUTPUT_PATH=results/${MODEL_ID}
BENCHMARKS=ai2_arc,mmlu_stem,gpqa_main_zeroshot,gpqa_diamond_zeroshot,gpqa_extended_zeroshot,openbookqa,sciq,epfl-mcq
DEVICE=cuda:0
BATCH_SIZE=auto:8

# Run Benchmark
lm-eval \
    --model hf \
    --model_args ${MODEL_ARGS} \
    --output_path ${OUTPUT_PATH} \
    --tasks ${BENCHMARKS} \
    --batch_size ${BATCH_SIZE} \
    --device ${DEVICE} \
    --log_samples

deactivate
