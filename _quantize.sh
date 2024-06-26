#!/bin/bash -l
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 4
#SBATCH --mem 32G
#SBATCH --time 02:00:00
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

# -------- Base Phi3 model
# GPTQ-8 bits
python scripts/quantize.py

# GPTQ-4 bits
python scripts/quantize.py hub_path="cs552-mlp/phi3-gptq-4bits" model.quantization_config.bits=4

# -------- ARC3 (cs552-mlp/phi3-lora-arc3)
# -- Merge
python scripts/merge.py peft_model_id="cs552-mlp/phi3-lora-arc3"

# -- Quantize
# 8 bits
python scripts/quantize.py \
    model.pretrained_model_name_or_path="output/merged/phi3-lora-arc3" \
    hub_path="cs552-mlp/phi3-lora-arc3-gptq" \
    model.quantization_config.bits=8

# 4 bits
python scripts/quantize.py \
    model.pretrained_model_name_or_path="output/merged/phi3-lora-arc3" \
    hub_path="cs552-mlp/phi3-lora-arc3-gptq-4bits" \
    model.quantization_config.bits=4

# 3 bits
python scripts/quantize.py \
    model.pretrained_model_name_or_path="output/merged/phi3-lora-arc3" \
    hub_path="cs552-mlp/phi3-lora-arc3-gptq-3bits" \
    model.quantization_config.bits=3

# 2 bits
python scripts/quantize.py \
    model.pretrained_model_name_or_path="output/merged/phi3-lora-arc3" \
    hub_path="cs552-mlp/phi3-lora-arc3-gptq-2bits" \
    model.quantization_config.bits=2


deactivate
