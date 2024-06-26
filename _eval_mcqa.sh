#!/bin/bash -l
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 4
#SBATCH --mem 32G
#SBATCH --time 00:30:00
#SBATCH --gres gpu:1
#SBATCH --account cs-552
#SBATCH --qos cs-552
#SBATCH --reservation cs-552
#SBATCH --output=output/slurm/slurm-%j.out

# Load modules
module load gcc/11.3.0
module load intel/2021.6.0
module load cuda/12.1.1
module load python/3.10.4

# Activate venv
source ~/venvs/mnlp-project/bin/activate

# Run the quantisation
python evaluator.py

deactivate
