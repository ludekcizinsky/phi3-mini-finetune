#!/bin/bash

# Ensuring the script stops if any command fails
set -e

# Default settings for the flags 
SKIP_GIT_LFS=false
SKIP_VENV=false

# ---------- GIT LFS
# Check if the --skip-git-lfs option was provided on the command line
if [[ " $* " == *" --skip-git-lfs "* ]]; then
	SKIP_GIT_LFS=true
fi

if [ "$SKIP_GIT_LFS" = false ]; then
	# Add the Git LFS repository using the script provided by packagecloud
	echo "Downloading Git LFS binaries"
	wget https://github.com/git-lfs/git-lfs/releases/download/v2.13.3/git-lfs-linux-amd64-v2.13.3.tar.gz
	tar -xzvf git-lfs-linux-amd64-v2.13.3.tar.gz
	mkdir -p ~/bin
	mv git-lfs ~/bin/
	rm -f git-lfs-linux-amd64-v2.13.3.tar.gz
	export PATH=$HOME/bin:$PATH
	echo 'export PATH=$HOME/bin:$PATH' >>~/.bashrc

	# Set up Git LFS
	echo "Setting up Git LFS..."
	~/bin/git-lfs install
fi

# ---------- VENV
# Check if the --skip-venv option was provided on the command line
if [[ " $* " == *" --skip-venv "* ]]; then
    SKIP_VENV=true
fi

if [ "$SKIP_VENV" = false ]; then
    # Load nvidia toolkit version 12.1
    echo "Loading modules..."
    module load gcc/11.3.0
    module load intel/2021.6.0
    module load cuda/12.1.1
	module load python/3.10.4

	# Create venv
	echo "Creating virtual environment..."
	virtualenv --system-site-packages ~/venvs/mnlp-project

	# Activate venv
	echo "Activating virtual environment..."
	source ~/venvs/mnlp-project/bin/activate

	# Install packages
	echo "Installing packages..."
	# - Pip upgrade
	pip install --upgrade pip

	# - torch and triton first
	pip install torch==2.2.0 triton

	# - unsloth
	pip install "unsloth[cu121-torch220]@git+https://github.com/unslothai/unsloth.git"

	# - other requirements
	pip install -r requirements.txt

	# Loading .env
	echo "Loading .env"
fi

# ---------- ENVIRONMENT VARIABLES
# Load environment variables from .env file
if [ -f .env ]; then
  export $(grep -v '^#' .env | xargs)
fi

# Setup keys
# - W&B
echo "Setting up W&B API key"
python -m wandb login $WANDB
# - HF
echo "Setting up HF login"
git config --global credential.helper store
pip install -U "huggingface_hub[cli]"
huggingface-cli login --token $HUGGINGFACE_TOKEN --add-to-git-credential
