"""
This module allows you to quantise given model under given settings.
"""

import sys
from dotenv import load_dotenv
import os
import shutil

# Python Libraries
import warnings

# Hydra
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig
os.environ["HYDRA_FULL_ERROR"] = "1"

# Root Path Setup
import rootutils

root_path = rootutils.setup_root(__file__)
rootutils.set_root(
    path=root_path,
    project_root_env_var=True,
)
sys.path.append(str(root_path))

# Ignore certain warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# For GPU memory usage
from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo

# Load environment variables from the .env file
load_dotenv()

def print_gpu_utilization(stage):
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(handle)
    print(f"[{stage}] GPU memory occupied: {info.used//1024**3} GB.")

@hydra.main(version_base=None, config_path="../configs", config_name="quantize")
def main(cfg: DictConfig):

    # Init the tokenizer
    tokenizer = instantiate(cfg.tokenizer)
    print("✅ Successfully instantiated the tokenizer.")

    # Quantise the model
    quantised_model = instantiate(cfg.model)
    print("✅ Successfully instantiated the quantised model.")

    # Push to HF
    if cfg.hub_path is not None:
        hf_token = os.getenv("HF_TOKEN")
        quantised_model.push_to_hub(cfg.hub_path, token=hf_token)
        tokenizer.push_to_hub(cfg.hub_path, token=hf_token)
        print(f"✅ Successfully pushed the model and tokenizer to the Hugging Face Hub. Find it at {cfg.hub_path}")


if __name__ == "__main__":
    main()
