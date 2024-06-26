"""
This module allows you to quantise given model under given settings.
"""

import sys
from dotenv import load_dotenv
import os

# Python Libraries
import warnings

# Hydra
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig
os.environ["HYDRA_FULL_ERROR"] = "1"

# Root Path Setup
import rootutils

# HF + PyTorch
from transformers import AutoModelForCausalLM
from peft import PeftModel
import torch

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

    # - Load the base model
    base_model = AutoModelForCausalLM.from_pretrained(
        cfg.model.pretrained_model_name_or_path, 
        device_map="auto"
    )
    print_gpu_utilization("Base model loaded")

    # - Load the PEFT model
    model = PeftModel.from_pretrained(base_model, cfg.peft_model_id)
    print_gpu_utilization("PEFT model loaded")
    merged_model = model.merge_and_unload(progressbar=True)
    print_gpu_utilization("Merged model loaded")

    # - Save the model to disk
    merged_path = os.path.join("output", "merged", cfg.peft_model_id.split("/")[-1])
    merged_model.save_pretrained(merged_path, safe_serialization=True, max_shard_size="2GB")
    tokenizer.save_pretrained(merged_path)

    # - cleanup
    del base_model
    del merged_model
    torch.cuda.empty_cache() 
    print("✅ Successfully saved the merged model to disk.")

if __name__ == "__main__":
    main()
