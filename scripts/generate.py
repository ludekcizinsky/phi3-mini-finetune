"""
This module can be used for getting the sense of the given model's
generation capabilities.
"""

# Unsloth
from unsloth import PatchDPOTrainer, FastLanguageModel

# Path to the model
PatchDPOTrainer()

# Python Libraries
import warnings
from tqdm import tqdm

# Hydra
import hydra
from omegaconf import DictConfig

# HF + PyTorch
from transformers import pipeline
from transformers.utils import logging as transformers_logging

# W&B
import wandb

# Root Path Setup
import rootutils
import sys

root_path = rootutils.setup_root(__file__)
rootutils.set_root(
    path=root_path,
    project_root_env_var=True,
)
sys.path.append(str(root_path))

# Custom Imports
from src.utils import setup_exp, read_json

# Ignore certain warnings
warnings.filterwarnings("ignore", category=FutureWarning)


@hydra.main(version_base=None, config_path="../configs", config_name="generate")
def main(cfg: DictConfig):
    # --- Setup the experiment
    has_custom_prompts = cfg.custom_prompts_path is not None
    if has_custom_prompts:
        console, tokenizer, model, cfg = setup_exp(cfg, model_only=True)
    else:
        console, tokenizer, model, _, val_recipe, cfg = setup_exp(
            cfg, init_with_peft=False, model_only=False
        )

    # --- Inference setup
    console.rule("[bold]Generation", style="red")
    # - Enable native 2x faster inference
    FastLanguageModel.for_inference(model)

    # - Create the pipeline
    transformers_logging.set_verbosity(50)
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
    transformers_logging.set_verbosity(30)

    # - Sample a few prompts
    if has_custom_prompts:
        samples = [{k: v} for k, v in read_json(cfg.custom_prompts_path).items()]
    else:
        samples = val_recipe.shuffle(seed=cfg.trainer.args.data_seed).select(
            range(cfg.num_samples)
        )

    # --- Generation
    data = []
    for sample in tqdm(samples, desc="Generating responses"):
        prompt = sample["prompt"]
        outputs = pipe(
            prompt,
            max_new_tokens=2048,
            do_sample=True,
            temperature=1.0,
            top_k=50,
            top_p=0.9,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )
        generated_answer = outputs[0]["generated_text"][len(prompt) :].strip()
        data.append([prompt, generated_answer])

    # --- Log to W&B
    if cfg.wandb.enabled:
        table = wandb.Table(data=data, columns=["prompt", "generated_text"])
        wandb.log({"generations": table})
        console.print(f"Logged generations to W&B")
        wandb.finish()
    else:
        console.print(f"Did not log to W&B as run_id is not provided.")


if __name__ == "__main__":
    main()
