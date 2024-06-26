"""
Inspired from:
https://huggingface.co/microsoft/Phi-3-mini-128k-instruct/resolve/main/sample_finetune.py
https://www.philschmid.de/dpo-align-llms-in-2024-with-trl

This script is used to fine-tune a model using specified strategy. Currently, the script supports
two strategies: SFT and DPO.
"""

# Unsloth
# from unsloth import PatchDPOTrainer

# Path to the model
# PatchDPOTrainer()

import sys

# Python Libraries
import warnings

# Hydra
import hydra

# Root Path Setup
import rootutils

# HF + PyTorch
import torch
from datasets import concatenate_datasets

# W&B
import wandb
from omegaconf import DictConfig
from transformers.utils import logging as transformers_logging

root_path = rootutils.setup_root(__file__)
rootutils.set_root(
    path=root_path,
    project_root_env_var=True,
)
sys.path.append(str(root_path))

# Custom Imports
from src.utils import setup_exp

# Ignore certain warnings
warnings.filterwarnings("ignore", category=FutureWarning)


@hydra.main(version_base=None, config_path="../configs", config_name="finetune")
def main(cfg: DictConfig):

    # --- Setup the experiment
    console, tokenizer, model, train_recipe, val_recipe, val_recipe_sub, cfg = (
        setup_exp(cfg)
    )

    # --- Decide on which val recipe to use during train and eval
    if cfg.recipe.full_train:
        # Set the validation datasets to None
        during_train_val = None
        final_val = None

        # Assert that the user has not specified evaluation during training or final evaluation
        assert (
            cfg.trainer.args.evaluation_strategy == "no"
        ), "Cannot evaluate during training when using the full dataset for training"
        assert (
            not cfg.eval
        ), "Cannot do final evaluation when using the full dataset for training"

        # And finally, concat the training and validation datasets
        train_recipe = concatenate_datasets([train_recipe, val_recipe])
        console.print(
            f"Using the full dataset for training. Size of the dataset is {len(train_recipe)}."
        )
    else:
        during_train_val = val_recipe_sub
        final_val = val_recipe if cfg.full_size_final_val else val_recipe_sub

    # --- Training
    console.rule("\n[bold]Training", style="red")
    transformers_logging.set_verbosity(40)  # error lvl
    # - Init the model
    trainer = hydra.utils.instantiate(
        cfg.trainer,
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_recipe,
        eval_dataset=during_train_val,
        _recursive_=True,
    )

    n_params = trainer.get_num_trainable_parameters() / 1e9
    console.print(
        f"Total number of trainable parameters in billion: {n_params:.2f} which is {100*n_params/(3.8+n_params):.2f} %."
    )

    # - Train
    train_result = trainer.train()
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()
    print("\n")

    # --- Evaluate model, if specified
    if cfg.eval:
        console.rule("\n[bold]Final Evaluation", style="red")
        final_val = final_val.map(trainer.tokenize_row, num_proc=1)
        trainer.evaluate(final_val)
        print("\n")

    # --- Save the model, finish W&B run, free up resources
    trainer.save_model(cfg.trainer.args.output_dir)
    if cfg.wandb.enabled:
        wandb.finish()
    del model
    del trainer
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
