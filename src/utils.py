"""
This module contains utility functions that are used throughout the project.
"""

import json
import logging
import os
import sys
from contextlib import contextmanager
from typing import Any, List

# Hydra
import hydra
import numpy as np

# HF + PyTorch
import torch
import transformers
import wandb
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf, open_dict
from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizer
from transformers.trainer_callback import ProgressCallback
from trl.commands.cli_utils import init_zero_verbose

import datasets
from datasets import Dataset


def read_json(path, mode="r", **kwargs):
    return json.loads(read_file(path, mode=mode, **kwargs))


def write_json(data, path):
    return write_file(json.dumps(data, indent=2), path)


def to_jsonl(data):
    return json.dumps(data).replace("\n", "")


def read_file(path, mode="r", **kwargs):
    """Reads a file and returns its content."""
    with open(path, mode=mode, **kwargs) as f:
        return f.read()


def write_file(data, path, mode="w", **kwargs):
    with open(path, mode=mode, **kwargs) as f:
        f.write(data)


def read_jsonl(path, mode="r", **kwargs):
    ls = []
    with open(path, mode, **kwargs) as f:
        for line in f:
            ls.append(json.loads(line))
    return ls


def write_jsonl(data, path, mode="w"):
    assert isinstance(data, list)
    lines = [to_jsonl(elem) for elem in data]
    write_file("\n".join(lines) + "\n", path, mode=mode)


def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )


def print_dict_as_table(data, console, max_columns_per_table):
    """
    Prints a dictionary as a table with keys as columns and values in a single row.
    If the number of columns exceeds max_columns_per_table, multiple tables are printed.

    Args:
        data (dict): A dictionary to be displayed as a table.
        console (Console): A Rich console object for output.
        max_columns_per_table (int): Maximum number of columns allowed per table.
    """

    # Function to create and print a table
    def create_and_print_table(keys, values):
        table = Table(show_header=True, header_style="italic blue")
        for key in keys:
            table.add_column(key, style="dim", width=18)
        table.add_row(*values)
        console.print(table)

    keys = list(data.keys())
    values = [str(data[key]) for key in keys]

    # Process keys and values in chunks according to max_columns_per_table
    for i in range(0, len(keys), max_columns_per_table):
        chunk_keys = keys[i : i + max_columns_per_table]
        chunk_values = values[i : i + max_columns_per_table]
        create_and_print_table(chunk_keys, chunk_values)


def on_log(self, args, state, control, logs=None, **kwargs):
    if state.is_local_process_zero and self.training_bar is not None:
        _ = logs.pop("total_flos", None)


def log_prompt_and_answer(console, prompt, generated_answer):
    """
    Log prompt and generated answer using the provided console.
    """
    console.print("\n[b]Prompt:[/b]")
    console.print(prompt)
    console.print("\n[b]Generated Answer:[/b]")
    console.print(generated_answer.strip())
    console.rule(style="blue")


def setup_logging():
    """
    Setup logging with RichHandler and set log level from configuration.
    """
    init_zero_verbose()
    console = Console()

    # Define log format and handlers with RichHandler
    rich_handler = RichHandler(console=console, show_time=False)
    FORMAT = "%(message)s"
    logging.basicConfig(
        format=FORMAT,
        datefmt="[%X]",
        handlers=[rich_handler],
        level=logging.INFO,
    )

    # Set log level from configuration
    log_level = "WARNING"
    logger = logging.getLogger(__name__)
    logger.setLevel(log_level.upper())

    # Set verbosity for datasets and transformers
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    ProgressCallback.on_log = (
        on_log  # Hacky way to stop getting the annoying dicts logged in the stdout
    )

    return console


def setup_env_vars(cfg):
    """
    Setup environment variables for parallelism and W&B.

    Args:
        cfg (DictConfig): Configuration object.
    """
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["WANDB_PROJECT"] = cfg.wandb.project
    os.environ["WANDB_ENTITY"] = cfg.wandb.entity
    os.environ["WANDB_LOG_MODEL"] = cfg.wandb.checkpoint
    os.environ["WANDB_DIR"] = cfg.wandb.output_dir
    os.environ["HYDRA_FULL_ERROR"] = "1"


def setup_wandb(cfg, console):
    """
    Setup W&B for logging.

    Args:
        cfg (DictConfig): Configuration object.
    """

    if cfg.wandb.enabled:
        console.rule("[bold]Wandb Config", style="red")

        # Get the command that executed the script
        command = "python " + " ".join(sys.argv)

        # Add it to the config
        with open_dict(cfg):
            cfg.wandb.command = command

        wandb.init(
            entity=cfg.wandb.entity,
            project=cfg.wandb.project,
            config=OmegaConf.to_container(cfg, resolve=True),
            resume=False,
            settings=wandb.Settings(start_method="thread"),
            reinit=False,
            group=cfg.wandb.group,
            force=True,
            notes=cfg.wandb.notes,
            tags=cfg.wandb.tags,
        )
        print("\n")
    else:
        with open_dict(cfg):
            cfg.trainer.args.report_to = "none"


@contextmanager
def temporary_log_level(level):
    """
    Temporarily set the log level of the logger.

    Args:
        level (int): The log level to be set.
    """

    logger = logging.getLogger()
    original_level = logger.getEffectiveLevel()
    logger.setLevel(level)
    try:
        yield
    finally:
        logger.setLevel(original_level)


def length_extractor(batch, tokenizer, column_names):
    """
    Extract the length of each sentence in the batch using the tokenizer.

    Args:
        batch (dict): A batch of data.

    Returns:
        dict: A dictionary containing the token lengths of each sentence.
    """
    result = {}
    for col in column_names:
        tokenized_lengths = [len(tokenizer.encode(sentence)) for sentence in batch[col]]
        result[col] = tokenized_lengths
    return result


def set_model_dtypes(model, dtype):
    """
    Set the model parameters to a specific dtype.

    Args:
        model (transformers.PreTrainedModel): A model object.
        dtype (torch.dtype): The dtype to be set.
    """

    model.torch_dtype = dtype
    model.bnb_4bit_compute_dtype = dtype

    return model


def get_xpercentile_maxlen_data_sft(
    cfg,
    tokenizer: PreTrainedTokenizer,
    train_dataset: Dataset,
    eval_dataset: Dataset,
    percentile_value: float = 95,
    console: Console = None,
    data_recipe_info: dict = dict()
):
    """
    Filter the datasets based on the x-percentile of the token lengths.

    Args:
        cfg (DictConfig): Configuration object.
        tokenizer (PreTrainedTokenizer): A tokenizer object.
        train_dataset (Dataset): Training
        eval_dataset (Dataset): Evaluation dataset.
        percentile_value (float): The percentile value to be used.
        console (Console): A Rich console object for output.

    Returns:
        Tuple[Dataset, Dataset]: Filtered training and evaluation datasets.
    """

    if console is not None:
        console.rule(
            "[bold]Filtering datasets based on x-percentile of sequence length",
            style="red",
        )

    def compute_length_batch(examples, key):
        return {
            f"{key}_length": [
                len(tokenizer(text)["input_ids"]) for text in examples[key]
            ]
        }

    # Compute the length of the datasets before filtering
    train_n, eval_n = len(train_dataset), len(eval_dataset)
    
    # Compute lengths in parallel using map with batched processing
    train_dataset = train_dataset.map(
        lambda examples: compute_length_batch(examples, "text"),
        batched=True,
        num_proc=cfg.hardware.num_workers,
    )

    eval_dataset = eval_dataset.map(
        lambda examples: compute_length_batch(examples, "text"),
        batched=True,
        num_proc=cfg.hardware.num_workers,
    )

    # Compute percentiles for the training dataset
    max_seq_length = np.percentile(train_dataset["text_length"], percentile_value)

    # Up the lengths to the next multiple of 2
    max_seq_length = ((int(max_seq_length) + 1) // 2) * 2

    # Filter training/val dataset based on max_seq_length
    train_dataset = train_dataset.filter(
        lambda x: x["text_length"] <= max_seq_length
    )
    eval_dataset = eval_dataset.filter(
        lambda x: x["text_length"] <= max_seq_length
    )

    if console is not None:
        # Print the lengths
        print("\n")
        console.print("[bold]Sequence max lengths:")
        console.print(f"\t - Text: {max_seq_length}")

        # Print the percentage of data retained
        train_retained = 100 * len(train_dataset) / train_n
        eval_retained = 100 * len(eval_dataset) / eval_n
        console.print("\n[bold]Percentage of data retained:")
        console.print(f"\t - Train: {train_retained:.2f}%")
        console.print(f"\t - Eval: {eval_retained:.2f}%")
        print("\n")

    # Save the max lengths in data recipe info
    data_recipe_info["max_seq_length"] = max_seq_length

    # Finally, set the max_seq_length in the config
    with open_dict(cfg):

        # Unsloth
        if cfg.model_lib == "unsloth":
            # We use packing which packs several shorter sequences up to the specified max_seq_length 
            cfg.model.max_seq_length = cfg.trainer.max_seq_length
            cfg.peft_config.max_seq_length = cfg.trainer.max_seq_length

    return train_dataset, eval_dataset, cfg


def get_xpercentile_maxlen_data_dpo(
    cfg,
    tokenizer: PreTrainedTokenizer,
    train_dataset: Dataset,
    eval_dataset: Dataset,
    percentile_value: float = 95,
    console: Console = None,
    data_recipe_info: dict = dict(),
):
    """
    Filter the datasets based on the x-percentile of the token lengths.

    Args:
        cfg (DictConfig): Configuration object.
        tokenizer (PreTrainedTokenizer): A tokenizer object.
        train_dataset (Dataset): Training
        eval_dataset (Dataset): Evaluation dataset.
        percentile_value (float): The percentile value to be used.
        console (Console): A Rich console object for output.

    Returns:
        Tuple[Dataset, Dataset]: Filtered training and evaluation datasets.
    """

    if console is not None:
        console.rule(
            "[bold]Filtering datasets based on x-percentile of sequence length",
            style="red",
        )

    def compute_length_batch(examples, key):
        return {
            f"{key}_length": [
                len(tokenizer(text)["input_ids"]) for text in examples[key]
            ]
        }

    # Compute the length of the datasets before filtering
    train_n, eval_n = len(train_dataset), len(eval_dataset)

    # Compute lengths in parallel using map with batched processing
    train_dataset = train_dataset.map(
        lambda examples: compute_length_batch(examples, "prompt"),
        batched=True,
        num_proc=cfg.hardware.num_workers,
    )
    train_dataset = train_dataset.map(
        lambda examples: compute_length_batch(
            {
                "prompt_chosen": [
                    examples["prompt"][i] + examples["chosen"][i]
                    for i in range(len(examples["prompt"]))
                ]
            },
            "prompt_chosen",
        ),
        batched=True,
        num_proc=cfg.hardware.num_workers,
    )
    train_dataset = train_dataset.map(
        lambda examples: compute_length_batch(
            {
                "prompt_rejected": [
                    examples["prompt"][i] + examples["rejected"][i]
                    for i in range(len(examples["prompt"]))
                ]
            },
            "prompt_rejected",
        ),
        batched=True,
        num_proc=cfg.hardware.num_workers,
    )

    # Compute percentiles for the training dataset
    prompt_length = np.percentile(train_dataset["prompt_length"], percentile_value)
    max_seq_length_chosen = np.percentile(
        train_dataset["prompt_chosen_length"], percentile_value
    )
    max_seq_length_rejected = np.percentile(
        train_dataset["prompt_rejected_length"], percentile_value
    )
    max_seq_length = max(max_seq_length_chosen, max_seq_length_rejected)

    # Filter training dataset based on max_seq_length
    train_dataset = train_dataset.filter(
        lambda x: x["prompt_chosen_length"] <= max_seq_length
    )

    # Up the lengths to the next multiple of 2
    prompt_length = ((int(prompt_length) + 1) // 2) * 2
    max_seq_length = ((int(max_seq_length) + 1) // 2) * 2

    # Filter evaluation dataset based on max_seq_length from training dataset
    eval_dataset = eval_dataset.map(
        lambda examples: compute_length_batch(
            {
                "prompt_chosen": [
                    examples["prompt"][i] + examples["chosen"][i]
                    for i in range(len(examples["prompt"]))
                ]
            },
            "prompt_chosen",
        ),
        batched=True,
        num_proc=cfg.hardware.num_workers,
    )
    eval_dataset = eval_dataset.filter(
        lambda x: x["prompt_chosen_length"] <= max_seq_length
    )

    if console is not None:
        # Print the lengths
        print("\n")
        console.print("[bold]Sequence max lengths:")
        console.print(f"\t - Prompt: {prompt_length}")
        console.print(f"\t - Chosen/Rejected: {max_seq_length}")

        # Print the percentage of data retained
        train_retained = 100 * len(train_dataset) / train_n
        eval_retained = 100 * len(eval_dataset) / eval_n
        console.print("\n[bold]Percentage of data retained:")
        console.print(f"\t - Train: {train_retained:.2f}%")
        console.print(f"\t - Eval: {eval_retained:.2f}%")
        print("\n")

    # Save the max lengths in data recipe info
    data_recipe_info["max_seq_length"] = max_seq_length
    data_recipe_info["max_prompt_length"] = prompt_length

    # Finally, set the max_seq_length in the config
    with open_dict(cfg):

        # DPO
        cfg.trainer.max_length = max_seq_length  # prompt + chosen/rejected
        cfg.trainer.max_target_length = max_seq_length  # chosen/rejected
        cfg.trainer.max_prompt_length = prompt_length  # prompt

        # Unsloth
        if cfg.model_lib == "unsloth":
            cfg.model.max_seq_length = max_seq_length
            cfg.peft_config.max_seq_length = max_seq_length

    return train_dataset, eval_dataset, cfg


def init_model_tokenizer(cfg, console):

    # --- Main setup of model and tokenizer
    console.rule("\n[bold]Initialisation of model and tokenizer", style="red")
    # - HF
    if cfg.model_lib == "hf":

        # - Model using QLoRA
        console.print(f"Instantiating model <{cfg.model._target_}>")
        with temporary_log_level(logging.ERROR):
            model: AutoModelForCausalLM = set_model_dtypes(
                instantiate(cfg.model), torch.bfloat16
            )

        # - Tokenizer
        console.print(f"Instantiating tokenizer <{cfg.tokenizer._target_}>")
        tokenizer: AutoTokenizer = hydra.utils.call(cfg.tokenizer)

    # - Unsloth
    elif cfg.model_lib == "unsloth":
        # Init the model and tokenizer
        model, tokenizer = instantiate(cfg.model)

        # Ensure we use fp16
        with open_dict(cfg):
            cfg.trainer.args.fp16 = True
            cfg.trainer.args.bf16 = False

    else:
        raise ValueError("Model not supported")

    # --- Other setups
    if cfg.tokenizer.pad_token == "eos_token":
        console.print("Setting pad token to <eos_token>")
        tokenizer.pad_token = tokenizer.eos_token

    # Based on the settings described in the official microsoft cookbook
    # https://github.com/microsoft/Phi-3CookBook/blob/main/code/04.Finetuning/Phi-3-finetune-qlora-python.ipynb
    if not cfg.is_dpo:
        tokenizer.pad_token = tokenizer.unk_token
        tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
        tokenizer.padding_side = 'right' 
    print("\n")

    return model, tokenizer, cfg


def filter2str(info: dict):

    # Helper function
    kwargs2str = lambda kw: " / ".join(
        [f"{k}: {v}" for k, v in kw.items() if v is not None]
    )

    if info.get("filtering_strategy") is None:
        return "None"
    else:
        return (
            f"{info.get('filtering_strategy')} ({kwargs2str(info['filtering_kwargs'])})"
        )


def recipe2str(datasets, ratios, info, split):

    # Merge the datasets and ratios into a collection of tuples
    merged = [(dataset, ratio) for dataset, ratio in zip(datasets, ratios)]

    # Sort the merge by the dataset ID
    merged.sort(key=lambda x: x[0].ID)

    # Create a string representation of the recipe
    try:
        recipe_str_loc = "\n".join(
            [
                f"{dataset.ID} ({ratio})\n\t - {filter2str(dataset.filter_info)}"
                for dataset, ratio in merged
            ]
        )

        info[f"{split}_recipe_str_loc"] = recipe_str_loc
        recipe_str_wandb = "/".join(
            [
                f"[{i}] {dataset.ID} [{ratio}, {filter2str(dataset.filter_info)}]"
                for i, (dataset, ratio) in enumerate(merged)
            ]
        )
        info[f"{split}_recipe_str"] = recipe_str_wandb

    except Exception as _:
        recipe_str_loc = "\n".join(
            [
                f"{dataset.ID} ({ratio})\n"
                for dataset, ratio in merged
            ]
        )

        info[f"{split}_recipe_str_loc"] = recipe_str_loc
        recipe_str_wandb = "/".join(
            [
                f"[{i}] {dataset.ID} [{ratio}]"
                for i, (dataset, ratio) in enumerate(merged)
            ]
        )
        info[f"{split}_recipe_str"] = recipe_str_wandb



def recinfo2str(info: dict, console):

    console.rule(f"[bold]Final recipe information", style="red")
    # Train
    console.print(f"[bold]Training:")
    console.print("Recipe: ", info["train_recipe_str_loc"])
    console.print(f"Original Size: {info['train_len_before']}")
    console.print(f"Final Size: {info['train_len_after']}")
    print("\n")
    # Val
    console.print(f"[bold]Validation:")
    console.print("Recipe: ", info["val_recipe_str_loc"])
    console.print(f"Original Size: {info['val_len_before']}")
    console.print(f"Final Size: {info['val_len_after']}")
    print("\n")


def init_data_recipes(console, cfg, tokenizer):

    # --- Keep track of the data recipe info to log to W&B
    data_recipe_info = dict()

    # --- Init the individual datasets
    console.rule("[bold]Load Datasets", style="red")
    # - Train
    train_datasets: List[Dataset] = [
        instantiate(cfg, tokenizer=tokenizer) for cfg in cfg.recipe.train_datasets
    ]
    recipe2str(train_datasets, cfg.recipe.train_ratio, data_recipe_info, "train")

    # - Val
    val_datasets: List[Dataset] = [instantiate(cfg, tokenizer=tokenizer) for cfg in cfg.recipe.val_datasets]
    recipe2str(val_datasets, cfg.recipe.val_ratio, data_recipe_info, "val")

    # --- Init the recipes
    # - Train
    train_recipe: Dataset = instantiate(
        cfg.recipe, datasets=train_datasets, ratio=cfg.recipe.train_ratio, _recursive_=False
    )
    data_recipe_info["train_len_before"] = len(train_recipe)
    # - Val
    val_recipe: Dataset = instantiate(
        cfg.recipe, datasets=val_datasets, ratio=cfg.recipe.val_ratio, _recursive_=False
    )
    data_recipe_info["val_len_before"] = len(val_recipe)
    print("\n")

    # --- Find the max sequence length using x-percentile, and return the filtered datasets
    get_perc_func = get_xpercentile_maxlen_data_dpo if cfg.is_dpo else get_xpercentile_maxlen_data_sft
    train_recipe, val_recipe, cfg = get_perc_func(
        cfg,
        tokenizer,
        train_recipe,
        val_recipe,
        cfg.recipe.keep_percentile,
        console,
        data_recipe_info,
    )
    data_recipe_info["train_len_after"] = len(train_recipe)
    data_recipe_info["val_len_after"] = len(val_recipe)

    # --- Subset eval dataset to max_eval_samples
    if cfg.recipe.max_eval_samples is not None:
        val_recipe_sub = val_recipe.shuffle(seed=cfg.trainer.args.data_seed).select(
            range(min(len(val_recipe), cfg.recipe.max_eval_samples))
        )
        data_recipe_info["val_len_after"] = len(val_recipe)
    else:
        val_recipe_sub = None

    # --- Print the data recipe info
    recinfo2str(data_recipe_info, console)

    # --- Log to W&B
    if cfg.wandb.enabled:
        wandb.log(data_recipe_info)

    return train_recipe, val_recipe, val_recipe_sub, cfg


def summarise_config(cfg, console):
    console.rule("[bold]Training Config Summary", style="red")
    print_dict_as_table(
        OmegaConf.to_container(cfg.trainer.args, resolve=True), console, 8
    )
    print("\n")

    console.rule("[bold]PEFT Config Summary", style="red")
    peft_conf = cfg.trainer.peft_config if cfg.model_lib == "hf" else cfg.peft_config
    print_dict_as_table(OmegaConf.to_container(peft_conf, resolve=True), console, 8)
    print("\n")


def setup_exp(cfg, init_with_peft=True, model_only=False):

    # --- Setup Logging & Env Vars
    console = setup_logging()
    setup_env_vars(cfg)

    # --- W&B setup
    setup_wandb(cfg, console)

    # --- Summary of the configurations
    summarise_config(cfg, console)

    # --- Model and Tokeniser Loading
    model, tokenizer, cfg = init_model_tokenizer(cfg, console)
    if model_only:
        return console, tokenizer, model, cfg

    # --- Dataset Loading
    train_recipe, val_recipe, val_recipe_sub, cfg = init_data_recipes(
        console, cfg, tokenizer
    )

    # (Unsloth only) Now we have the max seq lenght in the peft config,
    # thus we can instantiate the model in peft
    if cfg.model_lib == "unsloth" and cfg.init_with_peft:
        console.rule("[bold]Init PEFT Model", style="red")
        model = instantiate(cfg.peft_config, model=model)
        print("\n")

    return console, tokenizer, model, train_recipe, val_recipe, val_recipe_sub, cfg
