from typing import List, Literal, Optional

import numpy as np
from datasets import Dataset, DatasetInfo, concatenate_datasets

RatioModeType = Literal["absolute", "relative"]


class DataRecipe(Dataset):
    def __init__(
        self,
        datasets: List[Dataset],
        ratio: List[float],
        ratio_mode: RatioModeType = "absolute",
        max_size: Optional[int] = None,
        **kwargs,
    ):

        # Subset the datasets
        if ratio_mode == "absolute":
            datasets = self._absolute_ratio(datasets, ratio, max_size)
        elif ratio_mode == "relative":
            datasets = self._relative_ratio(datasets, ratio, max_size)
        else:
            raise ValueError(f"Unknown ratio mode: {ratio_mode}")

        # Parse the kwargs
        # self.num_workers = kwargs.get("num_workers", 1)

        # Concatenate the datasets
        # self.tokenizer = tokenizer
        data_recipe = concatenate_datasets(datasets)
        # data_tokenized, self.max_counts = self.tokenize_dataset(data_recipe)
        # print(self.max_counts)

        # Initialise as HF dataset
        super().__init__(
            arrow_table=data_recipe._data,
            info=DatasetInfo(
                description=f"Concatenated {len(datasets)} datasets with ratio {ratio} and ratio mode {ratio_mode}."
            ),
        )

    def _absolute_ratio(
        self, datasets: List[Dataset], ratio: List[float], max_size: int
    ) -> List[Dataset]:
        """
        Selects a subset of each dataset individually. Can either be specified
        by an absolute number of samples or by a relative number of samples.

        Example 1:
            ratio = [0.2, 0.3, 0.3]
            datasets = [d1, d2, d3]
            20% of d1, 30% of d2, and 30% of d3 are selected.

        Example 2:
            ratio = [100, 200, 300]
            datasets = [d1, d2, d3]
            100 samples from d1, 200 samples from d2, and 300 samples from d3 are selected.

        Args:
            datasets (List[Dataset]): A list of datasets.
            ratio (List[float]): A list of ratios.

        Returns:
            A list of datasets.
        """
        # Convert the ratios to absolute numbers, if necessary
        if all([0 < r <= 1 for r in ratio]):
            n_samples = [int(r * len(d)) for r, d in zip(ratio, datasets)]
        else:
            n_samples = [int(r) for r in ratio]

        # Scale the number of samples to the maximum data
        scale = min(1, max_size / sum(n_samples)) if max_size is not None else 1
        n_samples = [int(n * scale) for n in n_samples]

        # Fill up the remaining samples
        remaining = max_size - sum(n_samples) if max_size is not None else 0
        for i in range(remaining):
            n_samples[i % len(n_samples)] += 1

        # Sample from each dataset
        datasets = [
            d.select(range(min(n, len(d)))) for d, n in zip(datasets, n_samples)
        ]

        return datasets

    def tokenize_and_count_tokens(self, batch):
        """
        Tokenize the text and count the number of tokens for each entry in the batch.

        Args:
            batch (dict): A batch from the dataset.
            tokenizer (PreTrainedTokenizer): The tokenizer to use.
            column_names (List[str]): List of text column names to process.

        Returns:
            dict: A dictionary with tokenized data and token counts.
        """
        tokenized_batch = {}
        token_counts = {col: [] for col in self.columns_names}
        token_counts = {}

        for col in self.columns_names:
            # Tokenize the text
            tokenized_text = self.tokenizer(
                batch[col], truncation=True, padding=True, return_tensors="pt"
            )
            tokenized_batch[col] = tokenized_text["input_ids"]

            # Count tokens
            token_counts[f"{col}_count"] = [
                len(ids) for ids in tokenized_text["input_ids"]
            ]

        return {**tokenized_batch, **token_counts}

    def tokenize_dataset(self, dataset):
        """
        Tokenize and count tokens in the dataset.

        Args:
            dataset (Dataset): The dataset to process.
            tokenizer (PreTrainedTokenizer): The tokenizer to use.
            column_names (List[str]): List of text column names to tokenize.

        Returns:
            Dataset: The tokenized dataset.
            dict: Maximum token counts for each column.
        """

        # Parse column names
        self.columns_names = dataset.data.column_names

        # Apply the tokenizer and count tokens
        tokenized_data = dataset.map(
            lambda batch: self.tokenize_and_count_tokens(batch),
            batched=True,
            num_proc=self.num_workers,
            remove_columns=self.columns_names,  # Remove original text columns
        )

        # Calculate maximum token counts
        max_token_counts = {
            col: max(tokenized_data[f"{col}_count"]) for col in self.columns_names
        }

        return tokenized_data, max_token_counts
