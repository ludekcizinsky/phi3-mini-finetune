from typing import Dict, Literal
from xml.etree.ElementInclude import include

from datasets import Dataset
from pandas import DataFrame

from ..base_dataset import BaseDataset


class PreferenceData(BaseDataset):
    """The preference data."""

    ID = "preference"
    PATH = None  # The preference data has been preprocessed and saved to disk
    FilteringStrategy = Literal[
        "none",
        "keep_first",
        "global_threshold",
        "local_tolerance",
    ]

    def __init__(
        self,
        split: str,
        filtering_strategy: FilteringStrategy,
        **filtering_kwargs,
    ):
        self.save_args(
            filtering_strategy=filtering_strategy, filtering_kwargs=filtering_kwargs
        )
        super().__init__(split)

        # Save filtering information
        self.filter_info = dict()
        self.filter_info[f"filtering_strategy"] = filtering_strategy
        self.filter_info["filtering_kwargs"] = filtering_kwargs

    def filter_data(self, data: Dataset) -> Dataset:
        # No filtering
        if self.filtering_strategy == "none":
            return data

        # Transform to DataFrame (for filtering)
        df = data.to_pandas()

        # Add agreement column
        choice = df["overall"]
        df["agreement"] = (
            (df["correctness"] == choice).astype(int)
            + (df["relevance"] == choice).astype(int)
            + (df["clarity"] == choice).astype(int)
            + (df["completeness"] == choice).astype(int)
        )

        match self.filtering_strategy:
            case "keep_first":
                df = self._keep_first(df)
            case "global_threshold":
                df = self._global_threshold(df, **self.filtering_kwargs)
            case "local_tolerance":
                df = self._local_tolerance(df, **self.filtering_kwargs)
            case _:
                raise ValueError(f"Unknown strategy: {self.filtering_strategy}")

        # Transform back to Dataset
        data = Dataset.from_pandas(df)

        return data

    def prepare_data(self, data: Dataset) -> Dataset:
        """
        Prepare the data into the format required by e.g. the DPOTrainer.
        It formats the data into a dictionary with the following keys; `prompt`,
        `chosen`, and `rejected`. The `prompt` is the question, `chosen` is the
        chosen answer, and `rejected` is the rejected answer.

        Args:
            data (Dataset): The data to prepare.

        Returns:
            Dataset: The prepared data.
        """

        def format(example: Dict) -> Dict:
            chosen = example["A"] if example["overall"] == "A" else example["B"]
            rejected = example["B"] if example["overall"] == "A" else example["A"]

            return {
                "prompt": example["question_complete"],
                "chosen": chosen,
                "rejected": rejected,
            }

        return data.map(format, remove_columns=data.column_names)

    def _keep_first(self, df: DataFrame) -> DataFrame:
        """
        Keep the first row for each question.

        ```
        PreferenceData(
            split="train",
            filtering_strategy="keep_first"
        )
        ```

        Args:
            df (DataFrame): The data to filter.

        Returns:
            DataFrame: The filtered data.
        """
        return df.groupby("question_id").first()

    def _global_threshold(self, df: DataFrame, mode: str, threshold: int) -> DataFrame:
        """
        Filter the data based on a global threshold value. If mode is "least", the
        function will keep all rows where the agreement is at least the threshold.
        If mode is "most", the function will keep all rows where the agreement is
        at most the threshold.

        ```
        PreferenceData(
            split="train",
            filtering_strategy="global_threshold",
            filtering_kwargs=dict(mode="most", threshold=2)
        )
        ```

        Args:
            df (DataFrame): The data to filter.
            mode (str): The mode of filtering.
            threshold (int): The threshold value.

        Returns:
            DataFrame: The filtered data.
        """
        if mode == "least":
            return df[df["agreement"] >= threshold]
        elif mode == "most":
            return df[df["agreement"] <= threshold]
        else:
            raise ValueError(f"Unknown mode: {mode}")

    def _local_tolerance(self, df: DataFrame, mode: str, tolerance: int) -> DataFrame:
        """
        Filter the data based on a local tolerance value. If mode is "least",
        the function will keep all rows where the agreement is at least the
        maximum agreement found for a quest in the group minus the tolerance. If
        mode is "most", the function will keep all rows where the agreement is
        at most the minimum agreement found for a quest in the group plus the
        tolerance.

        ```
        PreferenceData(
            split="train",
            filtering_strategy="local_threshold",
            filtering_kwargs=dict(mode="most", tolerance=0)
        )
        ```

        Args:
            df (DataFrame): The data to filter.
            mode (str): The mode of filtering.
            tolerance (int): The tolerance value.

        Returns:
            DataFrame: The filtered data.
        """

        def _helper(group):
            max_agreement = group["agreement"].max()
            min_agreement = group["agreement"].min()
            if mode == "least":
                return group[group["agreement"] >= max_agreement - tolerance]
            elif mode == "most":
                return group[group["agreement"] <= min_agreement + tolerance]
            else:
                raise ValueError(f"Unknown mode: {mode}")

        return df.groupby("question_id").apply(_helper)
