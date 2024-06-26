from typing import Dict

from datasets import Dataset

from ..base_dataset import BaseDataset
from datasets import concatenate_datasets


class HardCodedData(BaseDataset):
    """The preference data."""

    ID = "hardcoded"
    PATH = None  # The hardcoded data has been preprocessed and saved to disk

    def __init__(self, *args, **kwargs):
        self.duplicate = kwargs.get("duplicate", 1)
        super().__init__(*args, **kwargs)

    def filter_data(self, data: Dataset) -> Dataset:
        return data

    def prepare_data(self, data: Dataset) -> Dataset:
        def _format_data(example: Dict) -> Dict:
            text = f"Question: {example['question']}\nAnswer: {example['answer']}"
            return {"text": text}

        data = data.map(_format_data, remove_columns=data.column_names)

        # Duplicate each sample `self.duplicate` times
        data = concatenate_datasets([data] * self.duplicate)

        return data
