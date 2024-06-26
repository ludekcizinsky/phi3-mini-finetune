from os import remove
from typing import Dict

from datasets import Dataset

from ..base_dataset import BaseDataset


class OrcaMathData(BaseDataset):
    """The OrcaMath dataset.

    200k math word problems from the ORCA dataset.

    HF - https://huggingface.co/datasets/microsoft/orca-math-word-problems-200k

    Paper - https://arxiv.org/pdf/2402.14830
    """

    ID = "orcamath"
    PATH = "microsoft/orca-math-word-problems-200k"

    def __init__(self, filter_on_length=False, *args, **kwargs):
        self.save_args(filter_on_length=filter_on_length)
        super().__init__(*args, **kwargs)

    def prepare_data(self, data: Dataset) -> Dataset:
        def _format_data(example: Dict) -> Dict:
            text = f"Question: {example['question']}\nAnswer: {example['answer']}"
            return {"text": text}

        if self.filter_on_length:
            data = data.filter(lambda x: len(x["answer"]) >= 1000)

        data = data.map(_format_data, remove_columns=data.column_names)

        return data

    def filter_data(self, data: Dataset) -> Dataset:
        return data
