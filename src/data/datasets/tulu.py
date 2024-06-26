from typing import List

from langdetect import detect

from datasets import Dataset

from ..base_dataset import BaseDataset
from .tulu_base import TuluDatasetIDs


class TuluData(BaseDataset):
    """The TuluData dataset.
    Prefer to use the specific parts of Tulu that extend TuluBaseDataset, which uses this class for instantiation of the Tulu dataset.
    """

    ID = "tulu"
    PATH = "allenai/tulu-v2-sft-mixture"

    def __init__(
        self,
        sub_datasets: List[TuluDatasetIDs] = None,
        filter_out_english=False,
        *args,
        **kwargs
    ):
        self.save_args(sub_datasets=sub_datasets, filter_out_english=filter_out_english)
        super().__init__(*args, **kwargs)

    def filter_data(self, data: Dataset) -> Dataset:
        # Filter out hard coded examples
        data = data.filter(lambda x: x["dataset"] != "hardcoded")

        # Filter to only include the specified sub-datasets
        if self.sub_datasets is not None:
            sub_dataset_strings = [
                sub_dataset.value for sub_dataset in self.sub_datasets
            ]
            data = data.filter(lambda x: x["dataset"] in sub_dataset_strings)

        # Filter out non-english examples
        def _is_english(example):
            example = self._format_data(example)
            text = example["text"]
            try:
                return detect(text) == "en"
            except:
                return False

        if self.filter_out_english:
            data = data.filter(_is_english)

        return data

    def prepare_data(self, data: Dataset) -> Dataset:
        # Format the data to be a conversation between user and assistant
        data = data.map(self._format_data, remove_columns=data.column_names)

        return data

    def _format_data(self, example: dict) -> dict:
        messages = example["messages"]
        text = ""
        for message in messages:
            text += message["role"].upper() + ": "
            text += message["content"] + "\n"
        return {"text": text}
