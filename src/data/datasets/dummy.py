from typing import Optional

from datasets import Dataset

from ..base_dataset import BaseDataset


class DummyData(BaseDataset):
    """A dummy dataset to show the behaviour of the BaseDataset class."""

    ID = "dummy"
    PATH = "lhoestq/demo1"

    def __init__(self, split: str, keep: Optional[int] = None):
        self.save_args(keep=keep)
        super().__init__(split)

    def prepare_data(self, data: Dataset) -> Dataset:
        return data

    def filter_data(self, data: Dataset) -> Dataset:
        if self.keep is None:
            return data
        return data.select(range(self.keep))
