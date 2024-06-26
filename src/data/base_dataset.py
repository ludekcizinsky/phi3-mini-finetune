import os
from abc import ABC, abstractmethod

from datasets import Dataset, load_dataset, load_from_disk

from ..paths import DATA_DIR
from typing import Optional


class BaseDataset(Dataset, ABC):
    """
    An abstract class that defines the structure of a dataset. It is based
    on the Hugging Face Dataset class and uses the load_dataset function to
    load the data into a local folder. The class provides two abstract methods
    that need to be implemented by the child classes:
    - prepare_data: A method that prepares the data for further processing.
    - filter_data: A method that filters the data based on some criteria.

    Every dataset instance used in our project should inherit from this class.
    """

    ID = None
    PATH: str = None
    DATA_FILES: Optional[str] = None  # For local datasets
    NAME: Optional[str] = None  # For data subsets

    def __init__(self, split: str = "train", **kwargs):
        """
        Initialises the dataset by loading the data from the given path and split.
        Then, it prepares and filters the data based on the child class implementation.

        Args:
            path (str): The path to the dataset.
            split (str, optional): The split of the dataset to load. Defaults to "train".

        Returns:
            None
        """

        # Assert the path is defined
        self._assert_class_args()

        # Try to load from disk, else from HF
        data_dir = os.path.join(DATA_DIR, self.ID)
        if self.NAME:
            name = self.NAME.name.lower()
            data_dir = os.path.join(data_dir, name)
        if os.path.exists(data_dir):
            collection = load_from_disk(data_dir)
        else:
            name = self.NAME.value if self.NAME else None
            collection = load_dataset(self.PATH, data_files=self.DATA_FILES, name=name)
            collection.save_to_disk(data_dir)

        # Get split
        dataset = collection[split]

        # Initialise as HF Dataset
        super().__init__(arrow_table=self.prepare_data(self.filter_data(dataset)).data)

    def filter_data(self, data: Dataset) -> Dataset:
        """
        Filters the data based on arguments. Can be overridden by child classes.

        Args:
            data (Dataset): The dataset to filter.

        Returns:
            Dataset: The filtered dataset.
        """
        return data

    @abstractmethod
    def prepare_data(self, data: Dataset) -> Dataset:
        pass

    def save_args(self, **kwargs):
        """
        Saves the arguments passed to the class as attributes.

        Args:
            **kwargs: Arbitrary keyword arguments.

        Returns:
            None
        """
        for key, value in kwargs.items():
            setattr(self, key, value)

    def _assert_class_args(self):
        """
        Makes assertions about the PATH attribute of the child class.
        """
        assert (
            self.ID is not None
        ), "The ID attribute must be defined in the child class."

    def write_data(
        self, path: str = "./datasets/", custom_filename=None, custom_split=None
    ):
        """
        Writes the dataset to the given path as a .jsonl file

        Args:
            path (str): The path to write the dataset to.
            custom_filename (str): A custom filename for the dataset with extension. If None, the ID is used.
            split (str): The split of the dataset to write. If None, the split that defined the dataset is used.

        Returns:
            None
        """

        # Create path
        if custom_filename is None:
            filename = self.ID
        else:
            filename = custom_filename
        filename += ".jsonl"
        path = os.path.join(path, filename)

        # Write JSON
        self.to_json(path_or_buf=path)
