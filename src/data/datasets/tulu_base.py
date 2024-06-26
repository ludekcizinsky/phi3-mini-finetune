import os
from abc import ABC
from enum import Enum

from langdetect import detect

from datasets import Dataset

from ..base_dataset import BaseDataset


class TuluDatasetIDs(Enum):
    FLAN_V2 = "flan_v2"
    FLAN_V2_COT = "cot"
    OASST1 = "oasst1"
    LIMA = "lima"
    GPT4_ALPACA = "gpt4_alpaca"
    CODE_ALPACA = "code_alpaca"
    SHAREGPT = "sharegpt"
    WIZARDLM = "wizardlm"
    OPEN_ORCA = "open_orca"
    SCIENCE_EVIDENCE_INFERENCE = "science.evidence_inference"
    SCIENCE_QASPER_TRUNCATED_4000 = "science.qasper_truncated_4000"
    SCIENCE_SCIFACT_JSON = "science.scifact_json"
    SCIENCE_SCITLDR_AIC = "science.scitldr_aic"
    SCIENCE_SCIERC_NER = "science.scierc_ner"
    SCIENCE_SCIERC_RELATION = "science.scierc_relation"


class TuluBaseDataset(BaseDataset, ABC):
    """A base dataset for all datasets that are in Tulu. Should not be instantiated itself."""

    ID = "tulu"
    SPLITS = ["train"]
    PATH = "allenai/tulu-v2-sft-mixture"
    TULU_DATASET_ID: TuluDatasetIDs = None
    DATASET_SIZE = None
    ENGLISH_PERCENTAGE = None

    def __init__(self, filter_out_english=False, *args, **kwargs):
        self.save_args(filter_out_english=filter_out_english)
        super().__init__(*args, **kwargs)

    def prepare_data(self, data: Dataset) -> Dataset:

        # Filter to only include the specific Tulu dataset
        data = data.filter(lambda x: x["dataset"] == self.TULU_DATASET_ID.value)

        # Format the data to be a conversation between user and assistant
        data = data.map(self._format_data, remove_columns=data.column_names)

        return data

    def _format_data(self, example):
        messages = example["messages"]
        text = ""
        for message in messages:
            text += message["role"].upper() + ": "
            text += message["content"] + "\n"
        return {"text": text}

    def _assert_class_args(self):
        assert (
            self.TULU_DATASET_ID is not None
        ), "TULU_DATASET_ID must be set in the subclass."
        return super()._assert_class_args()

    def filter_data(self, data: Dataset) -> Dataset:
        """
        Filters the data based on arguments. Can be overridden by child classes.

        Args:
            data (Dataset): The dataset to filter.

        Returns:
            Dataset: The filtered dataset.
        """

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
