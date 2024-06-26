from abc import ABC
from enum import Enum
from datasets import Dataset
from typing import Dict

from ..base_dataset import BaseDataset


class ArcDatasetName(Enum):
    Easy = "ARC-Easy"
    Challenge = "ARC-Challenge"


class ArcBaseData(BaseDataset, ABC):
    """A base dataset for all datasets that are in ARC. Should not be instantiated itself."""

    ID = "arc"
    PATH = "allenai/ai2_arc"
    NAME: ArcDatasetName = None

    def __init__(self, *args, **kwargs):
        self.tokenizer = kwargs.get("tokenizer", None)
        super().__init__(*args, **kwargs)

    def prepare_data(self, data: Dataset) -> Dataset:
        self.data_to_write = data.map(
            lambda ex: self._format_data(ex, use_write_format=True),
            remove_columns=data.column_names,
        )
        data = data.map(self._format_data, remove_columns=data.column_names)

        return data

    def _format_data(self, example: Dict, use_write_format=False) -> Dict:
        output_dict = {}

        # Parse the data
        question = example["question"]
        answer_letters = example["choices"]["label"]
        answer_options = example["choices"]["text"]
        correct_answer_letter = example["answerKey"]

        # Create question with options
        enumerated_answer_options = ""
        for letter, option in zip(answer_letters, answer_options):
            enumerated_answer_options += f"{letter}. {option}\n"

        # Add messages that will be used in ChatML formatter
        messages = []

        # - User question
        question_with_options = f"{question}\n Input:{enumerated_answer_options}"
        user = {"content": question_with_options, "role": "user"}
        messages.append(user)

        # - Assistant response
        correct_answer_option = answer_options[
            answer_letters.index(correct_answer_letter)
        ]
        correct_answer = f"{correct_answer_letter}. {correct_answer_option}"
        response = f"The correct answer is {correct_answer}"

        assistant = {"content": f"{response}", "role": "assistant"}
        messages.append(assistant)
        output_dict["text"] = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=False, tokenize=False
        )

        # If we want to save the data to the cs552 format
        if use_write_format:
            output_dict["question"] = question_with_options
            output_dict["answer"] = correct_answer_letter

        return output_dict
