import os
import random
from typing import Dict

from datasets import Dataset

from ..base_dataset import BaseDataset


class SciQData(BaseDataset):
    """The SciQ Dataset.

    The SciQ dataset contains 13,679 crowdsourced science exam questions about Physics, Chemistry and Biology, among others. The questions are in multiple-choice format with 4 answer options each. For the majority of the questions, an additional paragraph with supporting evidence for the correct answer is provided.

    HF - https://huggingface.co/datasets/allenai/sciq

    Paper - https://www.semanticscholar.org/paper/Crowdsourcing-Multiple-Choice-Science-Questions-Welbl-Liu/932a5de79d8a8ebb75ea0c43493450fd9922e738
    """

    ID = "sciq"
    PATH = "allenai/sciq"

    def __init__(self, *args, **kwargs):
        self.include_explanation = kwargs.get("include_explanation", True)
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
        explanation = example["support"]
        correct = example["correct_answer"]
        answers = [
            example["distractor1"],
            example["distractor2"],
            example["distractor3"],
        ]

        # Add correct answer in at random position
        random.seed(42)
        position = random.randint(0, 3)
        answers.insert(position, correct)

        # Get correct answer
        correct_letter = chr(65 + position)
        correct_text = f"{correct_letter}. {correct}"

        # Create question with options
        answer_text = ""
        for i, answer in enumerate(answers):
            answer_text += f"{chr(65 + i)}. {answer}\n"

        # Add messages that will be used in ChatML formatter
        messages = []

        # - User question
        question_with_options = f"{question}\n Input: {answer_text}"
        user = {"content": question_with_options, "role": "user"}
        messages.append(user)

        # - Assistant response
        response = (
            f"{explanation}\nTherefore the correct answer is {correct_text}"
            if self.include_explanation
            else f"The correct answer is {correct_text}"
        )
        assistant = {"content": f"{response}", "role": "assistant"}
        messages.append(assistant)
        output_dict["text"] = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=False, tokenize=False
        )

        # If we want to save the data to the cs552 format
        if use_write_format:
            output_dict["question"] = question_with_options
            output_dict["answer"] = correct_letter

        return output_dict
