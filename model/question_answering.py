# https://huggingface.co/transformers/task_summary.html#extractive-question-answering

from transformers import AutoTokenizer, AutoModelForQuestionAnswering, pipeline
from model.result import Result
import torch

# question answering model
model = "bert-large-uncased-whole-word-masking-finetuned-squad"


class QuestionAnswering:
    def __init__(self):
        self.qa = pipeline("question-answering")

    def answer(self, question, context):
        result = self.qa(question=question, context=context)
        # print(f"QA: '{result['answer']}', score: {round(result['score'], 3)}")
        return Result(result['score'], result['answer'])