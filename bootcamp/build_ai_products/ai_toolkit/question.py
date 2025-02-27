from typing import List

from pydantic import BaseModel, Field


class QuizQuestion(BaseModel):
    question: str = Field(description="The question text")
    correct_answer: str = Field(description="The correct answer to the question")
    incorrect_answers: List[str] = Field(description="List of incorrect answers")
