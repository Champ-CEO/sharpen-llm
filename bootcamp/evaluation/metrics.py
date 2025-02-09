import json
from typing import Any

from opik.evaluation.metrics import base_metric, score_result


def _remove_thinking_from_response(response: str) -> str:
    close_tag = "</think>"
    tag_length = len(close_tag)
    return response[response.find(close_tag) + tag_length :].strip()


class AccuracyMetric(base_metric.BaseMetric):
    def __init__(self, name: str, field: str):
        self.name = name
        self.field = field

    def score(self, expected_output: str, output: str, **ignored_kwargs: Any):
        output = _remove_thinking_from_response(output)
        text = output.replace("```json", "").replace("```", "")
        response = json.loads(text)

        return score_result.ScoreResult(
            value=expected_output[self.field] == response[self.field],
            name=self.name,
        )
