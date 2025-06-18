import json
from typing import List

from google import genai
from google.genai.types import Content


class Tokenizer:
    """
    Tiny wrapper around Gemini's model tokenizer.
    Computes number of tokens used for full turns (history + user + model).
    """

    def __init__(self, model_name: str):
        self.client = genai.Client()
        self.model = model_name

    def _count(self, text: str) -> int:
        return self.client.models.count_tokens(
            model=self.model,
            contents=text,
        ).total_tokens

    def count_tokens(
        self,
        history: List[Content],
        new_user_msg: str,
        assistant_reply: str,
    ) -> tuple[int, int]:
        """
        Count (input_tokens, output_tokens) used in a full interaction:
        - input: history + new user message + function call responses
        - output: assistant reply text only
        """
        total_in = 0
        for content in history:
            if content.role in {"user", "model"}:
                part = content.parts[0]
                if part.text:
                    total_in += self._count(part.text)
                elif part.function_response:
                    total_in += self._count(json.dumps(part.function_response.response))

        total_in += self._count(new_user_msg)
        total_out = self._count(assistant_reply)

        return total_in, total_out
