from typing import Optional

from google import genai
from google.genai.types import Content, Part


class LLMChatSession:
    """
    Wrapper around a single Gemini chat session.

    This class hides the details of the Google GenAI SDK and exposes
    a minimal interface to send messages and receive streamed chunks.
    """

    def __init__(
        self,
        client: genai.Client,
        model: str,
        system_instruction: str,
        history: Optional[list[Content]] = None,
        tools: Optional[list] = None,
    ):
        self.chat = client.chats.create(
            model=model,
            history=history,
            config=genai.types.GenerateContentConfig(
                system_instruction=system_instruction,
                tools=tools or [],
            ),
        )

    def stream(self, input: str | list[Part]):
        """
        Unified interface: input can be string (user message) or list of Part (function call).
        """
        return self.chat.send_message_stream(input)
