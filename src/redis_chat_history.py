import json
import os
from typing import List, Optional

import redis
from google.genai.types import Content, FunctionCall, Part

from logger import logger


class RedisChatHistory:
    """Encapsulates all Redis persistence logic, now preserving function_call parts."""

    def __init__(self, redis_url: Optional[str] = None):
        redis_url = redis_url or os.getenv("REDIS_URL")
        if not redis_url:
            raise EnvironmentError("REDIS_URL environment variable is not set and no URL was provided.")

        self._client: redis.Redis = redis.Redis.from_url(redis_url, decode_responses=True)
        self._client.ping()
        logger.info("Successfully connected to Redis.")

    def load_history(self, user_id: str) -> List[Content]:
        """Return a list of `Content` objects for `user_id` (may be empty)."""
        key = f"chat_history:{user_id}"
        json_history = self._client.lrange(key, 0, -1)
        if not json_history:
            logger.debug("No history found for user %s in Redis.", user_id)
            return []

        logger.debug("Loaded history for user %s from Redis.", user_id)
        return [self._deserialize_message(raw) for raw in json_history]

    def save_message(self, user_id: str, message: Content) -> None:
        """Append a single message to Redis chat history list."""
        key = f"chat_history:{user_id}"
        serialized = self._serialize_message(message)
        self._client.rpush(key, serialized)

    def _serialize_message(self, message: Content) -> str:
        """
        Serialize a Content to JSON, handling both text and function_call parts.
        """
        parts_list = []
        for part in message.parts:
            entry: dict = {}
            # Only include text if it's not None or empty
            if part.text:
                entry["text"] = part.text
            # If this part invoked or returned a function call, include it
            if getattr(part, "function_response", None):
                fc: FunctionCall = part.function_response
                entry["function_response"] = {
                    "name": fc.name,
                    "response": fc.response,
                }
            parts_list.append(entry)

        return json.dumps(
            {
                "role": message.role,
                "parts": parts_list,
            },
            ensure_ascii=False,
        )

    def _deserialize_message(self, raw: str) -> Content:
        """
        Reconstruct a Content from the JSON we saved, restoring both text
        and function_call parts.
        """
        data = json.loads(raw)
        parts: List[Part] = []

        for entry in data.get("parts", []):
            text = entry.get("text", "")
            fc_data = entry.get("function_response")
            if fc_data:
                fc = Part.from_function_response(
                    name=fc_data["name"],
                    response=fc_data["response"],
                )
                parts.append(fc)
            else:
                parts.append(Part(text=text))

        return Content(role=data["role"], parts=parts)

    def pop_last_conversation(self, user_id: str) -> None:
        """
        Removes all consecutive 'model' messages from the end of the history,
        followed by the last 'user' message if present.
        """
        key = f"chat_history:{user_id}"
        while True:
            raw = self._client.lindex(key, -1)
            if not raw:
                break
            data = json.loads(raw)
            if data.get("role") == "model":
                self._client.rpop(key)
            else:
                break

        # Now check if last is a user message and remove it too
        raw = self._client.lindex(key, -1)
        if raw:
            data = json.loads(raw)
            if data.get("role") == "user":
                self._client.rpop(key)
                parts = data.get("parts", [])
                if parts:
                    user_text = parts[0].get("text")

        return user_text

    def increment_usage_counter(self, user_id: str) -> int:
        """
        Increments the user's message usage counter and returns the new count.
        """
        key = f"user_message_count:{user_id}"
        return self._client.incr(key)

    def get_usage_count(self, user_id: str) -> int:
        key = f"user_message_count:{user_id}"
        value = self._client.get(key)
        return int(value) if value else 0

    def reset_usage_counter(self, user_id: str) -> None:
        key = f"user_message_count:{user_id}"
        self._client.delete(key)
