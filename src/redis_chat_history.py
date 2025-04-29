import json
import os
from typing import List, Optional

import redis
from google.genai.types import Content, Part

from logger import logger


class RedisChatHistory:
    """Encapsulates all Redis persistence logic (composition instead of inheritance).

    Parameters
    ----------
    redis_url : str | None
        A redis connection string (e.g. "redis://localhost:6379/0"). If *None*, the
        URL is read from the ``REDIS_URL`` environment variable.
    """

    def __init__(self, redis_url: Optional[str] = None):
        redis_url = redis_url or os.getenv("REDIS_URL")
        if not redis_url:
            raise EnvironmentError("REDIS_URL environment variable is not set and no URL was provided.")

        # One connection per process is usually enough; Redis manages socket pool.
        self._client: redis.Redis = redis.Redis.from_url(redis_url, decode_responses=True)
        self._client.ping()
        logger.info("Successfully connected to Redis.")

    # ------------- public API -------------

    def load_history(self, user_id: str) -> List[Content]:
        """Return a list of ``Content`` objects for ``user_id`` (may be empty)."""
        key = f"chat_history:{user_id}"
        json_history_list = self._client.lrange(key, 0, -1)
        if not json_history_list:
            logger.debug("No history found for user %s in Redis.", user_id)
            return []

        logger.debug("Loaded history for user %s from Redis.", user_id)
        return [self._deserialize_message(m) for m in json_history_list]

    def save_message(self, user_id: str, message: Content) -> None:
        """Append a single message to Redis chat history list."""
        key = f"chat_history:{user_id}"
        self._client.rpush(key, self._serialize_message(message))

    # ------------- internal helpers -------------

    def _serialize_message(self, message: Content) -> str:
        """Serialize ``Content`` to JSON. Store only role+plain‑text parts."""
        return json.dumps(
            {
                "role": message.role,
                "parts": [part.text for part in message.parts],
            }
        )

    def _deserialize_message(self, raw: str) -> Content:
        data = json.loads(raw)
        parts = [Part(text=p) for p in data.get("parts", [])]
        return Content(role=data["role"], parts=parts)
