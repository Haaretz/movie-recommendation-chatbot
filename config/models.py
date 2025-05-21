import os
from dataclasses import dataclass
from typing import List, Literal, Set

from google.genai.types import Content
from pydantic import BaseModel, Field


@dataclass
class ChatContext:
    history: List[Content]
    seen: Set[str]
    message: str
    user_id: str
    remaining_user_messages: int


class ChatConfig(BaseModel):
    max_user_messages_per_session: int = 10
    warn_template: str
    warn_last_message: str
    blocked_message: str
    long_request: str
    token_limit: int
    chat_ttl_seconds: int
    non_paying_messages: str


class FieldsConfig(BaseModel):
    fields_for_frontend: List[str]
    fields_for_llm: List[str]


class EmbeddingConfig(BaseModel):
    embedding_model_name: str
    embedding_dimensionality: int


class QdrantConfig(BaseModel):
    MIN_SCORE_THRESHOLD: float
    SEARCH_LIMIT: int
    qdrant_url: str = Field(default_factory=lambda: os.getenv("QDRANT_URL", ""))
    qdrant_collection_name: str
    embedding_metric: Literal["cosine", "dot", "euclidean"]


class LLMConfig(BaseModel):
    GOOGLE_API_KEY: str = Field(default_factory=lambda: os.getenv("GOOGLE_API_KEY", ""))
    llm_model_name: str


class AppConfig(BaseModel):
    fields: FieldsConfig
    embedding: EmbeddingConfig
    qdrant: QdrantConfig
    llm: LLMConfig
    bucket_name: str
    chat: ChatConfig
