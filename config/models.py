import os
from typing import List, Literal

from pydantic import BaseModel, Field


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
