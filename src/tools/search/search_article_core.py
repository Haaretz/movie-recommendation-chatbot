from typing import Any, Dict, List, Union

from qdrant_client import models

from config.models import EmbeddingConfig, QdrantConfig
from logger import logger
from src.tools.search.utillity.embedding import Embedding
from src.tools.search.utillity.qdrant import QdrantClientManager


class SearchArticle(QdrantClientManager, Embedding):

    HNSW_EF = 128
    SCROLL_LIMIT = 10

    def __init__(self, qdrant_config: QdrantConfig, embedding_config: EmbeddingConfig):
        """Initialize SearchArticle with structured configs."""
        self.MIN_SCORE_THRESHOLD = qdrant_config.MIN_SCORE_THRESHOLD
        self.SEARCH_LIMIT = qdrant_config.SEARCH_LIMIT
        self.qdrant_collection_name = qdrant_config.qdrant_collection_name

        # Initialize Qdrant client and Embedding
        QdrantClientManager.__init__(self, qdrant_config)
        Embedding.__init__(self, embedding_config)

    def _extract_and_translate_payload_from_points(
        self, points: List[Union[models.ScoredPoint, models.PointStruct]]
    ) -> List[Dict[str, Any]]:
        """Extract payload and process points (either ScoredPoint or PointStruct), handling duplicates."""
        documents = []
        for point in points:
            payload = point.payload
            if "publish_time" in payload:
                payload["publish_time"] = payload["publish_time"].split("T")[0]
            documents.append(payload)
        return documents

    @staticmethod
    def _create_qdrant_filter(
        streaming: list[str],
        genres: list[str],
        review_type: str,
        seen_ids: set[str],
    ) -> models.Filter:
        must_conditions = []
        must_not_conditions = []

        # --- Step 1: Create must conditions based on input parameters ---

        if streaming:
            must_conditions.append(
                models.FieldCondition(key="distribution_platform", match=models.MatchAny(any=streaming))
            )
        if genres:
            must_conditions.append(models.FieldCondition(key="genre", match=models.MatchAny(any=genres)))
        if review_type:
            replacements = {"movie": "Movie", "series": "Series"}
            review_type = replacements.get(review_type, review_type)

            must_conditions.append(models.FieldCondition(key="review_type", match=models.MatchValue(value=review_type)))

        must_conditions.append(models.FieldCondition(key="create_by_AI", match=models.MatchValue(value=False)))

        # --- Step 2: Create must_not conditions based on seen_ids ---

        if seen_ids:
            must_not_conditions.append(
                models.FieldCondition(
                    key="id",  # or "payload.article_id" if that's where your ID lives
                    match=models.MatchAny(any=list(seen_ids)),
                )
            )

        qdrant_filter = models.Filter(must=must_conditions)

        return qdrant_filter

    def retrieve_relevant_documents(
        self, query: str, streaming: list[str], genres: list[str], review_type: str, seen_ids: set[str]
    ) -> str:
        """
        Retrieve relevant documents from Qdrant using vector search and payload filters.

        Embeds the query, creates a Qdrant filter, performs vector search, and processes results.

        Args:
            query: Search query string.
            brand: Newspaper brand to filter by.
            writer_name: Writer name to filter by.
            publish_time_start: Start date for publish time range filter.
            publish_time_end: End date for publish time range filter.
            primary_section: Primary section to filter by.
            secondary_section: Secondary section to filter by.
            tags: Tags to filter by.
            article_type: Article type to filter by.

        Returns:
            Formatted string of relevant documents. Returns NO_RESULT if no relevant documents found.
        """

        query_embedding_vector = self.embed_query(query)
        qdrant_filter = self._create_qdrant_filter(streaming, genres, review_type, seen_ids)

        search_params = models.SearchParams(hnsw_ef=self.HNSW_EF, exact=False)

        try:
            search_result = self.client_qdrant.search(
                collection_name=self.qdrant_collection_name,
                query_vector=query_embedding_vector,
                limit=self.SEARCH_LIMIT,
                score_threshold=self.MIN_SCORE_THRESHOLD,
                query_filter=qdrant_filter,
                search_params=search_params,
            )
        except Exception as e:
            logger.error(f"Error in search: {e}")
            logger.exception(e)
            return "qdrant search error"
        # search_result = self._filter_search_results_by_score(search_result)
        documents = self._extract_and_translate_payload_from_points(search_result)

        return documents


if __name__ == "__main__":
    from config.loader import load_config

    app_config = load_config()

    searcher = SearchArticle(
        qdrant_config=app_config.qdrant,
        embedding_config=app_config.embedding,
    )

    result = searcher.retrieve_relevant_documents(
        query="הסרט החדש של לוקה גואדנינו שמבוסס על ספר של בורוז על הומו שחי במקסיקו בשנות ה-50 ומחפש גברים צעירים",
        streaming=[],
        genres=[],
        review_type="movie",
    )
    print(result)
