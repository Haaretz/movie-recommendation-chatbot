from typing import Any, Dict, List, Union

from qdrant_client import models

from logger import logger
from src.tools.search.utillity.embedding import Embedding
from src.tools.search.utillity.qdrant import QdrantClientManager


class SearchArticle(QdrantClientManager, Embedding):

    HNSW_EF = 128
    SCROLL_LIMIT = 10

    def __init__(self, config: Dict[str, Any]):
        """Initialize SearchArticle class, inheriting from QdrantClientManager and Embedding."""
        self.config = config
        self.MIN_SCORE_THRESHOLD = config["qdrant"]["MIN_SCORE_THRESHOLD"]
        self.SEARCH_LIMIT = config["qdrant"]["SEARCH_LIMIT"]

        QdrantClientManager.__init__(self, config["qdrant"]["qdrant_url"])
        Embedding.__init__(
            self,
            config["embedding"]["embedding_model_name"],
            config["embedding"]["embedding_dimensionality"],
        )

        self.qdrant_collection_name = config["qdrant"].get("qdrant_collection_name")

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
    ) -> models.Filter:
        must_conditions = []

        if len(streaming) > 0:
            must_conditions.append(
                models.FieldCondition(key="distribution_platform", match=models.MatchAny(any=streaming))
            )
        if len(genres) > 0:
            must_conditions.append(models.FieldCondition(key="genre", match=models.MatchAny(any=genres)))
        if review_type:
            replacements = {"movie": "Movie", "series": "Series"}
            review_type = replacements.get(review_type, review_type)

            must_conditions.append(models.FieldCondition(key="review_type", match=models.MatchValue(value=review_type)))

        must_conditions.append(models.FieldCondition(key="create_by_AI", match=models.MatchValue(value=False)))

        qdrant_filter = models.Filter(must=must_conditions)

        return qdrant_filter

    def retrieve_relevant_documents(self, query: str, streaming: list[str], genres: list[str], review_type: str) -> str:
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
        qdrant_filter = self._create_qdrant_filter(streaming, genres, review_type)

        search_params = models.SearchParams(hnsw_ef=self.HNSW_EF, exact=False)

        try:
            search_result = self.client_qdrant.search(
                collection_name=self.qdrant_collection_name,
                query_vector=query_embedding_vector,
                limit=self.SEARCH_LIMIT,
                # score_threshold=self.MIN_SCORE_THRESHOLD,
                score_threshold=0.6,
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
    from config.load_config import load_config

    config = load_config("config/config.yaml")

    SA = SearchArticle(config)

    result = SA.retrieve_relevant_documents(
        query="הסרט החדש של לוקה גואדנינו שמבוסס על ספר של בורוז על הומו שחי במקסיקו בשנות ה-50 ומחפש גברים צעירים",
        streaming=[],
        genres=[],
        review_type="movie",
    )
    print(result)
