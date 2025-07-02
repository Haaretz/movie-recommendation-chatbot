import datetime
from typing import Any, Dict, List, Set, Union

from qdrant_client import models

from config.models import ChatConfig, EmbeddingConfig, QdrantConfig
from logger import get_logger
from src.tools.search.utillity.embedding import Embedding
from src.tools.search.utillity.qdrant import QdrantClientManager

logger = get_logger(__name__)


class SearchArticle(QdrantClientManager, Embedding):

    HNSW_EF = 128
    SCROLL_LIMIT = 10
    _num_results_retrieved = 5

    def __init__(
        self,
        qdrant_config: QdrantConfig,
        embedding_config: EmbeddingConfig,
        chat_config: ChatConfig,
        excluded_ids: Set[str],
    ):
        """Initialize SearchArticle with structured configs."""
        self.MIN_SCORE_THRESHOLD = qdrant_config.MIN_SCORE_THRESHOLD
        self.SEARCH_LIMIT = qdrant_config.SEARCH_LIMIT
        self.qdrant_collection_name = qdrant_config.qdrant_collection_name
        self.days_until_not_current_in_theaters = chat_config.days_until_not_current_in_theaters
        self.excluded_ids = excluded_ids

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
            if payload["movie"] and datetime.datetime.now() - datetime.datetime.strptime(
                payload["publish_time"], "%Y-%m-%dT%H:%M:%SZ"
            ) > datetime.timedelta(days=self.days_until_not_current_in_theaters):
                if payload.get("distribution_platform") and "בתי קולנוע" in payload["distribution_platform"]:
                    payload["distribution_platform"].remove("בתי קולנוע")

            distribution = payload.get("distribution_platform")
            if isinstance(distribution, list) and "Amazon Prime Video" in distribution:
                # change from 'Amazon Prime Video' to 'Prime Video'
                payload["distribution_platform"] = [
                    platform.replace("Amazon Prime Video", "Amazon")
                    for platform in payload.get("distribution_platform", [])
                ]
            documents.append(payload)
        return documents

    def _create_qdrant_filter(
        self,
        streaming: List[str],
        genres: List[str],
        review_type: str,
        writer_filter: List[str],
        seen_ids: Set[str],
    ) -> models.Filter:
        """
        Build a Qdrant filter with OR-logic between three alternative blocks:
        1. create_by_AI == False  AND  stars > 3
        2. section_tertiary_id == '0000017d-d7fd-d8a6'  AND  tone ∈ {'positive', 'mixed'}
        3. writer_name == 'ניב הדס'  AND  tone == 'positive'
        Caller-supplied filters (streaming, genres, review_type, seen_ids) are mandatory.
        """

        # ---------- mandatory filters (always applied) ----------
        must_conditions: List[models.Condition] = []

        if streaming:
            must_conditions.append(
                models.FieldCondition(
                    key="distribution_platform",
                    match=models.MatchAny(any=streaming),
                )
            )

        if genres:
            must_conditions.append(
                models.FieldCondition(
                    key="genre",
                    match=models.MatchAny(any=genres),
                )
            )

        if review_type:
            replacements = {"movie": "Movie", "series": "Series"}
            must_conditions.append(
                models.FieldCondition(
                    key="review_type",
                    match=models.MatchValue(value=replacements.get(review_type, review_type)),
                )
            )
        if writer_filter:
            must_conditions.append(
                models.FieldCondition(
                    key="writer_name",
                    match=models.MatchAny(any=writer_filter),
                )
            )

        # ---------- optional NOT-conditions ----------
        must_not_conditions: List[models.Condition] = []
        if seen_ids:
            must_not_conditions.append(
                models.FieldCondition(
                    key="article_id",
                    match=models.MatchAny(any=set(seen_ids).union(self.excluded_ids)),
                )
            )
        else:
            must_not_conditions.append(
                models.FieldCondition(
                    key="article_id",
                    match=models.MatchAny(any=list(self.excluded_ids)),
                )
            )

        # ---------- alternative block 1 ----------
        block_a = models.Filter(
            must=[
                models.FieldCondition(key="create_by_AI", match=models.MatchValue(value=False)),
                models.FieldCondition(key="stars", range=models.Range(gt=3)),
            ]
        )

        # ---------- alternative block 2 ----------
        block_b = models.Filter(
            must=[
                models.FieldCondition(
                    key="section_tertiary_id",
                    match=models.MatchValue(value="0000017d-d7fd-d8a6-af7f-dfff3f470000"),
                ),
                models.FieldCondition(
                    key="tone",
                    match=models.MatchAny(any=["positive", "mixed"]),
                ),
            ]
        )

        # ---------- alternative block 3 ----------
        block_c = models.Filter(
            must=[
                models.FieldCondition(
                    key="writer_name",
                    match=models.MatchValue(value="ניב הדס"),
                ),
                models.FieldCondition(
                    key="tone",
                    match=models.MatchValue(value="positive"),
                ),
            ]
        )

        # ---------- combine everything ----------
        return models.Filter(
            must=must_conditions,
            must_not=must_not_conditions,
            should=[block_a, block_b, block_c],  # OR across the three blocks
        )

    def _order_search_results_by_date(
        self, search_result: List[Union[models.ScoredPoint, models.PointStruct]]
    ) -> List[Union[models.ScoredPoint, models.PointStruct]]:
        """
        Order search results by publish_time in descending order and remove duplicates by name.
        For series with multiple entries (e.g. different seasons), keeps only the most recent one.

        Args:
            search_result: List of search results (either ScoredPoint or PointStruct).

        Returns:
            Ordered list of search results without duplicates by name.
        """
        # Sort the results based on the publish_time field in descending order
        sorted_results = sorted(
            search_result,
            key=lambda x: x.payload.get("publish_time", ""),
            reverse=True,
        )

        # Remove duplicates by name, keeping the most recent (first in sorted list) for same series in different seasons
        seen_names = set()
        unique_results = []

        for result in sorted_results:
            name = result.payload.get("name", "")
            if name not in seen_names:
                seen_names.add(name)
                unique_results.append(result)

        return unique_results[: self.SEARCH_LIMIT]

    def retrieve_relevant_documents(
        self,
        query: str,
        streaming: list[str],
        genres: list[str],
        review_type: str,
        writer_filter: list[str],
        seen_ids: set[str],
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
        qdrant_filter = self._create_qdrant_filter(streaming, genres, review_type, writer_filter, seen_ids)

        search_params = models.SearchParams(hnsw_ef=self.HNSW_EF, exact=False)
        try:
            search_result = self.client_qdrant.search(
                collection_name=self.qdrant_collection_name,
                query_vector=query_embedding_vector,
                limit=self._num_results_retrieved,
                score_threshold=self.MIN_SCORE_THRESHOLD,
                query_filter=qdrant_filter,
                search_params=search_params,
            )
            search_result = self._order_search_results_by_date(search_result)
        except Exception as e:
            logger.error(f"Error in search: {e}")
            logger.exception(e)
            return "qdrant search error"
        # search_result = self._filter_search_results_by_score(search_result)
        documents = self._extract_and_translate_payload_from_points(search_result)
        logger.info(f"ids seen: {seen_ids}")
        recevied = [d["article_id"] for d in documents]
        logger.info(f"search result: {recevied}")
        logger.info(f"does seen in received: {seen_ids.intersection(recevied)}")

        return documents


if __name__ == "__main__":
    from config.loader import load_config

    app_config = load_config()

    searcher = SearchArticle(
        qdrant_config=app_config.qdrant,
        embedding_config=app_config.embedding,
        chat_config=app_config.chat,
        excluded_ids=set(),
    )

    result = searcher.retrieve_relevant_documents(
        query="הסרט החדש של לוקה גואדנינו שמבוסס על ספר של בורוז על הומו שחי במקסיקו בשנות ה-50 ומחפש גברים צעירים",
        streaming=[],
        genres=[],
        review_type="movie",
        seen_ids=set(),
        writer_filter=[],
    )
    print(result)
