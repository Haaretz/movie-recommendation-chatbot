from typing import Any, Dict, List

from qdrant_client import models

from logger import logger
from src.tools.search.search_article_filters import SearchArticleFilters
from src.tools.search.search_article_results import SearchArticleResults
from src.tools.search.utillity.embedding import Embedding
from tools.search.utillity.qdrant import QdrantClientManager


class SearchArticle(QdrantClientManager, Embedding, SearchArticleFilters, SearchArticleResults):

    SEARCH_LIMIT = 15
    HNSW_EF = 128
    SCROLL_LIMIT = 10

    def __init__(self, config: Dict[str, Any]):
        """Initialize SearchArticle class, inheriting from QdrantClientManager and Embedding."""
        self.config = config
        self.MIN_SCORE_THRESHOLD = config["qdrant"]["MIN_SCORE_THRESHOLD"]
        self.SEARCH_LIMIT = config["qdrant"]["SEARCH_LIMIT"]
        self.SCROLL_LIMIT = config["qdrant"]["SCROLL_LIMIT"]

        QdrantClientManager.__init__(self, config["qdrant"]["qdrant_url"])
        Embedding.__init__(
            self,
            config["embedding"]["embedding_model_name"],
            config["embedding"]["embedding_dimensionality"],
        )
        SearchArticleFilters.__init__(self)
        SearchArticleResults.__init__(self, config["return_fields"], logger)

        self.qdrant_collection_name = config["qdrant"].get("qdrant_collection_name")
        self.translation_mapping = config["return_fields"]

    def retrieve_documents_by_payload(
        self,
        brand: str,
        writer_name: str,
        publish_time_start: str,
        publish_time_end: str,
        primary_section: str,
        secondary_section: str,
        tags: str,
        article_type: str,
        url: List[str],
    ) -> str:
        """
        Retrieve documents from Qdrant based on payload filters only (no vector search).

        Utilizes the scroll API for efficient payload-based filtering.

        Args:
            brand: Newspaper brand to filter by.
            writer_name: Writer name to filter by.
            publish_time_start: Start date for publish time range filter.
            publish_time_end: End date for publish time range filter.
            primary_section: Primary section to filter by.
            secondary_section: Secondary section to filter by.
            tags: Tags to filter by.
            article_type: Article type to filter by.

        Returns:
            Formatted string of documents matching the payload filters. Returns NO_RESULT if no documents found.
        """
        logger.info(
            f"Retrieving documents by payload filter: brand='{brand}', writer_name='{writer_name}', publish_time_start='{publish_time_start}', publish_time_end='{publish_time_end}', primary_section='{primary_section}', secondary_section='{secondary_section}', tags='{tags}', article_type='{article_type}'"
        )

        qdrant_filter = self._create_qdrant_filter(
            brand,
            writer_name,
            publish_time_start,
            publish_time_end,
            primary_section,
            secondary_section,
            tags,
            article_type,
            url,
        )

        all_points = []
        next_page_offset = None
        while True:
            scroll_batch, next_page_offset = self.client_qdrant.scroll(
                collection_name=self.qdrant_collection_name,
                scroll_filter=qdrant_filter,
                limit=self.SCROLL_LIMIT,
                offset=next_page_offset,
            )
            if not scroll_batch:
                break
            all_points.extend(scroll_batch)
            if next_page_offset is None:
                break

        documents = self._extract_and_translate_payload_from_points(all_points)

        # logger.info(f"Payload-filtered documents:\n{documents}")

        return documents

    def retrieve_relevant_documents(
        self,
        query: str,
        brand: str,
        writer_name: str,
        publish_time_start: str,
        publish_time_end: str,
        primary_section: str,
        secondary_section: str,
        tags: str,
        article_type: str,
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
        logger.info(
            f"Retrieving relevant documents for query: '{query}', brand='{brand}', writer_name='{writer_name}', publish_time_start='{publish_time_start}',ֿ\
             publish_time_end='{publish_time_end}' primary_section='{primary_section}', secondary_section='{secondary_section}', tags='{tags}', article_type='{article_type}'"
        )

        query_embedding_vector = self.embed_query(query)
        qdrant_filter = self._create_qdrant_filter(
            brand,
            writer_name,
            publish_time_start,
            publish_time_end,
            primary_section,
            secondary_section,
            tags,
            article_type,
            url=[],
        )

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
    from config.load_config import load_config

    config = load_config("config/config.yaml")

    SA = SearchArticle(config)
    # result = SA.retrieve_documents_by_payload(
    #     url = ['00000194-273e-da14-adb7-777e7ed20000'],
    #     brand = None,
    #     writer_name =None,
    #     publish_time_start = None,
    #     publish_time_end = None,
    #     primary_section = None,
    #     secondary_section = None,
    #     tags = None,
    #     article_type = None,
    # )
    # print(result)

    result = SA.retrieve_relevant_documents(
        query="הכיבוש",
        brand=None,
        writer_name=None,
        publish_time_start=None,
        publish_time_end=None,
        primary_section=None,
        secondary_section=["מאמר מערכת"],
        # secondary_section=None,
        tags=None,
        article_type=None,
    )
    print(result)
