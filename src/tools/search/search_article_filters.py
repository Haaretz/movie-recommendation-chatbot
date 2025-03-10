from typing import List

from qdrant_client import models

from constant import BRAND_MAPPING


class SearchArticleFilters:
    def __init__(self):
        pass

    def _create_qdrant_filter(
        self,
        brand: str,
        writer_name: str,
        publish_time_start: str,
        publish_time_end: str,
        primary_section: str,
        secondary_section: str,
        tags: str,
        article_type: str,
        url: list[str],
    ) -> models.Filter:
        """
        Create a Qdrant filter object based on provided filter parameters.

        Handles mapping of brand names and constructs filter conditions for various fields.

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
            Qdrant Filter object, or None if no filter conditions are added.
        """
        filter_conditions = []

        self._add_brand_filter_condition(brand, filter_conditions)
        self._add_embedding_type_filter_condition(filter_conditions)
        self._add_writer_name_filter_condition(writer_name, filter_conditions)
        self._add_publish_time_filter_condition(publish_time_start, publish_time_end, filter_conditions)
        self._add_primary_section_filter_condition(primary_section, filter_conditions)
        self._add_secondary_section_filter_condition(secondary_section, filter_conditions)
        self._add_article_type_filter_condition(article_type, filter_conditions)
        self._add_tags_filter_condition(tags, filter_conditions)
        self._add_url_filter_condition(url, filter_conditions)

        return models.Filter(must=filter_conditions) if filter_conditions else None

    def _add_url_filter_condition(self, url: list[str], filter_conditions: List[models.FieldCondition]):
        """Adds url filter condition if url is provided and not ALL."""
        if len(url) > 0:
            url = [path.split("/")[-1] for path in url]
            filter_conditions.append(models.FieldCondition(key="article_id", match=models.MatchAny(any=url)))

    def _add_brand_filter_condition(self, brand: str, filter_conditions: List[models.FieldCondition]):
        """Adds brand filter condition if brand is provided and not ALL."""
        if brand:
            filter_conditions.append(
                models.FieldCondition(key="brand", match=models.MatchValue(value=BRAND_MAPPING.get(brand, brand)))
            )

    def _add_embedding_type_filter_condition(self, filter_conditions: List[models.FieldCondition]):
        """Adds embedding type filter condition (currently fixed to 'full_article')."""
        embedding_type = "full_article"  # Consider making this configurable if needed
        if embedding_type != "all":
            filter_conditions.append(
                models.FieldCondition(key="embedding_type", match=models.MatchValue(value=embedding_type))
            )

    def _add_writer_name_filter_condition(self, writer_name: str, filter_conditions: List[models.FieldCondition]):
        """Adds writer name filter condition if writer_name is provided and not ALL."""
        if writer_name:
            filter_conditions.append(models.FieldCondition(key="writer_name", match=models.MatchAny(any=writer_name)))

    def _add_publish_time_filter_condition(
        self, publish_time_start: str, publish_time_end: str, filter_conditions: List[models.FieldCondition]
    ):
        """Adds publish time range filter condition if both start and end times are provided."""
        if publish_time_start and publish_time_end:
            filter_conditions.append(
                models.FieldCondition(
                    key="publish_time",
                    range=models.DatetimeRange(gte=publish_time_start, lte=publish_time_end),
                )
            )

    def _add_primary_section_filter_condition(
        self, primary_section: str, filter_conditions: List[models.FieldCondition]
    ):
        """Adds primary section filter condition if primary_section is provided and not ALL."""
        if primary_section:
            filter_conditions.append(
                models.FieldCondition(key="section_primary", match=models.MatchAny(any=primary_section))
            )

    def _add_secondary_section_filter_condition(
        self, secondary_section: str, filter_conditions: List[models.FieldCondition]
    ):
        """Adds secondary section filter condition if secondary_section is provided and not ALL."""
        if secondary_section:
            filter_conditions.append(
                models.FieldCondition(key="section_secondary", match=models.MatchAny(any=secondary_section))
            )

    def _add_article_type_filter_condition(self, article_type: str, filter_conditions: List[models.FieldCondition]):
        """Adds article type filter condition if article_type is provided and not ALL."""
        if article_type:
            filter_conditions.append(
                models.FieldCondition(key="article_type", match=models.models.MatchValue(value=article_type))
            )

    def _add_tags_filter_condition(self, tags: str, filter_conditions: List[models.FieldCondition]):
        """Adds tags filter condition if tags is provided and not ALL."""
        if tags:
            filter_conditions.append(models.FieldCondition(key="tags", match=models.MatchAny(any=tags)))
