from qdrant_client import models


class SearchArticleFilters:
    def __init__(self):
        pass

    def _create_qdrant_filter(
        self,
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
            must_conditions.append(models.FieldCondition(key="review_type", match=models.MatchValue(value=review_type)))

        must_conditions.append(models.FieldCondition(key="create_by_AI", match=models.MatchValue(value=False)))

        qdrant_filter = models.Filter(must=must_conditions)

        return qdrant_filter
