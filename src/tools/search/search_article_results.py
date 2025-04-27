from typing import Any, Dict, List, Union

from qdrant_client import models


class SearchArticleResults:
    def __init__(self, logger):
        self.logger = logger

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
