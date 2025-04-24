from typing import Any, Dict, List, Union

from qdrant_client import models


class SearchArticleResults:
    def __init__(self, return_fields_names, logger):
        self.return_fields_names = return_fields_names
        self.logger = logger

    def _extract_and_translate_payload_from_points(
        self, points: List[Union[models.ScoredPoint, models.PointStruct]]
    ) -> List[Dict[str, Any]]:
        """Extract payload and process points (either ScoredPoint or PointStruct), handling duplicates."""
        documents = []
        for point in points:
            payload = point.payload
            # article = {key: payload[key] for key in payload if key in self.return_fields_names}
            if "publish_time" in payload:
                payload["publish_time"] = payload["publish_time"].split("T")[0]
            documents.append(payload)
        return documents
