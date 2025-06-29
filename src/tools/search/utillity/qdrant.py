from qdrant_client import QdrantClient

from config.loader import load_config
from config.models import QdrantConfig
from logger import get_logger

logger = get_logger(__name__)


class QdrantClientManager:
    def __init__(self, qdrant_config: QdrantConfig):
        # Use unpacking to initialize the client directly
        self.client_qdrant = QdrantClient(
            url=qdrant_config.qdrant_url,
        )

    def close(self):
        """
        Closes the Qdrant client connection.
        """
        if self.client_qdrant:
            self.client_qdrant.close()
            logger.info("Qdrant client connection closed.")
        else:
            logger.warning("Qdrant client was not initialized, no need to close.")


if __name__ == "__main__":
    # Load full config (from all YAMLs + env)
    app_config = load_config()

    # Pass only the qdrant config to the manager
    qdrant_manager = QdrantClientManager(app_config.qdrant)
