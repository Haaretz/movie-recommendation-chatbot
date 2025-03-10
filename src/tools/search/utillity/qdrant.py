from qdrant_client import QdrantClient

from logger import logger


class QdrantClientManager:
    def __init__(self, config):
        self.config = config
        self.client_qdrant = QdrantClient(config)

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

    from config.load_config import load_config

    config = load_config("config/config.yaml")
