import os

import vertexai
from vertexai.language_models import TextEmbeddingInput, TextEmbeddingModel

from config.loader import load_config


class Embedding:
    def __init__(self, EmbeddingConfig):
        # Initialize VertexAI with default project or from env
        vertexai.init(project=os.environ.get("VERTEXAI_PROJECT", "htz-data"))

        # Load the embedding model by name
        model_name = EmbeddingConfig.embedding_model_name
        embedding_dimensionality = EmbeddingConfig.embedding_dimensionality

        self.model = TextEmbeddingModel.from_pretrained(model_name)
        self.embedding_dimensionality = embedding_dimensionality

    def embed_query(self, text):
        kwargs = dict(output_dimensionality=self.embedding_dimensionality)
        try:
            if isinstance(text, list):
                inputs = [TextEmbeddingInput(t, "RETRIEVAL_QUERY") for t in text]
                embedding_response = self.model.get_embeddings(inputs, **kwargs)
                return [resp.values for resp in embedding_response]
            else:
                inputs = [TextEmbeddingInput(text, "RETRIEVAL_QUERY")]
                embedding_response = self.model.get_embeddings(inputs, **kwargs)[0]
                return embedding_response.values
        except Exception as e:
            error_message = f"error in embedding: {e}"
            raise RuntimeError(error_message) from e


if __name__ == "__main__":
    # Load full application configuration
    app_config = load_config()

    # Extract the relevant section for embedding
    embedding_config = app_config.embedding

    # Instantiate the embedding class using values from config
    embedding = Embedding(embedding_config)

    # Example single input
    text = "This is a test sentence."
    embeddings = embedding.embed_query(text)
    print(embeddings)
