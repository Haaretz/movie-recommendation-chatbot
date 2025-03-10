import os

import vertexai
from vertexai.language_models import TextEmbeddingInput, TextEmbeddingModel


class Embedding:
    def __init__(self, model: TextEmbeddingModel, embedding_dimensionality: int):
        vertexai.init(project=os.environ.get("VERTEXAI_PROJECT", "htz-data"))
        self.model = TextEmbeddingModel.from_pretrained(model)
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
    from config.load_config import load_config

    config = load_config("config/config.yaml")
    embedding_model_name = config["embedding"]["embedding_model_name"]
    embedding_dimensionality = config["embedding"]["embedding_dimensionality"]
    embedding = Embedding(embedding_model_name, embedding_dimensionality)
    text = "This is a test sentence."
    embeddings = embedding.embed_article(text)
    print(embeddings)
    text = ["This is a test sentence."] * 3
    embeddings = embedding.embed_article(text)
    print(embeddings)
