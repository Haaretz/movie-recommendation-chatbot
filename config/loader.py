import yaml

from config.models import (
    AppConfig,
    ChatConfig,
    EmbeddingConfig,
    FieldsConfig,
    LLMConfig,
    QdrantConfig,
)


def load_yaml_file(path: str) -> dict:
    with open(path, "r") as file:
        return yaml.safe_load(file)


def load_config() -> AppConfig:
    return AppConfig(
        fields=FieldsConfig(**load_yaml_file("config/fields.yaml")),
        embedding=EmbeddingConfig(**load_yaml_file("config/embedding.yaml")),
        qdrant=QdrantConfig(**load_yaml_file("config/qdrant.yaml")),
        llm=LLMConfig(**load_yaml_file("config/llm.yaml")),
        chat=ChatConfig(**load_yaml_file("config/chat.yaml")),
        bucket_name="ask_haaretz",
    )


if __name__ == "__main__":
    config = load_config()

    print(config.llm.llm_model_name)
    print(config.qdrant.qdrant_url)
