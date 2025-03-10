import os

import yaml


def load_config(filepath):
    with open(filepath, "r") as file:
        config = yaml.safe_load(file)
        if "llm" in config and os.getenv("GOOGLE_API_KEY") is not None:
            config["llm"]["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")
        if "qdrant" in config and os.getenv("QDRANT_URL") is not None:
            config["qdrant"]["qdrant_url"] = os.getenv("QDRANT_URL")
        return config


if __name__ == "__main__":
    config_filepath = "config/config.yaml"
    config = load_config(config_filepath)
    print(config)
