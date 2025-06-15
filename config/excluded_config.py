import json
from typing import Set


class ExcludedIdLoader:
    def __init__(self, filepath: str = "config/excluded_ids.jsonc"):
        self._excluded_ids: Set[str] = set()
        self._filepath = filepath
        self._load_ids()

    def _load_ids(self):
        try:
            with open(self._filepath, "r", encoding="utf-8") as f:
                clean_lines = [line for line in f if not line.strip().startswith("//")]
                json_content = "".join(clean_lines)
                self._excluded_ids = set(json.loads(json_content))
            print(f"ExcludedIdLoader: Loaded {len(self._excluded_ids)} IDs from {self._filepath}")
        except FileNotFoundError:
            print(f"ExcludedIdLoader: Warning: {self._filepath} not found.")
        except Exception as e:
            print(f"ExcludedIdLoader: Error loading {self._filepath}: {e}.")

    def get_excluded_ids(self) -> Set[str]:
        return self._excluded_ids


if __name__ == "__main__":
    # Load the excluded IDs when this module is run directly
    loader = ExcludedIdLoader()
    excluded_ids = loader.get_excluded_ids()
    print(f"Excluded IDs: {excluded_ids}")
