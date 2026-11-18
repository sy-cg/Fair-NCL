import importlib.util
import pickle
import sys
import types
import unittest
from pathlib import Path

import pandas as pd

_tqdm_stub = types.ModuleType("tqdm")
_tqdm_stub.tqdm = lambda value, **kwargs: value
sys.modules.setdefault("tqdm", _tqdm_stub)

from data.unified_preprocessing import UnifiedSequentialPreprocessor


def _load_semantic_similarity_module():
    repo_root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "semantic_similarity_under_test",
        repo_root / "methods" / "semantic_similarity.py",
    )
    module = importlib.util.module_from_spec(spec)
    torch_stub = types.ModuleType("torch")
    torch_stub.Tensor = object
    previous_torch = sys.modules.get("torch")
    sys.modules["torch"] = torch_stub
    try:
        spec.loader.exec_module(module)
    finally:
        if previous_torch is None:
            sys.modules.pop("torch", None)
        else:
            sys.modules["torch"] = previous_torch
    return module


class ProcessedDataPortabilityTests(unittest.TestCase):
    def test_pickle_payload_stores_metadata_tables_as_records(self):
        users = pd.DataFrame(
            {
                "raw_user_id": [10, 11],
                "gender_raw": pd.Series(["F", "M"], dtype="string"),
                "gender": [1, 0],
                "age_group": [0, 1],
                "user_id": [0, 1],
            }
        )
        items = pd.DataFrame(
            {
                "raw_item_id": [101, 102],
                "movie_id": [1, 2],
                "item_text": pd.Series(["Alpha", "Beta"], dtype="string"),
            }
        )
        data = {
            "dataset": "ml-1m",
            "train_data": [],
            "val_data": [],
            "test_data": [],
            "sequences": {},
            "users": users,
            "movies": items,
            "items": items,
            "user_map": {10: 0, 11: 1},
            "item_map": {101: 1, 102: 2},
            "item_id_start": 1,
            "num_users": 2,
            "num_items": 2,
            "metadata": {},
            "build_config": {},
        }

        payload = UnifiedSequentialPreprocessor._make_pickle_payload(data)

        self.assertIsInstance(payload["users"], list)
        self.assertIsInstance(payload["items"], list)
        self.assertIsInstance(payload["movies"], list)
        self.assertEqual(payload["users"][0]["gender_raw"], "F")
        self.assertEqual(payload["items"][0]["item_text"], "Alpha")
        pickle.dumps(payload)

    def test_semantic_text_extraction_accepts_record_metadata(self):
        semantic_similarity = _load_semantic_similarity_module()
        item_metadata = [
            {"movie_id": 1, "item_text": "Alpha"},
            {"movie_id": 2, "title": "Beta", "genres": "Drama"},
            {"movie_id": 3, "item_text": "Out of range"},
        ]

        item_ids, texts = semantic_similarity._extract_item_texts(
            item_metadata,
            num_items=2,
        )

        self.assertEqual(item_ids, [1, 2])
        self.assertEqual(texts, ["Alpha", "title: Beta. genres: Drama"])


if __name__ == "__main__":
    unittest.main()
