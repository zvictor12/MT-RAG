import json
from collections.abc import Mapping
from pathlib import Path

from mtrag.schemas import BgeFeatures


class BgeFeatureStore:
    """Compact dense NPZ plus inspectable sparse JSONL, keyed by query case ID."""

    def __init__(self, directory: Path) -> None:
        self.directory = directory
        self.dense_path = directory / "dense.npz"
        self.sparse_path = directory / "sparse.jsonl"

    def save(self, features: Mapping[str, BgeFeatures]) -> None:
        import numpy as np

        self.directory.mkdir(parents=True, exist_ok=True)
        keys = list(features)
        dense = np.asarray([features[key].dense for key in keys], dtype=np.float32)

        dense_tmp = self.directory / ".dense.npz.tmp"
        sparse_tmp = self.directory / ".sparse.jsonl.tmp"
        with dense_tmp.open("wb") as handle:
            np.savez_compressed(handle, keys=np.asarray(keys), dense=dense)
        with sparse_tmp.open("w", encoding="utf-8") as handle:
            for key in keys:
                handle.write(
                    json.dumps(
                        {"key": key, "sparse": features[key].sparse},
                        ensure_ascii=False,
                        separators=(",", ":"),
                    )
                )
                handle.write("\n")

        dense_tmp.replace(self.dense_path)
        sparse_tmp.replace(self.sparse_path)

    def load(self) -> dict[str, BgeFeatures]:
        import numpy as np

        with np.load(self.dense_path, allow_pickle=False) as archive:
            keys = [str(key) for key in archive["keys"]]
            dense = archive["dense"]

        sparse = {}
        with self.sparse_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                item = json.loads(line)
                sparse[item["key"]] = {
                    str(key): float(value)
                    for key, value in item["sparse"].items()
                }

        if set(keys) != set(sparse):
            raise RuntimeError("Dense and sparse BGE feature keys do not match")
        return {
            key: BgeFeatures(
                dense=tuple(float(value) for value in vector),
                sparse=sparse[key],
            )
            for key, vector in zip(keys, dense, strict=True)
        }
