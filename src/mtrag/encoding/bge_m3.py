import gc
from collections.abc import Sequence
from pathlib import Path

from mtrag.interfaces import BatchGuard, NoopGuard
from mtrag.schemas import BgeFeatures


class BGEM3QueryEncoder:
    """Encode queries once into compatible dense and sparse BGE-M3 features."""

    def __init__(
        self,
        model_path: Path,
        *,
        device: str = "cuda:0",
        batch_size: int = 32,
        max_length: int = 512,
        guard_chunk_size: int = 256,
        guard: BatchGuard | None = None,
    ) -> None:
        self.model_path = model_path
        self.device = device
        self.batch_size = batch_size
        self.max_length = max_length
        self.guard_chunk_size = guard_chunk_size
        self.guard = guard or NoopGuard()
        self._model = None

    def _load(self):
        if self._model is not None:
            return self._model

        from FlagEmbedding import BGEM3FlagModel
        from FlagEmbedding.inference.embedder.encoder_only import m3 as m3_module

        # FlagEmbedding creates two nested tqdm bars on every encode call.
        # The experiment emits one durable checkpoint progress line instead.
        m3_module.tqdm = lambda iterable, *args, **kwargs: iterable
        m3_module.trange = lambda *args, **kwargs: range(*args)

        self._model = BGEM3FlagModel(
            str(self.model_path),
            devices=self.device,
            use_fp16=self.device.startswith("cuda"),
            normalize_embeddings=True,
            batch_size=self.batch_size,
            query_max_length=self.max_length,
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=False,
        )
        return self._model

    def encode(self, texts: Sequence[str]) -> list[BgeFeatures]:
        if not texts:
            return []
        model = self._load()
        features: list[BgeFeatures] = []

        for start in range(0, len(texts), self.guard_chunk_size):
            self.guard.wait("gpu")
            batch = list(texts[start : start + self.guard_chunk_size])
            output = model.encode_queries(
                batch,
                batch_size=self.batch_size,
                max_length=self.max_length,
                return_dense=True,
                return_sparse=True,
                return_colbert_vecs=False,
            )
            for dense, sparse in zip(
                output["dense_vecs"],
                output["lexical_weights"],
                strict=True,
            ):
                features.append(
                    BgeFeatures(
                        dense=tuple(float(value) for value in dense),
                        sparse={str(key): float(value) for key, value in sparse.items()},
                    )
                )

        return features

    def close(self) -> None:
        if self._model is None:
            return

        try:
            self._model.stop_self_pool()
        finally:
            self._model = None
            gc.collect()
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass

    def __enter__(self) -> "BGEM3QueryEncoder":
        self._load()
        return self

    def __exit__(self, *_args) -> None:
        self.close()
