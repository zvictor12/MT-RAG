import gc
from collections.abc import Sequence
from numbers import Real
from pathlib import Path

from mtrag.interfaces import BatchGuard, NoopGuard


class BgeV2M3Scorer:
    """Thin GPU adapter around the BGE v2 M3 cross-encoder."""

    def __init__(
        self,
        model_path: Path,
        *,
        device: str = "cuda:0",
        batch_size: int = 8,
        max_length: int = 512,
        guard: BatchGuard | None = None,
        guard_chunk_size: int = 256,
    ) -> None:
        self.model_path = model_path
        self.device = device
        self.batch_size = batch_size
        self.max_length = max_length
        self.guard = guard or NoopGuard()
        self.guard_chunk_size = guard_chunk_size
        self._model = None

    def _load(self):
        if self._model is not None:
            return self._model

        from FlagEmbedding import FlagReranker
        from FlagEmbedding.inference.reranker.encoder_only import (
            base as reranker_module,
        )

        reranker_module.tqdm = lambda iterable, *args, **kwargs: iterable
        reranker_module.trange = lambda *args, **kwargs: range(*args)

        self._model = FlagReranker(
            str(self.model_path),
            devices=self.device,
            use_fp16=self.device.startswith("cuda"),
            batch_size=self.batch_size,
            max_length=self.max_length,
            normalize=False,
        )
        return self._model

    def score(self, pairs: Sequence[tuple[str, str]]) -> list[float]:
        if not pairs:
            return []
        model = self._load()
        scores: list[float] = []
        for start in range(0, len(pairs), self.guard_chunk_size):
            self.guard.wait("gpu")
            chunk = list(pairs[start : start + self.guard_chunk_size])
            chunk_scores = model.compute_score(
                chunk,
                batch_size=self.batch_size,
                max_length=self.max_length,
                normalize=False,
            )
            if isinstance(chunk_scores, Real):
                chunk_scores = [float(chunk_scores)]
            scores.extend(float(score) for score in chunk_scores)
        return scores

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

    def __enter__(self) -> "BgeV2M3Scorer":
        self._load()
        return self

    def __exit__(self, *_args) -> None:
        self.close()
