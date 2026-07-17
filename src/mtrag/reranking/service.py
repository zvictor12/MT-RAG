from collections.abc import Mapping, Sequence

from mtrag.interfaces import PairScorer
from mtrag.runtime.cache import SqliteCache, stable_key
from mtrag.schemas import SearchHit


def passage_text(hit: SearchHit) -> str:
    title = hit.title.strip() if hit.title else ""
    text = hit.text.strip() if hit.text else ""
    if not title and not text:
        raise ValueError(f"Candidate {hit.document_id} has no text")
    folded_title = title.casefold()
    folded_text = text.casefold()
    title_is_embedded = (
        folded_text == folded_title
        or folded_text.startswith(f"{folded_title}\n")
    )
    if not title or title_is_embedded:
        return text or title
    if not text:
        return title
    return f"{title}\n{text}"


class RerankService:
    def __init__(
        self,
        scorer: PairScorer,
        *,
        cache: SqliteCache | None = None,
        model_revision: str,
        max_length: int = 512,
        score_chunk_size: int = 256,
    ) -> None:
        if score_chunk_size <= 0:
            raise ValueError("score_chunk_size must be positive")
        self.scorer = scorer
        self.cache = cache
        self.model_revision = model_revision
        self.max_length = max_length
        self.score_chunk_size = score_chunk_size

    def rerank_many(
        self,
        queries: Mapping[str, str],
        candidates: Mapping[str, Sequence[SearchHit]],
        *,
        top_k: int,
    ) -> dict[str, list[SearchHit]]:
        entries: list[tuple[str, SearchHit, str, str]] = []
        missing: dict[str, tuple[str, str]] = {}
        scores: dict[str, float] = {}

        for task_id, hits in candidates.items():
            query = queries[task_id]
            for hit in hits:
                if not hit.has_passage:
                    continue
                passage = passage_text(hit)
                key = stable_key(
                    self.model_revision,
                    self.max_length,
                    query,
                    hit.document_id,
                    passage,
                )
                entries.append((task_id, hit, passage, key))
                cached = self.cache.get("reranker", key) if self.cache else None
                if cached is None:
                    missing.setdefault(key, (query, passage))
                else:
                    scores[key] = float(cached)

        missing_keys = list(missing)
        for start in range(0, len(missing_keys), self.score_chunk_size):
            keys = missing_keys[start : start + self.score_chunk_size]
            values = self.scorer.score([missing[key] for key in keys])
            for key, value in zip(keys, values, strict=True):
                scores[key] = value
            if self.cache:
                self.cache.put_many(
                    "reranker",
                    dict(zip(keys, values, strict=True)),
                )

        grouped: dict[str, list[tuple[float, SearchHit]]] = {}
        for task_id, hit, _passage, key in entries:
            grouped.setdefault(task_id, []).append((scores[key], hit))

        output = {task_id: [] for task_id in candidates}
        for task_id, scored_hits in grouped.items():
            scored_hits.sort(key=lambda item: (-item[0], item[1].document_id))
            output[task_id] = [
                SearchHit(
                    document_id=hit.document_id,
                    score=score,
                    rank=rank,
                    source=f"{hit.source}_reranked",
                    title=hit.title,
                    text=hit.text,
                    components={**hit.components, "pre_rerank_score": hit.score},
                )
                for rank, (score, hit) in enumerate(scored_hits[:top_k], start=1)
            ]
        return output
