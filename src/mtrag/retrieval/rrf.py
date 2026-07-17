from collections.abc import Mapping, Sequence

from mtrag.schemas import SearchHit


def rrf_fuse(
    runs: Mapping[str, Sequence[SearchHit]],
    *,
    rank_constant: int = 60,
    top_k: int = 20,
) -> list[SearchHit]:
    """Fuse ranked lists using only ranks, never incomparable raw scores."""
    fused_scores: dict[str, float] = {}
    representatives: dict[str, SearchHit] = {}
    components: dict[str, dict[str, float]] = {}

    for source, hits in sorted(runs.items()):
        for hit in hits:
            fused_scores[hit.document_id] = fused_scores.get(hit.document_id, 0.0) + (
                1.0 / (rank_constant + hit.rank)
            )
            representatives.setdefault(hit.document_id, hit)
            components.setdefault(hit.document_id, {})[f"{source}_rank"] = float(
                hit.rank
            )
            components[hit.document_id][f"{source}_score"] = hit.score

    ordered_ids = sorted(
        fused_scores,
        key=lambda document_id: (-fused_scores[document_id], document_id),
    )[:top_k]

    results = []
    for rank, document_id in enumerate(ordered_ids, start=1):
        hit = representatives[document_id]
        results.append(
            SearchHit(
                document_id=document_id,
                score=fused_scores[document_id],
                rank=rank,
                source="rrf",
                title=hit.title,
                text=hit.text,
                components=components[document_id],
            )
        )
    return results
