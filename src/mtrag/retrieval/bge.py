from collections.abc import Sequence

from mtrag.retrieval.elasticsearch import (
    SOURCE_FIELDS,
    ElasticsearchGateway,
    index_name,
)
from mtrag.schemas import SearchHit, SearchQuery


class DenseRetriever:
    def __init__(
        self,
        gateway: ElasticsearchGateway,
        *,
        candidate_multiplier: int = 10,
        rescore_oversample: float = 2.0,
    ) -> None:
        if candidate_multiplier <= 0:
            raise ValueError("candidate_multiplier must be positive")
        if rescore_oversample < 1.0:
            raise ValueError("rescore_oversample must be at least 1.0")
        self.gateway = gateway
        self.candidate_multiplier = candidate_multiplier
        self.rescore_oversample = rescore_oversample

    def search_many(
        self,
        queries: Sequence[SearchQuery],
        *,
        top_k: int,
    ) -> dict[str, list[SearchHit]]:
        searches = []
        for query in queries:
            if query.bge is None:
                raise ValueError(f"BGE features are missing for {query.task_id}")
            searches.append(
                (
                    index_name(query.domain, "dense"),
                    {
                        "size": top_k,
                        "_source": list(SOURCE_FIELDS),
                        "query": {
                            "knn": {
                                "field": "embedding",
                                "query_vector": list(query.bge.dense),
                                "k": top_k,
                                "num_candidates": max(
                                    100,
                                    top_k * self.candidate_multiplier,
                                ),
                                "rescore_vector": {
                                    "oversample": self.rescore_oversample,
                                },
                            }
                        },
                    },
                )
            )

        responses = self.gateway.msearch(searches, source="bge_dense")
        return {
            query.task_id: hits
            for query, hits in zip(queries, responses, strict=True)
        }


class SparseRetriever:
    def __init__(self, gateway: ElasticsearchGateway) -> None:
        self.gateway = gateway

    def search_many(
        self,
        queries: Sequence[SearchQuery],
        *,
        top_k: int,
    ) -> dict[str, list[SearchHit]]:
        searches = []
        for query in queries:
            if query.bge is None:
                raise ValueError(f"BGE features are missing for {query.task_id}")
            vector = {
                key: value
                for key, value in query.bge.sparse.items()
                if value > 0
            }
            searches.append(
                (
                    index_name(query.domain, "sparse"),
                    {
                        "size": top_k,
                        "_source": list(SOURCE_FIELDS),
                        "query": {
                            "sparse_vector": {
                                "field": "embedding",
                                "query_vector": vector,
                                "prune": False,
                            }
                        },
                    },
                )
            )

        responses = self.gateway.msearch(searches, source="bge_sparse")
        return {
            query.task_id: hits
            for query, hits in zip(queries, responses, strict=True)
        }
