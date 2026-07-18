from collections.abc import Sequence
from typing import cast

from mtrag.retrieval.elasticsearch import (
    SOURCE_FIELDS,
    ElasticsearchGateway,
    index_name,
)
from mtrag.schemas import BgeFeatures, SearchHit, SearchQuery


class DenseRetriever:
    def __init__(
        self,
        gateway: ElasticsearchGateway,
        *,
        candidate_multiplier: int = 10,
        rescore_oversample: float = 2.0,
    ) -> None:
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
            features = cast(BgeFeatures, query.bge)
            searches.append(
                (
                    index_name(query.domain, "dense"),
                    {
                        "size": top_k,
                        "_source": list(SOURCE_FIELDS),
                        "query": {
                            "knn": {
                                "field": "embedding",
                                "query_vector": list(features.dense),
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
            features = cast(BgeFeatures, query.bge)
            vector = {
                key: value
                for key, value in features.sparse.items()
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
