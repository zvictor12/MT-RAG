from collections.abc import Sequence

from mtrag.retrieval.elasticsearch import (
    SOURCE_FIELDS,
    ElasticsearchGateway,
    index_name,
)
from mtrag.schemas import SearchHit, SearchQuery


class ElserRetriever:
    def __init__(self, gateway: ElasticsearchGateway) -> None:
        self.gateway = gateway

    def search_many(
        self,
        queries: Sequence[SearchQuery],
        *,
        top_k: int,
    ) -> dict[str, list[SearchHit]]:
        searches = [
            (
                index_name(query.domain, "elser"),
                {
                    "size": top_k,
                    "_source": list(SOURCE_FIELDS),
                    "query": {
                        "semantic": {
                            "field": "semantic_text",
                            "query": query.text,
                        }
                    },
                },
            )
            for query in queries
        ]
        responses = self.gateway.msearch(searches, source="elser")
        return {
            query.task_id: hits
            for query, hits in zip(queries, responses, strict=True)
        }
