import json
from collections.abc import Sequence
from typing import Any

import requests
from requests.adapters import HTTPAdapter
from urllib3.util import Retry

from mtrag.schemas import SearchHit


SOURCE_FIELDS = ("doc_id", "title", "text", "url", "row_idx")


def index_name(domain: str, method: str) -> str:
    if method == "elser":
        return f"mtrag-{domain}-elser"
    return f"mtrag-{domain}-bge-m3-{method}"


class ElasticsearchGateway:
    def __init__(self, url: str, *, request_batch_size: int = 64) -> None:
        self.url = url.rstrip("/")
        self.request_batch_size = request_batch_size
        self.session = requests.Session()
        self.session.mount(
            "http://",
            HTTPAdapter(
                max_retries=Retry(
                    total=3,
                    connect=3,
                    read=0,
                    status=3,
                    backoff_factor=1,
                    allowed_methods={"GET", "POST"},
                    status_forcelist=(429, 502, 503, 504),
                )
            ),
        )

    def msearch(
        self,
        searches: Sequence[tuple[str, dict[str, Any]]],
        *,
        source: str,
    ) -> list[list[SearchHit]]:
        results: list[list[SearchHit]] = []
        for start in range(0, len(searches), self.request_batch_size):
            chunk = searches[start : start + self.request_batch_size]
            payload = "".join(
                json.dumps({"index": index}) + "\n" + json.dumps(body) + "\n"
                for index, body in chunk
            )
            response = self.session.post(
                f"{self.url}/_msearch",
                data=payload.encode("utf-8"),
                headers={"content-type": "application/x-ndjson"},
                timeout=300,
            )
            response.raise_for_status()
            items = response.json().get("responses", [])
            if len(items) != len(chunk):
                raise RuntimeError("Elasticsearch returned an incomplete _msearch response")

            for item in items:
                if "error" in item:
                    raise RuntimeError(f"Elasticsearch search failed: {item['error']}")
                hits = []
                for hit in item["hits"]["hits"]:
                    document = hit.get("_source", {})
                    candidate = SearchHit(
                        document_id=str(document.get("doc_id", hit["_id"])),
                        score=float(hit["_score"]),
                        rank=len(hits) + 1,
                        source=source,
                        title=document.get("title"),
                        text=document.get("text"),
                    )
                    if candidate.has_passage:
                        hits.append(candidate)
                results.append(hits)
        return results
