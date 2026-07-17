import sys

from elasticsearch import Elasticsearch

from mtrag.config import settings


query = " ".join(sys.argv[1:]).strip()
if not query:
    raise SystemExit("usage: smoke_search.py QUERY")

es = Elasticsearch(settings.elasticsearch_url)
response = es.search(
    index="mtrag-*-elser",
    size=5,
    query={
        "semantic": {
            "field": "semantic_text",
            "query": query,
        }
    },
    source=["doc_id", "title", "text", "url"],
)

for hit in response["hits"]["hits"]:
    source = hit["_source"]
    print(f"{hit['_score']:.4f}  {hit['_index']}  {source['doc_id']}")
    print(source.get("title") or source["text"][:160])
