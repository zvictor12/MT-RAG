from mtrag.config import settings
from mtrag.encoding import BGEM3QueryEncoder
from mtrag.retrieval.bge import DenseRetriever, SparseRetriever
from mtrag.retrieval.elasticsearch import ElasticsearchGateway
from mtrag.schemas import SearchQuery


query_text = "What is IBM Cloud Object Storage?"

with BGEM3QueryEncoder(
    settings.bge_model_path,
    batch_size=min(settings.bge_batch_size, 4),
    max_length=settings.bge_max_length,
) as encoder:
    features = encoder.encode([query_text])[0]

print("dense dimensions:", len(features.dense))
print("sparse non-zero features:", len(features.sparse))

query = SearchQuery(
    task_id="smoke",
    domain="cloud",
    text=query_text,
    bge=features,
)
gateway = ElasticsearchGateway(settings.elasticsearch_url)

for name, retriever in (
    ("dense", DenseRetriever(gateway)),
    ("sparse", SparseRetriever(gateway)),
):
    hit = retriever.search_many([query], top_k=1)["smoke"][0]
    print(f"{name}: {hit.document_id}  score={hit.score:.6f}")
    print(hit.title or (hit.text or "")[:160])
