from mtrag.config import settings
from mtrag.reranking import BgeV2M3Scorer


pairs = [
    (
        "What is IBM Cloud Object Storage?",
        "IBM Cloud Object Storage is a scalable cloud storage service for unstructured data.",
    ),
    (
        "What is IBM Cloud Object Storage?",
        "The Arizona Cardinals are a professional American football team.",
    ),
]

with BgeV2M3Scorer(
    settings.reranker_model_path,
    batch_size=min(settings.reranker_batch_size, 4),
    max_length=settings.reranker_max_length,
) as scorer:
    scores = scorer.score(pairs)

print("relevant:", scores[0])
print("irrelevant:", scores[1])
if scores[0] <= scores[1]:
    raise RuntimeError("Reranker smoke test produced the wrong ordering")
