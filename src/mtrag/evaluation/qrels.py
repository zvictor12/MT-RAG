"""Load the canonical MT-RAG Task A relevance judgements."""

from __future__ import annotations

import csv
import re
from collections.abc import Iterable
from pathlib import Path


DOMAINS = ("clapnq", "cloud", "govt", "fiqa")

COLLECTION_TO_DOMAIN = {
    "mt-rag-clapnq-elser-512-100-20240503": "clapnq",
    "mt-rag-ibmcloud-elser-512-100-20240502": "cloud",
    "mt-rag-govt-elser-512-100-20240611": "govt",
    "mt-rag-fiqa-beir-elser-512-100-20240501": "fiqa",
}

_DOMAIN_ALIASES = {
    "clapnq": "clapnq",
    "cloud": "cloud",
    "ibmcloud": "cloud",
    "govt": "govt",
    "fiqa": "fiqa",
}
_LEGACY_TASK_ID = re.compile(r"^(?P<prefix>.+)::(?P<turn>\d+)$")

Qrels = dict[str, dict[str, int]]


def normalize_task_id(task_id: str) -> str:
    """Convert only a trailing legacy ``::turn`` delimiter to ``<::>``.

    A global string replacement would corrupt an identifier containing ``::``
    elsewhere, so only the final numeric turn separator is considered.
    """
    if not isinstance(task_id, str) or not task_id:
        raise ValueError("task_id must be a non-empty string")
    if "<::>" in task_id:
        return task_id

    match = _LEGACY_TASK_ID.fullmatch(task_id)
    if match is None:
        return task_id
    return f"{match.group('prefix')}<::>{match.group('turn')}"


def normalize_domain(value: str) -> str:
    """Return the benchmark domain for a domain or collection name."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError("domain or collection must be a non-empty string")

    normalized = value.strip().lower()
    if normalized in COLLECTION_TO_DOMAIN:
        return COLLECTION_TO_DOMAIN[normalized]
    if normalized in _DOMAIN_ALIASES:
        return _DOMAIN_ALIASES[normalized]

    tokens = set(filter(None, re.split(r"[^a-z0-9]+", normalized)))
    matches = {
        domain
        for alias, domain in _DOMAIN_ALIASES.items()
        if alias in tokens
    }
    if len(matches) == 1:
        return matches.pop()
    raise ValueError(f"unknown MT-RAG domain or collection: {value!r}")


def qrels_path(benchmark_root: str | Path, domain: str) -> Path:
    """Build the current, canonical qrels path in mt-rag-benchmark."""
    canonical_domain = normalize_domain(domain)
    return (
        Path(benchmark_root)
        / "mtrag-human"
        / "retrieval_tasks"
        / canonical_domain
        / "qrels"
        / "dev.tsv"
    )


def load_qrels(path: str | Path) -> Qrels:
    """Read a TREC-style TSV qrels file and canonicalize its task IDs."""
    qrels: Qrels = {}
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        required = {"query-id", "corpus-id", "score"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(
                f"qrels must contain columns {sorted(required)}; "
                f"got {reader.fieldnames}"
            )

        for line_number, row in enumerate(reader, start=2):
            query_id = normalize_task_id(row["query-id"])
            document_id = row["corpus-id"]
            if not document_id:
                raise ValueError(f"empty corpus-id at {path}:{line_number}")
            try:
                relevance = int(row["score"])
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"invalid relevance at {path}:{line_number}"
                ) from error

            documents = qrels.setdefault(query_id, {})
            existing = documents.get(document_id)
            if existing is not None and existing != relevance:
                raise ValueError(
                    f"conflicting qrel for {query_id!r}/{document_id!r}"
                )
            documents[document_id] = relevance

    return qrels


def load_benchmark_qrels(
    benchmark_root: str | Path,
    domains: Iterable[str] = DOMAINS,
) -> dict[str, Qrels]:
    """Load qrels for each requested domain from the benchmark repository."""
    loaded: dict[str, Qrels] = {}
    for domain in domains:
        canonical_domain = normalize_domain(domain)
        if canonical_domain in loaded:
            raise ValueError(f"duplicate domain: {canonical_domain}")
        loaded[canonical_domain] = load_qrels(
            qrels_path(benchmark_root, canonical_domain)
        )
    return loaded
