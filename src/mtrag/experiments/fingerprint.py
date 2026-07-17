import hashlib
import json
from collections.abc import Mapping
from typing import Any


FINGERPRINT_VERSION = 1


def fingerprint(kind: str, inputs: Mapping[str, Any]) -> str:
    """Hash the semantic inputs of one experiment artifact."""

    payload = json.dumps(
        {
            "version": FINGERPRINT_VERSION,
            "kind": kind,
            "inputs": inputs,
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()
