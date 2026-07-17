import argparse
import time
from pathlib import Path

import requests

from mtrag.config import settings


DOMAINS = ("clapnq", "cloud", "govt", "fiqa")
MODES = ("dense", "sparse")
SNAPSHOT_ROOT = (
    Path(__file__).resolve().parents[1]
    / "artifacts"
    / "elasticsearch"
    / "snapshots"
    / "bge-m3"
)


def es_request(
    method: str,
    path: str,
    body: dict | None = None,
    *,
    timeout: int = 120,
    **kwargs,
) -> dict:
    response = requests.request(
        method,
        settings.elasticsearch_url + path,
        json=body,
        timeout=timeout,
        **kwargs,
    )
    if not response.ok:
        raise RuntimeError(f"{method} {path}: {response.status_code} {response.text}")
    return response.json() if response.content else {}


def wait_for_elasticsearch() -> None:
    for _ in range(60):
        try:
            es_request("GET", "/", timeout=2)
            return
        except (requests.RequestException, RuntimeError):
            time.sleep(2)
    raise RuntimeError("Elasticsearch did not start in 120 seconds")


def validate_snapshot(mode: str, domain: str) -> None:
    directory = SNAPSHOT_ROOT / mode / domain
    has_index = any(
        path.name != "index.latest" for path in directory.glob("index-*")
    )
    if not (directory / "indices").is_dir() or not has_index:
        archive = f"mtrag_bge_m3_{mode}_{domain}.zip"
        raise RuntimeError(f"Extract {archive} into {directory}")


def restore(mode: str, domain: str) -> None:
    index = f"mtrag-{domain}-bge-m3-{mode}"
    if requests.head(f"{settings.elasticsearch_url}/{index}", timeout=10).ok:
        count = es_request("GET", f"/{index}/_count")["count"]
        print(index, "already restored:", count)
        return

    repository = f"mtrag-bge-m3-{mode}-{domain}-repository"
    snapshot = f"mtrag-bge-m3-{mode}-{domain}"
    location = f"/snapshots/bge-m3/{mode}/{domain}"

    es_request(
        "PUT",
        f"/_snapshot/{repository}",
        {
            "type": "fs",
            "settings": {"location": location, "readonly": True},
        },
    )
    es_request("GET", f"/_snapshot/{repository}/{snapshot}")
    es_request(
        "POST",
        f"/_snapshot/{repository}/{snapshot}/_restore",
        {"indices": index, "include_global_state": False},
        timeout=60 * 60,
        params={"wait_for_completion": "true"},
    )
    es_request("DELETE", f"/_snapshot/{repository}")
    print(index, "restored:", es_request("GET", f"/{index}/_count")["count"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--domains", nargs="+", choices=DOMAINS, default=DOMAINS)
    parser.add_argument("--modes", nargs="+", choices=MODES, default=MODES)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for mode in args.modes:
        for domain in args.domains:
            validate_snapshot(mode, domain)
    wait_for_elasticsearch()
    for mode in args.modes:
        for domain in args.domains:
            restore(mode, domain)


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as error:
        raise SystemExit(str(error)) from None
