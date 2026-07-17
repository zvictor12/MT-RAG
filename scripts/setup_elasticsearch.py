import time
from pathlib import Path

import requests

from mtrag.config import settings


SNAPSHOT_REPOSITORY = "mtrag-repository"
SNAPSHOT_NAME = "mtrag-elser"
SNAPSHOT_LOCATION = "/snapshots/elser"
SNAPSHOT_DIR = (
    Path(__file__).resolve().parents[1]
    / "artifacts"
    / "elasticsearch"
    / "snapshots"
    / "elser"
)
MODEL_ID = ".elser_model_2_linux-x86_64"
DOMAINS = ("clapnq", "cloud", "govt", "fiqa")


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


def start_trial() -> None:
    result = es_request(
        "POST",
        "/_license/start_trial",
        params={"acknowledge": "true"},
    )
    license_info = es_request("GET", "/_license")["license"]
    license_type = license_info["type"]
    if license_info["status"] != "active" or license_type not in {
        "trial",
        "platinum",
        "enterprise",
    }:
        raise RuntimeError(f"ELSER license is not active: {license_info}")
    print("license:", license_type)


def validate_snapshot_files() -> None:
    if not (SNAPSHOT_DIR / "indices").is_dir():
        raise RuntimeError(
            f"ELSER snapshot is missing in {SNAPSHOT_DIR}. "
            "Extract mtrag_elser_snapshot.zip there first."
        )
    if not any(
        path.name != "index.latest" for path in SNAPSHOT_DIR.glob("index-*")
    ):
        raise RuntimeError(f"No Elasticsearch repository index found in {SNAPSHOT_DIR}")


def register_snapshot_repository() -> None:
    es_request(
        "PUT",
        f"/_snapshot/{SNAPSHOT_REPOSITORY}",
        {
            "type": "fs",
            "settings": {"location": SNAPSHOT_LOCATION, "readonly": True},
        },
    )
    es_request("GET", f"/_snapshot/{SNAPSHOT_REPOSITORY}/{SNAPSHOT_NAME}")
    print("ELSER snapshot: verified")


def create_elser_endpoint() -> None:
    path = f"/_inference/sparse_embedding/{settings.elser_inference_id}"
    if requests.get(settings.elasticsearch_url + path, timeout=10).ok:
        print("ELSER endpoint: already exists")
    else:
        es_request(
            "PUT",
            path,
            {
                "service": "elasticsearch",
                "service_settings": {
                    "model_id": MODEL_ID,
                    "num_threads": 2,
                    "adaptive_allocations": {
                        "enabled": True,
                        "min_number_of_allocations": 1,
                        "max_number_of_allocations": 1,
                    },
                },
                "chunking_settings": {"strategy": "none"},
            },
            timeout=31 * 60,
            params={"timeout": "30m"},
        )

    last_error = ""
    for _ in range(120):
        response = requests.post(
            settings.elasticsearch_url + path,
            json={"input": "test query"},
            timeout=120,
        )
        if response.ok:
            print("ELSER endpoint: ready")
            return
        last_error = f"{response.status_code} {response.text}"
        time.sleep(15)
    raise RuntimeError(f"ELSER model did not become ready: {last_error}")


def restore_snapshot() -> None:
    indices = [f"mtrag-{domain}-elser" for domain in DOMAINS]
    missing = [
        index
        for index in indices
        if not requests.head(f"{settings.elasticsearch_url}/{index}", timeout=10).ok
    ]
    if not missing:
        print("indices: already restored")
        return

    es_request(
        "POST",
        f"/_snapshot/{SNAPSHOT_REPOSITORY}/{SNAPSHOT_NAME}/_restore",
        {
            "indices": ",".join(missing),
            "include_global_state": False,
        },
        timeout=60 * 60,
        params={"wait_for_completion": "true"},
    )

    for index in indices:
        count = es_request("GET", f"/{index}/_count")["count"]
        print(index, count)


def main() -> None:
    validate_snapshot_files()
    wait_for_elasticsearch()
    register_snapshot_repository()
    start_trial()
    create_elser_endpoint()
    restore_snapshot()


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as error:
        raise SystemExit(str(error)) from None
