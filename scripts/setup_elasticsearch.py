import time
from pathlib import Path

import requests

from mtrag.config import settings


SNAPSHOT_ROOT = (
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
    license_info = es_request("GET", "/_license")["license"]
    license_type = license_info["type"]
    if license_info["status"] == "active" and license_type in {
        "trial",
        "platinum",
        "enterprise",
    }:
        print("license:", license_type)
        return

    es_request(
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


def validate_snapshot_files(domain: str) -> None:
    directory = SNAPSHOT_ROOT / domain
    has_index = any(
        path.name != "index.latest" for path in directory.glob("index-*")
    )
    if not (directory / "indices").is_dir() or not has_index:
        archive = f"mtrag_elser_{domain}.zip"
        raise RuntimeError(f"Extract {archive} into {directory}")


def create_elser_endpoint() -> None:
    path = f"/_inference/sparse_embedding/{settings.elser_inference_id}"
    endpoint_exists = requests.get(
        settings.elasticsearch_url + path,
        timeout=10,
    ).ok
    if endpoint_exists:
        probe = requests.post(
            settings.elasticsearch_url + path,
            json={"input": "test query"},
            timeout=120,
        )
        if probe.ok:
            print("ELSER endpoint: ready")
            return
        print("ELSER endpoint: restarting failed deployment")
        es_request("DELETE", path, params={"force": "true"})

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


def restore_snapshot(domain: str) -> None:
    index = f"mtrag-{domain}-elser"
    count_response = requests.get(
        f"{settings.elasticsearch_url}/{index}/_count",
        timeout=10,
    )
    if count_response.ok:
        count = count_response.json()["count"]
        print(index, "already restored:", count)
        return
    if requests.head(f"{settings.elasticsearch_url}/{index}", timeout=10).ok:
        es_request("DELETE", f"/{index}")

    repository = f"mtrag-elser-{domain}-repository"
    snapshot = f"mtrag-elser-{domain}"
    location = f"/snapshots/elser/{domain}"

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


def main() -> None:
    for domain in DOMAINS:
        validate_snapshot_files(domain)
    wait_for_elasticsearch()
    start_trial()
    create_elser_endpoint()
    for domain in DOMAINS:
        restore_snapshot(domain)


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as error:
        raise SystemExit(str(error)) from None
