import subprocess

import requests

from mtrag.config import settings


def check_gpu() -> bool:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total,driver_version",
            "--format=csv,noheader",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    print("GPU:", result.stdout.strip() or result.stderr.strip())
    return result.returncode == 0


def check_elasticsearch() -> bool:
    try:
        info = requests.get(settings.elasticsearch_url, timeout=5).json()
        health = requests.get(
            f"{settings.elasticsearch_url}/_cluster/health",
            timeout=5,
        ).json()
        license_info = requests.get(
            f"{settings.elasticsearch_url}/_license",
            timeout=5,
        ).json()
        indices = requests.get(
            f"{settings.elasticsearch_url}/_cat/indices/mtrag-*",
            params={"format": "json", "h": "index,docs.count,store.size"},
            timeout=10,
        ).json()
    except (requests.RequestException, ValueError, KeyError) as error:
        print("Elasticsearch: unavailable:", error)
        return False

    print(
        "Elasticsearch:",
        info["version"]["number"],
        health["status"],
        f"license={license_info['license']['type']}",
    )
    if indices:
        for index in sorted(indices, key=lambda item: item["index"]):
            print(
                " ",
                index["index"],
                f"docs={index['docs.count']}",
                f"size={index['store.size']}",
            )
    else:
        print("  indices: none restored")
    return True


def check_ollama() -> bool:
    try:
        version = requests.get(f"{settings.ollama_url}/api/version", timeout=5).json()
        tags = requests.get(f"{settings.ollama_url}/api/tags", timeout=10).json()
    except (requests.RequestException, ValueError) as error:
        print("Ollama: unavailable:", error)
        return False

    models = {model["name"] for model in tags.get("models", [])}
    model_ready = settings.ollama_model in models
    print(
        "Ollama:",
        version.get("version", "unknown"),
        f"model={settings.ollama_model}",
        "ready" if model_ready else "not pulled",
    )
    return model_ready


checks = (check_gpu(), check_elasticsearch(), check_ollama())
raise SystemExit(0 if all(checks) else 1)
